"""The webcam web server: one browser session drives the animation pipeline.

build_app() wires the HTTP API, the WebSocket and the static page around a
pipeline object (webcam/vid2vid.py or vid2vid_trt.py). It holds no model code,
so tests can run it with a stand-in pipeline.

Routes (all /api routes except POST /api/session need a session cookie):
  POST /api/session                open a session with the access key
  POST /api/logout                 close it
  GET  /api/settings               frame size and queue limit for the page
  GET  /api/queue                  how many sessions are connected
  GET  /api/preferences            the page's appearance (webcam/preferences.py)
  PUT  /api/preferences            change some of it; answers the stored object
  POST /api/upload_reference_image the reference portrait (multipart, field ref_image)
  WS   /api/ws/{user_id}           webcam frames in, animated frames out
  GET  /                           the page (webcam/static/)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from webcam import preferences, security
from webcam.connection_manager import ConnectionManager, ServerFullException
from webcam.util import pil_to_frame

STATIC_DIR = Path(__file__).resolve().parent / "static"
FRAME_SIZE = (512, 512)
THROTTLE = 0.001
MAX_SESSION_BODY = 4096
MAX_PREFS_BODY = 4096


def frame_to_tensor(img):
    """RGB PIL image -> HxWx3 uint8 tensor, as the pipelines expect."""
    import numpy as np
    import torch

    if img.size != FRAME_SIZE:
        img = img.resize(FRAME_SIZE)
    return torch.from_numpy(np.array(img, dtype=np.uint8))


def build_app(config, pipeline, guard: security.AccessGuard, secure: bool,
              prefs: preferences.PreferenceStore | None = None) -> FastAPI:
    """Return the FastAPI app. `secure` is True when the server speaks TLS.
    Without `prefs` the appearance is kept in memory only."""
    prefs = prefs or preferences.PreferenceStore(None)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(security.SecurityHeadersMiddleware, secure=secure)
    conn_manager = ConnectionManager()
    state = SimpleNamespace(stop_event=None, task=None)
    max_users = max(1, int(config.max_queue_size or 1))

    def session_of(request: Request) -> str:
        token = request.cookies.get(security.SESSION_COOKIE)
        if not guard.valid(token):
            raise HTTPException(status_code=401, detail="Sign in with the access key")
        return token

    def check_origin(request: Request) -> None:
        if not security.same_origin(request.headers.get("origin"),
                                    request.headers.get("host"), secure):
            raise HTTPException(status_code=403, detail="Cross-origin request refused")

    @app.post("/api/session")
    async def open_session(request: Request):
        check_origin(request)
        client = request.client.host if request.client else "unknown"
        if guard.locked_out(client):
            raise HTTPException(status_code=429, detail="Too many wrong keys; wait a minute")
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_SESSION_BODY:
                raise HTTPException(status_code=413, detail="Request too large")
        try:
            offered = json.loads(body or b"{}").get("key", "")
        except (ValueError, AttributeError):
            offered = ""
        if not isinstance(offered, str):
            offered = ""
        token = guard.open_session(offered, client)
        if token is None:
            raise HTTPException(status_code=401, detail="Wrong access key")
        response = Response(status_code=204)
        response.headers["set-cookie"] = security.cookie_header(
            token, secure, max_age=12 * 3600)
        return response

    @app.post("/api/logout")
    async def logout(request: Request):
        check_origin(request)
        guard.close_session(request.cookies.get(security.SESSION_COOKIE))
        response = Response(status_code=204)
        response.headers["set-cookie"] = security.cookie_header("", secure, max_age=0)
        return response

    @app.get("/api/settings")
    async def settings(request: Request):
        session_of(request)
        return JSONResponse({"width": FRAME_SIZE[0], "height": FRAME_SIZE[1],
                             "max_queue_size": max_users})

    @app.get("/api/queue")
    async def queue_size(request: Request):
        session_of(request)
        return JSONResponse({"queue_size": conn_manager.get_user_count()})

    @app.get("/api/preferences")
    async def get_preferences(request: Request):
        session_of(request)
        return JSONResponse(prefs.get())

    @app.put("/api/preferences")
    async def put_preferences(request: Request):
        check_origin(request)
        session_of(request)
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_PREFS_BODY:
                raise HTTPException(status_code=413, detail="Request too large")
        try:
            change = json.loads(body)
        except ValueError:
            raise HTTPException(status_code=400, detail="Send a JSON object") from None
        try:
            stored = await asyncio.to_thread(prefs.update, change)
        except preferences.PreferenceError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        except OSError:
            logging.exception("Could not save the preferences")
            raise HTTPException(status_code=500, detail="Could not save the preferences") from None
        return JSONResponse(stored)

    @app.post("/api/upload_reference_image")
    async def upload_reference_image(request: Request):
        session_of(request)
        check_origin(request)
        length = request.headers.get("content-length")
        if length is None or not length.isdigit():
            raise HTTPException(status_code=411, detail="Content-Length required")
        if int(length) > security.MAX_UPLOAD_BYTES + 65536:
            raise HTTPException(status_code=413, detail="Image too large")
        try:
            form = await request.form(max_files=1, max_fields=1,
                                      max_part_size=security.MAX_UPLOAD_BYTES)
        except Exception:
            raise HTTPException(status_code=400, detail="Malformed upload") from None
        upload = form.get("ref_image")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status_code=400, detail="Field ref_image is missing")
        data = await upload.read(security.MAX_UPLOAD_BYTES + 1)
        try:
            img = security.decode_image(data, security.UPLOAD_FORMATS,
                                        security.MAX_UPLOAD_BYTES)
        except security.ImageRejected as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        try:
            pipeline.fuse_reference(img)
        except Exception:
            logging.exception("reference image failed")
            raise HTTPException(status_code=500, detail="Failed to process reference image")
        return {"status": "ok"}

    @app.websocket("/api/ws/{user_id}")
    async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
        headers = websocket.scope["headers"]
        if not (security.same_origin(security.header(headers, b"origin"),
                                     security.header(headers, b"host"), secure)
                and guard.valid(security.read_cookie(headers))):
            await websocket.close(code=1008)
            return
        sender_task = None
        try:
            await conn_manager.connect(user_id, websocket, max_users)
            sender_task = asyncio.create_task(push_results_to_client(user_id, websocket))
            if state.task is None or state.task.done():
                start_prediction_thread(user_id)
            await handle_websocket_input(user_id, websocket)
        except ServerFullException as e:
            logging.info("refused a session: %s", e)
        except WebSocketDisconnect:
            logging.info("user disconnected: %s", user_id)
        except Exception:
            logging.exception("websocket error")
        finally:
            if sender_task is not None:
                sender_task.cancel()
            await conn_manager.disconnect(user_id, pipeline)
            if state.stop_event is not None:
                state.stop_event.set()

    async def handle_websocket_input(user_id: uuid.UUID, websocket: WebSocket):
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if message.get("text") is not None:
                try:
                    status = json.loads(message["text"]).get("status")
                except (ValueError, AttributeError):
                    continue
                if status == "pause":
                    await conn_manager.update_data(user_id, SimpleNamespace(restart=True))
                elif status == "resume":
                    await conn_manager.send_json(user_id, {"status": "send_frame"})
            elif message.get("bytes"):
                try:
                    img = security.decode_image(message["bytes"], security.FRAME_FORMATS,
                                                security.MAX_FRAME_BYTES)
                except security.ImageRejected:
                    continue
                pipeline.accept_new_params(SimpleNamespace(image=frame_to_tensor(img)))

    async def push_results_to_client(user_id: uuid.UUID, websocket: WebSocket):
        min_fps, max_fps, smoothing = 10, 30, 0.8
        last_burst_time = time.time()
        last_queue_size = 0
        sleep_time = ema_interval = 1 / 40
        try:
            while True:
                size = await conn_manager.get_output_queue_size(user_id)
                if size > last_queue_size:
                    now = time.time()
                    elapsed = now - last_burst_time
                    if size > 0 and elapsed > 0:
                        ema_interval = smoothing * ema_interval + (1 - smoothing) * elapsed / size
                        sleep_time = min(max(ema_interval, 1 / max_fps), 1 / min_fps)
                    last_burst_time = now
                last_queue_size = size
                frame = await conn_manager.get_frame(user_id)
                if frame is None:
                    await asyncio.sleep(0.001)
                    continue
                await websocket.send_bytes(frame)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            pass
        except Exception:
            logging.exception("sending results failed")

    def start_prediction_thread(user_id):
        state.stop_event = threading.Event()

        def prediction_loop(uid, loop, stop_event):
            while not stop_event.is_set():
                images = pipeline.produce_outputs()
                if not images:
                    time.sleep(THROTTLE)
                    continue
                frames = [pil_to_frame(i) for i in images]
                asyncio.run_coroutine_threadsafe(
                    conn_manager.put_frames_to_output_queue(uid, frames), loop)

        state.task = asyncio.create_task(asyncio.to_thread(
            prediction_loop, user_id, asyncio.get_running_loop(), state.stop_event))

    async def cleanup():
        if state.stop_event is not None:
            state.stop_event.set()
        if state.task is not None:
            state.task.cancel()
            try:
                await state.task
            except (asyncio.CancelledError, Exception):
                pass
        await conn_manager.disconnect_all(pipeline)

    app.state.cleanup = cleanup
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
