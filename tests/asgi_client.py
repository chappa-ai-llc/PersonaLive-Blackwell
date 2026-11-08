"""A minimal ASGI test client (stdlib only), so the tests need no httpx."""

import asyncio
import json as jsonlib


class Response:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = {}
        self.cookies = []
        for k, v in headers:
            k = k.decode().lower()
            v = v.decode()
            if k == "set-cookie":
                self.cookies.append(v)
            self.headers[k] = v
        self.body = body

    def json(self):
        return jsonlib.loads(self.body)


def request(app, method, path, headers=None, body=b"", client=("127.0.0.1", 50000)):
    async def run():
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method, "path": path, "raw_path": path.encode(),
            "query_string": b"", "scheme": "http", "root_path": "",
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "client": client, "server": ("127.0.0.1", 7860),
        }
        sent = {"done": False}

        async def receive():
            if sent["done"]:
                await asyncio.sleep(3600)
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}

        status, resp_headers, chunks = None, [], []

        async def send(message):
            nonlocal status, resp_headers
            if message["type"] == "http.response.start":
                status = message["status"]
                resp_headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        await app(scope, receive, send)
        return Response(status, resp_headers, b"".join(chunks))

    return asyncio.run(run())


def websocket_handshake(app, path, headers=None):
    """Open a WebSocket and return the first message the app sends:
    'websocket.accept' or 'websocket.close' (a refused handshake)."""
    async def run():
        scope = {
            "type": "websocket", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "path": path, "raw_path": path.encode(), "query_string": b"",
            "scheme": "ws", "root_path": "", "subprotocols": [],
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "client": ("127.0.0.1", 50001), "server": ("127.0.0.1", 7860),
        }
        queue = asyncio.Queue()
        await queue.put({"type": "websocket.connect"})
        first = asyncio.get_running_loop().create_future()

        async def receive():
            return await queue.get()

        async def send(message):
            if not first.done():
                first.set_result(message)

        task = asyncio.create_task(app(scope, receive, send))
        message = await asyncio.wait_for(first, 10)
        await queue.put({"type": "websocket.disconnect", "code": 1000})
        try:
            await asyncio.wait_for(task, 10)
        except Exception:
            task.cancel()
        return message["type"]

    return asyncio.run(run())
