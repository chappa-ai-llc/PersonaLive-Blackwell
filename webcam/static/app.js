// PersonaLive webcam page. Plain JavaScript, no build step and no third-party
// code: the server's Content-Security-Policy allows only this file.
"use strict";

const $ = (id) => document.getElementById(id);
const SIZE = 512;

const state = {
  ws: null,
  status: "disconnected", // connected | wait | send_frame | paused | disconnected
  stream: null,
  frameBlob: null,
  refBlob: null,
  refSent: false,
  running: false,
  pusher: null,
  capturing: false,
  lastFrameUrl: null,
};

function warn(text) {
  $("warning").textContent = text;
  $("warning").hidden = !text;
  if (text) info("");
}

function info(text) {
  $("info").textContent = text;
  $("info").hidden = !text;
}

async function api(path, options = {}) {
  const res = await fetch(path, { credentials: "same-origin", ...options });
  if (res.status === 401) {
    showSignin();
    throw new Error("Sign in with the access key.");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new Error(detail);
  }
  return res;
}

// ---- Sign-in -------------------------------------------------------------

function showSignin() {
  $("app").hidden = true;
  $("signin").hidden = false;
  $("key").focus();
}

async function signIn(key) {
  const res = await fetch("/api/session", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  if (res.status === 204) return true;
  let detail = "Sign-in failed.";
  try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
  warn(detail);
  return false;
}

async function start() {
  // The printed address carries the key in the fragment (#key=...), which the
  // browser never sends to a server. Use it once, then remove it.
  const m = location.hash.match(/^#key=([A-Za-z0-9_\-]+)$/);
  if (m) {
    history.replaceState(null, "", location.pathname + location.search);
    await signIn(m[1]);
  }
  try {
    const settings = await (await api("/api/settings")).json();
    if (settings.width !== SIZE || settings.height !== SIZE) {
      warn(`Unexpected frame size ${settings.width}x${settings.height}.`);
    }
    $("signin").hidden = true;
    $("app").hidden = false;
  } catch (e) {
    if ($("signin").hidden) warn(e.message);
  }
}

$("signin").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const key = $("key").value.trim();
  if (await signIn(key)) {
    $("key").value = "";
    warn("");
    await start();
  }
});

// ---- Reference portrait --------------------------------------------------

function cropSquare(source, width, height) {
  const canvas = $("work");
  const ctx = canvas.getContext("2d");
  const side = Math.min(width, height);
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.drawImage(source, (width - side) / 2, (height - side) / 2, side, side, 0, 0, SIZE, SIZE);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
}

$("refFile").addEventListener("change", async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  try {
    await img.decode();
  } catch (_) {
    URL.revokeObjectURL(url);
    warn("That file is not an image this browser can read.");
    return;
  }
  state.refBlob = await cropSquare(img, img.naturalWidth, img.naturalHeight);
  URL.revokeObjectURL(url);
  const preview = $("refPreview");
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(state.refBlob);
  preview.hidden = false;
  $("refEmpty").hidden = true;
  state.refSent = false;
  $("fuse").disabled = false;
});

$("fuse").addEventListener("click", async () => {
  if (!state.refBlob) {
    warn("Choose a portrait first.");
    return;
  }
  const form = new FormData();
  form.append("ref_image", state.refBlob, "reference.jpg");
  $("fuse").disabled = true;
  try {
    await api("/api/upload_reference_image", { method: "POST", body: form });
    state.refSent = true;
    warn("");
    info("Reference portrait fused.");
  } catch (e) {
    $("fuse").disabled = false;
    warn(e.message);
  }
});

// ---- Camera --------------------------------------------------------------

async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === "videoinput");
  const select = $("devices");
  select.replaceChildren();
  cams.forEach((cam, i) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Camera ${i + 1}`;
    select.append(opt);
  });
  select.hidden = cams.length === 0;
}

function useStream(stream) {
  stopStream();
  state.stream = stream;
  const video = $("camera");
  video.srcObject = stream;
  $("camEmpty").hidden = true;
  if (!state.capturing) {
    state.capturing = true;
    video.requestVideoFrameCallback ? video.requestVideoFrameCallback(captureFrame)
      : requestAnimationFrame(captureFrame);
  }
}

function stopStream() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  $("camera").srcObject = null;
}

async function startCamera(deviceId) {
  const video = { width: SIZE, height: SIZE };
  if (deviceId) video.deviceId = { exact: deviceId };
  useStream(await navigator.mediaDevices.getUserMedia({ audio: false, video }));
  await listCameras();
}

async function captureFrame() {
  const video = $("camera");
  if (!state.stream) {
    state.capturing = false;
    return;
  }
  if (video.videoWidth > 0) {
    const canvas = $("work");
    const ctx = canvas.getContext("2d");
    const w = video.videoWidth;
    const h = video.videoHeight;
    const side = Math.min(w, h);
    ctx.save();
    ctx.scale(-1, 1); // mirror, as the user sees themselves
    ctx.translate(-SIZE, 0);
    ctx.drawImage(video, (w - side) / 2, (h - side) / 2, side, side, 0, 0, SIZE, SIZE);
    ctx.restore();
    state.frameBlob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.9));
  }
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(captureFrame);
  else requestAnimationFrame(captureFrame);
}

$("devices").addEventListener("change", (ev) => {
  if (state.stream) startCamera(ev.target.value).catch((e) => warn(e.message));
});

$("share").addEventListener("click", async () => {
  try {
    useStream(await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false }));
  } catch (e) {
    warn(`Screen sharing failed: ${e.message}`);
  }
});

// ---- Animation stream ----------------------------------------------------

function connect() {
  return new Promise((resolve, reject) => {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${location.host}/api/ws/${crypto.randomUUID()}`);
    ws.binaryType = "blob";
    state.ws = ws;
    ws.onclose = () => {
      state.status = "disconnected";
      if (state.running) {
        warn("The connection to the server closed.");
        stopAnimation();
      }
      reject(new Error("The server refused the connection."));
    };
    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        drawFrame(event.data);
        return;
      }
      let data;
      try { data = JSON.parse(event.data); } catch (_) { return; }
      switch (data.status) {
        case "connected":
          state.status = "connected";
          resolve();
          break;
        case "send_frame":
        case "wait":
          if (state.status !== "paused") state.status = data.status;
          break;
        case "timeout":
        case "error":
          state.status = "disconnected";
          warn(data.message || "The session ended.");
          reject(new Error(data.message || data.status));
          break;
      }
    };
  });
}

async function drawFrame(blob) {
  const canvas = $("output");
  try {
    const bitmap = await createImageBitmap(blob);
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
    }
    canvas.getContext("2d", { alpha: false }).drawImage(bitmap, 0, 0);
    bitmap.close();
    $("outEmpty").hidden = true;
  } catch (_) { /* skip a frame that does not decode */ }
}

function pushFrames() {
  const fps = Number($("fps").value);
  const ws = state.ws;
  if (ws && ws.readyState === WebSocket.OPEN && state.frameBlob &&
      (state.status === "connected" || state.status === "send_frame")) {
    ws.send(state.frameBlob);
  }
  state.pusher = setTimeout(pushFrames, 1000 / fps);
}

async function startAnimation() {
  if (!state.refSent) {
    warn("Fuse a reference portrait first.");
    return;
  }
  $("toggle").disabled = true;
  try {
    if (!state.stream) await startCamera($("devices").value || undefined);
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.status = "connected";
      state.ws.send(JSON.stringify({ status: "resume" }));
    } else {
      await connect();
    }
    state.running = true;
    $("outEmpty").textContent = "Waiting for frames...";
    $("snapshot").disabled = false;
    $("toggle").textContent = "Stop animation";
    warn("");
    pushFrames();
  } catch (e) {
    warn(e.message);
    stopAnimation();
  } finally {
    $("toggle").disabled = false;
  }
}

function stopAnimation() {
  state.running = false;
  clearTimeout(state.pusher);
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ status: "pause" }));
  }
  state.status = "paused";
  stopStream();
  $("camEmpty").hidden = false;
  $("snapshot").disabled = true;
  $("toggle").textContent = "Start animation";
}

$("toggle").addEventListener("click", () => (state.running ? stopAnimation() : startAnimation()));

$("fps").addEventListener("input", (ev) => { $("fpsValue").textContent = ev.target.value; });

$("snapshot").addEventListener("click", () => {
  $("output").toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `personalive-${Date.now()}.jpg`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, "image/jpeg", 0.95);
});

start();
