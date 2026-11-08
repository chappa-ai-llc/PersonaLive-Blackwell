"""Security controls for the webcam server (inference_online.py).

Everything here is plain Python with no GPU or model imports, so the tests in
tests/ can check it on any machine:

- AccessGuard: the access key and the browser sessions it opens. Every API
  call and the WebSocket need a session; a session is a random token in an
  HttpOnly, SameSite=Strict cookie, opened by POST /api/session with the key.
- SecurityHeadersMiddleware: security headers on every HTTP response.
- same_origin: the Origin check for state-changing requests and the WebSocket.
- decode_image: bounded, format-restricted image decoding for uploads and
  webcam frames.
"""

from __future__ import annotations

import hmac
import io
import os
import secrets
import stat
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit

from PIL import Image

SESSION_COOKIE = "personalive_session"
KEY_ENV = "PERSONALIVE_ACCESS_KEY_FILE"
MIN_KEY_LENGTH = 32

# Image limits. Webcam frames are 512x512 JPEGs of about 50-150 kB.
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
FRAME_FORMATS = ("JPEG",)
UPLOAD_FORMATS = ("JPEG", "PNG", "WEBP")


class AccessKeyError(Exception):
    """The access key file is missing, unreadable, too short or too open."""


def load_access_key(path: Optional[str]) -> tuple[str, bool]:
    """Return (key, generated).

    With a path (the --access-key-file option, or PERSONALIVE_ACCESS_KEY_FILE,
    the *_FILE convention Docker secrets use), the key is read from that file,
    which must not be readable by group or others. Without one, a new random
    key is made for this run only and never written to disk.
    """
    if not path:
        return secrets.token_urlsafe(32), True
    p = Path(path)
    try:
        st = p.stat()
    except OSError as e:
        raise AccessKeyError(f"cannot read the access key file {p}: {e.strerror}") from None
    if os.name == "posix" and st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise AccessKeyError(
            f"the access key file {p} is readable by other users; run: chmod 600 {p}")
    key = p.read_text(encoding="utf-8").strip()
    if len(key) < MIN_KEY_LENGTH:
        raise AccessKeyError(
            f"the access key in {p} is shorter than {MIN_KEY_LENGTH} characters")
    return key, False


class AccessGuard:
    """Checks the access key and keeps the sessions it opened (in memory).

    Wrong keys are counted per client address: after max_failures within
    failure_window seconds, that client is refused until the window passes.
    """

    def __init__(self, key: str, session_ttl: float = 12 * 3600,
                 max_failures: int = 10, failure_window: float = 60.0):
        self._key = key.encode("utf-8")
        self._ttl = session_ttl
        self._sessions: dict[str, float] = {}
        self._failures: dict[str, deque[float]] = {}
        self._max_failures = max_failures
        self._window = failure_window
        self._lock = threading.Lock()

    def locked_out(self, client: str, now: Optional[float] = None) -> bool:
        """True while this client sent too many wrong keys within the window."""
        now = time.monotonic() if now is None else now
        with self._lock:
            for addr in list(self._failures):
                times = self._failures[addr]
                while times and now - times[0] > self._window:
                    times.popleft()
                if not times:
                    del self._failures[addr]
            return len(self._failures.get(client, ())) >= self._max_failures

    def open_session(self, offered: str, client: str,
                     now: Optional[float] = None) -> Optional[str]:
        """Return a new session token when the offered key is right, else None."""
        now = time.monotonic() if now is None else now
        if self.locked_out(client, now):
            return None
        if not hmac.compare_digest(offered.encode("utf-8"), self._key):
            with self._lock:
                self._failures.setdefault(client, deque()).append(now)
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions = {t: exp for t, exp in self._sessions.items() if exp > now}
            self._sessions[token] = now + self._ttl
        return token

    def valid(self, token: Optional[str], now: Optional[float] = None) -> bool:
        if not token:
            return False
        now = time.monotonic() if now is None else now
        with self._lock:
            expires = self._sessions.get(token)
            if expires is None:
                return False
            if expires <= now:
                del self._sessions[token]
                return False
            return True

    def close_session(self, token: Optional[str]) -> None:
        if token:
            with self._lock:
                self._sessions.pop(token, None)


def cookie_header(token: str, secure: bool, max_age: int) -> str:
    attrs = [f"{SESSION_COOKIE}={token}", "Path=/", "HttpOnly", "SameSite=Strict",
             f"Max-Age={max_age}"]
    if secure:
        attrs.append("Secure")
    return "; ".join(attrs)


def read_cookie(headers: Iterable[tuple[bytes, bytes]]) -> Optional[str]:
    for name, value in headers:
        if name.lower() != b"cookie":
            continue
        for part in value.decode("latin-1").split(";"):
            k, _, v = part.strip().partition("=")
            if k == SESSION_COOKIE:
                return v
    return None


def header(headers: Iterable[tuple[bytes, bytes]], wanted: bytes) -> Optional[str]:
    for name, value in headers:
        if name.lower() == wanted:
            return value.decode("latin-1")
    return None


def same_origin(origin: Optional[str], host: Optional[str], secure: bool) -> bool:
    """True when the Origin header names this server (scheme, host and port).

    Browsers always send Origin on WebSocket handshakes and on cross-origin or
    non-GET fetches, so a missing or foreign Origin is refused.
    """
    if not origin or not host or origin == "null":
        return False
    parts = urlsplit(origin)
    expected_scheme = "https" if secure else "http"
    return parts.scheme == expected_scheme and parts.netloc.lower() == host.lower()


# Content-Security-Policy: only this server's own files and connections.
# 'self' covers same-origin ws:/wss: (CSP Level 3); blob: carries the frames.
CSP = "; ".join([
    "default-src 'none'",
    "script-src 'self'",
    "style-src 'self'",
    "img-src 'self' blob: data:",
    "media-src 'self' blob: mediastream:",
    "connect-src 'self'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


def security_headers(secure: bool, api: bool) -> list[tuple[bytes, bytes]]:
    h = [
        (b"content-security-policy", CSP.encode()),
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        # same-origin, not no-referrer: with no-referrer browsers send
        # "Origin: null" on same-origin POSTs and the Origin check refuses them.
        (b"referrer-policy", b"same-origin"),
        (b"permissions-policy",
         b"camera=(self), display-capture=(self), microphone=(), geolocation=(), "
         b"usb=(), payment=()"),
        (b"cross-origin-opener-policy", b"same-origin"),
        (b"cross-origin-resource-policy", b"same-origin"),
    ]
    if secure:
        h.append((b"strict-transport-security", b"max-age=31536000"))
    if api:
        h.append((b"cache-control", b"no-store"))
    return h


class SecurityHeadersMiddleware:
    """Pure ASGI middleware: adds security_headers() to every HTTP response."""

    def __init__(self, app, secure: bool):
        self.app = app
        self.secure = secure

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        extra = security_headers(self.secure, scope["path"].startswith("/api/"))
        names = {n for n, _ in extra}

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                kept = [(n, v) for n, v in message.get("headers", [])
                        if n.lower() not in names and n.lower() != b"server"]
                message = dict(message, headers=kept + extra)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class ImageRejected(ValueError):
    """The bytes are not an acceptable image."""


def decode_image(data: bytes, formats: tuple[str, ...], max_bytes: int,
                 max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    """Decode an image with bounded size and only the listed formats.

    The header is checked before any pixel is decoded, so an oversized or
    decompression-bomb image is refused without allocating it. Returns an RGB
    image.
    """
    if not data:
        raise ImageRejected("empty image")
    if len(data) > max_bytes:
        raise ImageRejected(f"image larger than {max_bytes} bytes")
    try:
        img = Image.open(io.BytesIO(data), formats=list(formats))
    except Exception:
        raise ImageRejected("not a supported image") from None
    w, h = img.size
    if w < 1 or h < 1 or w > max_side or h > max_side:
        raise ImageRejected(f"image sides must be 1 to {max_side} pixels")
    try:
        img.load()
        return img.convert("RGB")
    except Exception:
        raise ImageRejected("image data is damaged") from None


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")
