"""Tests for the webcam server's security controls (webcam/security.py and
webcam/server.py), with a stand-in pipeline: no GPU or weights needed.

Run from the repo root:  python -m unittest discover -s tests -v
"""

import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from asgi_client import request, websocket_handshake  # noqa: E402
from webcam import security  # noqa: E402
from webcam.server import build_app  # noqa: E402

KEY = "k" * 40
ORIGIN = "http://127.0.0.1:7860"
HOST = "127.0.0.1:7860"


class FakePipeline:
    def __init__(self):
        self.references = []
        self.params = []

    def fuse_reference(self, img):
        self.references.append(img)

    def accept_new_params(self, params):
        self.params.append(params)

    def produce_outputs(self):
        return []

    def close(self):
        pass


def jpeg(size=(64, 64), fmt="JPEG"):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 100, 50)).save(buf, format=fmt)
    return buf.getvalue()


def multipart(field, filename, data, content_type="image/jpeg"):
    boundary = "testboundary1234"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = FakePipeline()
        self.guard = security.AccessGuard(KEY)
        config = SimpleNamespace(max_queue_size=1)
        self.app = build_app(config, self.pipeline, self.guard, secure=False)

    def sign_in(self, key=KEY, origin=ORIGIN, client=("127.0.0.1", 50000)):
        headers = {"host": HOST, "content-type": "application/json"}
        if origin:
            headers["origin"] = origin
        return request(self.app, "POST", "/api/session", headers,
                       json.dumps({"key": key}).encode(), client=client)

    def session_cookie(self):
        r = self.sign_in()
        self.assertEqual(r.status, 204)
        cookie = r.cookies[0]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        return cookie.split(";")[0]

    # --- sign-in ----------------------------------------------------------

    def test_api_needs_a_session(self):
        for path in ("/api/settings", "/api/queue"):
            r = request(self.app, "GET", path, {"host": HOST})
            self.assertEqual(r.status, 401, path)

    def test_right_key_opens_a_session(self):
        cookie = self.session_cookie()
        r = request(self.app, "GET", "/api/settings", {"host": HOST, "cookie": cookie})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.json()["width"], 512)

    def test_wrong_key_is_refused(self):
        r = self.sign_in(key="x" * 40)
        self.assertEqual(r.status, 401)
        self.assertEqual(r.cookies, [])

    def test_sign_in_needs_same_origin(self):
        self.assertEqual(self.sign_in(origin="http://evil.example").status, 403)
        self.assertEqual(self.sign_in(origin=None).status, 403)
        self.assertEqual(self.sign_in(origin="null").status, 403)

    def test_repeated_wrong_keys_lock_out_that_client_only(self):
        for _ in range(10):
            self.sign_in(key="x" * 40)
        self.assertEqual(self.sign_in().status, 429)
        other = self.sign_in(client=("10.1.2.3", 40000))
        self.assertEqual(other.status, 204)

    def test_forged_cookie_is_refused(self):
        r = request(self.app, "GET", "/api/settings",
                    {"host": HOST, "cookie": f"{security.SESSION_COOKIE}=forged"})
        self.assertEqual(r.status, 401)

    def test_logout_ends_the_session(self):
        cookie = self.session_cookie()
        r = request(self.app, "POST", "/api/logout",
                    {"host": HOST, "origin": ORIGIN, "cookie": cookie})
        self.assertEqual(r.status, 204)
        r = request(self.app, "GET", "/api/settings", {"host": HOST, "cookie": cookie})
        self.assertEqual(r.status, 401)

    # --- headers ----------------------------------------------------------

    def test_security_headers_on_page_and_api(self):
        for path in ("/", "/app.js", "/api/settings"):
            r = request(self.app, "GET", path, {"host": HOST})
            self.assertIn("default-src 'none'", r.headers["content-security-policy"], path)
            self.assertIn("frame-ancestors 'none'", r.headers["content-security-policy"])
            self.assertEqual(r.headers["x-content-type-options"], "nosniff")
            self.assertEqual(r.headers["referrer-policy"], "same-origin")
            self.assertNotIn("server", r.headers)
            self.assertNotIn("access-control-allow-origin", r.headers)
        r = request(self.app, "GET", "/api/settings", {"host": HOST})
        self.assertEqual(r.headers["cache-control"], "no-store")

    def test_hsts_only_over_tls(self):
        tls_app = build_app(SimpleNamespace(max_queue_size=1), FakePipeline(),
                            security.AccessGuard(KEY), secure=True)
        r = request(tls_app, "GET", "/", {"host": HOST})
        self.assertIn("max-age=31536000", r.headers["strict-transport-security"])
        r = request(self.app, "GET", "/", {"host": HOST})
        self.assertNotIn("strict-transport-security", r.headers)

    def test_no_api_docs_exposed(self):
        for path in ("/docs", "/redoc", "/openapi.json"):
            r = request(self.app, "GET", path, {"host": HOST})
            self.assertEqual(r.status, 404, path)

    def test_page_loads_no_third_party_code(self):
        html = (REPO / "webcam/static/index.html").read_text()
        self.assertNotIn("http://", html.replace('href="https://github.com/GVCLab', ""))
        self.assertNotIn("<script>", html)
        self.assertNotIn("style=", html)

    # --- reference upload -------------------------------------------------

    def upload(self, data, cookie, origin=ORIGIN, content_type="image/jpeg"):
        body, ctype = multipart("ref_image", "ref.jpg", data, content_type)
        headers = {"host": HOST, "cookie": cookie, "content-type": ctype,
                   "content-length": str(len(body))}
        if origin:
            headers["origin"] = origin
        return request(self.app, "POST", "/api/upload_reference_image", headers, body)

    def test_upload_reaches_the_pipeline(self):
        cookie = self.session_cookie()
        r = self.upload(jpeg((300, 200)), cookie)
        self.assertEqual(r.status, 200)
        self.assertEqual(self.pipeline.references[0].mode, "RGB")

    def test_upload_needs_session_and_origin(self):
        self.assertEqual(self.upload(jpeg(), cookie="").status, 401)
        cookie = self.session_cookie()
        self.assertEqual(self.upload(jpeg(), cookie, origin="http://evil.example").status, 403)
        self.assertEqual(self.pipeline.references, [])

    def test_upload_refuses_non_images_and_other_formats(self):
        cookie = self.session_cookie()
        self.assertEqual(self.upload(b"not an image", cookie).status, 400)
        gif = io.BytesIO()
        Image.new("RGB", (8, 8)).save(gif, format="GIF")
        self.assertEqual(self.upload(gif.getvalue(), cookie).status, 400)

    def test_upload_refuses_oversized_bodies(self):
        cookie = self.session_cookie()
        headers = {"host": HOST, "origin": ORIGIN, "cookie": cookie,
                   "content-type": "multipart/form-data; boundary=x",
                   "content-length": str(security.MAX_UPLOAD_BYTES + 100000)}
        r = request(self.app, "POST", "/api/upload_reference_image", headers, b"")
        self.assertEqual(r.status, 413)

    # --- websocket --------------------------------------------------------

    def test_websocket_refuses_without_session(self):
        path = "/api/ws/123e4567-e89b-12d3-a456-426614174000"
        self.assertEqual(websocket_handshake(self.app, path, {"host": HOST, "origin": ORIGIN}),
                         "websocket.close")

    def test_websocket_refuses_cross_origin(self):
        cookie = self.session_cookie()
        path = "/api/ws/123e4567-e89b-12d3-a456-426614174000"
        headers = {"host": HOST, "origin": "http://evil.example", "cookie": cookie}
        self.assertEqual(websocket_handshake(self.app, path, headers), "websocket.close")

    def test_websocket_accepts_signed_in_same_origin(self):
        cookie = self.session_cookie()
        path = "/api/ws/123e4567-e89b-12d3-a456-426614174000"
        headers = {"host": HOST, "origin": ORIGIN, "cookie": cookie}
        self.assertEqual(websocket_handshake(self.app, path, headers), "websocket.accept")


class DecodeImageTest(unittest.TestCase):
    def test_accepts_listed_format(self):
        img = security.decode_image(jpeg((32, 16)), ("JPEG",), 10**6)
        self.assertEqual(img.size, (32, 16))
        self.assertEqual(img.mode, "RGB")

    def test_refuses_unlisted_format(self):
        with self.assertRaises(security.ImageRejected):
            security.decode_image(jpeg(fmt="PNG"), ("JPEG",), 10**6)

    def test_refuses_too_many_bytes(self):
        with self.assertRaises(security.ImageRejected):
            security.decode_image(jpeg(), ("JPEG",), 10)

    def test_refuses_huge_dimensions_before_decoding(self):
        # A small PNG that claims 20000x20000 pixels (a decompression bomb).
        buf = io.BytesIO()
        Image.new("1", (20000, 20000)).save(buf, format="PNG")
        with self.assertRaises(security.ImageRejected):
            security.decode_image(buf.getvalue(), ("PNG",), 10**7)

    def test_refuses_truncated_image(self):
        with self.assertRaises(security.ImageRejected):
            security.decode_image(jpeg((256, 256))[:300], ("JPEG",), 10**6)


class AccessKeyTest(unittest.TestCase):
    def test_generated_key_is_long_and_fresh(self):
        a, gen_a = security.load_access_key(None)
        b, _ = security.load_access_key(None)
        self.assertTrue(gen_a)
        self.assertGreaterEqual(len(a), 40)
        self.assertNotEqual(a, b)

    def test_key_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "key"
            p.write_text(KEY + "\n")
            os.chmod(p, 0o600)
            self.assertEqual(security.load_access_key(str(p)), (KEY, False))

    @unittest.skipUnless(os.name == "posix", "file modes are POSIX")
    def test_key_file_readable_by_others_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "key"
            p.write_text(KEY)
            os.chmod(p, 0o644)
            with self.assertRaises(security.AccessKeyError):
                security.load_access_key(str(p))

    def test_short_key_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "key"
            p.write_text("short")
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
            with self.assertRaises(security.AccessKeyError):
                security.load_access_key(str(p))

    def test_missing_key_file_is_refused(self):
        with self.assertRaises(security.AccessKeyError):
            security.load_access_key("/nonexistent/key")


class OriginTest(unittest.TestCase):
    def test_same_origin(self):
        self.assertTrue(security.same_origin("http://127.0.0.1:7860", "127.0.0.1:7860", False))
        self.assertTrue(security.same_origin("https://gpu.lan:7860", "gpu.lan:7860", True))

    def test_other_origins(self):
        self.assertFalse(security.same_origin("http://gpu.lan:7860", "gpu.lan:7860", True))
        self.assertFalse(security.same_origin("http://127.0.0.1:7861", "127.0.0.1:7860", False))
        self.assertFalse(security.same_origin("http://evil.example", "127.0.0.1:7860", False))
        self.assertFalse(security.same_origin(None, "127.0.0.1:7860", False))

    def test_loopback(self):
        self.assertTrue(security.is_loopback("127.0.0.1"))
        self.assertTrue(security.is_loopback("::1"))
        self.assertFalse(security.is_loopback("0.0.0.0"))
        self.assertFalse(security.is_loopback("192.168.1.5"))


class LaunchRulesTest(unittest.TestCase):
    """inference_online.main refuses unsafe settings before loading any model."""

    def run_main(self, argv):
        import inference_online
        return inference_online.main(argv)

    def test_remote_listen_without_tls_is_refused(self):
        self.assertEqual(self.run_main(["--host", "0.0.0.0"]), 2)

    def test_half_tls_config_is_refused(self):
        self.assertEqual(self.run_main(["--ssl-certfile", "cert.pem"]), 2)

    def test_bad_key_file_is_refused(self):
        self.assertEqual(self.run_main(["--access-key-file", "/nonexistent/key"]), 2)


if __name__ == "__main__":
    unittest.main()
