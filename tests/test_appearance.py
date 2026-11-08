"""Tests for the page's appearance: the preference API (webcam/preferences.py),
the vendored style files and the page rules from the style guide.

Run from the repo root:  python -m unittest discover -s tests -v
"""

import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from asgi_client import request  # noqa: E402
from webcam import preferences, security  # noqa: E402
from webcam.server import build_app  # noqa: E402
import check_style  # noqa: E402

KEY = "k" * 40
ORIGIN = "http://127.0.0.1:7860"
HOST = "127.0.0.1:7860"
STATIC = REPO / "webcam" / "static"


class PreferenceApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sub" / "preferences.json"
        self.store = preferences.PreferenceStore(self.path)
        self.app = build_app(SimpleNamespace(max_queue_size=1), None,
                             security.AccessGuard(KEY), secure=False, prefs=self.store)
        res = request(self.app, "POST", "/api/session",
                      {"host": HOST, "origin": ORIGIN, "content-type": "application/json"},
                      json.dumps({"key": KEY}).encode())
        self.assertEqual(res.status, 204)
        self.cookie = res.cookies[0].split(";")[0]

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, body, origin=ORIGIN, cookie=True):
        headers = {"host": HOST, "content-type": "application/json"}
        if origin:
            headers["origin"] = origin
        if cookie:
            headers["cookie"] = self.cookie
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        return request(self.app, "PUT", "/api/preferences", headers, data)

    def get(self, cookie=True):
        headers = {"host": HOST}
        if cookie:
            headers["cookie"] = self.cookie
        return request(self.app, "GET", "/api/preferences", headers)

    def test_default(self):
        res = self.get()
        self.assertEqual(res.status, 200)
        self.assertEqual(res.json(), preferences.DEFAULT)
        self.assertEqual(res.headers.get("cache-control"), "no-store")

    def test_needs_session(self):
        self.assertEqual(self.get(cookie=False).status, 401)
        self.assertEqual(self.put({"mode": "dark"}, cookie=False).status, 401)

    def test_put_needs_same_origin(self):
        self.assertEqual(self.put({"mode": "dark"}, origin=None).status, 403)
        self.assertEqual(self.put({"mode": "dark"}, origin="http://evil.example").status, 403)

    def test_partial_update_is_stored_and_saved(self):
        res = self.put({"theme": "ember", "type": {"head": "barlow-condensed", "text": 16}})
        self.assertEqual(res.status, 200)
        body = res.json()
        self.assertEqual(body["theme"], "ember")
        self.assertEqual(body["mode"], "system")
        self.assertEqual(body["type"]["head"], "barlow-condensed")
        self.assertEqual(body["type"]["text"], 16)
        self.assertEqual(body["type"]["body"], "theme")
        self.assertEqual(self.get().json(), body)
        self.assertEqual(json.loads(self.path.read_text()), body)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        # A new store (a restart) reads the same record.
        self.assertEqual(preferences.PreferenceStore(self.path).get(), body)

    def test_every_vendored_theme_and_auto_accepted(self):
        themes = json.loads((STATIC / "style" / "vendor.json").read_text())["themes"]
        for theme in ["auto", *themes]:
            self.assertEqual(self.put({"theme": theme}).status, 200, theme)

    def test_bad_values_refused(self):
        bad = [
            {"theme": "neon"}, {"mode": "dim"}, {"scale": 3}, {"scale": True}, {"scale": "1"},
            {"colour": "red"}, {"type": {"head": "comic-sans"}},
            {"type": {"body": "barlow-condensed"}},  # a heading-only font
            {"type": {"text": 20}}, {"type": {"label": 8}}, {"type": {"label": True}},
            {"type": {"heading": "huge"}}, {"type": {"size": 1}}, {"type": "big"}, ["mode"],
        ]
        for change in bad:
            res = self.put(change)
            self.assertEqual(res.status, 400, change)
            self.assertTrue(res.json()["detail"], change)
        self.assertEqual(self.put(b"not json").status, 400)
        self.assertEqual(self.get().json(), preferences.DEFAULT)
        self.assertFalse(self.path.exists())

    def test_body_limit(self):
        res = self.put(b"{" + b" " * (8 * 1024) + b"}")
        self.assertEqual(res.status, 413)

    def test_damaged_file_falls_back_per_field(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"theme": "gone", "mode": "dark", "scale": 9,
                                         "type": {"head": "archivo", "text": 99}}))
        value = preferences.PreferenceStore(self.path).get()
        self.assertEqual(value["theme"], "auto")
        self.assertEqual(value["mode"], "dark")
        self.assertEqual(value["scale"], 1)
        self.assertEqual(value["type"]["head"], "archivo")
        self.assertEqual(value["type"]["text"], 0)
        self.path.write_text("{not json")
        self.assertEqual(preferences.PreferenceStore(self.path).get(), preferences.DEFAULT)


class StyleFilesTest(unittest.TestCase):
    def test_vendored_copy_matches_vendor_json(self):
        self.assertEqual(check_style.integrity(), [])

    def test_integrity_catches_a_hand_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            style = Path(tmp)
            for name in ("vendor.json", "themes.css", "type.json"):
                (style / name).write_bytes((STATIC / "style" / name).read_bytes())
            (style / "themes.css").write_text("/* edited */", encoding="utf-8")
            (style / "extra.css").write_text("", encoding="utf-8")
            problems = check_style.integrity(style)
            self.assertTrue(any(p.startswith("changed: themes.css") for p in problems))
            self.assertIn("not in vendor.json: extra.css", problems)
            self.assertTrue(any(p.startswith("missing: fonts/") for p in problems))

    def test_every_theme_is_split_out(self):
        css = (STATIC / "style" / "themes.css").read_text(encoding="utf-8")
        themes = json.loads((STATIC / "style" / "vendor.json").read_text())["themes"]
        self.assertEqual(set(check_style.theme_sections(css)), set(themes))

    def test_no_bare_viewport_units(self):
        # STANDARD.md section 7: every vh/vw/dvh length divides by --zoom.
        pattern = re.compile(r"\d(?:\.\d+)?d?v[hw]\b")
        for f in [*STATIC.glob("*.css"), *STATIC.glob("*.js"), *STATIC.glob("*.html")]:
            for line in f.read_text(encoding="utf-8").splitlines():
                for m in pattern.finditer(line):
                    self.assertIn("/ var(--zoom, 1)", line[m.end():m.end() + 20],
                                  f"{f.name}: bare viewport unit in: {line.strip()}")

    def test_page_loads_nothing_from_elsewhere(self):
        for f in [*STATIC.glob("*.css"), *STATIC.glob("*.js"), STATIC / "style" / "themes.css"]:
            text = f.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"url\(\s*[\"']?(https?:)?//", f.name)
            self.assertNotIn("@import", text, f.name)
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertNotRegex(html, r"<(script|link)[^>]+(src|href)=\"(https?:)?//")
        self.assertIn("font-src 'self'", security.CSP)

    def test_page_sets_a_default_theme(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn(f'data-preset="{preferences.DEFAULT_THEME}"', html)
        self.assertIn(f'DEFAULT_THEME = "{preferences.DEFAULT_THEME}"',
                      (STATIC / "appearance.js").read_text(encoding="utf-8"))

    def test_page_scales_match_server(self):
        js = (STATIC / "preferences.js").read_text(encoding="utf-8")
        m = re.search(r"const SCALES = \[([^\]]+)\]", js)
        self.assertEqual(tuple(float(x) for x in m.group(1).split(",")),
                         tuple(float(x) for x in preferences.SCALES))

    def test_app_css_defines_no_colours(self):
        # Colours come from the theme tokens only. The plates on top of the
        # camera picture are the one exception: they sit on video, not a theme surface.
        css = (STATIC / "app.css").read_text(encoding="utf-8")
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        allowed = {"#fff"}
        for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b", css):
            self.assertIn(m.group(0), allowed, m.group(0))
        self.assertNotRegex(css, r"(?m)^\s*--[a-z0-9-]+\s*:", "app.css redefines a token")


if __name__ == "__main__":
    unittest.main()
