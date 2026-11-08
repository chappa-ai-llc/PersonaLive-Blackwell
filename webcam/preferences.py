"""The page's appearance preference: theme, light or dark, UI scale and type.

The shape follows the chappa.ai style guide (STANDARD.md sections 5 and 5.1):

    {"theme": "auto", "mode": "system", "scale": 1,
     "type": {"head": "theme", "body": "theme", "label": 0, "heading": "theme", "text": 0}}

The server is the record; the page caches it in localStorage only to paint
before the request returns. The webcam server has one user, the holder of the
access key, so there is one record, kept in a JSON file (--prefs-file).

The allowed values come from the vendored style files (static/style/), so a
new theme or font taken with `sg.py vendor` is accepted without a code change.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

STYLE_DIR = Path(__file__).resolve().parent / "static" / "style"
PREFS_ENV = "PERSONALIVE_PREFS_FILE"

# "auto": no stored theme; the page picks high-contrast when the system asks
# for more contrast (prefers-contrast: more), else DEFAULT_THEME.
DEFAULT_THEME = "modernist"
MODES = ("system", "light", "dark")
SCALES = (0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2)
DEFAULT = {
    "theme": "auto", "mode": "system", "scale": 1,
    "type": {"head": "theme", "body": "theme", "label": 0, "heading": "theme", "text": 0},
}


class PreferenceError(ValueError):
    """A value the page may not store."""


def load_catalogue(style_dir: Path = STYLE_DIR) -> dict:
    """Theme ids, font choices and size ranges from the vendored files."""
    vendor = json.loads((style_dir / "vendor.json").read_text(encoding="utf-8"))
    type_ = json.loads((style_dir / "type.json").read_text(encoding="utf-8"))
    return {
        "themes": tuple(vendor["themes"]),
        "head": tuple(c["id"] for c in type_["choices"] if "head" in c["roles"]),
        "body": tuple(c["id"] for c in type_["choices"] if "body" in c["roles"]),
        "headings": tuple(h["id"] for h in type_["headings"]),
        "label": (type_["sizes"]["label"]["min"], type_["sizes"]["label"]["max"]),
        "text": (type_["sizes"]["text"]["min"], type_["sizes"]["text"]["max"]),
    }


def default_path() -> Path:
    """preferences.json in the user's configuration folder."""
    if os.name == "nt" and os.getenv("APPDATA"):
        base = Path(os.environ["APPDATA"])
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "personalive" / "preferences.json"


def _whole(value, low: int, high: int, name: str) -> int:
    # 0 means "theme default"; bool is an int in Python, so refuse it first.
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreferenceError(f"type.{name} must be a whole number")
    if value != 0 and not low <= value <= high:
        raise PreferenceError(f"type.{name} must be 0 (theme default) or {low} to {high}")
    return value


def merge(current: dict, change: object, cat: dict) -> dict:
    """Return current with the fields in change applied, or raise PreferenceError.

    change may hold only some fields (PUT sends what changed); unknown fields
    are refused so a typo does not pass silently.
    """
    if not isinstance(change, dict):
        raise PreferenceError("send a JSON object")
    unknown = set(change) - set(DEFAULT)
    if unknown:
        raise PreferenceError(f"unknown field: {sorted(unknown)[0]}")
    out = json.loads(json.dumps(current))
    if "theme" in change:
        if change["theme"] != "auto" and change["theme"] not in cat["themes"]:
            raise PreferenceError("theme must be auto or one of: " + ", ".join(cat["themes"]))
        out["theme"] = change["theme"]
    if "mode" in change:
        if change["mode"] not in MODES:
            raise PreferenceError("mode must be system, light or dark")
        out["mode"] = change["mode"]
    if "scale" in change:
        s = change["scale"]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or s not in SCALES:
            raise PreferenceError("scale must be one of " + ", ".join(str(x) for x in SCALES))
        out["scale"] = s
    if "type" in change:
        t = change["type"]
        if not isinstance(t, dict):
            raise PreferenceError("type must be an object")
        unknown = set(t) - set(DEFAULT["type"])
        if unknown:
            raise PreferenceError(f"unknown field: type.{sorted(unknown)[0]}")
        for role in ("head", "body"):
            if role in t:
                if t[role] != "theme" and t[role] not in cat[role]:
                    raise PreferenceError(f"type.{role} must be theme or one of: "
                                          + ", ".join(cat[role]))
                out["type"][role] = t[role]
        if "heading" in t:
            if t["heading"] != "theme" and t["heading"] not in cat["headings"]:
                raise PreferenceError("type.heading must be theme or one of: "
                                      + ", ".join(cat["headings"]))
            out["type"]["heading"] = t["heading"]
        for name in ("label", "text"):
            if name in t:
                out["type"][name] = _whole(t[name], *cat[name], name)
    return out


class PreferenceStore:
    """The one preference record. path=None keeps it in memory (tests)."""

    def __init__(self, path: Optional[Path], catalogue: Optional[dict] = None):
        self.path = Path(path) if path else None
        self.catalogue = catalogue or load_catalogue()
        self._lock = threading.Lock()
        self._value = self._read()

    def _read(self) -> dict:
        if self.path is None or not self.path.exists():
            return json.loads(json.dumps(DEFAULT))
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
            # Re-validate: a hand edit, or a theme a newer style guide dropped,
            # falls back to the default for that field instead of failing.
            value = json.loads(json.dumps(DEFAULT))
            for field in ("theme", "mode", "scale"):
                try:
                    value = merge(value, {field: stored[field]}, self.catalogue)
                except (KeyError, TypeError, PreferenceError):
                    pass
            for field in DEFAULT["type"]:
                try:
                    value = merge(value, {"type": {field: stored["type"][field]}},
                                  self.catalogue)
                except (KeyError, TypeError, PreferenceError):
                    pass
            return value
        except (OSError, ValueError):
            return json.loads(json.dumps(DEFAULT))

    def get(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._value))

    def update(self, change: object) -> dict:
        with self._lock:
            value = merge(self._value, change, self.catalogue)
            if self.path is not None:
                self._write(value)
            self._value = value
            return json.loads(json.dumps(value))

    def _write(self, value: dict) -> None:
        # Write a private temp file next to the record, then rename over it,
        # so a crash never leaves half a file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".preferences-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
