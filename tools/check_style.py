#!/usr/bin/env python3
"""Check the webcam page's vendored style files (webcam/static/style/).

Two checks, both run before a release (tools/release.sh runs the first):

1. Integrity, standard library only: every file listed in vendor.json is
   present with its SHA-256, and no other file is in the folder. A hand edit
   fails here; take a new version with `sg.py vendor` instead.
2. Contrast: each theme in themes.css is run through the style guide's own
   `sg.py audit-css` (`--aaa` for a theme whose header says AAA). This needs a
   clone of chappa-ai-llc/style-guide: --style-guide DIR, or the environment
   variable STYLE_GUIDE_DIR, or a sibling folder ../style-guide.

Usage:
  python tools/check_style.py [--style-guide DIR] [--integrity-only]

Exit 0 when every check passes, 1 when one fails, 2 when the style guide is
needed and not found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STYLE = REPO / "webcam" / "static" / "style"
THEME_HEADER = re.compile(r'^/\* style-guide theme "([a-z0-9-]+)" [0-9.]+ \(STANDARD', re.M)


def integrity(style: Path = STYLE) -> list[str]:
    """Problems with the vendored copy; [] when it matches vendor.json."""
    problems = []
    listed = json.loads((style / "vendor.json").read_text(encoding="utf-8"))["files"]
    for rel, digest in sorted(listed.items()):
        p = style / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            problems.append(f"changed: {rel} (re-run sg.py vendor; never edit by hand)")
    present = {p.relative_to(style).as_posix() for p in style.rglob("*") if p.is_file()}
    for rel in sorted(present - set(listed) - {"vendor.json"}):
        problems.append(f"not in vendor.json: {rel}")
    return problems


def theme_sections(css: str) -> dict[str, str]:
    """Each theme's part of themes.css: its tokens and its components."""
    marks = [(m.start(), m.group(1)) for m in THEME_HEADER.finditer(css)]
    sections: dict[str, str] = {}
    for i, (start, theme) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(css)
        sections[theme] = sections.get(theme, "") + css[start:end]
    return sections


def find_style_guide(given: str | None) -> Path | None:
    for candidate in (given, os.getenv("STYLE_GUIDE_DIR"), REPO.parent / "style-guide"):
        if candidate and (Path(candidate) / "tools" / "sg.py").is_file():
            return Path(candidate)
    return None


def contrast(guide: Path, style: Path = STYLE) -> list[str]:
    css = (style / "themes.css").read_text(encoding="utf-8")
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        for theme, text in theme_sections(css).items():
            f = Path(tmp) / f"{theme}.css"
            f.write_text(text, encoding="utf-8")
            cmd = [sys.executable, str(guide / "tools" / "sg.py"), "audit-css", str(f)]
            if "AAA" in text.split("\n", 1)[0]:
                cmd.append("--aaa")
            run = subprocess.run(cmd, capture_output=True, text=True)
            last = (run.stdout.strip().splitlines() or ["(no output)"])[-1]
            print(f"  {theme}: {last}")
            if run.returncode != 0:
                problems.append(f"contrast: {theme}\n{run.stdout}{run.stderr}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the vendored style files.")
    ap.add_argument("--style-guide", help="clone of chappa-ai-llc/style-guide")
    ap.add_argument("--integrity-only", action="store_true",
                    help="skip the contrast audit (no style guide needed)")
    args = ap.parse_args(argv)

    problems = integrity()
    print(f"integrity: {len(problems)} problem(s)")
    if not args.integrity_only:
        guide = find_style_guide(args.style_guide)
        if guide is None:
            print("contrast: style guide not found; give --style-guide DIR or set STYLE_GUIDE_DIR",
                  file=sys.stderr)
            return 2
        print(f"contrast (sg.py audit-css from {guide}):")
        problems += contrast(guide)
    for p in problems:
        print("FAIL", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
