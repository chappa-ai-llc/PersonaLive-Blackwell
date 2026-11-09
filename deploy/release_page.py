#!/usr/bin/env python3
"""Make the GitHub release page for a release tag, and for any older release tag that has none.

    python3 deploy/release_page.py vX.Y.Z

Each page's notes are that version's section of CHANGELOG.md in this checkout, plus how to verify the signed tag.
Older tags without a page get one first, oldest first, without the Latest mark; the named tag gets the Latest mark.
A tag that already has a page is left alone, so a second run changes nothing.

Settings (environment):
  GITHUB_TOKEN_FILE  file holding a GitHub token with contents: write on the repository (required)
  RELEASE_REPO       OWNER/NAME; default: read from `git remote get-url origin`
  GITHUB_API         API base; default https://api.github.com

Exit 0: done or nothing to do. 1: GitHub refused, or the named tag has no CHANGELOG section. 2: bad arguments or setup.
Standard library only.
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
TITLE = "PersonaLive-Blackwell {version}"
VERIFY = (
    "\n\n---\n\n**Verify this release.** The tag `{tag}` is signed with the project's release key "
    "(fingerprint `SHA256:wZo0qZggJ3mUGdpCZhkuUk88ZE6d8t/da3ctY1Mvt7Y`). Check it before installing:\n\n"
    "```bash\ngit fetch --tags\ngit -c gpg.ssh.allowedSignersFile=.release-signers tag -v {tag}\n```\n\n"
    "See `SECURITY.md`, \"Verifying a release\"."
)


class Refused(Exception):
    pass


def changelog_section(text, version):
    """The body of the `## [version]` section, without its heading; None when absent or empty."""
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not m:
        return None
    body = m.group(1).strip()
    return body or None


def repo_from_origin():
    out = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True)
    m = re.search(r"github\.com[:/]([A-Za-z0-9._-]+/[A-Za-z0-9._-]+?)(?:\.git)?/?$", out.stdout.strip())
    return m.group(1) if m else None


class GitHub:
    def __init__(self, repo, token, api="https://api.github.com"):
        self.repo, self.token, self.api = repo, token, api.rstrip("/")

    def call(self, method, path, body=None):
        req = urllib.request.Request(
            self.api + path, method=method, data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json",
                     "User-Agent": "personalive-blackwell-release"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            try:
                detail = json.load(e).get("message", "")
            except ValueError:
                detail = ""
            return e.code, {"message": detail}

    def pages(self, path):
        page, items = 1, []
        while True:
            status, got = self.call("GET", f"{path}?per_page=100&page={page}")
            if status != 200:
                raise Refused(f"GET {path}: {status} {got.get('message', '')}")
            items += got
            if len(got) < 100:
                return items
            page += 1

    def release_tags(self):
        return [t["name"] for t in self.pages(f"/repos/{self.repo}/tags") if TAG_RE.match(t["name"])]

    def tags_with_pages(self):
        return {r["tag_name"] for r in self.pages(f"/repos/{self.repo}/releases")}

    def make(self, tag, body, latest):
        status, got = self.call("POST", f"/repos/{self.repo}/releases", {
            "tag_name": tag, "name": TITLE.format(version=tag[1:]), "body": body,
            "draft": False, "prerelease": False, "make_latest": "true" if latest else "false"})
        if status != 201:
            raise Refused(f"making the page for {tag}: {status} {got.get('message', '')}")
        return got.get("html_url", "")


def plan(target, tags, have):
    """The tags to make pages for, oldest first: older tags with no page, then the target if it has none."""
    key = lambda t: tuple(int(x) for x in TAG_RE.match(t).groups())
    older = sorted((t for t in tags if TAG_RE.match(t) and key(t) < key(target) and t not in have), key=key)
    return older + ([target] if target not in have else [])


def run(target, gh, changelog_text, out=print):
    tags = gh.release_tags()
    if target not in tags:
        raise Refused(f"{gh.repo} has no tag {target}")
    todo = plan(target, tags, gh.tags_with_pages())
    if not todo:
        out(f"{target} already has a page; nothing to do")
        return 0
    for tag in todo:
        notes = changelog_section(changelog_text, tag[1:])
        if notes is None:
            if tag == target:
                raise Refused(f"CHANGELOG.md has no section [{tag[1:]}]")
            out(f"warning: CHANGELOG.md has no section [{tag[1:]}]; no page for {tag}")
            continue
        url = gh.make(tag, notes + VERIFY.format(tag=tag), latest=(tag == target))
        out(f"page made: {url}")
    return 0


def main(argv):
    if len(argv) != 2 or not TAG_RE.match(argv[1]):
        print(__doc__, file=sys.stderr)
        return 2
    token_file = os.environ.get("GITHUB_TOKEN_FILE")
    repo = os.environ.get("RELEASE_REPO") or repo_from_origin()
    if not token_file or not repo:
        print("release_page: needs GITHUB_TOKEN_FILE, and RELEASE_REPO or a GitHub origin remote", file=sys.stderr)
        return 2
    token = Path(token_file).read_text(encoding="utf-8").strip()
    gh = GitHub(repo, token, os.environ.get("GITHUB_API", "https://api.github.com"))
    try:
        return run(argv[1], gh, (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    except Refused as e:
        print(f"release_page: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
