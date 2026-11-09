#!/usr/bin/env bash
# Make the signed release tag vX.Y.Z on main, with that version's CHANGELOG
# section as the tag message, after the checks pass.
#
#   tools/release.sh 0.2.1
#
# Needs: a clean main that matches origin/main, a "## [X.Y.Z]" section in
# CHANGELOG.md, version = "X.Y.Z" in pyproject.toml, and Git set up to sign
# tags (git config gpg.format / user.signingkey). PYTHON picks the interpreter
# for the tests (default python3). It never pushes: push the tag yourself with
# the command it prints, so your own pre-push checks run.
set -euo pipefail
cd "$(dirname "$0")/.."

version="${1:-}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: tools/release.sh X.Y.Z" >&2
  exit 2
fi
tag="v$version"
py="${PYTHON:-python3}"

fail() { echo "release.sh: $*" >&2; exit 1; }

[[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || fail "not on main"
[[ -z "$(git status --porcelain)" ]] || fail "the working tree is not clean"
git fetch --quiet origin main
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || fail "main differs from origin/main"
git rev-parse -q --verify "refs/tags/$tag" >/dev/null && fail "$tag already exists (tags are never moved)"
grep -q "^version = \"$version\"$" pyproject.toml || fail "pyproject.toml version is not $version"

notes="$(awk -v h="## [$version]" 'index($0, h) == 1 {p = 1; next} p && /^## \[/ {exit} p' CHANGELOG.md)"
[[ -n "${notes//[[:space:]]/}" ]] || fail "CHANGELOG.md has no section [$version]"
grep -q '^\*\*Purpose:\*\*' <<<"$notes" || fail "the [$version] section has no **Purpose:** line"

"$py" -m unittest discover -s tests
"$py" tools/check_style.py --integrity-only

msgfile="$(mktemp)"
trap 'rm -f "$msgfile"' EXIT
printf 'PersonaLive-Blackwell %s\n\n%s\n' "$version" "$(sed -e '/./,$!d' <<<"$notes")" >"$msgfile"
git tag -s --cleanup=verbatim -F "$msgfile" "$tag"
git tag -v "$tag" 2>/dev/null || git -c gpg.ssh.allowedSignersFile=.release-signers tag -v "$tag"

echo
echo "Signed $tag. Push it with:"
echo "  git push origin $tag"
