#!/usr/bin/env bash
# Release step for a pushed tag vX.Y.Z: make its GitHub release page (and any
# older tag's missing page) from CHANGELOG.md. There is nothing to deploy: users
# install from the signed tag. Run with bash, since the checkout may be noexec:
#
#   bash deploy/release.sh v0.2.1
#
# Needs GITHUB_TOKEN_FILE (a token with contents: write); see release_page.py.
set -euo pipefail
exec python3 "$(dirname "$0")/release_page.py" "$@"
