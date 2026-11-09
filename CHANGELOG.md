# Changelog

All notable changes to this fork. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] - 2026-11-09

**Purpose:** documentation and release tooling only. Each release now states what it is for, has a release page, and
anyone can check that its tag was signed by this project. No change to the software or its dependencies.

**Config changes:** none.

### Added

- `SECURITY.md`, "Verifying a release": the release key, its fingerprint, and how to check a tag with `git tag -v`.
- `.release-signers`: the release key in Git's allowed-signers format.
- `tools/release.sh`: runs the checks, then makes the signed tag with that version's notes as the tag message.
- `deploy/release.sh` and `deploy/release_page.py`: make the GitHub release page for a tag from its `CHANGELOG.md`
  section, with how to verify the signature, and add the missing pages for 0.1.0 and 0.2.0. Standard library only.

### Changed

- `tools/lock.sh` takes `uv` from `PATH` or from `UV`, and stops with a clear message when neither is set.
- The 0.1.0 and 0.2.0 notes below now open with a purpose statement.

### Removed

- The project-management files (the task board's data and the internal design document) are no longer kept in this
  repository. They describe how the maintainers run the project, not how to use it; the README, `SECURITY.md` and
  this changelog cover users.

## [0.2.0] - 2026-11-08

**Purpose:** lets you choose how the webcam page looks (theme, light or dark, UI scale and fonts). The server saves the
choice in a file on your own machine. The animation pipeline and its dependencies are unchanged.

**Config changes:** none needed. The new `--prefs-file` option (`PERSONALIVE_PREFS_FILE`) has a default:
`personalive/preferences.json` under `~/.config` on Linux or `%APPDATA%` on Windows.

### Added

- The webcam page takes the shared chappa.ai style guide (1.7.0): five themes (Modernist, Flight manual, Aurora,
  Ember, High contrast), light, dark or system, UI scale 80% to 200%, and heading font, body font, text, heading and
  label size. A menu in the upper right holds Preferences and Sign out (#5).
- `GET` and `PUT /api/preferences`: the appearance is saved on the server, in `--prefs-file` (mode 600), and checked
  against the themes and fonts the page ships; anything else is refused with 400.
- Automatic theme: High contrast when the system asks for more contrast (`prefers-contrast: more`), else Modernist.
  Windows contrast themes keep their own colours.
- `tools/check_style.py`: checks the vendored themes and fonts against their SHA-256 list and runs the style guide's
  contrast audit on every theme.

### Security

- The CSP adds `font-src 'self'` for the bundled fonts; the page still loads nothing from other sites.

### Fixed

- The empty portrait tile drew a broken-image box: a CSS `display` rule overrode the `hidden` attribute.

## [0.1.0] - 2026-11-08

**Purpose:** the first tagged release of this fork: a security sweep after a dependency scan found 373 known advisories and 320 Dependabot
alerts in the four dependency files inherited from upstream.

**Config changes:** install from the new lock with `pip install --require-hashes -r requirements.txt` in a fresh
Python 3.12+ environment; `requirements_base.txt` and `requirements_trt.txt` are gone. The webcam server now listens
on 127.0.0.1 and needs the access key it prints at start; to reach it from another machine, give `--ssl-certfile`,
`--ssl-keyfile` and `--access-key-file` (README, "Running the webcam server"). Do not run `apply_patches.py` from an
older checkout: it overwrites installed PyTorch files.

### Security

- Dependencies: one hash-pinned lock (`requirements.txt`, generated from `pyproject.toml` by `tools/lock.sh`) with 0
  known advisories. Moves include torch 2.9.1 to 2.15.0, diffusers 0.27.0 to 0.42.1, transformers 4.36.2 to 5.22.2,
  pillow 11.3.0 to 13.0.0, starlette 0.48.0 to 1.10.0, python-multipart 0.0.20 to 0.0.33, protobuf 4.25.8 to 7.36.2,
  onnx 1.20.0 to 1.23.2, urllib3 2.6.2 to 2.8.0 and certifi to 2026.10.23.
- Removed the SvelteKit front end and its npm lockfile (about 30 vulnerable packages, and the no-npm rule); the page
  is now plain HTML, CSS and JavaScript with no third-party code (it also loaded a script from a CDN).
- Webcam server: listens on 127.0.0.1 by default and refuses any other address without TLS; access key and session
  cookie for every API call and the WebSocket; no wildcard CORS; Origin checks; security headers; size, format and
  dimension limits on images; no API docs pages; one session at a time by default.
- `apply_patches.py` and `patches/` removed: they overwrote PyTorch's installed `torch/cuda/__init__.py` with a copy
  from 2.9.1 and edited onnx-graphsurgeon in place. PyTorch 2.11 and later accept GB10 (sm_121) on their own, and
  onnx-graphsurgeon 0.6 handles newer ONNX.
- All checkpoints load with `torch.load(..., weights_only=True)`.
- `tools/download_weights.py` downloads from pinned Hugging Face commits and checks SHA-256 against
  `tools/weights.lock.json` (the pins are still to be recorded; see SECURITY.md).
- Dropped MediaPipe, decord, scikit-image, markdown2, accelerate, xFormers and other unused packages.

### Changed

- Offline inference crops faces with the LivePortrait keypoints, as the webcam path does, instead of MediaPipe, and
  reads videos with PyAV instead of decord. `--use_xformers` is now a switch, off by default.
- `--acceleration` defaults to `none` (PyTorch attention); the launcher still selects `tensorrt`.
- Python 3.12 or later.

### Added

- `tests/test_security.py`: 34 tests of the webcam server's controls, standard library only.
- `tools/smoke_random_weights.py`: runs the whole online pipeline on CPU with random weights, to check a dependency
  upgrade without a GPU or the real weights.
- `SECURITY.md`.
- `docs/MDD-personalive-blackwell.md`, the design document.
