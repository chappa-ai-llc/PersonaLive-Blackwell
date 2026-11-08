# Changelog

All notable changes to this fork. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-08

The first tagged release: a security sweep after the BUILD office's scan found 373 advisories and 320 Dependabot
alerts in the four dependency files.

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
- `docs/MDD-personalive-blackwell.md`, the design document (replaces the office kick-off brief).
