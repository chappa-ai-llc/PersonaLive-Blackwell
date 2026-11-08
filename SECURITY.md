# Security

As of 0.1.0 (2026-11-08). This repository is public, so this page says what the code does to stay safe, what is still
open, and how to report a problem.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting on this repository (Security tab, "Report a vulnerability"), not a
public issue. Problems in the model itself belong upstream at
[GVCLab/PersonaLive](https://github.com/GVCLab/PersonaLive).

## Dependencies

- **One lock file with hashes.** `pyproject.toml` lists the direct dependencies; `requirements.txt` pins every package,
  direct or not, to one version with its SHA-256 hashes, and `pip install --require-hashes -r requirements.txt`
  refuses anything else. `tools/lock.sh` regenerates it with uv.
- **No known advisories.** On 2026-11-08 the lock had 0 open advisories in OSV for all 90 packages. Each lower bound
  in `pyproject.toml` is the first release that fixed every advisory known for that package, and the transitive
  floors under `[tool.uv] constraint-dependencies` stop a resolver from going below them.
- **No npm.** The web page is plain HTML, CSS and JavaScript in `webcam/static/`, with no build step and no
  third-party code. The SvelteKit front end and its npm tree were removed in 0.1.0. The themes and fonts in
  `webcam/static/style/` are vendored from the chappa.ai style guide with a SHA-256 for each file, which
  `tests/test_appearance.py` and `tools/check_style.py` check; the fonts are served by this server (`font-src 'self'`).
- **Fewer packages.** Removed in 0.1.0 because nothing needed them, or because they could not be made safe:
  MediaPipe (its face-mesh API needs protobuf 4, which has advisories), decord (unmaintained), scikit-image, markdown2,
  accelerate (an advisory with no fix), xFormers (no build for the current PyTorch), python-dotenv and the
  `onnxruntime`, `colored` and `jax` packages.
- **TensorRT.** NVIDIA publishes `tensorrt-cu13` and `tensorrt-cu13-libs` on PyPI only as installer packages
  ("wheel-stub"). pip checks the installer against the hash in `requirements.txt`; the installer then downloads the
  real wheel from `https://pypi.nvidia.com/` and checks it against the SHA-256 that index publishes. Locking those
  wheels directly is tracked in the issues.

## Model weights

- `tools/download_weights.py` downloads each file from a fixed Hugging Face commit and checks its SHA-256 against
  `tools/weights.lock.json`. A file that does not match is deleted and refused.
- **Open:** the pins in `tools/weights.lock.json` are not recorded yet, because the machine that prepared 0.1.0 could
  not reach Hugging Face. Until they are, the script downloads nothing and says why. A maintainer records them with
  `python tools/download_weights.py --pin` and reviews the diff before committing it.
- Every checkpoint is loaded with `torch.load(..., weights_only=True)`, which refuses pickled code.

## Webcam server

`inference_online.py` (code in `webcam/server.py`, `webcam/security.py` and `webcam/preferences.py`, tests in `tests/test_security.py` and `tests/test_appearance.py`):

- **Listens on 127.0.0.1 by default.** Any other address needs TLS (`--ssl-certfile`, `--ssl-keyfile`), or the server
  refuses to start. TLS 1.2 uses forward-secret AEAD suites only.
- **Access key, always.** Every API call and the WebSocket need a session, opened with the access key. The key is
  either new for each run (printed with the address, kept in memory only) or read from `--access-key-file` /
  `PERSONALIVE_ACCESS_KEY_FILE`, a file that must be at least 32 characters and not readable by other users. Sessions
  are HttpOnly, SameSite=Strict cookies that last 12 hours. Ten wrong keys in a minute lock that client out for the
  rest of the minute.
- **Same origin only.** No CORS. Sign-in, sign-out, uploads, saving the appearance and the WebSocket check the
  `Origin` header against the server's own address.
- **Security headers** on every response: a Content-Security-Policy that allows only the server's own files,
  `frame-ancestors 'none'`, `nosniff`, `Referrer-Policy: same-origin`, a Permissions-Policy that allows the camera and
  screen capture for this page only, and HSTS over TLS. No `Server` header and no API docs pages.
- **Bounded input.** Reference images: JPEG, PNG or WebP, at most 10 MB and 4096 pixels a side. Webcam frames: JPEG,
  at most 2 MB. Image headers are checked before any pixel is decoded.
- **Appearance preference.** `PUT /api/preferences` takes at most 4096 bytes and accepts only the theme, mode, scale
  and type values the page ships (anything else is a 400). It is written atomically to `--prefs-file` with mode 600.
- One browser session at a time by default (`--max-queue-size`).

## Not covered

- The web page is a local tool, not a hosted service: it has no user accounts and no self-assessment page.
- Output frames are not watermarked. Use the animation only with the consent of the person in the portrait.
