"""PersonaLive webcam server: animate a portrait from your webcam in the browser.

Start it with launcher.py (picks the GPU and builds the TensorRT engine), or
directly:

    python inference_online.py [--acceleration tensorrt]

It prints the address to open, with the access key in it. See README.md,
"Running the webcam server", for remote use over TLS.
"""

import asyncio
import logging
import multiprocessing as mp
import signal
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

from webcam import security
from webcam.config import parse_args

# Forward-secret AEAD suites only for TLS 1.2; TLS 1.3 suites are always strong.
TLS_CIPHERS = "ECDHE+AESGCM:ECDHE+CHACHA20"


def main(argv=None) -> int:
    config = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    secure = bool(config.ssl_certfile and config.ssl_keyfile)
    if bool(config.ssl_certfile) != bool(config.ssl_keyfile):
        print("Give both --ssl-certfile and --ssl-keyfile, or neither.", file=sys.stderr)
        return 2
    if not security.is_loopback(config.host) and not secure:
        print(f"Refusing to listen on {config.host} without TLS: other machines would "
              "see your webcam and the access key in clear text, and browsers allow "
              "the camera only over HTTPS. Give --ssl-certfile and --ssl-keyfile, or "
              "listen on 127.0.0.1 (the default).", file=sys.stderr)
        return 2
    try:
        key, generated = security.load_access_key(config.access_key_file)
    except security.AccessKeyError as e:
        print(e, file=sys.stderr)
        return 2
    config.pretty_print()

    import torch
    import uvicorn

    from webcam.preferences import PreferenceStore, default_path
    from webcam.server import build_app

    mp.set_start_method("spawn", force=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.acceleration == "tensorrt":
        from webcam.vid2vid_trt import Pipeline
    else:
        from webcam.vid2vid import Pipeline
    pipeline = Pipeline(config, device)

    guard = security.AccessGuard(key)
    prefs = PreferenceStore(Path(config.prefs_file) if config.prefs_file else default_path())
    app = build_app(config, pipeline, guard, secure, prefs)

    def stop(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        try:
            asyncio.run(app.state.cleanup())
        except Exception as e:
            print(f"Cleanup failed: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    scheme = "https" if secure else "http"
    shown_host = "127.0.0.1" if config.host in ("0.0.0.0", "::") else config.host
    if ":" in shown_host:
        shown_host = f"[{shown_host}]"
    url = f"{scheme}://{shown_host}:{config.port}/"
    if generated:
        print(f"\nOpen {url}#key={key}\n(the key is new for this run; "
              f"--access-key-file keeps one across runs)\n")
    else:
        print(f"\nOpen {url} and enter the access key from {config.access_key_file}\n")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
        ssl_ciphers=TLS_CIPHERS,
        server_header=False,
        proxy_headers=False,
        ws_max_size=security.MAX_FRAME_BYTES + 1024,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
