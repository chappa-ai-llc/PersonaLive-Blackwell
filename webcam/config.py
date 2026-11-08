"""Command-line options of the webcam server (inference_online.py).

Every option can also come from an environment variable, named in its help.
Defaults are the safe choice: the server listens on this machine only, needs
an access key, and serves one session at a time.
"""

from typing import NamedTuple, Optional
import argparse
import os

from webcam.security import KEY_ENV


class Args(NamedTuple):
    host: str
    port: int
    max_queue_size: int
    ssl_certfile: Optional[str]
    ssl_keyfile: Optional[str]
    access_key_file: Optional[str]
    acceleration: str
    config_path: str

    def pretty_print(self):
        print()
        for field, value in self._asdict().items():
            print(f"{field}: {value}")
        print()


def parse_args(argv=None) -> Args:
    parser = argparse.ArgumentParser(description="PersonaLive webcam server")
    parser.add_argument(
        "--host", default=os.getenv("HOST", "127.0.0.1"),
        help="Address to listen on (env HOST; default 127.0.0.1, this machine only). "
             "Any other address needs --ssl-certfile and --ssl-keyfile.")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("PORT", "7860")),
        help="Port (env PORT; default 7860)")
    parser.add_argument(
        "--max-queue-size", dest="max_queue_size", type=int,
        default=int(os.getenv("MAX_QUEUE_SIZE", "1")),
        help="Most browser sessions at once (env MAX_QUEUE_SIZE; default 1: "
             "the pipeline animates one face at a time)")
    parser.add_argument(
        "--ssl-certfile", dest="ssl_certfile", default=os.getenv("SSL_CERTFILE"),
        help="TLS certificate chain, PEM (env SSL_CERTFILE)")
    parser.add_argument(
        "--ssl-keyfile", dest="ssl_keyfile", default=os.getenv("SSL_KEYFILE"),
        help="TLS private key, PEM (env SSL_KEYFILE)")
    parser.add_argument(
        "--access-key-file", dest="access_key_file", default=os.getenv(KEY_ENV),
        help=f"File holding the access key, at least 32 characters, mode 600 (env {KEY_ENV}). "
             "Without it a new key is made for each run and printed with the address.")
    parser.add_argument(
        "--acceleration", default=os.getenv("ACCELERATION", "none"),
        choices=["none", "xformers", "tensorrt"],
        help="none: PyTorch attention (default); tensorrt: the compiled engine "
             "(launcher.py builds it); xformers: only if you installed a build "
             "that matches your PyTorch (env ACCELERATION)")
    parser.add_argument(
        "--config_path", default="./configs/prompts/personalive_online.yaml",
        help="Model configuration")
    return Args(**vars(parser.parse_args(argv)))
