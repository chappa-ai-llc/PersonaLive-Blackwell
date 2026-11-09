"""Download the model weights into ./pretrained_weights, pinned and verified.

Every file comes from a fixed Hugging Face commit and is checked against the
SHA-256 recorded in tools/weights.lock.json, so a changed or replaced upstream
file is refused instead of loaded.

    python tools/download_weights.py          download and verify (the normal step)
    python tools/download_weights.py --pin    maintainers: record the current
                                              upstream commit and file hashes in
                                              the lock file, then review the diff

Run from the repo root.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK = Path(__file__).resolve().parent / "weights.lock.json"
DEST = REPO_ROOT / "pretrained_weights"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# Hugging Face repo -> (folder under pretrained_weights, files in the repo, and
# the prefix to strip from a file's path when saving it). The base model comes
# from Lambda's verified "lambda" org: the old "lambdalabs" name now belongs to
# an unrelated account (SECURITY.md, "Model weights").
SOURCES = {
    "lambda/sd-image-variations-diffusers": (
        "sd-image-variations-diffusers",
        ["unet/config.json", "unet/diffusion_pytorch_model.bin",
         "image_encoder/config.json", "image_encoder/pytorch_model.bin",
         "model_index.json"],
        "",
    ),
    "stabilityai/sd-vae-ft-mse": (
        "sd-vae-ft-mse",
        ["config.json", "diffusion_pytorch_model.bin"],
        "",
    ),
    "huaichang/PersonaLive": (
        "personalive",
        [f"pretrained_weights/personalive/{n}.pth" for n in (
            "denoising_unet", "motion_encoder", "motion_extractor",
            "pose_guider", "reference_unet", "temporal_module")],
        "pretrained_weights/personalive/",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target(repo: str, name: str) -> Path:
    folder, _, strip = SOURCES[repo]
    return DEST / folder / name[len(strip):]


def fetch(repo: str, name: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    cached = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision))
    out = target(repo, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    shutil.copyfile(cached, tmp)
    return tmp


def download(lock: dict) -> int:
    failed = False
    for repo, (_, files, _) in SOURCES.items():
        entry = lock.get(repo) or {}
        revision = entry.get("revision") or ""
        hashes = entry.get("files") or {}
        if not SHA_RE.match(revision) or any(not HASH_RE.match(hashes.get(n) or "") for n in files):
            print(f"{repo}: not pinned in {LOCK.name}. A maintainer records the pins with "
                  "--pin; see SECURITY.md, 'Model weights'.", file=sys.stderr)
            failed = True
            continue
        for name in files:
            out = target(repo, name)
            if out.exists() and sha256(out) == hashes[name]:
                continue
            print(f"{repo}@{revision[:12]}: {name}")
            tmp = fetch(repo, name, revision)
            got = sha256(tmp)
            if got != hashes[name]:
                tmp.unlink()
                print(f"  REFUSED: SHA-256 {got} is not the pinned {hashes[name]}",
                      file=sys.stderr)
                failed = True
                continue
            tmp.replace(out)
    if failed:
        return 1
    print(f"All weights present and verified in {DEST}")
    return 0


def pin() -> int:
    """Record each repo's current commit and the SHA-256 of every listed file.

    Large files are stored in Git LFS, whose pointer carries the file's SHA-256,
    so their hashes come from the Hub's metadata at that commit and nothing big
    is downloaded. Small files are downloaded and hashed. A listed file missing
    at the commit is refused: an upstream repo was emptied or replaced.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    lock = {}
    for repo, (_, files, _) in SOURCES.items():
        revision = api.model_info(repo).sha
        info = api.model_info(repo, revision=revision, files_metadata=True)
        siblings = {s.rfilename: s for s in info.siblings or []}
        missing = [n for n in files if n not in siblings]
        if missing:
            print(f"{repo}@{revision}: REFUSED, missing {', '.join(missing)}. "
                  "The upstream repo changed; check who owns it now.", file=sys.stderr)
            return 1
        print(f"{repo}: pinning commit {revision}")
        hashes = {}
        for name in files:
            lfs = siblings[name].lfs
            if lfs:
                hashes[name] = lfs.sha256
            else:
                tmp = fetch(repo, name, revision)
                hashes[name] = sha256(tmp)
                tmp.replace(target(repo, name))
        lock[repo] = {"revision": revision, "files": hashes}
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {LOCK}. Review the diff before committing it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pin", action="store_true",
                        help="record the current upstream commits and hashes (maintainers)")
    args = parser.parse_args()
    if args.pin:
        return pin()
    return download(json.loads(LOCK.read_text()))


if __name__ == "__main__":
    sys.exit(main())
