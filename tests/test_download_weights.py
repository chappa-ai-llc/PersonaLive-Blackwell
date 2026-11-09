"""tools/download_weights.py: pinning and verification, with a fake Hugging Face Hub."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import download_weights as dw  # noqa: E402

REV = "a" * 40
SMALL = b'{"small": true}'


def lfs_hash(name):
    return hashlib.sha256(name.encode()).hexdigest()


class DownloadWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(dw, "LOCK", self.tmp / "weights.lock.json"))
        self.enterContext(mock.patch.object(dw, "DEST", self.tmp / "pretrained_weights"))

    def hub(self, drop=()):
        """Install a fake huggingface_hub: every listed file exists unless in `drop`;
        *.json files are plain Git files, everything else is in LFS."""
        tmp = self.tmp

        class FakeApi:
            def model_info(self, repo, revision=None, files_metadata=False):
                files = [n for n in dw.SOURCES[repo][1] if n not in drop]
                sib = [types.SimpleNamespace(
                    rfilename=n,
                    lfs=None if n.endswith(".json") else types.SimpleNamespace(sha256=lfs_hash(n)))
                    for n in files]
                return types.SimpleNamespace(sha=REV, siblings=sib)

        def hf_hub_download(repo_id, filename, revision):
            assert revision == REV
            p = tmp / "cache" / repo_id / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(SMALL)
            return str(p)

        self.enterContext(mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(
            HfApi=FakeApi, hf_hub_download=hf_hub_download)}))

    def saved(self):
        return [p for p in dw.DEST.rglob("*") if p.is_file()]

    def test_pin_takes_lfs_hashes_from_metadata_and_hashes_small_files(self):
        self.hub()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dw.pin(), 0)
        lock = json.loads(dw.LOCK.read_text())
        self.assertEqual(set(lock), set(dw.SOURCES))
        for repo, (_, files, _) in dw.SOURCES.items():
            self.assertEqual(lock[repo]["revision"], REV)
            for n in files:
                want = hashlib.sha256(SMALL).hexdigest() if n.endswith(".json") else lfs_hash(n)
                self.assertEqual(lock[repo]["files"][n], want)
        # Only the small files were downloaded.
        self.assertTrue(self.saved())
        self.assertTrue(all(p.suffix == ".json" for p in self.saved()))

    def test_pin_refuses_a_repo_missing_a_listed_file(self):
        self.hub(drop={"unet/diffusion_pytorch_model.bin"})
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dw.pin(), 1)
        self.assertFalse(dw.LOCK.exists())
        self.assertIn("REFUSED", err.getvalue())

    def test_download_refuses_a_hash_mismatch(self):
        self.hub()
        repo = "stabilityai/sd-vae-ft-mse"
        lock = {repo: {"revision": REV, "files": {n: "0" * 64 for n in dw.SOURCES[repo][1]}}}
        self.enterContext(mock.patch.object(dw, "SOURCES", {repo: dw.SOURCES[repo]}))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dw.download(lock), 1)
        self.assertEqual(self.saved(), [])

    def test_lock_file_is_complete_and_uses_the_verified_base_repo(self):
        lock = json.loads((Path(dw.__file__).parent / "weights.lock.json").read_text())
        self.assertEqual(set(lock), set(dw.SOURCES))
        self.assertFalse(any(r.startswith("lambdalabs/") for r in lock))
        for repo, (_, files, _) in dw.SOURCES.items():
            self.assertRegex(lock[repo]["revision"] or "", dw.SHA_RE)
            for n in files:
                self.assertRegex(lock[repo]["files"].get(n) or "", dw.HASH_RE)


if __name__ == "__main__":
    unittest.main()
