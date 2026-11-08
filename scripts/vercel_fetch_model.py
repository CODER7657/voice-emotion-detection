"""Vercel build step: download the pinned model into models/ and verify its sha256.

Copied into the deploy bundle as ``vercel_build.py`` and run by Vercel's Python
build (``[tool.vercel.scripts] build``) after the dependencies are installed.
It exists because the Hobby plan caps a CLI upload at 100 MB of source files
(vercel.com/docs/limits) and the model is 190 MB: the bundle uploads without it
and the build fetches the public release asset instead. Standard library only.

    MODEL_URL=<https url> MODEL_SHA256=<hex> python vercel_build.py
(the bundle builder bakes both values in as defaults)
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = ""  # filled in by scripts/build_vercel_bundle.py
MODEL_SHA256 = ""  # filled in by scripts/build_vercel_bundle.py
DEST = Path(__file__).resolve().parent / "models" / "wavlm_pooled_fp16w.onnx"
TIMEOUT_S = 120


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, expected: str, dest: Path, opener=urllib.request.urlopen) -> str:
    """Stream ``url`` to ``dest`` (atomically) and return its sha256; raise on mismatch."""
    if not url.startswith("https://"):
        raise ValueError("model URL must be https")
    expected = expected.strip().lower()
    if len(expected) != 64:
        raise ValueError("a 64-hex-digit sha256 is required")
    if dest.is_file() and _sha256(dest) == expected:
        return expected
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    digest = hashlib.sha256()
    try:
        with opener(url, timeout=TIMEOUT_S) as resp, open(tmp, "wb") as out:
            while block := resp.read(1 << 20):
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"downloaded model sha256 {digest.hexdigest()} != pinned {expected}")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return digest.hexdigest()


def main() -> int:
    url = os.getenv("MODEL_URL") or MODEL_URL
    sha = os.getenv("MODEL_SHA256") or MODEL_SHA256
    digest = fetch(url, sha, DEST)
    print(f"model ready: {DEST.name} sha256 {digest} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
