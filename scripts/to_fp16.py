"""Store the served WavLM model's weights as fp16 (compute stays fp32).

    python scripts/to_fp16.py                                   # models/ in, models/ out
    python scripts/to_fp16.py --src models/wavlm_pooled_fp32.onnx --dst models/wavlm_pooled_fp16w.onnx

Halves the file (378 MB -> 190 MB) for the GitHub Release and the deploy bundle.
``vemo.serve.Model`` upcasts the weights back to fp32 in memory before building the
session (``vemo.weights.load_model_bytes``), so the only change from the fp32 model is
the rounding of the weights. Writes ``SHA256SUMS`` next to the output for the release.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vemo.serve import ONNX_FILE, ONNX_FP32_FILE  # noqa: E402
from vemo.weights import sha256_file, store_fp16  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("models") / ONNX_FP32_FILE)
    ap.add_argument("--dst", type=Path, default=Path("models") / ONNX_FILE)
    ap.add_argument("--expect-src-sha256", help="refuse to convert unless the input matches")
    args = ap.parse_args(argv)

    if args.expect_src_sha256:
        actual = sha256_file(args.src)
        if actual != args.expect_src_sha256.strip().lower():
            print(f"{args.src}: sha256 {actual} does not match the expected value", file=sys.stderr)
            return 1

    report = store_fp16(args.src, args.dst)
    digest = sha256_file(args.dst)
    (args.dst.parent / "SHA256SUMS.fp16w").write_text(f"{digest}  {args.dst.name}\n", encoding="utf-8")
    print(f"{report['converted']} initializers -> fp16; {report['src_mb']} MB -> {report['dst_mb']} MB")
    print(f"sha256 {digest}  {args.dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
