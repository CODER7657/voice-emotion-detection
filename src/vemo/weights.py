"""fp16-stored weights with fp32 compute, and integrity checks for the model file.

Decided design (roadmap voice #2): the WavLM ONNX graph is stored with its
float32 weights rounded to float16, which halves the download and the deploy
bundle, and the loader casts them back to float32 in memory before the session
is built. Compute stays fp32, so the only difference from the fp32 model is the
rounding of the weights themselves (int8 quantisation was measured and lost 9
points of macro-F1; fp16 storage does not touch activations).

Which initializers were converted is recorded in the model's metadata, so the
loader upcasts exactly those and nothing that was float16 to begin with.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

#: metadata_props key listing the initializers stored as fp16 by ``store_fp16``.
META_KEY = "vemo.fp16_initializers"


def store_fp16(src: Path | str, dst: Path | str) -> dict:
    """Write a copy of ``src`` with every float32 initializer stored as float16.

    The result is not meant to run as is (its consumers still expect float32);
    load it with ``load_model_bytes``. Returns a small report.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(src))
    if any(p.key == META_KEY for p in model.metadata_props):
        raise ValueError(f"{src} already stores fp16 weights")

    converted = []
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT:
            continue
        weights = numpy_helper.to_array(init)
        if not np.all(np.isfinite(weights.astype(np.float16))):
            raise ValueError(f"initializer {init.name!r} overflows float16")
        init.CopyFrom(numpy_helper.from_array(weights.astype(np.float16), init.name))
        converted.append(init.name)

    entry = model.metadata_props.add()
    entry.key, entry.value = META_KEY, json.dumps(converted)
    onnx.save(model, str(dst))
    return {
        "converted": len(converted),
        "src_mb": round(Path(src).stat().st_size / 1e6, 1),
        "dst_mb": round(Path(dst).stat().st_size / 1e6, 1),
    }


def load_model_bytes(path: Path | str) -> bytes:
    """Serialized model ready for ``onnxruntime.InferenceSession``.

    An fp16-stored model has its recorded initializers upcast to float32; any
    other model is returned unchanged, so the loader works for both files.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    names = next((json.loads(p.value) for p in model.metadata_props if p.key == META_KEY), None)
    if names is None:
        return model.SerializeToString()

    wanted = set(names)
    for init in model.graph.initializer:
        if init.name in wanted:
            if init.data_type != onnx.TensorProto.FLOAT16:
                raise ValueError(f"initializer {init.name!r} is listed as fp16 but is not")
            weights = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(weights, init.name))
            wanted.discard(init.name)
    if wanted:
        raise ValueError(f"fp16 initializers missing from the graph: {sorted(wanted)[:3]}")

    kept = [p for p in model.metadata_props if p.key != META_KEY]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    return model.SerializeToString()


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path | str, expected: str) -> None:
    """Raise if the file is not byte-for-byte the pinned release asset."""
    actual = sha256_file(path)
    if actual != expected.strip().lower():
        raise ValueError(f"{Path(path).name}: sha256 {actual} does not match pinned {expected}")
