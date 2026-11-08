"""Assemble a self-contained Vercel deploy directory for the speech-emotion API.

    python scripts/build_vercel_bundle.py --model models/wavlm_pooled_fp16w.onnx \
        [--fetch-url <release asset url>] [--out .vercel-bundle] [--sha256 <pinned>] \
        [--region bom1] [--measure-deps]

Two ways to get the model into the function:

* copy (default): the model is copied into ``models/`` of the bundle. The
  upload is then about 190 MB, over the Hobby plan's 100 MB cap on source files
  uploaded by the CLI (vercel.com/docs/limits; Pro allows 1 GB).
* ``--fetch-url``: the model is NOT copied. The bundle gets ``vercel_build.py``
  (scripts/vercel_fetch_model.py with the URL and hash baked in), wired as the
  Vercel build command, which downloads the public release asset during
  Vercel's build and fails the build unless its sha256 matches the pin.
  ``--model`` is still required: it pins the hash and counts in the size check.

Layout (mirrors the repo, so api/index.py finds vemo and models/ the same way):

    api/index.py                 FastAPI ``app`` (the function entrypoint)
    src/vemo/{__init__,head,trim,weights}.py
    models/head.json
    models/wavlm_pooled_fp16w.onnx   copy mode (fetch mode: written by vercel_build.py)
    vercel_build.py              fetch mode only
    pyproject.toml               the ONLY dependency source (pins from api/requirements.txt),
                                 Python 3.12, tool.vercel.entrypoint = api.index:app
    .python-version, vercel.json, .vemo-bundle (marker: safe to delete on rebuild)
    public/                      --web-dist only: the built site, same origin as the API

No requirements.txt is emitted: Vercel's Python builder installs from the native
pyproject.toml when one exists and only converts requirements.txt when it does
not (vercel/vercel packages/python/src/install.ts, checked 2026-11-08), and the
bundle needs pyproject.toml anyway for the entrypoint and build hook.

The bundle, the model and the installed dependencies must stay under Vercel's
Python function limit: 500 MB uncompressed (vercel.com/docs/functions/limitations,
checked 2026-11-08). ``--measure-deps`` installs the pinned requirements for
CPython 3.12 manylinux x86_64 into a temporary directory to measure them too.
"""

from __future__ import annotations

import argparse
import json
import tomllib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vemo.weights import sha256_file, verify_sha256  # noqa: E402

LIMIT_MB = 500  # Vercel Python function, uncompressed
VEMO_MODULES = ("__init__.py", "head.py", "trim.py", "weights.py")
ONNX_FILE = "wavlm_pooled_fp16w.onnx"
MARKER = ".vemo-bundle"
MAX_DURATION_S = 60  # cold start (load + upcast 190 MB) plus one inference; Hobby max is 300
NL = "\n"


WEB_HEADERS = ROOT / "web" / "security-headers.json"


def vercel_config(region: str, headers: dict[str, str] | None = None) -> dict:
    # Memory is not set here: with Fluid compute it is a dashboard setting, and the
    # Hobby default (2 GB / 1 vCPU) is already the Hobby maximum.
    cfg: dict = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        # Without this the project's dashboard preset decides; "Other" (the default for a
        # CLI-created project) builds a static site with no function and every path 404s.
        "framework": "fastapi",
        "regions": [region],
        "functions": {
            "api/index.py": {
                "maxDuration": MAX_DURATION_S,
                "includeFiles": "{models/**,src/vemo/**}",
            }
        },
    }
    if headers:
        # One policy for the site (public/, served by Vercel's CDN) and the API responses.
        cfg["headers"] = [{"source": "/(.*)",
                           "headers": [{"key": k, "value": v} for k, v in headers.items()]}]
    return cfg


def pyproject(requirements: list[str], fetch: bool = False) -> str:
    deps = "".join(f'    "{r}",{NL}' for r in requirements)
    text = (
        "[project]\n"
        'name = "vemo-api"\n'
        'version = "1.0.0"\n'
        'requires-python = ">=3.12,<3.13"\n'
        f"dependencies = [\n{deps}]\n\n"
        "[tool.vercel]\n"
        'entrypoint = "api.index:app"\n'
    )
    if fetch:
        text += '\n[tool.vercel.scripts]\nbuild = "python vercel_build.py"\n'
    return text


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def bundle_dependencies(bundle: Path) -> list[str]:
    """The dependency list Vercel will install: [project].dependencies of the bundle's pyproject."""
    with open(bundle / "pyproject.toml", "rb") as handle:
        return list(tomllib.load(handle)["project"]["dependencies"])


def measure_deps(bundle: Path, run=subprocess.run) -> int:
    """Install the bundle's own pyproject dependencies (py3.12 manylinux) and measure them."""
    deps = bundle_dependencies(bundle)
    with tempfile.TemporaryDirectory() as tmp:
        reqs = Path(tmp) / "requirements-from-pyproject.txt"
        reqs.write_text(NL.join(deps) + NL, encoding="utf-8")
        target = Path(tmp) / "site"
        run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
             "--target", str(target), "--only-binary=:all:", "--python-version", "3.12",
             "--implementation", "cp", "--platform", "manylinux_2_28_x86_64",
             "--platform", "manylinux2014_x86_64", "-r", str(reqs)],
            check=True, timeout=600,
        )
        return dir_size(target) if target.exists() else 0


def _bake(src: str, name: str, value: str) -> str:
    placeholder = f'{name} = ""'
    if src.count(placeholder) != 1:
        raise ValueError(f"vercel_fetch_model.py must contain exactly one `{placeholder}` placeholder "
                         f"to bake the {name} into; found {src.count(placeholder)}")
    baked = src.replace(placeholder, f"{name} = {value!r}", 1)
    if baked == src:
        raise ValueError(f"failed to bake {name} into vercel_build.py")
    return baked


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline=NL)


def build(model: Path, out: Path, sha256: str | None = None, region: str = "bom1",
          head: Path = ROOT / "models" / "head.json", fetch_url: str | None = None,
          web_dist: Path | None = None) -> Path:
    model, head, out = Path(model), Path(head), Path(out)
    if web_dist is not None and not (Path(web_dist) / "index.html").is_file():
        raise FileNotFoundError(f"no index.html in {web_dist} (run `npm run build` in web/)")
    if not head.is_file():
        raise FileNotFoundError(f"head not found: {head}")
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    if fetch_url is not None and not fetch_url.startswith("https://"):
        raise ValueError("--fetch-url must be https")
    if sha256:
        verify_sha256(model, sha256)
    pinned = (sha256 or sha256_file(model)).strip().lower()
    if out.exists():
        if (out / MARKER).is_file():
            shutil.rmtree(out)
        elif not out.is_dir() or any(out.iterdir()):
            raise FileExistsError(f"{out} exists and was not created by this script (no {MARKER} marker); "
                                  "refusing to delete it")
    (out / "api").mkdir(parents=True)
    (out / "src" / "vemo").mkdir(parents=True)
    (out / "models").mkdir()

    shutil.copy2(ROOT / "api" / "index.py", out / "api" / "index.py")
    for name in VEMO_MODULES:
        shutil.copy2(ROOT / "src" / "vemo" / name, out / "src" / "vemo" / name)
    shutil.copy2(head, out / "models" / "head.json")
    if fetch_url:
        src = (ROOT / "scripts" / "vercel_fetch_model.py").read_text(encoding="utf-8")
        src = _bake(_bake(src, "MODEL_URL", fetch_url), "MODEL_SHA256", pinned)
        _write(out / "vercel_build.py", src)
    else:
        shutil.copy2(model, out / "models" / ONNX_FILE)

    reqs = [ln.strip() for ln in (ROOT / "api" / "requirements.txt").read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    _write(out / "pyproject.toml", pyproject(reqs, fetch=bool(fetch_url)))
    _write(out / ".python-version", "3.12\n")
    headers = None
    if web_dist is not None:
        # The site is served from the same origin as the API: no CORS, one deploy.
        shutil.copytree(web_dist, out / "public")
        headers = json.loads(WEB_HEADERS.read_text(encoding="utf-8"))
    _write(out / "vercel.json", json.dumps(vercel_config(region, headers), indent=2) + NL)
    _write(out / MARKER, "Created by scripts/build_vercel_bundle.py; rebuilt (deleted) on each run." + NL)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, required=True, help="path to wavlm_pooled_fp16w.onnx")
    ap.add_argument("--head", type=Path, default=ROOT / "models" / "head.json")
    ap.add_argument("--out", type=Path, default=ROOT / ".vercel-bundle")
    ap.add_argument("--sha256", help="refuse to build unless the model hashes to this")
    ap.add_argument("--region", default="bom1")
    ap.add_argument("--fetch-url", help="download the model at Vercel build time from this https URL "
                                        "instead of uploading it (Hobby caps CLI uploads at 100 MB)")
    ap.add_argument("--web-dist", type=Path, help="built site (web/dist) to serve from public/ with "
                                                  "web/security-headers.json")
    ap.add_argument("--measure-deps", action="store_true",
                    help="also install the pinned requirements (py3.12 manylinux) to measure them")
    args = ap.parse_args(argv)

    out = build(args.model, args.out, args.sha256, args.region, args.head, args.fetch_url, args.web_dist)
    bundle = dir_size(out)
    print(f"bundle (uploaded): {out} = {bundle / 1e6:.1f} MB")
    total = bundle
    if args.fetch_url:
        model_bytes = args.model.stat().st_size
        total += model_bytes
        print(f"model (fetched during the Vercel build) = {model_bytes / 1e6:.1f} MB")
    if args.measure_deps:
        deps = measure_deps(out)
        total += deps
        print(f"dependencies (py3.12 manylinux, installed) = {deps / 1e6:.1f} MB")
    print(f"total = {total / 1e6:.1f} MB of {LIMIT_MB} MB Vercel Python limit (uncompressed)")
    if total >= LIMIT_MB * 1e6:
        print("ERROR: over the Vercel Python function size limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
