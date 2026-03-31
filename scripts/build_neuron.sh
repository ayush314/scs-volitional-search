#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCS_DIR="${ROOT_DIR}/external/SCSInSCIMechanisms"

python - <<'PY'
import importlib
try:
    importlib.import_module("neuron")
except ImportError as exc:
    raise SystemExit(
        "NEURON is not installed in the active Python environment. "
        "Run `uv pip install -e \".[optim,dev]\"` first."
    ) from exc
PY

if ! command -v nrnivmodl >/dev/null 2>&1; then
  echo "nrnivmodl was not found on PATH. Install NEURON command-line tools first." >&2
  exit 1
fi

pushd "${SCS_DIR}" >/dev/null
nrnivmodl mod_files
popd >/dev/null

SCS_DIR_ENV="${SCS_DIR}" python - <<'PY'
import os
import platform
from pathlib import Path
from neuron import h

repo = Path(os.environ["SCS_DIR_ENV"])
preferred = [
    repo / "arm64" / "libnrnmech.dylib",
    repo / "x86_64" / "libnrnmech.dylib",
    repo / "arm64" / "libnrnmech.so",
    repo / "x86_64" / "libnrnmech.so",
]
candidates = [path for path in preferred if path.exists()]
if not candidates:
    candidates = sorted(repo.glob("**/libnrnmech.dylib")) + sorted(repo.glob("**/libnrnmech.so"))
if not candidates:
    raise SystemExit("Built NEURON library not found after nrnivmodl mod_files.")

library = candidates[0]
try:
    h.nrn_load_dll(str(library))
except RuntimeError as exc:
    print(f"Failed to load {library}: {exc}", flush=True)
    if platform.system() == "Darwin":
        print("", flush=True)
        print("macOS remediation:", flush=True)
        print(f"  xattr -dr com.apple.quarantine {library.parent}", flush=True)
        print(f"  codesign --force --sign - {library}", flush=True)
    raise SystemExit(1) from exc
print(f"Loaded NEURON mechanisms from {library}")
PY

echo "NEURON mechanisms built successfully."
