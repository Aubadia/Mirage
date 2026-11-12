#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-prepare
#SBATCH --output=result/serving/logs/%x-%j.out
#SBATCH --error=result/serving/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch from the Mirage repository." >&2
    exit 2
fi

source "$HOME/.bashrc"
set -euo pipefail

mirage_root="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "${mirage_root}/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=serving
mirage_setup
export MIRAGE_PYTHON MIRAGE_JOB_DIR

export MIRAGE_SERVE_ENV=/work/asciibase64/Mirage/environments/vllm/qwen3-coder-next
export UV_CACHE_DIR=/work/asciibase64/Mirage/cache/uv
export PIP_CACHE_DIR=/work/asciibase64/Mirage/cache/pip
export HF_HOME=/work/asciibase64/Mirage/cache/huggingface
export VLLM_CACHE_ROOT=/work/asciibase64/Mirage/cache/vllm
export XDG_CACHE_HOME=/work/asciibase64/Mirage/cache/xdg
export FLASHINFER_WORKSPACE_BASE=/work/asciibase64/Mirage/cache/flashinfer
export UV_PYTHON_DOWNLOADS=never
export TMPDIR="/work/asciibase64/Mirage/cache/tmp/prepare-${SLURM_JOB_ID}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

budget="$(mirage_budget 13800 300)"
srun --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --gpus-per-task=1 \
    timeout --signal=TERM --kill-after=30s "${budget}s" bash -s <<'SH'
set -euo pipefail

mkdir -p -- "$TMPDIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" \
    "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE"

# Reuse the interpreter, never the testing environment's site-packages.
if [[ ! -d "$MIRAGE_SERVE_ENV" ]]; then
    uv venv --seed --no-python-downloads \
        --python "$MIRAGE_PYTHON" "$MIRAGE_SERVE_ENV"
fi
if [[ ! -x "$MIRAGE_SERVE_ENV/bin/python" ]]; then
    echo "The serving directory exists but contains no usable interpreter." >&2
    exit 2
fi

uv pip install --python "$MIRAGE_SERVE_ENV/bin/python" \
    --torch-backend=auto --only-binary=:all: \
    'vllm>=0.15.0' huggingface_hub

export PATH="$MIRAGE_SERVE_ENV/bin:$PATH"
vllm download-kernels

uv pip freeze --python "$MIRAGE_SERVE_ENV/bin/python" \
    > "$MIRAGE_JOB_DIR/serving-requirements.txt"

"$MIRAGE_SERVE_ENV/bin/python" - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys

from huggingface_hub import HfApi, snapshot_download
import torch
import vllm

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Preparation requires one Slurm-visible CUDA GPU")

run_directory = Path(os.environ["MIRAGE_JOB_DIR"])
model = "Qwen/Qwen3-Coder-Next-FP8"
revision = HfApi().model_info(model, revision=os.environ.get("MIRAGE_MODEL_REVISION")).sha
destination = Path("/work/asciibase64/Mirage/models") / "Qwen3-Coder-Next-FP8" / revision
metadata = {
    "model": model,
    "revision": revision,
    "destination": str(destination),
    "python": sys.version,
    "interpreter": sys.executable,
    "vllm": vllm.__version__,
    "torch": str(torch.__version__),
    "transformers": importlib.metadata.version("transformers"),
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "status": "downloading",
}
record = run_directory / "preparation.json"
record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Downloading {model} at revision {revision}", flush=True)

snapshot_download(repo_id=model, revision=revision, local_dir=destination, max_workers=4)

# Require all indexed shards before publishing the usable serving configuration.
index = json.loads((destination / "model.safetensors.index.json").read_text())
for shard in set(index["weight_map"].values()):
    path = destination / shard
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty model shard: {path}")

settings = {
    "MIRAGE_SERVE_PYTHON": str(Path(os.environ["MIRAGE_SERVE_ENV"]) / "bin/python"),
    "MIRAGE_MODEL_PATH": str(destination),
    "MIRAGE_MODEL_REVISION": revision,
    "MIRAGE_VLLM_MODEL": model,
}
output = run_directory / "serving.env"
output.write_text(
    "".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()),
    encoding="utf-8",
)
metadata["status"] = "prepared"
record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Serving configuration: {output}", flush=True)
PY
SH
