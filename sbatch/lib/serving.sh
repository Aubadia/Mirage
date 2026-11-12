#!/bin/bash
# One prepared model service and one testing step share the allocation.

mirage_stop_service() {
    local pid attempt
    for pid in "${MIRAGE_TEST_PID:-}" "${MIRAGE_SERVICE_PID:-}"; do
        [[ -n "$pid" ]] || continue
        kill -TERM "$pid" 2>/dev/null || true
        for attempt in {1..20}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    MIRAGE_TEST_PID=""
    MIRAGE_SERVICE_PID=""
}

mirage_start_service() {
    : "${MIRAGE_SERVE_PYTHON:?Select a prepared vLLM environment; no automatic installation}"
    : "${MIRAGE_MODEL_PATH:?Set a prepared local model snapshot under /work/asciibase64/Mirage/models}"
    : "${MIRAGE_MODEL_REVISION:?Record the prepared model revision}"
    local serve_cli="${MIRAGE_SERVE_PYTHON%/*}/vllm"
    if [[ "$MIRAGE_SERVE_PYTHON" != /* || ! -x "$MIRAGE_SERVE_PYTHON" || ! -x "$serve_cli" ]]; then
        echo "Serving requires an absolute Python path and a sibling vllm executable." >&2
        return 2
    fi
    MIRAGE_MODEL_PATH="$(realpath -e -- "$MIRAGE_MODEL_PATH")"
    if [[ "$MIRAGE_MODEL_PATH" != /work/asciibase64/Mirage/models/* || ! -f "$MIRAGE_MODEL_PATH/config.json" ]]; then
        echo "Expected a prepared model snapshot under /work/asciibase64/Mirage/models/." >&2
        return 2
    fi
    export MIRAGE_SERVE_PYTHON MIRAGE_MODEL_PATH MIRAGE_MODEL_REVISION
    export MIRAGE_VLLM_MODEL="${MIRAGE_VLLM_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
    export MIRAGE_MAX_MODEL_LEN="${MIRAGE_MAX_MODEL_LEN:-16384}"
    export MIRAGE_MAX_TOKENS="${MIRAGE_MAX_TOKENS:-4096}"
    export MIRAGE_GENERATION_CONCURRENCY="${MIRAGE_GENERATION_CONCURRENCY:-2}"
    export MIRAGE_VLLM_PORT="${MIRAGE_VLLM_PORT:-8000}"
    export MIRAGE_VLLM_BASE_URL="http://127.0.0.1:${MIRAGE_VLLM_PORT}/v1"
    "$MIRAGE_PYTHON" - <<'PY'
import os
import socket

port = int(os.environ["MIRAGE_VLLM_PORT"])
context = int(os.environ["MIRAGE_MAX_MODEL_LEN"])
output = int(os.environ.get("MIRAGE_MAX_TOKENS", "2048"))
concurrency = int(os.environ["MIRAGE_GENERATION_CONCURRENCY"])
if not 0 < port < 65536 or not 0 < output < context or concurrency <= 0:
    raise SystemExit("Invalid serving port, context/output budget or concurrency")
with socket.socket() as probe:
    probe.bind(("127.0.0.1", port))
PY
    export HF_HOME=/work/asciibase64/Mirage/cache/huggingface
    export VLLM_CACHE_ROOT=/work/asciibase64/Mirage/cache/vllm
    export FLASHINFER_WORKSPACE_BASE=/work/asciibase64/Mirage/cache/flashinfer
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    mkdir -p -- "$HF_HOME" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "${MIRAGE_JOB_DIR}/serving"

    mirage_split_resources
    export OMP_NUM_THREADS=2
    export TORCHINDUCTOR_COMPILE_THREADS=2
    srun --exclusive --exact --ntasks=1 --cpus-per-task=4 --gpus-per-node=1 --gpus-per-task=1 \
        --mem="${MIRAGE_STEP_MEMORY_MB}M" \
        --output="${MIRAGE_JOB_DIR}/serving/stdout.log" \
        --error="${MIRAGE_JOB_DIR}/serving/stderr.log" \
        env -u PYTHONPATH bash -c '
            set -euo pipefail
            "$MIRAGE_SERVE_PYTHON" - "$1" <<'"'"'PY'"'"'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Serving must see exactly one allocated GPU")
config = Path(os.environ["MIRAGE_MODEL_PATH"]) / "config.json"
metadata = {
    "job_id": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
    "step_id": os.environ["SLURM_STEP_ID"], "python": sys.executable,
    "vllm": importlib.metadata.version("vllm"), "torch": str(torch.__version__),
    "transformers": importlib.metadata.version("transformers"),
    "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "model": os.environ["MIRAGE_VLLM_MODEL"], "model_path": str(config.parent),
    "declared_revision": os.environ["MIRAGE_MODEL_REVISION"],
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "endpoint": os.environ["MIRAGE_VLLM_BASE_URL"],
    "max_model_len": int(os.environ["MIRAGE_MAX_MODEL_LEN"]),
    "max_num_seqs": int(os.environ["MIRAGE_GENERATION_CONCURRENCY"]),
}
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2))
PY
            exec "${MIRAGE_SERVE_PYTHON%/*}/vllm" serve "$MIRAGE_MODEL_PATH" \
                --served-model-name "$MIRAGE_VLLM_MODEL" --host 127.0.0.1 \
                --port "$MIRAGE_VLLM_PORT" --tensor-parallel-size 1 \
                --max-model-len "$MIRAGE_MAX_MODEL_LEN" \
                --max-num-seqs "$MIRAGE_GENERATION_CONCURRENCY" \
                --generation-config vllm
        ' mirage-serving "${MIRAGE_JOB_DIR}/serving/environment.json" &
    MIRAGE_SERVICE_PID=$!

    local ready_budget
    ready_budget="$(mirage_budget "${MIRAGE_SERVE_START_TIMEOUT:-900}" 300)"
    "$MIRAGE_PYTHON" - "$ready_budget" "$MIRAGE_SERVICE_PID" <<'PY'
import json
import os
import sys
import time
import urllib.request

deadline = time.monotonic() + int(sys.argv[1])
base = os.environ["MIRAGE_VLLM_BASE_URL"]
while time.monotonic() < deadline:
    try:
        os.kill(int(sys.argv[2]), 0)
    except ProcessLookupError:
        raise SystemExit("Serving step exited before readiness; inspect its logs")
    try:
        with urllib.request.urlopen(base.removesuffix("/v1") + "/health", timeout=2) as response:
            if response.status != 200:
                raise ValueError("Unhealthy service")
        with urllib.request.urlopen(base + "/models", timeout=2) as response:
            models = json.load(response)
        if os.environ["MIRAGE_VLLM_MODEL"] in {item["id"] for item in models["data"]}:
            break
    except (OSError, ValueError, KeyError):
        pass
    time.sleep(1)
else:
    raise SystemExit("Serving readiness deadline expired")
PY
    mirage_check_environment
    mirage_python - "${MIRAGE_JOB_DIR}/serving/environment.json" <<'PY'
import json
from pathlib import Path
import sys
import torch

serving = json.loads(Path(sys.argv[1]).read_text())
if serving["gpu_uuid"] == str(torch.cuda.get_device_properties(0).uuid):
    raise SystemExit("Serving and testing steps share a GPU unexpectedly")
print("Serving and testing GPU isolation verified.")
PY
}

mirage_parallel_run() {
    local argument
    for argument in "$@"; do
        case "$argument" in
            --apis|--apis=*|--cases|--cases=*|--seed|--seed=*|\
            --backend|--backend=*|--precision|--precision=*|\
            --size|--size=*|--epsilon|--epsilon=*|--replay-failures|--replay-failures=*|\
            --canonicals-per-api|--canonicals-per-api=*|--targets-per-canonical|--targets-per-canonical=*|\
            --paths-per-target|--paths-per-target=*|--forward-only) ;;
            --*) echo "Unsupported or managed option: ${argument}" >&2; return 2 ;;
        esac
    done
    mirage_start_service
    local budget
    budget="$(mirage_budget "${MIRAGE_TIME_LIMIT:-13800}" 300)"
    srun --exclusive --exact --ntasks=1 --cpus-per-task=4 --gpus-per-node=1 --gpus-per-task=1 \
        --mem="${MIRAGE_STEP_MEMORY_MB}M" \
        "$MIRAGE_PYTHON" -m mirage run --provider vllm --device cuda \
        --workers "${MIRAGE_WORKERS:-2}" \
        --generation-concurrency "$MIRAGE_GENERATION_CONCURRENCY" \
        --time-limit "$budget" --worker-timeout "${MIRAGE_WORKER_TIMEOUT:-600}" \
        --output "${MIRAGE_JOB_DIR}/cases" "$@" > "${MIRAGE_JOB_DIR}/run.json" &
    MIRAGE_TEST_PID=$!
    while kill -0 "$MIRAGE_TEST_PID" 2>/dev/null; do
        if ! kill -0 "$MIRAGE_SERVICE_PID" 2>/dev/null; then
            echo "Serving exited during the batch." >&2
            return 1
        fi
        sleep 1
    done
    local status=0
    wait "$MIRAGE_TEST_PID" || status=$?
    MIRAGE_TEST_PID=""
    return "$status"
}
