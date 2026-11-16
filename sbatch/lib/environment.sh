#!/bin/bash
# Shared setup for batch jobs; sourcing this file alone does not launch work.

mirage_finish() {
    local status="$1"
    local ended_at
    trap - EXIT
    if declare -F mirage_stop_service >/dev/null; then
        mirage_stop_service || status=1
    fi
    ended_at="$(date +%s)"
    echo "========================================"
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Total Cost: $((ended_at - MIRAGE_STARTED_AT)) s"
    echo "Exit Status: ${status}"
    echo "========================================"
    exit "$status"
}

mirage_setup() {
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "Submit this script with sbatch." >&2
        return 2
    fi
    : "${SLURM_CPUS_PER_TASK:?Missing Slurm CPU allocation}"

    MIRAGE_STARTED_AT="$(date +%s)"
    trap 'mirage_finish "$?"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    echo "========================================"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Slurm Job: ${SLURM_JOB_ID}"
    echo "========================================"

    cd -- "${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Missing submission directory}}"
    MIRAGE_PROJECT_ROOT="$(pwd -P)"
    if [[ ! -f src/mirage/cli.py || ! -d tests ]]; then
        echo "MIRAGE_PROJECT_ROOT must name the Mirage repository." >&2
        return 2
    fi

    MIRAGE_PYTHON="${MIRAGE_PYTHON:-${MIRAGE_PROJECT_ROOT}/.venv/bin/python}"
    if [[ "$MIRAGE_PYTHON" != /* ]]; then
        MIRAGE_PYTHON="${MIRAGE_PROJECT_ROOT}/${MIRAGE_PYTHON}"
    fi
    if [[ ! -x "$MIRAGE_PYTHON" ]]; then
        echo "MIRAGE_PYTHON is not executable: ${MIRAGE_PYTHON}" >&2
        return 2
    fi

    # Do not inherit an accidental login-node /tmp dependency installation.
    export PYTHONPATH="${MIRAGE_PROJECT_ROOT}/src${MIRAGE_EXTRA_PYTHONPATH:+:${MIRAGE_EXTRA_PYTHONPATH}}"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export TORCHINDUCTOR_COMPILE_THREADS="$SLURM_CPUS_PER_TASK"

    MIRAGE_JOB_DIR="${MIRAGE_OUTPUT_ROOT:-${MIRAGE_PROJECT_ROOT}/result/${MIRAGE_CATEGORY:-experiments}}/${SLURM_JOB_ID}"
    mkdir -p -- "$MIRAGE_JOB_DIR"
    MIRAGE_JOB_DIR="$(cd -- "$MIRAGE_JOB_DIR" && pwd -P)"

    local scratch_root
    scratch_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
    MIRAGE_CACHE_DIR="$(mktemp -d "${scratch_root%/}/mirage-${SLURM_JOB_ID}.XXXXXX")"
    export TORCHINDUCTOR_CACHE_DIR="${MIRAGE_CACHE_DIR}/inductor"
    export TRITON_CACHE_DIR="${MIRAGE_CACHE_DIR}/triton"
    export XDG_CACHE_HOME="${MIRAGE_CACHE_DIR}/xdg"

    # Slurm owns CUDA visibility. Never select a GPU by its physical node index.
    printf 'Project: %s\nPython: %s\nArtifacts: %s\n' \
        "$MIRAGE_PROJECT_ROOT" "$MIRAGE_PYTHON" "$MIRAGE_JOB_DIR"
}

mirage_python() {
    local memory_args=()
    if [[ -n "${MIRAGE_STEP_MEMORY_MB:-}" ]]; then
        memory_args=(--mem="${MIRAGE_STEP_MEMORY_MB}M")
    fi
    srun --exclusive --exact --ntasks=1 --gpus-per-node=1 --gpus-per-task=1 \
        --cpus-per-task="${MIRAGE_STEP_CPUS:-$SLURM_CPUS_PER_TASK}" \
        "${memory_args[@]}" \
        "$MIRAGE_PYTHON" "$@"
}

mirage_split_resources() {
    # Step defaults can reserve both GPUs and all host memory despite --exact.
    local total_memory="${SLURM_MEM_PER_NODE:-}"
    if [[ -z "$total_memory" && "${SLURM_MEM_PER_GPU:-}" =~ ^[1-9][0-9]*$ ]]; then
        total_memory=$((SLURM_MEM_PER_GPU * 2))
    fi
    if [[ -z "$total_memory" && "${SLURM_MEM_PER_CPU:-}" =~ ^[1-9][0-9]*$ ]]; then
        total_memory=$((SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK))
    fi
    if [[ ! "$total_memory" =~ ^[1-9][0-9]*$ || "$total_memory" -lt 2 ]]; then
        echo "Cannot divide the allocation's memory between concurrent steps." >&2
        return 2
    fi
    MIRAGE_STEP_CPUS=4
    MIRAGE_STEP_MEMORY_MB=$((total_memory / 2))
    printf 'Per-step CPUs: %s; memory: %s MiB\n' "$MIRAGE_STEP_CPUS" "$MIRAGE_STEP_MEMORY_MB"
}

mirage_check_environment() {
    mirage_python - > "${MIRAGE_JOB_DIR}/environment.json" <<'PY'
import json
import os
import sys

if sys.version_info < (3, 14):
    raise SystemExit("Mirage requires Python >= 3.14; select MIRAGE_PYTHON.")

try:
    import torch
except ImportError as error:
    raise SystemExit(
        "PyTorch is unavailable on the allocated node. Set MIRAGE_PYTHON or "
        "MIRAGE_EXTRA_PYTHONPATH to a prepared, compute-node-accessible environment."
    ) from error

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("This job requires exactly one Slurm-visible CUDA GPU.")

print(json.dumps({
    "job_id": os.environ["SLURM_JOB_ID"],
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "hostname": os.uname().nodename,
    "python": sys.version,
    "executable": sys.executable,
    "torch": str(torch.__version__),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}, indent=2))
PY
}

mirage_budget() {
    local requested="$1"
    local reserve="$2"
    local workers="${3:-1}"
    local remaining
    remaining="$(squeue --noheader --jobs="$SLURM_JOB_ID" --format=%L)"

    # %L is a duration, so this calculation is independent of node time zones.
    "$MIRAGE_PYTHON" - "$requested" "$reserve" "$remaining" "$workers" <<'PY'
import re
import sys

requested, reserve = (int(value) for value in sys.argv[1:3])
remaining = sys.argv[3].strip()
workers = int(sys.argv[4])
match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+)(?::(\d+))?", remaining)
if not match or requested <= 0 or reserve < 0 or workers < 1:
    raise SystemExit("Expected positive budget and a finite Slurm remaining time.")

days, first, second, third = match.groups()
if third is None:
    seconds = int(first) * 60 + int(second)
else:
    seconds = int(first) * 3600 + int(second) * 60 + int(third)
seconds += int(days or 0) * 86400
budget = min(requested, (seconds - reserve) // workers)
if budget <= 0:
    raise SystemExit("Insufficient allocation time remains to run and preserve results.")
print(budget)
PY
}
