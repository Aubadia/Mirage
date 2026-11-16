#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-node=3
#SBATCH --mem=192G
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --job-name=mirage-e2e-latency
#SBATCH --output=result/e2e-latency/logs/%x-%j.out
#SBATCH --error=result/e2e-latency/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch." >&2
    exit 2
fi
source "$HOME/.bashrc"
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=e2e-latency
mirage_setup
printf '%s\n' "$MIRAGE_CACHE_DIR" > "$MIRAGE_JOB_DIR/cache-path.txt"
source "$MIRAGE_PROJECT_ROOT/result/benchmark-revalidation-qwen38/serving.env"
source "$MIRAGE_PROJECT_ROOT/sbatch/lib/serving.sh"
export MIRAGE_MTP_TOKENS=3 MIRAGE_VLLM_PORT=18915 MIRAGE_MAX_MODEL_LEN=16384
export MIRAGE_MAX_TOKENS=4096 MIRAGE_GENERATION_CONCURRENCY=2
export MIRAGE_TEMPERATURE=0 MIRAGE_TOP_P=1
export TMPDIR="$MIRAGE_CACHE_DIR/tmp"
mkdir -p "$TMPDIR"
cleanup() {
    local status="$?"
    trap - EXIT
    mirage_stop_service || status=1
    if [[ -d "$MIRAGE_CACHE_DIR" && "${MIRAGE_CACHE_DIR##*/}" == mirage-"$SLURM_JOB_ID".* ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR" || status=1
    fi
    if [[ ! -e "$MIRAGE_CACHE_DIR" ]]; then
        touch "$MIRAGE_JOB_DIR/cache-removed"
    else
        status=1
    fi
    mirage_finish "$status"
}
trap cleanup EXIT
# Prepare/tests precede model loading; execute a frozen copy of project code.
mkdir "$MIRAGE_JOB_DIR/preflight-code"
cp -a tests sbatch benchmarks "$MIRAGE_JOB_DIR/preflight-code/"
# A candidate may remain experimental until the live quality review is complete.
preflight_engine="${MIRAGE_PREFLIGHT_ENGINE:-$MIRAGE_PROJECT_ROOT/src}"
if [[ ! -d "$preflight_engine/mirage" ]]; then
    echo "MIRAGE_PREFLIGHT_ENGINE must contain the frozen mirage package." >&2
    exit 2
fi
cp -a -- "$preflight_engine" "$MIRAGE_JOB_DIR/preflight-code/src"
srun --exclusive --exact --ntasks=1 --cpus-per-task=4 --gpus-per-node=1 --gpus-per-task=1 \
    --mem=16384M env PYTHONPATH="$MIRAGE_JOB_DIR/preflight-code/src:$MIRAGE_JOB_DIR/preflight-code/tests" \
    "$MIRAGE_PYTHON" -m unittest test_sbatch test_artifacts -v \
    > "$MIRAGE_JOB_DIR/preflight.stdout.log" 2> "$MIRAGE_JOB_DIR/preflight.stderr.log"
started="$(date +%s)"
mirage_start_service
export MIRAGE_SESSION_DIR="$MIRAGE_JOB_DIR" MIRAGE_STEP_CPUS=32
printf '{"startup_seconds":%s,"restart_count":0}\n' "$(( $(date +%s)-started ))" > "$MIRAGE_JOB_DIR/session.json"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
for name in MIRAGE_SESSION_DIR MIRAGE_PROJECT_ROOT MIRAGE_PYTHON MIRAGE_VLLM_BASE_URL \
    MIRAGE_VLLM_MODEL MIRAGE_MODEL_REVISION MIRAGE_STEP_MEMORY_MB MIRAGE_STEP_CPUS \
    MIRAGE_MAX_TOKENS MIRAGE_MTP_TOKENS MIRAGE_CACHE_DIR TMPDIR TORCHINDUCTOR_CACHE_DIR \
    TRITON_CACHE_DIR XDG_CACHE_HOME OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS \
    NUMEXPR_NUM_THREADS TORCHINDUCTOR_COMPILE_THREADS MIRAGE_TEMPERATURE MIRAGE_TOP_P; do
    printf 'export %s=%q\n' "$name" "${!name}" >> "$MIRAGE_JOB_DIR/session.env"
done
touch "$MIRAGE_JOB_DIR/ready"
while [[ ! -f "$MIRAGE_JOB_DIR/stop" ]]; do
    if ! kill -0 "$MIRAGE_SERVICE_PID" 2>/dev/null; then exit 1; fi
    sleep 2
done
