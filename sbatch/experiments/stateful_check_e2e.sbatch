#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --job-name=mirage-check-e2e
#SBATCH --output=/work/asciibase64/Mirage/results/stateful-check-e2e/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Mirage/results/stateful-check-e2e/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch." >&2
    exit 2
fi
source "$HOME/.bashrc"
set -euo pipefail
frozen="$(realpath -e -- "${1:?Pass the directory created by stateful_check_e2e.py prepare}")"
export MIRAGE_PROJECT_ROOT="$frozen/snapshot"
export MIRAGE_PYTHON="${MIRAGE_PYTHON:-${SLURM_SUBMIT_DIR:?}/.venv/bin/python}"
export MIRAGE_OUTPUT_ROOT=/work/asciibase64/Mirage/results/stateful-check-e2e
MIRAGE_CATEGORY=stateful-check-e2e
source "$MIRAGE_PROJECT_ROOT/sbatch/lib/environment.sh"
source "$MIRAGE_PROJECT_ROOT/sbatch/lib/serving.sh"
mirage_setup
driver="$MIRAGE_PROJECT_ROOT/benchmarks/stateful_check_e2e.py"
startup_started=0
service_seconds=0
test_started=0
cleanup() {
    local status="$?" now
    trap - EXIT
    mirage_stop_service || status=1
    now="$(date +%s)"
    if [[ -d "$MIRAGE_CACHE_DIR" && "${MIRAGE_CACHE_DIR##*/}" == mirage-"$SLURM_JOB_ID".* ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR" || status=1
    fi
    if [[ -e "$MIRAGE_CACHE_DIR" ]]; then status=1; fi
    printf '{"exit_status":%d,"startup_seconds":%d,"test_and_audit_seconds":%d,"elapsed_seconds":%d,"allocated_gpu_seconds_estimate":%d,"restart_count":0}\n' \
        "$status" "$service_seconds" "$(( test_started ? now-test_started : 0 ))" "$((now-MIRAGE_STARTED_AT))" \
        "$((2*(now-MIRAGE_STARTED_AT)))" > "$MIRAGE_JOB_DIR/cleanup.json"
    mirage_finish "$status"
}
trap cleanup EXIT
"$MIRAGE_PYTHON" "$driver" verify "$frozen"
cp -- "$frozen/protocol.json" "$frozen/protocol.sha256" "$MIRAGE_JOB_DIR/"
if [[ -z "${MIRAGE_SERVE_PYTHON:-}" ]]; then
    export MIRAGE_SERVE_PYTHON=/work/asciibase64/Mirage/environments/vllm/qwen3-coder-next/bin/python
fi
export MIRAGE_MODEL_REVISION=37561816e66b5b9153b1b64c10968c18c1e6f4a0
export MIRAGE_MODEL_PATH="/work/asciibase64/Mirage/models/Qwen3.8-27B/$MIRAGE_MODEL_REVISION"
export MIRAGE_VLLM_MODEL=Qwen/Qwen3.8-27B
export MIRAGE_MAX_TOKENS=4096 MIRAGE_MAX_MODEL_LEN=16384 MIRAGE_MTP_TOKENS=3
export MIRAGE_TEMPERATURE=0 MIRAGE_TOP_P=1 MIRAGE_GENERATION_CONCURRENCY=2
export MIRAGE_SERVE_START_TIMEOUT=1800 MIRAGE_VLLM_PORT="${MIRAGE_VLLM_PORT:-18931}"
export TMPDIR="$MIRAGE_CACHE_DIR/tmp"
mkdir -p -- "$TMPDIR"
startup_started="$(date +%s)"
export MIRAGE_GENERATION_STOP_FILE="$MIRAGE_JOB_DIR/serving-exited"
export MIRAGE_REQUEST_RECEIPTS="$MIRAGE_JOB_DIR/request-receipts"
mkdir -p "$MIRAGE_REQUEST_RECEIPTS"
mirage_start_service
service_seconds="$(( $(date +%s) - startup_started ))"
printf '{"health_http_status":200,"model_list_verified":true,"startup_seconds":%d,"model":"%s","revision":"%s","ready_epoch":%d}\n' \
    "$service_seconds" "$MIRAGE_VLLM_MODEL" "$MIRAGE_MODEL_REVISION" "$(date +%s)" > "$MIRAGE_JOB_DIR/service-ready.json"
export MIRAGE_SESSION_DIR="$MIRAGE_JOB_DIR" MIRAGE_STEP_CPUS=12
export TORCHINDUCTOR_CACHE_DIR="$MIRAGE_CACHE_DIR/test/inductor" TRITON_CACHE_DIR="$MIRAGE_CACHE_DIR/test/triton"
export XDG_CACHE_HOME="$MIRAGE_CACHE_DIR/test/xdg"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
budget="$(mirage_budget 43200 1800)"
overall="$(mirage_budget 43200 120)"
test_started="$(date +%s)"
timeout --signal=TERM --kill-after=60s "${overall}s" \
    srun --exclusive --exact --ntasks=1 --cpus-per-task=12 --gpus-per-node=1 --gpus-per-task=1 \
    --mem="${MIRAGE_STEP_MEMORY_MB}M" "$MIRAGE_PYTHON" "$driver" campaign "$frozen" "$MIRAGE_JOB_DIR" "$budget" "$overall" \
    > "$MIRAGE_JOB_DIR/campaign.stdout.log" 2> "$MIRAGE_JOB_DIR/campaign.stderr.log" &
MIRAGE_TEST_PID=$!
while kill -0 "$MIRAGE_TEST_PID" 2>/dev/null; do
    if ! kill -0 "$MIRAGE_SERVICE_PID" 2>/dev/null; then
        echo "vLLM exited; draining submitted tests and preserving per-API results" >&2
        touch "$MIRAGE_GENERATION_STOP_FILE"
        break
    fi
    sleep 2
done
status=0
wait "$MIRAGE_TEST_PID" || status=$?
MIRAGE_TEST_PID=""
if [[ -e "$MIRAGE_GENERATION_STOP_FILE" ]]; then status=1; fi
exit "$status"

