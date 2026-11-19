#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=192G
#SBATCH --time=1:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-protocol-accept
#SBATCH --output=/work/asciibase64/Mirage/results/stateful-check-e2e/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Mirage/results/stateful-check-e2e/logs/%x-%j.err

source "$HOME/.bashrc"
set -euo pipefail
frozen="$(realpath -e -- "${1:?frozen protocol}")"
archive_root=/work/asciibase64/Mirage/results/stateful-check-e2e
export MIRAGE_PROJECT_ROOT="$frozen/snapshot" MIRAGE_PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
export MIRAGE_OUTPUT_ROOT="$archive_root/validation"
export MIRAGE_EXTRA_PYTHONPATH="$MIRAGE_PROJECT_ROOT/tests:$MIRAGE_PROJECT_ROOT/benchmarks"
source "$MIRAGE_PROJECT_ROOT/sbatch/lib/environment.sh"
source "$MIRAGE_PROJECT_ROOT/sbatch/lib/serving.sh"
mirage_setup
cleanup() {
    local status="$?"
    trap - EXIT
    mirage_stop_service || status=1
    if [[ -d "$MIRAGE_CACHE_DIR" && "${MIRAGE_CACHE_DIR##*/}" == mirage-"$SLURM_JOB_ID".* ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR" || status=1
    fi
    mirage_finish "$status"
}
trap cleanup EXIT
export MIRAGE_STEP_CPUS=12 MIRAGE_TEST_CUDA=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export MIRAGE_ARCHIVE_JOB="$archive_root/804883" TMPDIR="$MIRAGE_CACHE_DIR/tmp"
mkdir -p "$TMPDIR"
"$MIRAGE_PYTHON" "$MIRAGE_PROJECT_ROOT/benchmarks/stateful_check_e2e.py" verify "$frozen"
driver="$MIRAGE_PROJECT_ROOT/benchmarks/archive_protocol_acceptance.py"
mirage_python -m unittest test_evidence_guided test_state_guidance test_guided_prefix test_http_provider test_api_check test_multi_api_scheduling test_observation_isolation > "$MIRAGE_JOB_DIR/tests.log" 2>&1
mirage_python "$driver" controls "$archive_root" "$MIRAGE_JOB_DIR" > "$MIRAGE_JOB_DIR/controls.log" 2>&1
export MIRAGE_SERVE_PYTHON=/work/asciibase64/Mirage/environments/vllm/qwen3-coder-next/bin/python
export MIRAGE_MODEL_REVISION=37561816e66b5b9153b1b64c10968c18c1e6f4a0
export MIRAGE_MODEL_PATH="/work/asciibase64/Mirage/models/Qwen3.8-27B/$MIRAGE_MODEL_REVISION"
export MIRAGE_VLLM_MODEL=Qwen/Qwen3.8-27B MIRAGE_MAX_TOKENS=4096 MIRAGE_MAX_MODEL_LEN=16384
export MIRAGE_MTP_TOKENS=3 MIRAGE_TEMPERATURE=0 MIRAGE_TOP_P=1 MIRAGE_GENERATION_CONCURRENCY=2
export MIRAGE_VLLM_PORT=18932 MIRAGE_SESSION_DIR="$MIRAGE_JOB_DIR" MIRAGE_REQUEST_RECEIPTS="$MIRAGE_JOB_DIR/request-receipts"
mkdir -p "$MIRAGE_REQUEST_RECEIPTS"
start="$(date +%s)"
mirage_start_service
echo "Service startup $(( $(date +%s) - start )) s, restarts 0"
# A failed real-model iteration keeps its service for another frozen iteration.
iteration="$MIRAGE_JOB_DIR"
while true; do
    start="$(date +%s)"
    status=0
    mirage_python "$driver" dispatch "$archive_root" "$iteration" > "$iteration/dispatch.log" 2>&1 || status=$?
    echo "Dispatch $iteration elapsed $(( $(date +%s) - start )) s status $status"
    if [[ "$status" == 0 ]]; then break; fi
    while [[ ! -s "$MIRAGE_JOB_DIR/retry-protocol" ]]; do
        [[ ! -e "$MIRAGE_JOB_DIR/stop" ]] && kill -0 "$MIRAGE_SERVICE_PID" || exit "$status"
        sleep 2
    done
    [[ ! -e "$MIRAGE_SESSION_DIR/cancellation-unconfirmed" ]] || exit "$status"
    read -r frozen < "$MIRAGE_JOB_DIR/retry-protocol"
    mv "$MIRAGE_JOB_DIR/retry-protocol" "$iteration/retry-protocol.used"
    MIRAGE_PROJECT_ROOT="$frozen/snapshot"
    export PYTHONPATH="$MIRAGE_PROJECT_ROOT/src:$MIRAGE_PROJECT_ROOT/tests:$MIRAGE_PROJECT_ROOT/benchmarks"
    "$MIRAGE_PYTHON" "$MIRAGE_PROJECT_ROOT/benchmarks/stateful_check_e2e.py" verify "$frozen"
    driver="$MIRAGE_PROJECT_ROOT/benchmarks/archive_protocol_acceptance.py"
    iteration="$(mktemp -d "$MIRAGE_JOB_DIR/iteration-XXXXXX")"
    mirage_python -m unittest test_evidence_guided test_state_guidance test_guided_prefix test_http_provider test_api_check test_multi_api_scheduling test_observation_isolation > "$iteration/tests.log" 2>&1
    mirage_python "$driver" controls "$archive_root" "$iteration" > "$iteration/controls.log" 2>&1
done
