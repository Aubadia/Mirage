#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --job-name=mirage-check-regression
#SBATCH --output=/work/asciibase64/Mirage/results/stateful-check-e2e/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Mirage/results/stateful-check-e2e/logs/%x-%j.err

source "$HOME/.bashrc"
set -euo pipefail
frozen="${1:?Pass a compute-accessible frozen source/test directory}"
export MIRAGE_PROJECT_ROOT="$frozen" MIRAGE_OUTPUT_ROOT=/work/asciibase64/Mirage/results/stateful-check-e2e/validation
export MIRAGE_PYTHON="${MIRAGE_PYTHON:-${SLURM_SUBMIT_DIR:?}/.venv/bin/python}"
export MIRAGE_EXTRA_PYTHONPATH="$frozen/tests:$frozen/benchmarks"
source "$frozen/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=validation
mirage_setup
export MIRAGE_TEST_COMPILE=1 MIRAGE_TEST_CUDA=1 MIRAGE_TEST_E2E=1
export MIRAGE_CHECK_TEST_OUTPUT_ROOT="$MIRAGE_JOB_DIR/check-runs"
mkdir -p "$MIRAGE_CHECK_TEST_OUTPUT_ROOT" "$MIRAGE_CACHE_DIR/tmp"
export TMPDIR="$MIRAGE_CACHE_DIR/tmp" OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
cleanup() {
    local status="$?"
    trap - EXIT
    if [[ -d "$MIRAGE_CACHE_DIR" && "${MIRAGE_CACHE_DIR##*/}" == mirage-"$SLURM_JOB_ID".* ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR" || status=1
    fi
    mirage_finish "$status"
}
trap cleanup EXIT
additional=()
if [[ -n "${2:-}" ]]; then
    export MIRAGE_ARCHIVE_JOB="$2"
    srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --gpus-per-task=1 \
        "$MIRAGE_PYTHON" "$frozen/benchmarks/archive_observation_acceptance.py" "$2" "$MIRAGE_JOB_DIR" \
        > "$MIRAGE_JOB_DIR/archive.log" 2>&1
    additional=(test_observation_isolation test_numerical_gradients test_oracle_routing)
fi
budget="$(mirage_budget 2550 120)"
timeout --signal=TERM --kill-after=20s "${budget}s" srun --exclusive --exact --ntasks=1 \
    --cpus-per-task=8 --gpus-per-node=1 --gpus-per-task=1 "$MIRAGE_PYTHON" -m unittest \
    "${additional[@]}" test_guided_inputs test_state_guidance test_guided_prefix test_multi_api_scheduling test_api_check -v \
    > "$MIRAGE_JOB_DIR/tests.log" 2>&1
