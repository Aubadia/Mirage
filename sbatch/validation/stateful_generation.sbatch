#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-stateful
#SBATCH --output=result/validation/logs/%x-%j.out
#SBATCH --error=result/validation/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch from the Mirage repository." >&2
    exit 2
fi

source "$HOME/.bashrc"
set -euo pipefail

mirage_root="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "${mirage_root}/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=validation
mirage_setup
mirage_check_environment
snapshot="${MIRAGE_JOB_DIR}/snapshot"
mkdir -p "$snapshot"
cp -a "${MIRAGE_PROJECT_ROOT}/src" "${MIRAGE_PROJECT_ROOT}/tests" "${MIRAGE_PROJECT_ROOT}/benchmarks" "${MIRAGE_PROJECT_ROOT}/sbatch" "$snapshot/"
cp -- "${BASH_SOURCE[0]}" "${MIRAGE_JOB_DIR}/submitted.sbatch"
find "$snapshot" -type f -name '*.py' -exec sha256sum {} + > "${MIRAGE_JOB_DIR}/source-sha256.txt"
trap 'mirage_status=$?; rm -rf -- "$MIRAGE_CACHE_DIR"; mirage_finish "$mirage_status"' EXIT

export MIRAGE_TEST_COMPILE=1
export MIRAGE_TEST_CUDA=1
export MIRAGE_TEST_E2E=1
if [[ "$#" -eq 0 ]]; then
    set -- test_capabilities
fi
budget="$(mirage_budget 13800 300)"
timeout --signal=TERM --kill-after=30s "${budget}s" \
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    env PYTHONPATH="${snapshot}/src:${snapshot}/tests" \
    "$MIRAGE_PYTHON" -m unittest -v "$@" \
    > "${MIRAGE_JOB_DIR}/tests.log" 2>&1
cat -- "${MIRAGE_JOB_DIR}/tests.log"

if [[ "${MIRAGE_STATEFUL_BENCHMARK:-0}" == 1 ]]; then
    budget="$(mirage_budget 1800 300)"
    timeout --signal=TERM --kill-after=30s "${budget}s" \
        srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
        env PYTHONPATH="${snapshot}/src:${snapshot}/benchmarks/pytorch_correctness" \
        "$MIRAGE_PYTHON" "$snapshot/benchmarks/pytorch_correctness/evaluate.py" validate-stateful \
        --output "${MIRAGE_JOB_DIR}/stateful-benchmark" --timeout 180 \
        > "${MIRAGE_JOB_DIR}/benchmark.log" 2>&1
    cat -- "${MIRAGE_JOB_DIR}/benchmark.log"
fi
