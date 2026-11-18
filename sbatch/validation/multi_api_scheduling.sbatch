#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-multi-api
#SBATCH --output=result/validation/logs/%x-%j.out
#SBATCH --error=result/validation/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch from the Mirage repository." >&2
    exit 2
fi
source "$HOME/.bashrc"
set -euo pipefail
mirage_root="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "$mirage_root/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=validation
mirage_setup
cleanup() {
    local status="$?" removed=false
    trap - EXIT
    if [[ -d "$MIRAGE_CACHE_DIR" && "${MIRAGE_CACHE_DIR##*/}" == mirage-"$SLURM_JOB_ID".* ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR" || status=1
    fi
    [[ -e "$MIRAGE_CACHE_DIR" ]] || removed=true
    printf '{"exit_status":%s,"cache_removed":%s}\n' "$status" "$removed" > "$MIRAGE_JOB_DIR/cleanup.json"
    mirage_finish "$status"
}
trap cleanup EXIT
snapshot="$MIRAGE_JOB_DIR/snapshot"
mkdir -p "$snapshot"
cp -a "$MIRAGE_PROJECT_ROOT/src" "$MIRAGE_PROJECT_ROOT/tests" "$MIRAGE_PROJECT_ROOT/benchmarks" "$MIRAGE_PROJECT_ROOT/sbatch" "$snapshot/"
find "$snapshot" -type f -name '*.py' -exec sha256sum {} + > "$MIRAGE_JOB_DIR/source-sha256.txt"
export MIRAGE_TEST_E2E=1 MIRAGE_TEST_CUDA=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
if [[ "$#" -eq 0 ]]; then
    set -- test_multi_api_scheduling test_pipeline test_exploration test_cli test_artifacts test_http_provider
fi
budget="$(mirage_budget 13800 180)"
status=0
timeout --signal=TERM --kill-after=30s "${budget}s" \
    srun --exclusive --exact --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-node=1 --gpus-per-task=1 \
    env PYTHONPATH="$snapshot/src:$snapshot/tests" \
    "$MIRAGE_PYTHON" -m unittest -v --failfast "$@" > "$MIRAGE_JOB_DIR/tests.log" 2>&1 || status=$?
cat "$MIRAGE_JOB_DIR/tests.log"
exit "$status"
