#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-correctness
#SBATCH --output=result/pytorch-correctness/logs/%x-%j.out
#SBATCH --error=result/pytorch-correctness/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch." >&2
    exit 2
fi
source "$HOME/.bashrc"
set -euo pipefail
mirage_root="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "$mirage_root/sbatch/lib/environment.sh"
MIRAGE_OUTPUT_ROOT="$mirage_root/result/pytorch-correctness"
mirage_setup
benchmark_finish() {
    local status="$1"
    trap - EXIT
    if [[ -n "${MIRAGE_CACHE_DIR:-}" ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR"
    fi
    mirage_finish "$status"
}
trap 'benchmark_finish "$?"' EXIT
# Freeze the runner, engine, catalog and tests before any computational process.
mkdir -p "$MIRAGE_JOB_DIR/snapshot/benchmarks/pytorch_correctness" "$MIRAGE_JOB_DIR/snapshot/tests" "$MIRAGE_JOB_DIR/snapshot/sbatch/replay"
cp -a "$mirage_root/src" "$MIRAGE_JOB_DIR/snapshot/"
cp "$mirage_root"/benchmarks/pytorch_correctness/{cases.py,run.py,candidates.json,README.md} "$MIRAGE_JOB_DIR/snapshot/benchmarks/pytorch_correctness/"
cp "$mirage_root/tests/test_pytorch_benchmark.py" "$MIRAGE_JOB_DIR/snapshot/tests/"
cp "$mirage_root/sbatch/replay/pytorch_correctness.sbatch" "$MIRAGE_JOB_DIR/snapshot/sbatch/replay/"
export PYTHONPATH="$MIRAGE_JOB_DIR/snapshot/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
find "$MIRAGE_JOB_DIR/snapshot" -type f ! -path '*/__pycache__/*' -exec sha256sum {} + > "$MIRAGE_JOB_DIR/snapshot-sha256.txt"
mirage_check_environment
mirage_python -m unittest discover -s "$MIRAGE_JOB_DIR/snapshot/tests" -p test_pytorch_benchmark.py -v > "$MIRAGE_JOB_DIR/runner-tests.txt" 2>&1
budget="$(mirage_budget 12000 180)"
set +e
mirage_python "$MIRAGE_JOB_DIR/snapshot/benchmarks/pytorch_correctness/run.py" \
    --output "$MIRAGE_JOB_DIR/run" --time-limit "$budget" "$@"
status=$?
set -e
# The job remains nonzero for oracle violations (3) and execution failures (2).
rm -rf -- "$MIRAGE_CACHE_DIR"
exit "$status"
