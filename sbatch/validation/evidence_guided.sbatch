#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --job-name=mirage-state-accept
#SBATCH --output=/work/asciibase64/Mirage/results/stateful-check-e2e/logs/%x-%j.out
#SBATCH --error=/work/asciibase64/Mirage/results/stateful-check-e2e/logs/%x-%j.err

source "$HOME/.bashrc"
set -euo pipefail
frozen="$(realpath -e -- "${1:?frozen protocol}")"
export MIRAGE_PROJECT_ROOT="$frozen/snapshot"
export MIRAGE_PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
export MIRAGE_EXTRA_PYTHONPATH="$MIRAGE_PROJECT_ROOT/tests:$MIRAGE_PROJECT_ROOT/benchmarks"
export MIRAGE_OUTPUT_ROOT=/work/asciibase64/Mirage/results/stateful-check-e2e/validation
source "$MIRAGE_PROJECT_ROOT/sbatch/lib/environment.sh"
mirage_setup
cleanup() {
    local status="$?"
    trap - EXIT
    if [[ -d "$MIRAGE_CACHE_DIR" && "${MIRAGE_CACHE_DIR##*/}" == mirage-"$SLURM_JOB_ID".* ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR" || status=1
    fi
    mirage_finish "$status"
}
trap cleanup EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 MIRAGE_TEST_CUDA=1
export MIRAGE_ARCHIVE_JOB=/work/asciibase64/Mirage/results/stateful-check-e2e/804883
export TMPDIR="$MIRAGE_CACHE_DIR/tmp"
mkdir -p "$TMPDIR"
"$MIRAGE_PYTHON" "$MIRAGE_PROJECT_ROOT/benchmarks/stateful_check_e2e.py" verify "$frozen"
mirage_python -m unittest test_evidence_guided test_state_guidance test_guided_prefix test_http_provider test_multi_api_scheduling test_api_check test_observation_isolation > "$MIRAGE_JOB_DIR/tests.log" 2>&1
mirage_python -m torch.distributed.run --standalone --nproc-per-node=2 "$MIRAGE_PROJECT_ROOT/tests/test_evidence_guided.py" > "$MIRAGE_JOB_DIR/distributed.log" 2>&1
mirage_python "$MIRAGE_PROJECT_ROOT/benchmarks/archive_state_acceptance.py" /work/asciibase64/Mirage/results/stateful-check-e2e/808947 "$MIRAGE_JOB_DIR/archive" > "$MIRAGE_JOB_DIR/archive.log" 2>&1
budget="$(mirage_budget 3500 100)"
timeout --kill-after=20 "$budget" srun --exclusive --exact --ntasks=1 --cpus-per-task=12 --gpus-per-node=1 --gpus-per-task=1 \
    "$MIRAGE_PYTHON" "$MIRAGE_PROJECT_ROOT/benchmarks/archive_pool_acceptance.py" \
    /work/asciibase64/Mirage/results/stateful-check-e2e/805400 \
    /work/asciibase64/Mirage/results/cuda-worker-scaling/806666/cohort.json "$MIRAGE_JOB_DIR"
