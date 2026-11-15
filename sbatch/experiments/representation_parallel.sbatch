#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-repr-parallel
#SBATCH --output=result/representation-parallel/logs/%x-%j.out
#SBATCH --error=result/representation-parallel/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit with sbatch.' >&2
    exit 2
fi
source "$HOME/.bashrc"
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=representation-parallel
mirage_setup
parallel_cleanup() {
    local status="$?"
    trap - EXIT
    if [[ -d "$MIRAGE_CACHE_DIR" && "${MIRAGE_CACHE_DIR##*/}" == mirage-"$SLURM_JOB_ID".* ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR"
    fi
    mirage_finish "$status"
}
trap parallel_cleanup EXIT
mirage_check_environment
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
protocol_root="$MIRAGE_PROJECT_ROOT/result/representation-parallel/protocol-01"
budget="$(mirage_budget 13800 300)"
timeout --signal=TERM --kill-after=30s "$((budget+30))s" \
    srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --gpus-per-task=1 \
    "$MIRAGE_PYTHON" "$protocol_root/experiment.py" run --protocol "$protocol_root" \
    --output "$MIRAGE_JOB_DIR/measurement" --max-seconds "$budget" \
    > "$MIRAGE_JOB_DIR/measurement.stdout.log" 2> "$MIRAGE_JOB_DIR/measurement.stderr.log"
