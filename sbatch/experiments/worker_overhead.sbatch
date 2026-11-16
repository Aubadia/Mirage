#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-worker-overhead
#SBATCH --output=result/worker-overhead/logs/%x-%j.out
#SBATCH --error=result/worker-overhead/logs/%x-%j.err

# Nano4 limits CPUs to 12 per GPU. This 32-CPU experiment was preflighted and
# submitted with explicit overrides: --partition=dev --gpus-per-node=3.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch." >&2
    exit 2
fi
source "$HOME/.bashrc"
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=worker-overhead
mirage_setup
cleanup() {
    local status="$?"
    trap - EXIT
    if [[ -d "$MIRAGE_CACHE_DIR" && "${MIRAGE_CACHE_DIR##*/}" == mirage-"$SLURM_JOB_ID".* ]]; then
        rm -rf -- "$MIRAGE_CACHE_DIR"
    fi
    mirage_finish "$status"
}
trap cleanup EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
for name in MIRAGE_PROJECT_ROOT MIRAGE_PYTHON MIRAGE_JOB_DIR MIRAGE_CACHE_DIR \
    TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR XDG_CACHE_HOME; do
    printf 'export %s=%q\n' "$name" "${!name}" >> "$MIRAGE_JOB_DIR/session.env"
done
budget="$(mirage_budget 3600 300)"
timeout --signal=TERM --kill-after=30 "$budget" \
    srun --exclusive --exact --ntasks=1 --cpus-per-task=32 --gpus-per-node=1 --gpus-per-task=1 \
    "$MIRAGE_PYTHON" "$MIRAGE_PROJECT_ROOT/result/worker-overhead/experiment.py" initial \
    --output "$MIRAGE_JOB_DIR/initial" > "$MIRAGE_JOB_DIR/initial.stdout.log" 2> "$MIRAGE_JOB_DIR/initial.stderr.log"
touch "$MIRAGE_JOB_DIR/ready"
while [[ ! -f "$MIRAGE_JOB_DIR/stop" ]]; do sleep 2; done
