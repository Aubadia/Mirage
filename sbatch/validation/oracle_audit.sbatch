#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=mirage-oracle
#SBATCH --output=result/validation/logs/%x-%j.out
#SBATCH --error=result/validation/logs/%x-%j.err
source "$HOME/.bashrc"
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=validation
mirage_setup
oracle_finish() {
    local status="$?"
    # Remove only disposable scratch owned by this allocation, including interruption.
    case "${MIRAGE_CACHE_DIR:-}" in /tmp/mirage-"$SLURM_JOB_ID".*) rm -rf -- "$MIRAGE_CACHE_DIR" || status=1 ;; esac
    mirage_finish "$status"
}
trap oracle_finish EXIT
# Preserve exact code for every iteration before any computational work.
mkdir -p "$MIRAGE_JOB_DIR/code"
cp -a src tests benchmarks sbatch "$MIRAGE_JOB_DIR/code/"
export PYTHONPATH="$MIRAGE_JOB_DIR/code/src"
mirage_check_environment
export MIRAGE_TEST_CUDA=1
args=(--stage "${MIRAGE_ORACLE_STAGE:-regression}" --output "$MIRAGE_JOB_DIR/run")
if [[ -n "${MIRAGE_ORACLE_CANDIDATE:-}" ]]; then args+=(--candidate "$MIRAGE_ORACLE_CANDIDATE"); fi
budget=$(mirage_budget 13800 180)
status=0
timeout --signal=TERM --kill-after=30 "$budget" srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
 "$MIRAGE_PYTHON" "$MIRAGE_JOB_DIR/code/benchmarks/oracle_generalization.py" "${args[@]}" \
 > "$MIRAGE_JOB_DIR/run.log" 2>&1 || status=$?
printf '{"exit_status":%s}\n' "$status" > "$MIRAGE_JOB_DIR/status.json"
exit "$status"
