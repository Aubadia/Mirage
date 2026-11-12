#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=mirage-tests
#SBATCH --output=artifacts/slurm/logs/%x-%j.out
#SBATCH --error=artifacts/slurm/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch from the Mirage repository." >&2
    exit 2
fi

source "$HOME/.bashrc"
set -euo pipefail

mirage_root="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "${mirage_root}/sbatch/lib/environment.sh"
mirage_setup
mirage_check_environment

export MIRAGE_TEST_COMPILE=1
export MIRAGE_TEST_CUDA=1
export MIRAGE_TEST_E2E=1

# Stop the test process before Slurm's hard limit, retaining its log on shared storage.
budget="$(mirage_budget 13800 300)"
timeout --signal=TERM --kill-after=30s "${budget}s" \
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    "$MIRAGE_PYTHON" -m unittest discover -s tests -v \
    > "${MIRAGE_JOB_DIR}/tests.log" 2>&1

cat -- "${MIRAGE_JOB_DIR}/tests.log"
