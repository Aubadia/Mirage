#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-control-tests
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
# This suite uses only standard-library stubs; no model or PyTorch installation.
budget="$(mirage_budget 900 120)"
timeout --signal=TERM --kill-after=30s "${budget}s" \
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    env PYTHONPATH="${PYTHONPATH}:${MIRAGE_PROJECT_ROOT}/tests" \
    "$MIRAGE_PYTHON" -m unittest -v \
    test_http_provider test_cli test_sbatch test_artifacts \
    > "${MIRAGE_JOB_DIR}/tests.log" 2>&1
cat -- "${MIRAGE_JOB_DIR}/tests.log"
