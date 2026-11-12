#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-vllm-smoke
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
source "${mirage_root}/sbatch/lib/serving.sh"
export MIRAGE_WORKERS=1
mirage_parallel_run --apis torch.matmul --cases 4 --canonicals-per-api 1 --targets-per-canonical 4 --paths-per-target 1 --seed 0
mirage_stop_service
"$MIRAGE_PYTHON" - "${MIRAGE_JOB_DIR}/run.json" <<'CHECK'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
if report['coverage']['api']['covered'] != 1:
    raise SystemExit('Target API was not verified')
print('Canonical path verified. Review MR results in run.json.')
CHECK
