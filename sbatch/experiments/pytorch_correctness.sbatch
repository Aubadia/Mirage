#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=4:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=mirage-properties
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
source "$mirage_root/result/serving/761235/serving.env"
source "$mirage_root/sbatch/lib/serving.sh"
started=$(date +%s)
mirage_start_service
export MIRAGE_SESSION_DIR="$MIRAGE_JOB_DIR"
printf '{"startup_seconds":%s,"restart_count":0}\n' "$(( $(date +%s)-started ))" > "$MIRAGE_JOB_DIR/session.json"
for name in MIRAGE_SESSION_DIR MIRAGE_PROJECT_ROOT MIRAGE_PYTHON MIRAGE_VLLM_BASE_URL MIRAGE_VLLM_MODEL MIRAGE_STEP_MEMORY_MB MIRAGE_MAX_TOKENS; do
    printf 'export %s=%q\n' "$name" "${!name:-}" >> "$MIRAGE_JOB_DIR/session.env"
done
touch "$MIRAGE_JOB_DIR/ready"
while [[ ! -f "$MIRAGE_JOB_DIR/stop" ]]; do
    if ! kill -0 "$MIRAGE_SERVICE_PID" 2>/dev/null; then exit 1; fi
    sleep 2
done
