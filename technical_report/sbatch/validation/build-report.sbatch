#!/usr/bin/env bash
#SBATCH --job-name=mirage-report
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=technical_report/validation/%x-%j.out
#SBATCH --error=technical_report/validation/%x-%j.err
set -eo pipefail
source "$HOME/.bashrc"
set -u
REPORT_START=$(date +%s)
REPORT_MODE=${1:-report}
REPORT_ROOT="$SLURM_SUBMIT_DIR/technical_report"
finish_report() {
    local code=$?
    trap - EXIT
    local end
    end=$(date +%s)
    printf 'end_time=%s elapsed_seconds=%s exit_status=%s\n' "$(date --iso-8601=seconds)" "$((end-REPORT_START))" "$code"
    printf '{"job_id":"%s","mode":"%s","start_epoch":%s,"end_epoch":%s,"elapsed_seconds":%s,"exit_status":%s}\n' "$SLURM_JOB_ID" "$REPORT_MODE" "$REPORT_START" "$end" "$((end-REPORT_START))" "$code" > "$REPORT_ROOT/validation/job-$SLURM_JOB_ID.json"
    exit "$code"
}
trap finish_report EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'start_time=%s job_id=%s mode=%s\n' "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" "$REPORT_MODE"
srun --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 timeout --signal=TERM --kill-after=20 900 bash "$REPORT_ROOT/scripts/build.sh" "$REPORT_MODE"
