#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=mirage-replay
#SBATCH --output=result/replay/logs/%x-%j.out
#SBATCH --error=result/replay/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch from the Mirage repository." >&2
    exit 2
fi

source "$HOME/.bashrc"
set -euo pipefail

mirage_root="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "${mirage_root}/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=replay
mirage_setup

if [[ "$#" -eq 0 ]]; then
    echo "Usage: sbatch sbatch/replay/replay.sbatch MANIFEST [replay options]" >&2
    exit 2
fi
for argument in "$@"; do
    case "$argument" in
        --worker-timeout|--worker-timeout=*)
            echo "Use MIRAGE_WORKER_TIMEOUT to set the replay worker limit." >&2
            exit 2
            ;;
        --allow-environment-change|--source-program|--source-program=*|\
        --followup-program|--followup-program=*|--size|--size=*|\
        --updates|--updates=*|--iteration|--iteration=*|--help)
            ;;
        --*)
            echo "Unsupported or abbreviated option: ${argument}" >&2
            exit 2
            ;;
    esac
done

worker_timeout="${MIRAGE_WORKER_TIMEOUT:-600}"
if [[ ! "$worker_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "MIRAGE_WORKER_TIMEOUT must be a positive integer." >&2
    exit 2
fi

mirage_check_environment
worker_timeout="$(mirage_budget "$worker_timeout" 300 2)"

mirage_python -m mirage replay "$@" \
    --worker-timeout "$worker_timeout" > "${MIRAGE_JOB_DIR}/replay.json"

echo "Replay result: ${MIRAGE_JOB_DIR}/replay.json"
