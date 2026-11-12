#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=mirage-run
#SBATCH --output=result/experiments/logs/%x-%j.out
#SBATCH --error=result/experiments/logs/%x-%j.err

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch from the Mirage repository." >&2
    exit 2
fi

source "$HOME/.bashrc"
set -euo pipefail

mirage_root="${MIRAGE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
source "${mirage_root}/sbatch/lib/environment.sh"
MIRAGE_CATEGORY=experiments
mirage_setup

: "${MIRAGE_PROVIDER:?Set MIRAGE_PROVIDER=vllm or module:factory}"
for argument in "$@"; do
    case "$argument" in
        --provider|--provider=*|--device|--device=*|\
        --output|--output=*|--time-limit|--time-limit=*|\
        --worker-timeout|--worker-timeout=*)
            echo "Managed option ${argument}; use the documented MIRAGE_* environment variables." >&2
            exit 2
            ;;
        --apis|--apis=*|--cases|--cases=*|--seed|--seed=*|\
        --backend|--backend=*|--precision|--precision=*|\
        --size|--size=*|--epsilon|--epsilon=*|\
        --canonicals-per-api|--canonicals-per-api=*|--targets-per-canonical|--targets-per-canonical=*|\
        --paths-per-target|--paths-per-target=*|--forward-only|\
        --workers|--workers=*|--generation-concurrency|--generation-concurrency=*|\
        --replay-failures|--replay-failures=*|--help)
            ;;
        --*)
            echo "Unsupported or abbreviated option: ${argument}" >&2
            exit 2
            ;;
    esac
done

mirage_check_environment
budget="$(mirage_budget "${MIRAGE_TIME_LIMIT:-13800}" 300)"
worker_timeout="${MIRAGE_WORKER_TIMEOUT:-600}"
if [[ ! "$worker_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "MIRAGE_WORKER_TIMEOUT must be a positive integer." >&2
    exit 2
fi

mirage_python -m mirage run \
    --provider "$MIRAGE_PROVIDER" --device cuda \
    --time-limit "$budget" --worker-timeout "$worker_timeout" \
    --output "${MIRAGE_JOB_DIR}/cases" "$@" > "${MIRAGE_JOB_DIR}/run.json"

echo "Batch result: ${MIRAGE_JOB_DIR}/run.json"
