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
mirage_parallel_run --apis torch.matmul --cases 1 --updates 2 --seed 0
manifest="$("$MIRAGE_PYTHON" - "${MIRAGE_JOB_DIR}/run.json" <<'PY_CHECK'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
case = report["experiments"][0]
if case["status"] != "executed" or case["verdict"] != "pass":
    raise SystemExit(f"Real-model smoke did not pass: {case}")
print(case["artifact"])
PY_CHECK
)"
mirage_stop_service
replay_timeout="$(mirage_budget 120 120 2)"
mirage_python -m mirage replay "$manifest" --worker-timeout "$replay_timeout" \
    > "${MIRAGE_JOB_DIR}/replay.json"
"$MIRAGE_PYTHON" - "${MIRAGE_JOB_DIR}/replay.json" <<'PY_CHECK'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
if result["verdict"] != "pass" or result["provider_calls"] != 0 or result["environment_changes"]:
    raise SystemExit(f"Replay did not reproduce the smoke result: {result}")
print("Real-model smoke and LLM-free replay passed.")
PY_CHECK
