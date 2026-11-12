#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --job-name=mirage-smoke
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
mirage_check_environment

budget="$(mirage_budget 600 300)"
mirage_python -m mirage run \
    --provider reference --apis torch.matmul --device cuda \
    --backend eager --precision fp32 --cases 1 --updates 2 --seed 0 \
    --time-limit "$budget" --worker-timeout 120 \
    --output "${MIRAGE_JOB_DIR}/cases" > "${MIRAGE_JOB_DIR}/run.json"

manifest="$("$MIRAGE_PYTHON" - "${MIRAGE_JOB_DIR}/run.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
case = result["experiments"][0]
if case["status"] != "executed" or case["verdict"] != "pass":
    raise SystemExit(f"Smoke case did not pass: {case}")
print(case["artifact"])
PY
)"

replay_timeout="$(mirage_budget 120 120 2)"
mirage_python -m mirage replay "$manifest" \
    --worker-timeout "$replay_timeout" > "${MIRAGE_JOB_DIR}/replay.json"

"$MIRAGE_PYTHON" - "${MIRAGE_JOB_DIR}/replay.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
if result["verdict"] != "pass" or result["provider_calls"] != 0 or result["environment_changes"]:
    raise SystemExit(f"Smoke replay did not reproduce the expected result: {result}")
print("Smoke run and LLM-free replay passed.")
PY
