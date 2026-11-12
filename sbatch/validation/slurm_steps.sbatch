#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=4:00:00
#SBATCH --job-name=mirage-step-isolation
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
source "${mirage_root}/sbatch/lib/serving.sh"
MIRAGE_CATEGORY=validation
mirage_setup
mirage_split_resources
export OMP_NUM_THREADS=2

# Hold the first GPU while the testing step starts, without loading a model.
srun --exclusive --exact --ntasks=1 --cpus-per-task=4 --gpus-per-node=1 --gpus-per-task=1 \
    --mem="${MIRAGE_STEP_MEMORY_MB}M" \
    "$MIRAGE_PYTHON" - "$MIRAGE_JOB_DIR" <<'PY' &
import json
from pathlib import Path
import sys
import time
import torch

root = Path(sys.argv[1])
assert torch.cuda.device_count() == 1
torch.ones(1, device="cuda")
(root / "held-gpu.json").write_text(json.dumps({"uuid": str(torch.cuda.get_device_properties(0).uuid)}))
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    if (root / "tested-gpu.json").exists():
        break
    time.sleep(0.1)
else:
    raise SystemExit("The second step did not complete while the first held its GPU")
PY
MIRAGE_SERVICE_PID=$!

mirage_python - "$MIRAGE_JOB_DIR" <<'PY'
import json
from pathlib import Path
import sys
import time
import torch

root = Path(sys.argv[1])
assert torch.cuda.device_count() == 1
torch.ones(1, device="cuda")
deadline = time.monotonic() + 90
while not (root / "held-gpu.json").exists():
    if time.monotonic() >= deadline:
        raise SystemExit("The first step did not start concurrently")
    time.sleep(0.1)
held = json.loads((root / "held-gpu.json").read_text())
tested = str(torch.cuda.get_device_properties(0).uuid)
assert held["uuid"] != tested, "Steps received the same GPU"
(root / "tested-gpu.json").write_text(json.dumps({"uuid": tested, "different_from_held": True}))
print("Concurrent Slurm steps received distinct GPUs.")
PY
wait "$MIRAGE_SERVICE_PID"
MIRAGE_SERVICE_PID=""
