#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --job-name=mirage-cuda-scaling
#SBATCH --output=/work/asciibase64/Mirage/results/cuda-worker-scaling/%j.out
#SBATCH --error=/work/asciibase64/Mirage/results/cuda-worker-scaling/%j.err

source "$HOME/.bashrc"
set -euo pipefail
export MIRAGE_PROJECT_ROOT=/work/asciibase64/Mirage/results/stateful-check-e2e/protocols/20261119-observed-check-180s-001/snapshot
export MIRAGE_PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
export MIRAGE_OUTPUT_ROOT=/work/asciibase64/Mirage/results/cuda-worker-scaling
source "$MIRAGE_PROJECT_ROOT/sbatch/lib/environment.sh"
mirage_setup
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
driver="${1:?supply frozen benchmark driver}"
cp -- "$driver" "$MIRAGE_JOB_DIR/driver.py"
sha256sum "$MIRAGE_JOB_DIR/driver.py"
budget="$(mirage_budget 14000 300)"
timeout --kill-after=30 "$budget" srun --exact --exclusive --ntasks=1 --cpus-per-task=12 --gpus-per-node=1 --gpus-per-task=1 "$MIRAGE_PYTHON" "$MIRAGE_JOB_DIR/driver.py" "$MIRAGE_JOB_DIR"
