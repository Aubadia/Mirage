"""Exercise shell lifecycle failure paths with local command stubs."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ServingShellTests(unittest.TestCase):
    def test_memory_is_divided_instead_of_inheriting_the_whole_job(self):
        script = r'''
set -euo pipefail
source "$1/sbatch/lib/environment.sh"
SLURM_MEM_PER_NODE=409600
mirage_split_resources
[[ "$MIRAGE_STEP_MEMORY_MB" == 204800 && "$MIRAGE_STEP_CPUS" == 4 ]]
unset SLURM_MEM_PER_NODE
SLURM_MEM_PER_GPU=204800
mirage_split_resources
[[ "$MIRAGE_STEP_MEMORY_MB" == 204800 ]]
'''
        result = subprocess.run(["bash", "-c", script, "test", str(ROOT)],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_test_step_failure_is_preserved_and_service_is_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = r'''
set -euo pipefail
source "$1/sbatch/lib/serving.sh"
MIRAGE_JOB_DIR="$2"
MIRAGE_PYTHON=unused-by-stub
MIRAGE_GENERATION_CONCURRENCY=2
MIRAGE_STEP_MEMORY_MB=1
mirage_start_service() {
    sleep 60 &
    MIRAGE_SERVICE_PID=$!
    echo "$MIRAGE_SERVICE_PID" > "$MIRAGE_JOB_DIR/service.pid"
}
mirage_budget() { echo 10; }
srun() { return 17; }
trap 'mirage_stop_service' EXIT
mirage_parallel_run --cases 1
'''
            result = subprocess.run(["bash", "-c", script, "test", str(ROOT), temporary],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 17, result.stderr)
            pid = int((Path(temporary) / "service.pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_managed_options_are_rejected_before_starting_service(self):
        script = r'''
set -euo pipefail
source "$1/sbatch/lib/serving.sh"
mirage_start_service() { echo unexpected-start; }
mirage_parallel_run --time-limit 999999
'''
        result = subprocess.run(["bash", "-c", script, "test", str(ROOT)],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("unexpected-start", result.stdout)

    def test_scripts_reject_execution_without_allocation(self):
        environment = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        for path in ROOT.glob("sbatch/*/*.sbatch"):
            with self.subTest(script=path.name):
                result = subprocess.run(["bash", str(path)], env=environment,
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("sbatch", result.stderr)
