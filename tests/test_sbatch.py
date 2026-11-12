"""Exercise shell lifecycle failure paths with local command stubs."""

import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ServingShellTests(unittest.TestCase):
    def test_development_failure_preserves_service_and_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            session = Path(temporary) / "session"
            for path in (project / "src", project / "tests", project / "sbatch/lib", session):
                path.mkdir(parents=True, exist_ok=True)
            (project / "src/marker.py").write_text("VALUE = 1\n")
            shutil.copyfile(ROOT / "sbatch/lib/environment.sh", project / "sbatch/lib/environment.sh")
            script = r'''
set -uo pipefail
source "$1/sbatch/lib/development.sh"
export MIRAGE_PROJECT_ROOT="$2" MIRAGE_SESSION_DIR="$3" MIRAGE_PYTHON="$4"
export SLURM_JOB_ID=fixture SLURM_STEP_ID=fixture MIRAGE_TEST_CACHE_DIR="$3/cache"
squeue() { echo '10:00'; }
sleep 60 &
service=$!
trap 'kill "$service" 2>/dev/null; wait "$service" 2>/dev/null || true' EXIT
status=0
mirage_iteration -c 'import marker; print(marker.VALUE); raise SystemExit(17)' || status=$?
[[ "$status" == 17 ]] || exit 1
kill -0 "$service" || exit 1
echo 'VALUE = 2' > "$MIRAGE_PROJECT_ROOT/src/marker.py"
mirage_iteration -c 'import marker; print(marker.VALUE)' || exit 1
kill -0 "$service" || exit 1
'''
            result = subprocess.run(["bash", "-c", script, "test", str(ROOT), str(project), str(session), sys.executable],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs = sorted((path / "stdout.log").read_text().strip() for path in session.glob("iteration-*"))
            self.assertEqual(outputs, ["1", "2"])

    def test_unconfirmed_cancellation_blocks_session_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "cancellation-unconfirmed").touch()
            result = subprocess.run(["bash", "-c",
                                     'source "$1/sbatch/lib/development.sh"; MIRAGE_SESSION_DIR="$2"; SLURM_STEP_ID=test; mirage_iteration -c pass',
                                     "test", str(ROOT), temporary], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unconfirmed", result.stderr)

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
