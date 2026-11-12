from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mirage.generator.provider import Request, Stage
from mirage.runtime.process import cancellation_scope, run_worker


class ProcessTests(unittest.TestCase):
    def test_cancellation_stops_process_and_preserves_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cancellable_provider.py").write_text(
                "import time\nclass Provider:\n"
                "    def generate(self, request, *, deadline):\n        time.sleep(60)\n")
            cancelled = threading.Event()
            timer = threading.Timer(0.3, cancelled.set)
            timer.start()
            started = time.monotonic()
            try:
                with patch.dict(os.environ, {"PYTHONPATH": temporary + os.pathsep + os.environ.get("PYTHONPATH", "")}):
                    with cancellation_scope(cancelled):
                        result = run_worker(self.provider_request("cancellable_provider:Provider"), root / "worker", 30)
            finally:
                timer.cancel()
                timer.join()
            self.assertEqual(result.outcome, "timeout")
            self.assertIn("cancelled", result.error)
            self.assertLess(time.monotonic() - started, 5)
            saved = json.loads((root / "worker" / "process.json").read_text())
            self.assertEqual(saved["outcome"], "timeout")

    def provider_request(self, specification):
        return {"mode": "provider", "provider": specification, "deadline": time.monotonic() + 10,
                "request": asdict(Request(Stage.CANONICAL, "test", {"api": "torch.matmul", "actions": []}))}

    def test_provider_process_returns_structured_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "worker"
            result = run_worker(self.provider_request("reference"), directory, 10)
            self.assertEqual(result.outcome, "completed")
            response = json.loads((directory / "response.json").read_text())
            self.assertEqual(response["model"], "reference-fixture")

    def test_deadline_stops_noncooperative_provider_and_retains_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "slow_provider.py").write_text(
                "import time\nclass Provider:\n"
                "    def generate(self, request, *, deadline):\n        time.sleep(60)\n")
            with patch.dict(os.environ, {"PYTHONPATH": temporary + os.pathsep + os.environ.get("PYTHONPATH", "")}):
                started = time.monotonic()
                result = run_worker(self.provider_request("slow_provider:Provider"), path / "worker", 1)
            self.assertEqual(result.outcome, "timeout")
            self.assertLess(time.monotonic() - started, 5)
            self.assertTrue((path / "worker" / "request.json").exists())
            self.assertTrue((path / "worker" / "stderr.txt").exists())

    def test_unknown_provider_error_is_not_invalid_program(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_worker(self.provider_request("does_not_exist:factory"), Path(temporary) / "worker", 10)
            self.assertEqual(result.outcome, "exception")
            self.assertIn("ModuleNotFoundError", result.error)

    def test_oom_and_abrupt_exit_have_separate_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "failing_provider.py").write_text(
                "import os\n"
                "class OutOfMemory:\n"
                "    def generate(self, request, *, deadline):\n        raise MemoryError('exhausted')\n"
                "class Crash:\n"
                "    def generate(self, request, *, deadline):\n        os._exit(23)\n")
            with patch.dict(os.environ, {"PYTHONPATH": temporary + os.pathsep + os.environ.get("PYTHONPATH", "")}):
                oom = run_worker(self.provider_request("failing_provider:OutOfMemory"), root / "oom", 10)
                crash = run_worker(self.provider_request("failing_provider:Crash"), root / "crash", 10)
            self.assertEqual(oom.outcome, "oom")
            self.assertEqual(crash.outcome, "crash")
            self.assertEqual(crash.returncode, 23)

    def test_exhausted_budget_never_launches_another_process(self):
        with tempfile.TemporaryDirectory() as temporary, patch("mirage.runtime.process.subprocess.Popen") as launch:
            result = run_worker(self.provider_request("reference"), Path(temporary) / "worker", 0)
            launch.assert_not_called()
            self.assertEqual(result.outcome, "timeout")
            self.assertIsNone(result.returncode)
