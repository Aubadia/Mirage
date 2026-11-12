from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import tempfile
import unittest
from unittest.mock import patch

from mirage.artifacts.store import ArtifactStore
from mirage.cli import main, parser


class CLITests(unittest.TestCase):
    def test_parallel_options_and_service_failure_exit_status(self):
        arguments = parser().parse_args(["run", "--provider", "vllm", "--workers", "2",
                                        "--generation-concurrency", "1"])
        self.assertEqual(arguments.workers, 2)
        self.assertEqual(arguments.generation_concurrency, 1)
        report = {"stop_reason": "provider_cancellation_unconfirmed", "experiments": []}
        with patch("mirage.workflow.run", return_value=report), redirect_stdout(StringIO()):
            self.assertEqual(main(["run", "--provider", "vllm"]), 1)

    def test_provider_is_explicit_and_limits_are_validated(self):
        for arguments in (["run"], ["run", "--provider", "reference", "--epsilon", "nan"],
                          ["run", "--provider", "reference", "--replay-failures", "-1"]):
            with redirect_stderr(StringIO()):
                self.assertEqual(main(arguments), 2)

    def test_report_uses_latest_report_per_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(temporary)
            for index in range(2):
                store.write(f"progress-{index}", "report", {"batch_id": "b", "coverage": {"experiments": index}})
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["report", temporary]), 0)
            self.assertEqual(json.loads(output.getvalue())["coverage"], {"experiments": 1})

    def test_replay_has_explicit_environment_control(self):
        arguments = parser().parse_args(["replay", "case.json", "--allow-environment-change"])
        self.assertTrue(arguments.allow_environment_change)
