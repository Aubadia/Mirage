from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import tempfile
import unittest

from mirage.artifacts.store import ArtifactStore
from mirage.cli import main, parser


class CLITests(unittest.TestCase):
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

    def test_replay_has_explicit_cross_version_and_minimization_controls(self):
        arguments = parser().parse_args(["replay", "case.json", "--allow-environment-change", "--size", "2"])
        self.assertTrue(arguments.allow_environment_change)
        self.assertEqual(arguments.size, 2)
