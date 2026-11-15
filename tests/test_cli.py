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

    def test_property_results_and_provider_failures(self):
        for status,expected in [('executed',0),('provider_failed',1)]:
            report={'stop_reason':'case_limit','records':[{'status':status}]}
            with patch('mirage.workflow.run',return_value=report), redirect_stdout(StringIO()):
                self.assertEqual(main(['run','--provider','vllm','--strategy','properties']),expected)

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

    def test_property_replay_failure_preserves_nonzero_exit(self):
        with patch('mirage.workflow.replay',return_value={'process':{'outcome':'exception'}}), redirect_stdout(StringIO()):
            self.assertEqual(main(['replay','case.json']),1)

    def test_replay_has_explicit_environment_control(self):
        arguments = parser().parse_args(["replay", "case.json", "--allow-environment-change"])
        self.assertTrue(arguments.allow_environment_change)


    def test_stateful_strategy_selects_its_independent_version(self):
        with patch('mirage.workflow.run', return_value={'stop_reason':'case_limit','records':[]}) as execute, redirect_stdout(StringIO()):
            main(['run','--provider','vllm','--strategy','stateful'])
        self.assertEqual(execute.call_args.args[0].contract_version, 4)

    def test_stateful_features_default_on_and_can_be_disabled(self):
        for switches, expected in (([], True),
                (["--no-representation-traversal", "--no-oracle-improvements"], False)):
            with patch('mirage.workflow.run', return_value={'stop_reason':'case_limit','records':[]}) as execute, redirect_stdout(StringIO()):
                self.assertEqual(main(['run','--provider','vllm','--strategy','stateful', *switches]), 0)
            options = execute.call_args.args[0]
            self.assertIs(options.representation_traversal, expected)
            self.assertIs(options.oracle_improvements, expected)
        with patch('mirage.workflow.run', return_value={'stop_reason':'case_limit','records':[]}) as execute, redirect_stdout(StringIO()):
            self.assertEqual(main(['run','--provider','vllm','--strategy','properties']), 0)
        self.assertFalse(execute.call_args.args[0].representation_traversal)
        self.assertFalse(execute.call_args.args[0].oracle_improvements)
