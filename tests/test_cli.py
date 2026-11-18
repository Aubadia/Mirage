from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import tempfile
import unittest
from unittest.mock import patch

from mirage.artifacts.store import ArtifactStore
from mirage.cli import main, parser


class CLITests(unittest.TestCase):
    def test_report_reads_multi_api_summary_without_treating_it_as_manifest(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'summary.json').write_text(json.dumps({'multi_api': True, 'stop_reason': 'completed', 'sessions': []}))
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(main(['report', tmp]), 0)
            self.assertTrue(json.loads(output.getvalue())['multi_api'])

    def test_shared_api_options_apply_to_training_and_stateful(self):
        from mirage.contracts.programs import RunOptions
        for strategy in ('training', 'stateful'):
            with self.subTest(strategy=strategy):
                switches = ['run', '--provider', 'vllm', '--strategy', strategy,
                            '--active-apis', '3', '--workers', '4', '--workers-per-api', '2',
                            '--generation-concurrency', '2', '--generation-per-api', '1']
                with patch('mirage.workflow.run', return_value={'stop_reason': 'completed'} ) as execute, redirect_stdout(StringIO()):
                    self.assertEqual(main(switches), 0)
                options = execute.call_args.args[0]
                self.assertEqual((options.active_api_limit, options.per_api_worker_limit,
                                  options.per_api_generation_limit), (3, 2, 1))
        with self.assertRaises(ValueError):
            RunOptions('unused', 'fixture', active_api_limit=0)
        with self.assertRaises(ValueError):
            RunOptions('unused', 'fixture', strategy='properties', active_api_limit=2)

    def test_multi_api_partial_failure_returns_nonzero(self):
        with patch('mirage.workflow.run', return_value={'stop_reason': 'completed_with_failures'}), redirect_stdout(StringIO()):
            self.assertEqual(main(['run', '--provider', 'vllm', '--active-apis', '2']), 1)

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


    def test_pipeline_defaults_and_explicit_legacy_opt_out(self):
        for switches, expected in (([], True), (["--no-generation-pipeline"], False)):
            with patch('mirage.workflow.run', return_value={'stop_reason':'case_limit','records':[]}) as execute, redirect_stdout(StringIO()):
                self.assertEqual(main(['run','--provider','vllm','--strategy','stateful', *switches]), 0)
            self.assertIs(execute.call_args.args[0].generation_pipeline, expected)
            self.assertEqual(execute.call_args.args[0].workers, 8 if expected else 1)

    def test_worker_default_depends_on_strategy_and_explicit_value_wins(self):
        from mirage.contracts.programs import RunOptions
        self.assertEqual(RunOptions('output', 'provider').workers, 1)
        self.assertEqual(RunOptions('output', 'provider', strategy='stateful',
            contract_version=4, generation_pipeline=True).workers, 8)
        for switches, expected in ((['--strategy', 'stateful', '--workers', '1'], 1),
                                    (['--strategy', 'properties'], 1)):
            with patch('mirage.workflow.run', return_value={'stop_reason':'case_limit','records':[]}) as execute, redirect_stdout(StringIO()):
                self.assertEqual(main(['run','--provider','vllm', *switches]), 0)
            self.assertEqual(execute.call_args.args[0].workers, expected)

    def test_alternating_mode_keeps_pipeline_budget_and_worker_default(self):
        with patch('mirage.workflow.run', return_value={'stop_reason':'case_limit','records':[]}) as execute, redirect_stdout(StringIO()):
            self.assertEqual(main(['run','--provider','vllm','--strategy','stateful',
                                   '--no-generation-overlap','--time-limit','180']), 0)
        options = execute.call_args.args[0]
        self.assertTrue(options.generation_pipeline)
        self.assertFalse(options.generation_overlap)
        self.assertEqual(options.workers, 8)
        self.assertEqual(options.time_limit, 180)
        for switches in ([], ['--strategy','stateful','--no-generation-pipeline']):
            with redirect_stderr(StringIO()):
                self.assertEqual(main(['run','--provider','vllm','--no-generation-overlap',*switches]), 2)
        from mirage.contracts.programs import RunOptions
        with self.assertRaisesRegex(ValueError, 'boolean'):
            RunOptions('output', 'provider', generation_overlap=0)

    def test_resume_incomplete_is_nonzero(self):
        with patch('mirage.runtime.pipeline.resume_pipeline', return_value={
                'stop_reason': 'pipeline_incomplete', 'records': []}), redirect_stdout(StringIO()):
            self.assertEqual(main(['resume', '/unused']), 1)
