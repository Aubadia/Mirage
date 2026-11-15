from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from mirage.artifacts.store import ArtifactStore
from mirage.cli import main


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ArtifactStore(self.directory.name)

    def test_round_trip_and_derivation_preserve_original(self):
        experiment = {"api": "torch.matmul", "seed": 9, "rule": "layout"}
        original = self.store.write("original", "experiment", {"experiment": experiment})
        before = original.read_bytes()
        self.store.write("child", "experiment", {"program": "modified"}, parent_id="original")
        self.assertEqual(self.store.read("child")["parent_id"], "original")
        self.assertEqual(self.store.read("original")["payload"]["experiment"]["seed"], 9)
        with self.assertRaises(FileExistsError):
            self.store.write("original", "experiment", {})
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(list(Path(self.directory.name).glob(".pending-*")), [])

    def test_missing_parent_and_nonfinite_data_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.store.write("child", "experiment", {}, parent_id="missing")
        with self.assertRaises(ValueError):
            self.store.write("bad", "execution", {"loss": float("nan")})
        self.assertEqual(self.store.list_manifests(), [])

    def test_corrupt_version_id_and_duplicate_keys_rejected(self):
        path = self.store.write("a", "generation", {})
        original = self.store.read("a")
        for field, value in (("schema_version", 99), ("artifact_id", "other"), ("created_at", "bad")):
            changed = dict(original, **{field: value})
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.store.read("a")
        path.write_text('{"schema_version": 1, "schema_version": 1}')
        with self.assertRaises(ValueError):
            self.store.read("a")

    def test_identifier_validation(self):
        for value in ("../outside", "", "a/b"):
            with self.assertRaises(ValueError):
                self.store.write(value, "experiment", {})

    def test_cli_inspect_report_and_execution_validation(self):
        path = self.store.write("a", "experiment", {"note": "測試"})
        for arguments, expected_key in ((["inspect", str(path)], "payload"),
                                        (["report", self.directory.name], "artifact_count")):
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(arguments), 0)
            result = json.loads(output.getvalue())
            self.assertIn(expected_key, result)
            if arguments[0] == "report":
                self.assertIsNone(result["coverage"])

        for arguments in (["run"], ["replay", str(path)], ["run", "--apis", "unknown"]):
            output, errors = StringIO(), StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(main(arguments), 2)
            self.assertEqual(output.getvalue(), "")
            self.assertTrue(errors.getvalue())

    def test_cli_malformed_manifest_is_an_error(self):
        path = Path(self.directory.name) / "bad.json"
        path.write_text("{")
        with redirect_stderr(StringIO()):
            self.assertEqual(main(["inspect", str(path)]), 2)


class ExplorationMetricsTests(unittest.TestCase):
    def test_separate_operand_and_optimizer_coverage_and_missing_followup(self):
        from mirage.artifacts.report import summarize_exploration_oracle
        from mirage.contracts.capabilities import SEMANTIC_COVERAGE_VERSION
        row = {'api': 'torch.sin', 'executable': True, 'api_executed': True,
               'inspection': {'valid': True, 'semantic_coverage': {
                   'eligible': True, 'version': SEMANTIC_COVERAGE_VERSION,
                   'signatures': [{'api': 'torch.sin', 'representation': {'offset': 'zero'},
                                   'state': {'gradient.backward_count': 1}}]},
                   'optimizer_coverage': {'signatures': [{'representation': 'strided'}]}},
               'gradient_audits': {'source': {'status': 'unsupported'}, 'followup': {'status': 'unknown'}}}
        result = summarize_exploration_oracle({'records': [row]}, 10, 1000, True)
        self.assertEqual(result['representation_units_union'], 1)
        self.assertEqual(result['coverage_by_member']['followup']['unmeasured_cases'], 1)
        self.assertEqual(result['effective_numerical_cases'], 0)
        self.assertFalse(result['traversal']['measured_inventory'])
        self.assertIsNone(result['traversal']['dispatch_fraction'])

    def test_traversal_denominator_and_effective_case_count(self):
        from mirage.artifacts.report import summarize_exploration_oracle
        row = {'inspection': {'valid': True}, 'gradient_audits': {
            'source': {'status': 'agreement'}, 'followup': {'status': 'violation'}},
            'verdict': 'suspected_failure', 'finished_seconds': 4,
            'automatic_replays': [{'replay_valid': True}]}
        report = {'records': [row], 'representation_traversal': {'enabled': True,
            'sources': [{'candidates': [{'status': s} for s in ('pending', 'reached', 'not_verified', 'rejected')]}]}}
        result = summarize_exploration_oracle(report, 10, 1000, True)
        self.assertEqual(result['traversal']['eligible_candidates'], 3)
        self.assertEqual(result['traversal']['dispatch_fraction'], 2/3)
        self.assertEqual(result['traversal']['attainment_fraction'], 1/2)
        self.assertEqual(result['effective_numerical_cases'], 1)
        self.assertEqual(result['effective_numerical_members'], 2)
        self.assertEqual(result['automatically_reproduced_cases'], 1)

    def test_previously_reached_is_tested_without_claiming_another_dispatch(self):
        from mirage.artifacts.report import summarize_exploration_oracle
        report = {'records': [], 'representation_traversal': {'enabled': True,
            'sources': [{'candidates': [{'status': s} for s in
                ('pending', 'previously_reached', 'reached', 'not_verified', 'ineligible')]}]}}
        traversal = summarize_exploration_oracle(report, 10, 1000, True)['traversal']
        self.assertEqual(traversal['eligible_candidates'], 4)
        self.assertEqual(traversal['previously_reached_candidates'], 1)
        self.assertEqual(traversal['tested_candidates'], 2)
        self.assertEqual(traversal['dispatched_candidates'], 2)
        self.assertEqual(traversal['reached_candidates'], 1)
        self.assertEqual(traversal['dispatch_fraction'], 1/2)
        self.assertEqual(traversal['attainment_fraction'], 1/2)


class ExperimentResumeTests(unittest.TestCase):
    """Metadata-only controls: no model, worker, Torch or Slurm command executes."""
    def setUp(self):
        import importlib.util
        from argparse import Namespace
        from unittest.mock import patch
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        driver = Path(__file__).resolve().parents[1]/'benchmarks/exploration_oracle.py'
        spec = importlib.util.spec_from_file_location('experiment_driver_test', driver)
        self.driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.driver)
        self.args = Namespace(protocol=self.root/'protocol.json', max_runs=32)
        self.args.protocol.write_text('{}')
        self.protocol = {'schedule': [{'index': i, 'directory': f'run-{i}'} for i in range(3)],
                         'search_seconds': 180, 'worker_timeout': 120, 'driver': str(driver)}
        for target, value in (('require_allocation', None), ('locked_protocol', self.protocol),
                              ('remaining_seconds', 10000)):
            handle = patch.object(self.driver, target, return_value=value)
            handle.start(); self.addCleanup(handle.stop)
        handle = patch.dict('os.environ', {'MIRAGE_SESSION_DIR': str(self.root/'session')})
        handle.start(); self.addCleanup(handle.stop)
        (self.root/'session').mkdir()

    def completed(self, index, status):
        folder = self.root/f'run-{index}'
        folder.mkdir()
        (folder/'completion.json').write_text(json.dumps({'exit_status': status}))

    def test_failure_is_skipped_on_resume_and_later_failure_does_not_stop_next(self):
        from unittest.mock import patch, Mock
        self.completed(0, 1)
        launched = []
        def spawn(command, **kwargs):
            index = int(command[-1]); launched.append(index)
            code = 2 if index == 1 else 0
            self.completed(index, code)
            return Mock(returncode=code, wait=Mock(return_value=code))
        with patch.object(self.driver.subprocess, 'Popen', side_effect=spawn), \
             patch.object(self.driver, 'score', return_value=1):
            self.assertEqual(self.driver.run(self.args), 1)
            self.assertEqual(launched, [1, 2])
            progress = json.loads((self.root/'progress.json').read_text())
            self.assertEqual(progress['attempted'], [0, 1, 2])
            self.assertEqual(progress['failed_indices'], [0, 1])
            self.assertEqual(self.driver.run(self.args), 1)
            self.assertEqual(launched, [1, 2])

    def test_incomplete_search_is_not_retried_or_bypassed(self):
        from unittest.mock import patch
        (self.root/'run-0').mkdir()
        with patch.object(self.driver.subprocess, 'Popen') as spawn:
            with self.assertRaises(RuntimeError):
                self.driver.run(self.args)
            spawn.assert_not_called()

    def test_uncertain_cancellation_stops_before_independent_dispatch(self):
        from unittest.mock import patch
        (self.root/'session/cancellation-unconfirmed').write_text('pending')
        with patch.object(self.driver.subprocess, 'Popen') as spawn:
            self.assertEqual(self.driver.run(self.args), 2)
            spawn.assert_not_called()

    def test_score_distinguishes_all_attempted_from_all_successful(self):
        from unittest.mock import patch, Mock
        for i in range(3):
            self.completed(i, int(i == 1))
            folder = self.root/f'run-{i}'/'search'; folder.mkdir()
            (folder/'summary.json').write_text('{"records": []}')
        source = self.root/'engine/mirage/artifacts/report.py'
        source.parent.mkdir(parents=True); source.write_text('# frozen scorer')
        self.protocol['engines'] = {'candidate': {'path': str(self.root/'engine')}}
        with patch.dict('sys.modules', {'torch': Mock()}):
            self.assertEqual(self.driver.score(self.args.protocol), 1)
        result = json.loads((self.root/'comparison.json').read_text())
        self.assertTrue(result['all_attempted'])
        self.assertFalse(result['all_successful'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['failed_indices'], [1])
