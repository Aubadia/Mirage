from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from mirage.artifacts.store import ArtifactStore
from mirage.cli import main


class ArtifactTests(unittest.TestCase):
    def test_search_metrics_distinguish_rejected_failed_and_legacy_dispatches(self):
        from mirage.artifacts.report import summarize_search
        family = {'planning_family_selection': {'family': 'tensor_state'}}
        cases = [
            (dict(family, planning_dispatched=False, planning_usage=None,
                  program_generation_dispatched=False), 0, 0, True),
            (dict(family, planning_dispatched=True, planning_usage=None,
                  program_generation_dispatched=False), 1, 0, False),
            (dict(family, planning_usage={'input_tokens': 2, 'output_tokens': 3},
                  first_blocking_stage='state_plan'), 1, 5, True),
        ]
        for row, calls, tokens, complete in cases:
            with self.subTest(row=row):
                summary = summarize_search({'records': [row]})
                self.assertEqual(summary['planning_calls'], calls)
                self.assertEqual(summary['planning_family_calls'], {'tensor_state': calls} if calls else {})
                self.assertEqual(summary['known_tokens'], tokens)
                self.assertEqual(summary['token_usage_complete'], complete)
                self.assertEqual(summary['attempts'], 1)

    def test_multi_api_reports_keep_per_api_denominators_and_nonadditive_time(self):
        from mirage.artifacts.report import summarize_multi_api
        rows = [dict(api='a', strategy='training', state='completed', session_wall_seconds=8,
                     report={'coverage': {'declared': 3}}),
                dict(api='b', strategy='stateful', state='completed', session_wall_seconds=7,
                     report={'coverage': {'declared': 5}})]
        report = summarize_multi_api(rows, {'cancelled': False}, {'workers': 2}, 10)
        self.assertEqual(report['elapsed_seconds'], 10)
        self.assertEqual(report['per_api_elapsed_seconds_sum'], 15)
        self.assertEqual([row['coverage']['declared'] for row in report['per_api']], [3, 5])
        self.assertNotIn('coverage', report)
        self.assertNotIn('verdict', report)
        self.assertEqual(report['completed_api_runs'], 2)

    def test_candidate_counts_keep_admission_and_attainment_separate(self):
        from mirage.artifacts.report import summarize_training_candidates
        summary = summarize_training_candidates([
            {"candidates": [{"status": "admitted", "reasons": []},
                            {"status": "rejected", "reasons": ["no optimizer"]}]}],
            [{"id": "canonical-t00-p0", "candidate_id": "c", "status": "executed", "verified": False}])
        self.assertEqual(summary["registered_instances"], 2)
        self.assertEqual(summary["registry"]["admitted"], 1)
        self.assertEqual(summary["executed_path_attempts"], 1)
        self.assertEqual(summary["attained_path_attempts"], 0)
        self.assertEqual(summary["mr_applicable_path_attempts"], 0)

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

    def test_workflow_report_serialization_preserves_bytes_and_rejections(self):
        from dataclasses import dataclass
        from enum import Enum, IntEnum
        from mirage.artifacts.store import json_value
        from mirage.workflow import write_json

        class Text(str, Enum):
            LABEL = '測試'
        class Count(IntEnum):
            ONE = 1
        @dataclass
        class Evidence:
            values: tuple
        class Mapping(dict):
            pass
        shared = {'score': 1.25, 'zero': -0.0, 'enabled': True, 'missing': None}
        value = {'rows': [shared, shared], 'special': Evidence((Text.LABEL, Count.ONE)),
                 'subclass': Mapping(nested=[Text.LABEL]), 'empty': (), 'large': 2**80,
                 'nested_mixed': [shared, (Text.LABEL, Count.ONE), {'flag': Text.LABEL}]}
        path = Path(self.directory.name) / 'report.json'
        for score in (1.25, 9.5):
            shared['score'] = score
            write_json(path, value)
            expected = json.dumps(json_value(value), indent=2, allow_nan=False).encode('utf-8')
            self.assertEqual(path.read_bytes(), expected)
            self.assertIs(value['nested_mixed'][0], shared)
            self.assertIsInstance(value['nested_mixed'][1], tuple)
            self.assertIs(value['nested_mixed'][2]['flag'], Text.LABEL)
        before = path.read_bytes()
        cyclic = []; cyclic.append(cyclic)
        for bad, error in (({'nested': {1: 'bad'}}, ValueError),
                           ({'nested': float('nan')}, ValueError),
                           ({'nested': float('inf')}, ValueError),
                           ({'nested': object()}, TypeError), (cyclic, RecursionError),
                           ({'cycle': cyclic, 1: 'bad key'}, ValueError)):
            with self.assertRaises(error):
                write_json(path, bad)
            self.assertEqual(path.read_bytes(), before)


    def test_report_partial_storage_failure_preserves_previous_json(self):
        from contextlib import contextmanager
        import errno
        from unittest.mock import patch
        from mirage.workflow import write_json

        path = Path(self.directory.name) / 'report.json'
        write_json(path, {'previous': True})
        before = path.read_bytes()
        original = tempfile.NamedTemporaryFile

        @contextmanager
        def partial_file(*args, **kwargs):
            with original(*args, **kwargs) as stream:
                original_write = stream.write
                def fail(payload):
                    original_write(payload[:7])
                    stream.flush()
                    raise OSError(errno.ENOSPC, 'Injected full output storage')
                stream.write = fail
                yield stream

        with patch('mirage.workflow.tempfile.NamedTemporaryFile', partial_file):
            with self.assertRaises(OSError) as raised:
                write_json(path, {'replacement': 'x' * 100})
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob('.report.json.*.pending')), [])

    def test_report_publication_failure_preserves_previous_json(self):
        import errno
        from unittest.mock import patch
        from mirage.workflow import write_json

        path = Path(self.directory.name) / 'report.json'
        write_json(path, {'previous': True})
        before = path.read_bytes()
        with patch.object(Path, 'replace', side_effect=OSError(errno.ENOSPC, 'Injected publication failure')):
            with self.assertRaises(OSError):
                write_json(path, {'replacement': True})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob('.report.json.*.pending')), [])

    def test_report_reader_never_observes_a_partial_write(self):
        from contextlib import contextmanager
        import threading
        from unittest.mock import patch
        from mirage.workflow import write_json

        path = Path(self.directory.name) / 'report.json'
        previous, replacement = {'previous': True}, {'replacement': 'x' * 100}
        write_json(path, previous)
        partial, release = threading.Event(), threading.Event()
        errors = []
        original = tempfile.NamedTemporaryFile

        @contextmanager
        def delayed_file(*args, **kwargs):
            with original(*args, **kwargs) as stream:
                original_write = stream.write
                def delayed(payload):
                    original_write(payload[:7])
                    stream.flush()
                    partial.set()
                    if not release.wait(10):
                        raise TimeoutError('Test did not release the pending write')
                    return 7 + original_write(payload[7:])
                stream.write = delayed
                yield stream

        def writer():
            try:
                write_json(path, replacement)
            except BaseException as error:
                errors.append(error)

        with patch('mirage.workflow.tempfile.NamedTemporaryFile', delayed_file):
            thread = threading.Thread(target=writer)
            thread.start()
            try:
                self.assertTrue(partial.wait(5))
                self.assertEqual(json.loads(path.read_text()), previous)
            finally:
                release.set()
                thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(json.loads(path.read_text()), replacement)
        self.assertEqual(list(path.parent.glob('.report.json.*.pending')), [])


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


class E2EMeasurementTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1]/'benchmarks/e2e_latency.py'
        spec = importlib.util.spec_from_file_location('e2e_measurement_test', path)
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)

    def test_roster_contains_only_api_device_and_identifier(self):
        catalog = {'selected': [str(i) for i in range(10)], 'candidates': [
            {'id': str(i), 'api': 'torch.sin', 'device': 'cpu', 'conditions': 'private fixture trigger'}
            for i in range(10)]}
        rows = self.driver.targets(catalog)
        self.assertEqual([r['id'] for r in rows], catalog['selected'])
        self.assertTrue(all(set(r) == {'id','api','device'} for r in rows))
        catalog['selected'][-1] = '0'
        with self.assertRaises(ValueError): self.driver.targets(catalog)

    def test_missing_idle_metrics_is_unknown_and_histogram_labels_survive(self):
        from unittest.mock import patch
        metrics = ('# HELP ignored\n'
            'vllm:num_requests_running{engine="0",model_name="test"} 0\n'
            'vllm:num_requests_waiting{engine="0",model_name="test"} 0\n'
            'vllm:request_decode_time_seconds_bucket{le="+Inf"} 2\n')
        self.assertEqual(len(self.driver.metric_values(metrics)), 3)
        with patch.object(self.driver, 'service_metrics', return_value=metrics):
            self.assertTrue(self.driver.confirmed_idle())
        with patch.object(self.driver, 'service_metrics', return_value=''):
            self.assertFalse(self.driver.confirmed_idle())

    def test_protocol_detects_engine_or_driver_changes(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; source.mkdir()
            (source/'sample.py').write_text('VALUE = 1\n')
            args = Namespace(output=root/'run',engine=source,label='fixture',budget=180,workers=8)
            with redirect_stdout(StringIO()): self.driver.prepare(args)
            protocol = args.output/'protocol.json'
            self.driver.verify(protocol)
            (args.output/'engine/sample.py').write_text('VALUE = 2\n')
            with self.assertRaises(ValueError): self.driver.verify(protocol)
            (args.output/'engine/sample.py').write_text('VALUE = 1\n')
            (args.output/'driver.py').write_text('# altered\n')
            with self.assertRaises(ValueError): self.driver.verify(protocol)

    def test_protocol_preserves_alternating_mode_and_legacy_overlap_default(self):
        from argparse import Namespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; source.mkdir()
            (source/'sample.py').write_text('VALUE = 1\n')
            args = Namespace(output=root/'run',engine=source,label='alternating',budget=180,
                             workers=32,generation_overlap=False)
            with redirect_stdout(StringIO()): self.driver.prepare(args)
            path = args.output/'protocol.json'
            protocol = self.driver.verify(path)
            self.assertFalse(protocol['generation_overlap'])
            self.assertEqual(protocol['model_budget_seconds'], 180)
            report = {'search_metrics':{},'pipeline_timing':{},'pipeline_complete':True,'stop_reason':'llm_time_limit'}
            for index, expected in ((0, False), (1, True)):
                if expected:
                    protocol.pop('generation_overlap')
                    path.write_text(json.dumps(protocol))
                with patch.object(self.driver, 'require_allocation'), \
                     patch.object(self.driver, 'model_environment', return_value={}), \
                     patch.dict('os.environ', {'SLURM_JOB_ID':'fixture','SLURM_STEP_ID':'fixture'}), \
                     patch('mirage.workflow.run_stateful', return_value=report) as run:
                    self.assertEqual(self.driver.one(Namespace(protocol=path,index=index)), 0)
                options = run.call_args.args[0]
                self.assertIs(options.generation_overlap, expected)
                self.assertTrue(options.generation_pipeline)
                self.assertEqual(options.workers, 32)

    def test_ordinary_api_failure_keeps_other_targets_and_is_not_complete_success(self):
        from argparse import Namespace
        from unittest.mock import patch, Mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'protocol.json'; path.write_text('{}')
            protocol = {'engine':str(root), 'driver':'frozen.py', 'targets':[
                {'id':str(i),'api':'torch.sin','device':'cpu'} for i in range(10)]}
            calls = []
            def spawn(command, **kwargs):
                index = int(command[-1]); calls.append(index)
                code = 1 if index == 2 else 0
                return Mock(returncode=code, poll=Mock(return_value=code))
            with patch.object(self.driver, 'require_allocation'), \
                 patch.object(self.driver, 'verify', return_value=protocol), \
                 patch.object(self.driver, 'model_environment'), \
                 patch.object(self.driver, 'confirmed_idle', return_value=True), \
                 patch.object(self.driver, 'remaining', return_value=10000), \
                 patch.object(self.driver, 'service_metrics', return_value=''), \
                 patch.object(self.driver.signal, 'signal'), \
                 patch.object(self.driver.subprocess, 'Popen', side_effect=spawn), \
                 redirect_stdout(StringIO()):
                self.assertEqual(self.driver.campaign(Namespace(protocol=path)), 1)
            report = json.loads((root/'campaign.json').read_text())
            self.assertEqual(calls, list(range(10)))
            self.assertEqual(report['state'], 'complete_with_failures')
            self.assertEqual(report['remaining_unattempted'], 0)


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
