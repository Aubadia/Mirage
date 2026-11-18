"""Name-only entry, provenance boundaries and durable failure reports."""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mirage.cli import main
from mirage.runtime.process import ProcessResult
from mirage.workflow import check_api


class CheckControlTests(unittest.TestCase):
    def resolved_spec(self, api, cuda=False):
        return {"version": "api-spec-1", "api": api, "status": "resolved",
                "environment": {"cuda_available": cuda}, "semantic_guarantees": []}

    def preflight(self, request, directory, timeout):
        directory.mkdir()
        if request["mode"] == "api_spec":
            summary = self.resolved_spec(request["api"])
        else:
            summary = {"configured": True}
        (directory / "summary.json").write_text(json.dumps(summary))
        return ProcessResult("completed", 0, 0.001, str(directory))

    def test_only_api_and_environment_are_needed_and_reports_remain_unassessed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "MIRAGE_PROVIDER": "test_stateful:stateful_fixture_provider"}, clear=True):
            output = Path(tmp) / "run"
            execution = {"stop_reason": "case_limit", "records": [], "search_metrics": {
                "attempts": 1, "runtime_valid": 1, "verified_apis": ["torch.sin"],
                "applicable_comparisons": 0, "numerical_audit_statuses": {},
                "numerically_reported_cases": 0}}
            with patch("mirage.workflow.run_worker", side_effect=self.preflight), patch(
                    "mirage.workflow.run_stateful", return_value=execution) as execute:
                report = check_api("torch.sin", output=output)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["correctness"], "unassessed")
            self.assertEqual(report["metrics"]["applicable_comparisons"], 0)
            options = execute.call_args.args[0]
            self.assertTrue(options.forward_only)
            self.assertEqual(options.provider, "test_stateful:stateful_fixture_provider")
            self.assertIsNotNone(execute.call_args.kwargs["wall_deadline"])
            with redirect_stdout(StringIO()) as displayed:
                self.assertEqual(main(["report", str(output)]), 0)
            self.assertEqual(json.loads(displayed.getvalue()), report)

    def test_missing_api_skips_before_loading_provider(self):
        def unresolved(request, directory, timeout):
            self.assertEqual(request["mode"], "api_spec")
            directory.mkdir()
            (directory / "summary.json").write_text(json.dumps({
                "status": "unavailable", "reason": "no such public target"}))
            return ProcessResult("completed", 0, 0.1, str(directory))
        with tempfile.TemporaryDirectory() as tmp, patch("mirage.workflow.run_worker", side_effect=unresolved) as worker:
            report = check_api("torch.mirage_missing_api", output=Path(tmp) / "run")
            self.assertEqual(report["reason"], "api_unavailable")
            self.assertEqual(report["exit_status"], 0)
            self.assertEqual(worker.call_count, 1)
            self.assertTrue(report["terminal"])

    def test_invalid_environment_and_private_api_have_durable_errors(self):
        for api, settings in (("torch.sin", {"MIRAGE_CHECK_TIME_LIMIT": "nan"}),
                              ("torch._private", {})):
            with self.subTest(api=api), tempfile.TemporaryDirectory() as tmp, patch.dict(
                    os.environ, settings, clear=True), patch("mirage.workflow.run_worker") as worker:
                root = Path(tmp) / "run"
                report = check_api(api, output=root)
                self.assertEqual(report["reason"], "configuration_error")
                self.assertEqual(report["exit_status"], 2)
                self.assertEqual(json.loads((root / "summary.json").read_text()), report)
                worker.assert_not_called()
                with self.assertRaises(FileExistsError):
                    check_api(api, output=root)

    def test_no_gpu_and_exhausted_budget_are_skips(self):
        for settings, reason in (({"MIRAGE_CHECK_DEVICE": "cuda"}, "requested_device_unavailable"),
                                 ({"MIRAGE_CHECK_TIME_LIMIT": "0.000001"}, "budget_exhausted")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp, patch.dict(
                    os.environ, settings, clear=True), patch(
                    "mirage.workflow.run_worker", side_effect=self.preflight) as worker:
                report = check_api("torch.sin", output=Path(tmp) / "run")
                self.assertEqual(report["reason"], reason)
                self.assertEqual(worker.call_count, 1 if reason == "requested_device_unavailable" else 0)

    def test_provider_configuration_failure_is_not_a_correctness_result(self):
        def workers(request, directory, timeout):
            if request["mode"] == "api_spec":
                return self.preflight(request, directory, timeout)
            return ProcessResult("exception", 1, 0.1, str(directory), "endpoint is not configured")
        with tempfile.TemporaryDirectory() as tmp, patch("mirage.workflow.run_worker", side_effect=workers):
            report = check_api("torch.sin", output=Path(tmp) / "run")
            self.assertEqual(report["exit_status"], 1)
            self.assertEqual(report["reason"], "provider_preflight_exception")
            self.assertIsNone(report["metrics"])
            self.assertEqual(report["correctness"], "unassessed")

    def test_controller_crash_publishes_terminal_report_and_preserves_trace(self):
        with tempfile.TemporaryDirectory() as tmp, patch("mirage.workflow.run_worker", side_effect=self.preflight), patch(
                "mirage.workflow.run_stateful", side_effect=RuntimeError("controlled failure")):
            root = Path(tmp) / "run"
            report = check_api("torch.sin", output=root)
            self.assertEqual(report["reason"], "controller_error")
            self.assertTrue(report["terminal"])
            self.assertIn("controlled failure", (root / "error.txt").read_text())

    def test_unassessed_verdict_is_a_serializable_metric(self):
        from mirage.artifacts.report import summarize_api_check
        from mirage.workflow import write_json
        native = {"records": [{"mr_verdict": None}], "search_metrics": {
            "attempts": 1, "runtime_valid": 0, "verified_apis": [],
            "applicable_comparisons": 0, "numerical_audit_statuses": {},
            "numerically_reported_cases": 0}}
        metrics = summarize_api_check(native)
        self.assertEqual(metrics["mr_verdicts"], {"unassessed": 1})
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "summary.json", metrics)

    def test_cli_propagates_check_terminal_exit_and_output_directory(self):
        terminal = {"exit_status": 1, "terminal": True}
        with patch("mirage.workflow.check_api", return_value=terminal) as check, redirect_stdout(StringIO()):
            self.assertEqual(main(["check", "torch.sin", "--output", "/tmp/example"]), 1)
        check.assert_called_once_with("torch.sin", output=Path("/tmp/example"))


class SpecTests(unittest.TestCase):
    def test_installed_schema_method_and_wrapper_provenance(self):
        from mirage.runtime.api_metadata import specification
        for api in ("torch.add", "Tensor.add_", "torch.nn.functional.relu"):
            with self.subTest(api=api):
                spec = specification(api)
                self.assertEqual(spec["status"], "resolved")
                self.assertEqual(spec["semantic_guarantees"], [])
                self.assertEqual(spec["runtime_observations"], [])
                self.assertEqual(spec["capabilities"]["device_support"], "unknown")
                self.assertTrue(spec["sources"][0]["sha256"])
                json.dumps(spec, allow_nan=False)
                if api == "torch.add":
                    self.assertTrue(spec["operator_schemas"], spec)
                    self.assertTrue(any(a["write"] is True for s in spec["operator_schemas"]
                                        for a in s["arguments"]))

    def test_schema_unavailable_is_preserved_without_rejecting_the_api(self):
        from mirage.runtime.api_metadata import specification
        with patch("mirage.runtime.api_metadata.operator_schemas", return_value=([], "unregistered wrapper")):
            spec = specification("torch.nn.functional.relu")
        self.assertEqual(spec["status"], "resolved")
        self.assertIn("operator_schemas", spec["unknown_fields"])
        self.assertEqual(spec["schema_reason"], "unregistered wrapper")

    def test_remote_doc_version_mismatch_and_offline_status(self):
        from mirage.runtime.api_metadata import official_documentation
        with patch("urllib.request.urlopen", return_value=BytesIO(b"PyTorch 2.14 documentation")):
            doc = official_documentation("torch.example", "2.15.0+cu132")
        self.assertEqual(doc["status"], "unavailable")
        self.assertEqual(doc["reason"], "documentation release not verified")
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            doc = official_documentation("torch.example", "2.15.0")
        self.assertEqual(doc["status"], "unavailable")
        with patch("urllib.request.urlopen", return_value=BytesIO(b"PyTorch 2.15 documentation")):
            doc = official_documentation("torch.example", "2.15.0+cu132")
        self.assertEqual(doc["status"], "available")
        self.assertEqual(doc["role"], "generation_context_only")


@unittest.skipUnless(os.environ.get("MIRAGE_TEST_E2E") == "1", "Slurm integration required")
class CheckEntryIntegrationTests(unittest.TestCase):
    def test_name_only_entry_runs_real_isolated_torch_and_reopens_report(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "MIRAGE_PROVIDER": "test_stateful:stateful_fixture_provider",
                "MIRAGE_CHECK_CASES": "1", "MIRAGE_CHECK_TIME_LIMIT": "120",
                "MIRAGE_CHECK_OUTPUT_ROOT": os.environ.get("MIRAGE_CHECK_TEST_OUTPUT_ROOT", tmp), "MIRAGE_CHECK_DEVICE": "cpu"}):
            with redirect_stdout(StringIO()) as stdout:
                self.assertEqual(main(["check", "torch.sin"]), 0, stdout.getvalue())
            result = json.loads(stdout.getvalue())
            root = Path(result["directory"])
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["metrics"]["runtime_valid"], 1)
            self.assertEqual(result["metrics"]["verified_apis"], ["torch.sin"])
            spec = json.loads((root / "api-spec.json").read_text())
            self.assertEqual(spec["status"], "resolved")
            records = json.loads((root / "execution/summary.json").read_text())["records"]
            self.assertTrue(records[0]["api_executed"])
            from mirage.artifacts.bundle import load_tensor_record
            source = load_tensor_record(Path(records[0]["directory"]) / "execution/source.pt")
            self.assertEqual(spec["framework_version"], source["environment"]["torch"])
            with redirect_stdout(StringIO()) as displayed:
                self.assertEqual(main(["report", str(root)]), 0)
            self.assertEqual(json.loads(displayed.getvalue()), result)

    def test_name_only_missing_name_keeps_spec_without_provider(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "MIRAGE_CHECK_OUTPUT_ROOT": os.environ.get("MIRAGE_CHECK_TEST_OUTPUT_ROOT", tmp), "MIRAGE_PROVIDER": "unconfigured"}):
            with redirect_stdout(StringIO()) as stdout:
                self.assertEqual(main(["check", "torch.mirage_missing_api"]), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["reason"], "api_unavailable")
            self.assertEqual(result["correctness"], "unassessed")
            self.assertTrue((Path(result["directory"]) / "api-spec.json").is_file())

    def test_missing_service_configuration_and_budget_exhaustion_keep_real_reports(self):
        for settings, expected in (({"MIRAGE_PROVIDER": "vllm", "MIRAGE_VLLM_BASE_URL": ""},
                                    "provider_preflight_exception"),
                                   ({"MIRAGE_CHECK_TIME_LIMIT": "0.000001"}, "budget_exhausted")):
            with self.subTest(reason=expected), tempfile.TemporaryDirectory() as tmp, patch.dict(
                    os.environ, dict(settings, MIRAGE_CHECK_OUTPUT_ROOT=os.environ.get(
                        "MIRAGE_CHECK_TEST_OUTPUT_ROOT", tmp))):
                with redirect_stdout(StringIO()) as stdout:
                    code = main(["check", "torch.sin"])
                result = json.loads(stdout.getvalue())
                self.assertEqual(code, 1 if expected == "provider_preflight_exception" else 0)
                self.assertEqual(result["reason"], expected)
                self.assertEqual(result["correctness"], "unassessed")
                self.assertTrue(result["terminal"])
                self.assertTrue((Path(result["directory"]) / "summary.json").is_file())


PAIR_SOURCE = """def build(device, seed):
    x = torch.tensor([1., 2., 3., 4.], device=device)
    return {'inputs': {'x': x}, 'parameters': {}}

def run(ctx, probe):
    x = ctx['inputs']['x']
    y = x[2:].clone()
    out = probe.call(torch.add, x[:2], y, out=x[:2])
    return {'output': out}
"""


class PairProvider:
    def generate(self, request, deadline=None):
        from mirage.generator.provider import Response
        source = PAIR_SOURCE
        if request.context['configuration']['api'] == 'torch.bitwise_not':
            source = DISCRETE_SOURCE
        if request.context['configuration']['check_claim'] == 'operand_relation':
            source = source.replace('x[2:].clone()', 'x[2:]')
        return Response(json.dumps({'source': source}), model='authored-fixture')


def pair_provider():
    return PairProvider()


class FragmentTests(unittest.TestCase):
    def test_linked_history_controls(self):
        from mirage.runtime.stateful import execute
        from mirage.runtime.probes import Recorder
        build = """def build(device, seed):
    x = torch.tensor([1., 2.], device=device, requires_grad=True)
    return {'inputs': {'x': x}, 'parameters': {}}
"""
        controls = [
            ('torch.no_grad', 'scope_exit', "cm = probe.call(torch.no_grad)", "with cm: x = x + 1"),
            ('torch.nn.Identity', 'callable_use', "f = probe.call(torch.nn.Identity)", "x = probe.call(f, x)"),
            ('Tensor.register_hook', 'callback_trigger', "h = probe.call(x.register_hook, lambda g: g)", "x.sum().backward()"),
            ('torch.optim.SGD', 'persistent_step', "opt = probe.call(torch.optim.SGD, [x], lr=0.1)\n    x.sum().backward()", "probe.advance(opt)"),
            ('torch.futures.Future', 'async_completion', "f = probe.call(torch.futures.Future)\n    f.set_result(x)", "x = probe.wait(f)"),
            ('torch.sin', 'accumulation', "y = probe.call(torch.sin, x)\n    y.sum().backward(retain_graph=True)", "y.sum().backward()"),
        ]
        for api, claim, setup, trigger in controls:
            for positive in (False, True):
                with self.subTest(api=api, positive=positive):
                    source = build + "\ndef run(ctx, probe):\n    x = ctx['inputs']['x']\n    " + setup
                    if positive:
                        source += '\n    ' + trigger
                    source += "\n    return {'x': x}\n"
                    record = execute(source, dict(api=api, device='cpu', check_claim=claim,
                                         requires_backward=claim == 'accumulation'), Recorder())
                    self.assertEqual(record['inspection']['valid'], positive, record['inspection']['claim_history'])

    @unittest.skipUnless(os.environ.get('MIRAGE_TEST_E2E') == '1', 'isolated E2E')
    def test_explicit_relation_uses_checkpoint_and_spec(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'MIRAGE_PROVIDER':'test_api_check:pair_provider', 'MIRAGE_CHECK_CASES':'2',
            'MIRAGE_CHECK_DEVICE':'cpu', 'MIRAGE_CHECK_TIME_LIMIT':'120',
            'MIRAGE_CHECK_DIFFERENTIAL':'0'}):
            root = Path(os.environ.get('MIRAGE_CHECK_TEST_OUTPUT_ROOT', tmp)) / ('relation-' + __import__('uuid').uuid4().hex)
            result = check_api('torch.add', output=root)
            self.assertEqual(result['status'], 'completed', result)
            ledger = json.loads((root/'execution/relation-ledger.json').read_text())
            reached = [item for item in ledger.values() if item['status'] == 'reached']
            self.assertTrue(reached, ledger)
            self.assertTrue(reached[0]['checkpoint'][0]['attained'])
            request = json.loads((root/'execution/attempt-0001/request.json').read_text())
            self.assertEqual(request['context']['api_context']['spec_version'], 'api-spec-1')


DISCRETE_SOURCE = """def build(device, seed):
    x = torch.tensor([1, 4, 8, 2], dtype=torch.int32, device=device)
    return {'inputs': {'x': x}, 'parameters': {}}

def run(ctx, probe):
    x = ctx['inputs']['x'][1::2]
    return {'output': probe.call(torch.bitwise_not, x)}
"""


class DeviceControls(unittest.TestCase):
    def test_write_destination_and_guard_verdicts(self):
        import torch
        from mirage.rules.catalog import representation_pair, RELATIONS
        from mirage.backends.pytorch import PyTorchBackend
        from mirage.contracts.capabilities import Evidence, EvidenceStatus
        backend = PyTorchBackend()
        spec = RELATIONS['materialized_value_effect']
        facts = {k: Evidence(EvidenceStatus.VERIFIED, 'control', True, 'comparison', 'adapter')
                 for k in spec.preconditions + spec.input_relations}
        def record(value, guard=True):
            return dict(target_calls=[dict(output=torch.tensor(1), writable_paths=[['args', 0]],
                 destination_values=[torch.tensor(value)], guard_preserved=guard)])
        with patch.object(backend, 'representation_evidence', return_value=facts):
            for value, guard, expected in [(1, True, 'pass'), (2, True, 'suspected_failure'), (1, False, 'suspected_failure')]:
                self.assertEqual(representation_pair(record(1), record(value, guard), backend)['verdict'], expected)
            missing = record(1)
            del missing['target_calls'][0]['destination_values']
            self.assertEqual(representation_pair(record(1), missing, backend)['verdict'], 'inapplicable')

    def test_device_outcomes_and_primary_precedence(self):
        import torch
        from mirage.artifacts.replay import primary_route
        from mirage.inspector.comparison import compare_saved_devices
        with tempfile.TemporaryDirectory() as tmp:
            members = []
            for i, val in enumerate((1, 2, 1, 2, 3)):
                p = Path(tmp)/f'{i}.pt'
                torch.save({'target_calls':[{'output':torch.tensor(val)}]}, p)
                members.append(str(p))
            self.assertEqual(compare_saved_devices({'members':members[:4]})['status'], 'stable_mismatch')
            self.assertEqual(compare_saved_devices({'members':members[:3]+members[-1:]})['status'], 'unstable_or_unconfirmed')
        for verdict in ('pass', 'suspected_failure'):
            self.assertEqual(primary_route(dict(inspection={'valid':True}, mr_verdict=verdict)), 'primary_mr')

    @unittest.skipUnless(os.environ.get('MIRAGE_TEST_CUDA') == '1', 'Slurm CUDA control')
    def test_name_only_real_cuda_replay_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'MIRAGE_PROVIDER':'test_api_check:pair_provider', 'MIRAGE_CHECK_CASES':'1',
                'MIRAGE_CHECK_DEVICE':'cpu', 'MIRAGE_CHECK_TIME_LIMIT':'180',
                'MIRAGE_CHECK_DIFFERENTIAL':'1'}):
            root = Path(os.environ.get('MIRAGE_CHECK_TEST_OUTPUT_ROOT', tmp)) / ('device-' + __import__('uuid').uuid4().hex)
            from mirage.runtime.process import run_worker
            def interrupt_gpu(request, directory, timeout):
                if request.get('device') == 'cuda':
                    directory.mkdir()
                    return ProcessResult('timeout', None, .01, str(directory))
                return run_worker(request, directory, timeout)
            with patch('mirage.artifacts.replay.run_worker', side_effect=interrupt_gpu):
                partial = check_api('torch.bitwise_not', output=root)
            first, = partial['differential']['rows']
            self.assertEqual(first['status'], 'incomplete')
            self.assertTrue((Path(first['directory'])/'0-cpu/member.pt').is_file())
            with redirect_stdout(StringIO()) as resumed:
                self.assertEqual(main(['resume', str(root)]), 0)
            result = json.loads(resumed.getvalue())
            row, = result['differential']['rows']
            self.assertNotEqual(row['directory'], first['directory'])
            self.assertEqual(json.loads((Path(first['directory'])/'summary.json').read_text())['status'], 'incomplete')
            self.assertEqual(row['status'], 'agreement', row)
            self.assertEqual(result['metrics']['applicable_comparisons'], 0)
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(main(['replay', row['manifest'], '--worker-timeout','90']), 0)
            self.assertEqual(json.loads(output.getvalue())['status'], 'agreement')
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(main(['resume', str(root)]), 0)
            self.assertEqual(json.loads(output.getvalue())['differential'], result['differential'])


class AdmissionControls(unittest.TestCase):
    def test_unmet_relation_retains_runtime_reason(self):
        from mirage.runtime.probes import Recorder
        from mirage.runtime.stateful import execute
        predicate = dict(dimension='tensor_relation', field='shared_storage', expected=True,
                         checkpoint='final', call_index=0, operand_path=['args', 0], peer_operand_path=['args', 1])
        record = execute(PAIR_SOURCE, dict(api='torch.add', device='cpu', requires_backward=False,
                         check_claim='operand_relation', predicates=[predicate]), Recorder())
        self.assertFalse(record['inspection']['valid'])
        evidence, = record['inspection']['predicates']
        self.assertFalse(evidence['observed'])
        self.assertEqual(evidence['evidence_status'], 'verified')

    def test_incomparable_preconditions_block_both_agreement_and_mismatch(self):
        from copy import deepcopy
        from mirage.runtime.probes import Recorder
        from mirage.runtime.stateful import execute
        from mirage.inspector.comparison import device_admission, aligned_device_records
        original = execute(DISCRETE_SOURCE, dict(api='torch.bitwise_not', device='cpu', check_claim='forward',
                                                requires_backward=False), Recorder())
        schemas = [original['target_calls'][0]['native_dispatch'][0]['schema']]
        self.assertIsNone(device_admission(original, schemas))
        for kind, reason in [('rng','random_inputs_or_history'), ('ties','single_valued_pointwise_policy_unavailable'),
                             ('placement','operand_placement_or_representation_unsupported'),
                             ('cotangent','cotangent_alignment_unavailable'), ('unlinked', 'exact_native_overload_unavailable')]:
            changed = deepcopy(original)
            if kind == 'rng':
                changed['run_rng_unchanged'] = False
            elif kind == 'ties':
                changed['target_calls'][0]['native_dispatch'][0]['tags'] = []
            elif kind == 'placement':
                changed['target_calls'][0]['metadata'][0]['device'] = 'cuda'
            elif kind == 'cotangent':
                changed['config']['requires_backward'] = True
            else:
                changed['target_calls'][0]['native_dispatch'] = []
            self.assertEqual(device_admission(changed, schemas), reason)
        changed = deepcopy(original)
        changed['target_calls'][0]['metadata'][0]['offset'] += 1
        self.assertEqual(aligned_device_records(original, changed), 'target_inputs_or_alias_topology_changed')
