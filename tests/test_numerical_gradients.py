"""Generic checker controls; injected bad evidence is never a framework finding."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from mirage.backends.pytorch import PyTorchBackend
from mirage.inspector.numerical_gradients import audit_record
from mirage.inspector.oracles import audit_execution
from mirage.runtime.operands import discover
from mirage.runtime.probes import Recorder
from mirage.runtime.representations import ReconstructionError, StorageCoordinateChart
from mirage.runtime.stateful import execute, snapshot
from dataclasses import asdict

SOURCE = '''import torch

def build(device, seed):
    x = torch.tensor([0.3, 0.8], dtype=torch.float64, device=device, requires_grad=True)
    return {'inputs': {'x': x}, 'parameters': {}}

def run(ctx, probe):
    x = ctx['inputs']['x']
    y = probe.call(torch.sin, x)
    y.sum().backward()
    return {'output': y, 'gradient': x.grad}
'''


def recorded(source=SOURCE, api='torch.sin'):
    return execute(source, {'api': api, 'device': 'cpu', 'seed': 0, 'requires_backward': True}, Recorder())


def chart_for(tree):
    return StorageCoordinateChart({'inputs': snapshot(tree), 'metadata':
        [asdict(d) for d, _ in discover(tree, PyTorchBackend())]})


OPTIMIZER_SOURCE = SOURCE.replace("return {'inputs': {'x': x}, 'parameters': {}}",
    "optimizer = torch.optim.SGD([x], lr=0.1)\n    return {'inputs': {'x': x}, 'parameters': {'x': x}, 'optimizer': optimizer}")
OPTIMIZER_SOURCE = OPTIMIZER_SOURCE.replace("    y.sum().backward()", "    y.sum().backward()\n    ctx['optimizer'].step()")


class ScopedLocalVJPTests(unittest.TestCase):
    def record(self, source=OPTIMIZER_SOURCE, **config):
        return execute(source, dict(api='torch.sin', device='cpu', seed=0,
                       requires_backward=True, oracle_improvements=True, **config), Recorder())

    def test_native_update_preserves_forward_snapshot_and_local_gradient(self):
        r = self.record()
        self.assertIsNone(r['program_certificate'])
        self.assertIsNone(r['numerical_source_certificate'])
        self.assertEqual(r['scoped_vjp_source_certificate'], 'explicit_scoped_local_vjp_v1')
        before = r['target_calls'][0]['inputs']['args'][0]
        self.assertTrue(torch.equal(before, torch.tensor([0.3, 0.8], dtype=torch.float64)))
        self.assertFalse(torch.equal(before, r['final_training_state']['parameters']['x']))
        self.assertTrue(r['gradient_evidence']['events'][0]['operands_unchanged_at_backward'])
        result = audit_execution(r)
        self.assertEqual(result['status'], 'agreement', result)
        self.assertEqual(result['admission']['route'], 'scoped_local_effects')
        old = deepcopy(r)
        old['config']['oracle_improvements'] = False
        self.assertEqual(audit_execution(old)['status'], 'unsupported')

    def test_synthetic_wrong_backward_remains_detectable_after_update(self):
        class WrongIdentity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()
            @staticmethod
            def backward(ctx, gradient):
                return 2 * gradient
        source = OPTIMIZER_SOURCE.replace('torch.sin', 'torch.clone')
        r = execute(source, dict(api='torch.clone', device='cpu', seed=0,
                    requires_backward=True, oracle_improvements=True), Recorder())
        # Match the synthetic observed VJP and replayed faulty backward; corrupting
        # only saved evidence must fail reconstruction, not report a defect.
        r['gradient_evidence']['events'][0]['inputs'][0]['gradient'].mul_(2.)
        with patch('mirage.inspector.numerical_gradients.resolve', return_value=WrongIdentity.apply):
            result = audit_execution(r)
        self.assertEqual(result['status'], 'violation', result)
        self.assertTrue(result['events'][0]['baseline_verified'])

    def test_snapshot_corruption_does_not_become_a_violation(self):
        r = self.record()
        r['target_calls'][0]['inputs']['args'][0].add_(0.4)
        result = audit_execution(r)
        self.assertEqual(result['status'], 'unknown', result)
        self.assertFalse(result['events'][0]['baseline_verified'])

    def test_missing_or_changed_runtime_witness_is_unassessed(self):
        r = self.record()
        for mutate in (lambda x: x['target_calls'][0].pop('local_vjp_snapshot'),
                       lambda x: x['gradient_evidence']['events'][0].pop('operands_unchanged_at_backward'),
                       lambda x: x['gradient_evidence']['events'][0].update(operands_unchanged_at_backward=False),
                       lambda x: x.update(run_rng_unchanged=False)):
            changed = deepcopy(r)
            mutate(changed)
            with patch('mirage.inspector.numerical_gradients.audit_record') as auditor:
                self.assertIn(audit_execution(changed)['status'], ('unknown', 'unsupported'))
                auditor.assert_not_called()

    def test_gradient_reset_and_multiple_calls_keep_independent_events(self):
        source = OPTIMIZER_SOURCE.replace("    y = probe.call", "    x.grad = None\n    y = probe.call")
        source = source.replace("    ctx['optimizer'].step()", "    ctx['optimizer'].step()\n    ctx['optimizer'].zero_grad(set_to_none=True)\n    z = probe.call(torch.sin, x)\n    z.sum().backward()\n    ctx['optimizer'].step()")
        r = self.record(source)
        self.assertEqual(len(r['target_calls']), 2)
        self.assertFalse(torch.equal(r['target_calls'][0]['inputs']['args'][0], r['target_calls'][1]['inputs']['args'][0]))
        result = audit_execution(r)
        self.assertEqual(result['status'], 'agreement', result)
        self.assertEqual({e['call'] for e in result['events']}, {0, 1})

    def test_actual_mutation_before_backward_invalidates_local_version_witness(self):
        source = SOURCE.replace('torch.sin', 'torch.clone').replace(
            '    y.sum().backward()', '    x.detach().add_(1.)\n    y.sum().backward()')
        r = execute(source, dict(api='torch.clone', device='cpu', seed=0,
                    requires_backward=True, oracle_improvements=True), Recorder())
        self.assertTrue(r['backward_executed'])
        self.assertFalse(r['gradient_evidence']['events'][0]['operands_unchanged_at_backward'])
        self.assertIsNone(r['scoped_vjp_source_certificate'])
        self.assertEqual(audit_execution(r)['status'], 'unsupported')

    def test_source_effect_certificate_remains_bounded(self):
        backend = PyTorchBackend()
        allowed = OPTIMIZER_SOURCE.replace("torch.optim.SGD([x]", "torch.optim.SGD(list({'x': x}.values())")
        self.assertEqual(backend.scoped_vjp_source_certificate(allowed), 'explicit_scoped_local_vjp_v1')
        for before, after in (("    y = probe.call", "    x.register_hook(lambda g: g)\n    y = probe.call"),
                              ("    y = probe.call", "    x.add_(1.)\n    y = probe.call"),
                              ("ctx['optimizer'].step()", "ctx['optimizer'].step(lambda: y.sum())"),
                              ("    y = probe.call", "    x.grad = torch.ones_like(x)\n    y = probe.call")):
            self.assertIsNone(backend.scoped_vjp_source_certificate(OPTIMIZER_SOURCE.replace(before, after)))

    def test_native_schema_return_binding_generalizes_beyond_target_tables(self):
        backend = PyTorchBackend()
        # These target names exercise schema-based return binding only; no API
        # names are added to production semantic or numerical allowlists.
        for api in ('torch.expm1', 'torch.log1p', 'torch.prod'):
            with self.subTest(api=api):
                source = OPTIMIZER_SOURCE.replace('torch.sin', api)
                diagnostic = backend.explain_certificate(source, local_vjp=True, scoped_vjp=True)
                self.assertEqual(diagnostic['certificate'], 'explicit_scoped_local_vjp_v1', diagnostic)
                self.assertTrue(diagnostic['native_return_bindings'])
                self.assertIsNone(backend.numerical_source_certificate(source))
                r = execute(source, dict(api=api, device='cpu', seed=0,
                            requires_backward=True, oracle_improvements=True), Recorder())
                result = audit_execution(r)
                self.assertEqual(result['status'], 'agreement', result)

    def test_native_schema_bound_receiver_is_a_tensor_argument(self):
        source = OPTIMIZER_SOURCE.replace('probe.call(torch.sin, x)', 'probe.call(x.expm1)')
        diagnostic = PyTorchBackend().explain_certificate(source, local_vjp=True, scoped_vjp=True)
        self.assertEqual(diagnostic['certificate'], 'explicit_scoped_local_vjp_v1', diagnostic)
        self.assertEqual(diagnostic['native_return_bindings'][0]['target'], 'torch.Tensor.expm1')

    def test_native_schema_binding_rejects_unknown_wrappers_and_mixed_returns(self):
        backend = PyTorchBackend()
        for api in ('torch.nn.functional.dropout', 'torch.max', 'torch.nonexistent'):
            source = OPTIMIZER_SOURCE.replace('torch.sin', api)
            self.assertIsNone(backend.scoped_vjp_source_certificate(source), api)
        source = OPTIMIZER_SOURCE.replace('torch.sin', 'torch.expm1')
        with patch('torch._C._jit_get_schemas_for_operator', return_value=[]):
            self.assertIsNone(backend.scoped_vjp_source_certificate(source))
        def misleading(value):
            return value
        misleading.__name__ = 'expm1'
        misleading.__module__ = 'torch'
        import math
        for impostor in (misleading, math.expm1):
            with patch('mirage.runtime.api_metadata.resolve', return_value=impostor):
                self.assertIsNone(backend.scoped_vjp_source_certificate(source))

    def test_exact_object_repeats_are_numerically_aggregated_without_semantic_attribution(self):
        source = OPTIMIZER_SOURCE.replace('probe.call(torch.sin, x)', 'probe.call(torch.mul, x, x)')
        config = dict(api='torch.mul', device='cpu', seed=0, requires_backward=True)
        old = execute(source, config, Recorder())
        self.assertEqual(old['gradient_evidence']['events'][0]['status'], 'unknown')
        r = execute(source, dict(config, oracle_improvements=True), Recorder())
        event = r['gradient_evidence']['events'][0]
        self.assertEqual(event['status'], 'complete', event)
        self.assertEqual(event['coordinate_binding'], 'exact_object_sum')
        self.assertEqual({tuple(x['path']) for x in event['inputs']}, {('args', 0), ('args', 1)})
        self.assertEqual(r['target_operand_gradients'], [])
        self.assertEqual(r['target_calls'][0]['backward_bindings'][0]['operand_paths'], [])
        self.assertEqual(audit_execution(r)['status'], 'agreement')
        class WrongProduct(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x * y
            @staticmethod
            def backward(ctx, gradient):
                x, y = ctx.saved_tensors
                return 2 * gradient * y, 2 * gradient * x
        for item in event['inputs']:
            item['gradient'].mul_(2.)
        with patch('mirage.inspector.numerical_gradients.resolve', return_value=WrongProduct.apply):
            result = audit_execution(r)
        self.assertEqual(result['status'], 'violation', result)
        self.assertTrue(result['events'][0]['baseline_verified'])

    def test_exact_object_aggregate_rejects_distinct_views_and_multiplicity_mismatch(self):
        backend = PyTorchBackend()
        x = torch.tensor([0.3, 0.8], dtype=torch.float64, requires_grad=True)
        operands = discover({'args': (x, x)}, backend)
        descriptors = [asdict(d) for d, _ in operands]
        edges = backend.gradient_edges(operands)
        node = (x * x).grad_fn
        self.assertEqual(len(backend.local_vjp_coordinate_edges(node, edges, descriptors)), 2)
        variants = []
        distinct = deepcopy(descriptors)
        distinct[1]['object_group'] = 1
        variants.append(distinct)
        different_coordinates = deepcopy(descriptors)
        different_coordinates[1]['offset'] += 1
        variants.append(different_coordinates)
        duplicate_path = deepcopy(descriptors)
        duplicate_path[1]['path'] = duplicate_path[0]['path']
        variants.append(duplicate_path)
        variants.append(descriptors[:1])
        for metadata in variants:
            self.assertIsNone(backend.local_vjp_coordinate_edges(node, edges, metadata))
        self.assertIsNone(backend.local_vjp_coordinate_edges(node, edges[:1], descriptors))
        self.assertIsNone(backend.local_vjp_coordinate_edges(node, [edges[0], edges[0]], descriptors))

    def test_worker_explicit_representation_overrides_optimizer_mr(self):
        import json
        import tempfile
        from pathlib import Path
        from mirage.runtime.worker import dispatch
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(api='torch.sin', device='cpu', seed=0, requires_backward=True,
                          oracle_improvements=True, representation_candidate={
                              'path': ['args', 0], 'action': 'offset', 'call_index': 0})
            dispatch({'mode': 'execute_stateful_pair', 'source': OPTIMIZER_SOURCE, 'config': config}, Path(tmp))
            result = json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(result['relation'], 'representation')
            self.assertTrue(result['representation_candidate_attained'], result)
            self.assertTrue(result['followup_inspection']['valid'])
            self.assertTrue(result['mutation_candidates'])
            self.assertEqual(result['audit_schedule']['reason'], 'unseen_eligible_unit')
            self.assertTrue(result['audit_candidate_units'])
            self.assertEqual(result['gradient_audits']['followup']['status'], 'agreement', result)
            config.update(attempt=1, oracle_seen_units=result['audit_candidate_units'])
            second = Path(tmp)/'second'
            second.mkdir()
            dispatch({'mode': 'execute_stateful_pair', 'source': OPTIMIZER_SOURCE, 'config': config,
                      'initial_reference': str(Path(tmp)/'source.pt'),
                      'reference_scope': 'initialization'}, second)
            repeat = json.loads((second/'summary.json').read_text())
            self.assertFalse(repeat['audit_schedule']['selected'])
            self.assertEqual(repeat['audit_candidate_units'], [])
            self.assertEqual(repeat['audit_admissions']['source']['status'], 'eligible')
            self.assertTrue(repeat['initialization_reference_verified'])
            self.assertNotIn('replay_followup_evidence_equal', repeat)


class StorageCoordinateTests(unittest.TestCase):
    def test_offset_stride_negative_and_object_alias_are_rebuilt(self):
        base = torch.arange(12, dtype=torch.float64, requires_grad=True)
        x = torch._neg_view(base.as_strided((2, 2), (4, 2), 1))
        chart = chart_for({'args': (x, x), 'kwargs': {}})
        tree, _ = chart.build()
        self.assertTrue(chart.verify(tree))
        self.assertIs(tree['args'][0], tree['args'][1])
        self.assertEqual(tree['args'][0].stride(), (4, 2))
        self.assertTrue(tree['args'][0].is_neg())

    def test_shared_views_and_overlapping_coordinates_pull_back_once(self):
        base = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
        left, right = base[:1].expand(3), base[1:]
        chart = chart_for({'args': (left, right), 'kwargs': {}})
        tree, _ = chart.build()
        self.assertTrue(chart.verify(tree))
        self.assertEqual(len(chart.values), 2)
        self.assertEqual(chart.pullback([
            {'path': ['args', 0], 'gradient': torch.ones(3, dtype=torch.float64)},
            {'path': ['args', 1], 'gradient': torch.ones(1, dtype=torch.float64)}]), [3., 1.])

    def test_json_metadata_sequences_preserve_coordinate_alignment(self):
        import json
        x = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
        tree = {'args': (x, x), 'kwargs': {}}
        metadata = [asdict(d) for d, _ in discover(tree, PyTorchBackend())]
        chart = StorageCoordinateChart({'inputs': snapshot(tree), 'metadata': json.loads(json.dumps(metadata))})
        rebuilt, _ = chart.build()
        self.assertTrue(chart.verify(rebuilt))
        self.assertEqual(chart.pullback([
            {'path': ['args', 0], 'gradient': torch.ones_like(x)},
            {'path': ['args', 1], 'gradient': torch.ones_like(x)}]), [2., 2.])

    def test_mixed_differentiability_aliases_are_unknown(self):
        x = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
        with self.assertRaisesRegex(ReconstructionError, 'differentiability'):
            chart_for({'args': (x, x.detach()), 'kwargs': {}})

    def test_missing_operand_cotangent_is_not_zero(self):
        x = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
        chart = chart_for({'args': (x,), 'kwargs': {}})
        with self.assertRaisesRegex(ReconstructionError, 'cotangents'):
            chart.pullback([])


class NumericalGradientTests(unittest.TestCase):
    def test_projection_noise_scale_survives_cancellation(self):
        from mirage.inspector.numerical_gradients import _scalar
        output = torch.tensor([1e10, 1e10], dtype=torch.float64)
        event = {'outputs': [{'path': '', 'slot': 0, 'gradient': torch.tensor([1., -1.], dtype=torch.float64)}]}
        value, mass = _scalar(output, event)
        self.assertEqual(value, 0.)
        self.assertEqual(mass, 2e10)

    def test_correct_dense_call_agrees_and_is_json_serializable(self):
        import json
        result = audit_record(recorded())
        self.assertEqual(result['status'], 'agreement', result)
        self.assertTrue(result['events'][0]['baseline_verified'])
        self.assertEqual(len(result['events'][0]['directions']), 2)
        json.dumps(result, allow_nan=False)

    def test_local_vjp_gradient_setup_allows_correct_sin_and_add_audits(self):
        sin_source = SOURCE.replace('requires_grad=True', 'requires_grad=False').replace(
            '    y = probe.call', '    x.requires_grad_(True)\n    y = probe.call')
        add_source = sin_source.replace("    return {'inputs': {'x': x}",
            "    other = torch.tensor([0.4, 0.6], dtype=torch.float64, device=device)\n"
            "    return {'inputs': {'x': x, 'other': other}")
        add_source = add_source.replace('    y = probe.call(torch.sin, x)',
            "    other = ctx['inputs']['other']\n"
            "    torch.Tensor.requires_grad_(other, True)\n"
            "    y = probe.call(torch.add, x, other)")
        for source, api in ((sin_source, 'torch.sin'), (add_source, 'torch.add')):
            with self.subTest(api=api):
                record = recorded(source, api)
                self.assertTrue(record['inspection']['valid'])
                self.assertIsNone(record['program_certificate'])
                self.assertEqual(record['numerical_source_certificate'], 'explicit_straight_line_local_vjp_v1')
                result = audit_execution(record)
                self.assertEqual(result['status'], 'agreement', result)
                self.assertTrue(result['events'][0]['baseline_verified'])

    def test_local_vjp_certificate_keeps_readonly_and_baseline_guards(self):
        source = SOURCE.replace('requires_grad=True', 'requires_grad=False').replace(
            '    y = probe.call', '    x.requires_grad_(True)\n    y = probe.call')
        original = recorded(source)
        for field, value in (('schema_read_only', False), ('write_effects', [['args', 0]]),
                             ('writable_paths', [['args', 0]]), ('execution_mode', 'compiled')):
            with self.subTest(field=field):
                changed = deepcopy(original)
                changed['target_calls'][0][field] = value
                result = audit_execution(changed)
                self.assertEqual(result['status'], 'unknown', result)
                self.assertFalse(result['events'][0]['baseline_verified'])
        changed = deepcopy(original)
        changed['target_calls'][0]['output'].add_(1)
        result = audit_execution(changed)
        self.assertEqual(result['status'], 'unknown', result)
        self.assertIn('baseline output', result['events'][0]['reason'])
        changed = deepcopy(original)
        changed['gradient_evidence']['events'][0]['inputs'][0]['gradient'].mul_(2)
        result = audit_execution(changed)
        self.assertEqual(result['status'], 'unknown', result)
        self.assertIn('original VJP', result['events'][0]['reason'])

    @unittest.skipUnless(__import__('os').environ.get('MIRAGE_TEST_CUDA') == '1', 'Slurm CUDA validation only')
    def test_cuda_reconstruction_preserves_device_and_normal_gradient(self):
        self.assertTrue(torch.cuda.is_available(), 'CUDA gate requires an allocated CUDA device')
        for dtype in ('torch.float32', 'torch.float64'):
            with self.subTest(dtype=dtype):
                source = SOURCE.replace('torch.float64', dtype)
                record = execute(source, {'api': 'torch.sin', 'device': 'cuda', 'seed': 0,
                                         'requires_backward': True}, Recorder())
                self.assertTrue(record['completed'], record.get('error'))
                chart = StorageCoordinateChart(record['target_calls'][0])
                tree, _ = chart.build()
                self.assertEqual(tree['args'][0].device.type, 'cuda')
                self.assertTrue(chart.verify(tree))
                result = audit_record(record)
                self.assertEqual(result['status'], 'agreement', result)
                self.assertTrue(result['events'][0]['baseline_verified'])

    def test_nonfinite_derived_slopes_remain_unknown_and_json_safe(self):
        import json
        # Finite scalar values can still overflow during subtraction or division.
        with patch('mirage.inspector.numerical_gradients._scalar',
                   side_effect=[(0., 1.), (1e308, 1e308), (-1e308, 1e308)]):
            result = audit_record(recorded())
        self.assertEqual(result['status'], 'unknown', result)
        self.assertIn('nonfinite finite-difference', result['events'][0]['reason'])
        json.dumps(result, allow_nan=False)

    def test_representation_bits_and_noncontiguity_are_not_materialized_away(self):
        for expression in ('torch._neg_view(-x)',
                           'torch.stack((x, x), dim=-1)[:, 0]',
                           'torch.cat((torch.zeros(1, dtype=x.dtype), x))[1:]'):
            with self.subTest(expression=expression):
                source = SOURCE.replace('y = probe.call(torch.sin, x)',
                    'x = ' + expression + '\n    y = probe.call(torch.sin, x)')
                result = audit_record(recorded(source))
                self.assertEqual(result['status'], 'agreement', result)

    def test_float32_and_scaled_cotangents_use_same_algorithm(self):
        source = SOURCE.replace('torch.float64', 'torch.float32').replace('y.sum().backward()', '(3*y).sum().backward()')
        self.assertEqual(audit_record(recorded(source))['status'], 'agreement')

    def test_repeated_backward_has_separate_assessments(self):
        source = SOURCE.replace('y.sum().backward()',
            'y.sum().backward(retain_graph=True)\n    (2*y).sum().backward()')
        result = audit_record(recorded(source))
        self.assertEqual(result['status'], 'agreement', result)
        self.assertEqual({e['invocation'] for e in result['events']}, {0, 1})

    def test_corrupted_observed_vjp_fails_baseline_instead_of_claiming_bug(self):
        record = recorded()
        record['gradient_evidence']['events'][0]['inputs'][0]['gradient'].mul_(2)
        result = audit_record(record)
        self.assertEqual(result['status'], 'unknown')
        self.assertIn('original VJP', result['events'][0]['reason'])

    def test_tiny_corrupted_observed_vjp_cannot_certify_reconstruction(self):
        for scale in (1e-18, 1e-40, 1e-300):
            for factor in (0., 2.):
                with self.subTest(scale=scale, factor=factor):
                    source = SOURCE.replace('torch.sin', 'torch.clone').replace(
                        'y.sum().backward()', f'({scale!r} * y).sum().backward()')
                    record = recorded(source, 'torch.clone')
                    record['gradient_evidence']['events'][0]['inputs'][0]['gradient'].mul_(factor)
                    result = audit_record(record)
                    self.assertEqual(result['status'], 'unknown', result)
                    event = result['events'][0]
                    self.assertFalse(event['baseline_verified'])
                    self.assertIn('original VJP', event['reason'])

    def test_injected_wrong_backward_is_detected_without_reference_derivative(self):
        class WrongIdentity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()
            @staticmethod
            def backward(ctx, gradient):
                return 2 * gradient
        source = SOURCE.replace('torch.sin', 'torch.clone')
        record = recorded(source, 'torch.clone')
        record['gradient_evidence']['events'][0]['inputs'][0]['gradient'].mul_(2)
        with patch('mirage.inspector.numerical_gradients.resolve', return_value=WrongIdentity.apply):
            result = audit_record(record)
        self.assertEqual(result['status'], 'violation', result)
        self.assertTrue(result['events'][0]['baseline_verified'])
        self.assertTrue(result['events'][0]['stable_steps'])
        self.assertTrue(result['events'][0]['one_sided_consistent'])

    def test_nonsmooth_zero_is_unknown_not_wrong_gradient(self):
        source = SOURCE.replace('[0.3, 0.8]', '[0.0, 0.8]').replace('torch.sin', 'torch.relu')
        result = audit_record(recorded(source, 'torch.relu'))
        self.assertEqual(result['status'], 'unknown', result)
        self.assertTrue(any(not d['one_sided_consistent'] for d in result['events'][0]['directions']))

    def test_incomplete_or_duplicate_invocations_do_not_certify(self):
        record = recorded()
        record['backward_invocations'][0]['completed'] = False
        self.assertEqual(audit_record(record)['status'], 'unknown')
        record = recorded()
        record['gradient_evidence']['events'].append(deepcopy(record['gradient_evidence']['events'][0]))
        self.assertEqual(audit_record(record)['status'], 'unknown')

    def test_mutation_and_context_are_outside_local_numeric_contract(self):
        for field, value in (('schema_read_only', False), ('execution_mode', 'compiled')):
            with self.subTest(field=field):
                record = recorded(); record['target_calls'][0][field] = value
                self.assertEqual(audit_record(record)['status'], 'unknown')
        record = recorded(); record['target_calls'][0]['entry_context']['cpu_enabled'] = True
        self.assertEqual(audit_record(record)['status'], 'unknown')

    def test_budget_and_truncated_evidence_do_not_become_agreement(self):
        record = recorded()
        self.assertEqual(audit_record(record, budget_seconds=0)['status'], 'unknown')
        record['gradient_evidence']['dropped_events'] = 1
        self.assertEqual(audit_record(record)['status'], 'unknown')
        del record['gradient_evidence']
        self.assertEqual(audit_record(record)['status'], 'unsupported')


    def test_nonlocal_reciprocal_steps_do_not_discredit_a_large_correct_vjp(self):
        source = SOURCE.replace('torch.float64', 'torch.float32').replace(
            '[0.3, 0.8]', '[1e-8, 0.8]').replace('torch.sin', 'torch.reciprocal')
        result = audit_record(recorded(source, 'torch.reciprocal'))
        self.assertEqual(result['status'], 'unknown', result)
        direction = result['events'][0]['directions'][0]
        self.assertFalse(direction['one_sided_consistent'])
        self.assertEqual(direction['status'], 'unknown')

    def test_singular_crossing_reference_stays_unknown_under_cotangent_scaling(self):
        # A generic local finite-difference oracle must reject a nonlocal stencil;
        # this is a correct derivative, not an expected framework failure.
        source = """import torch

def build(device, seed):
    x = torch.tensor([[1e-8, 0.0], [0.0, 2.0]], dtype=torch.float32, device=device, requires_grad=True)
    return {'inputs': {'x': x}, 'parameters': {}}

def run(ctx, probe):
    x = ctx['inputs']['x']
    sign, logabsdet = probe.call(torch.linalg.slogdet, x)
    (SCALE * (sign.sum() + logabsdet.sum())).backward()
    return {'sign': sign, 'logabsdet': logabsdet, 'gradient': x.grad}
"""
        for scale in (1., 1e-18):
            with self.subTest(scale=scale):
                record = recorded(source.replace('SCALE', repr(scale)), 'torch.linalg.slogdet')
                result = audit_record(record)
                self.assertEqual(result['status'], 'unknown', result)
                direction = result['events'][0]['directions'][0]
                self.assertFalse(direction['stable_steps'])
                self.assertFalse(direction['one_sided_consistent'])
                self.assertEqual(direction['status'], 'unknown')

    def test_stable_wrong_backward_remains_detectable_across_magnitude_and_scaling(self):
        for factor in (2., 1e8):
            class WrongIdentity(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x.clone()
                @staticmethod
                def backward(ctx, gradient):
                    return factor * gradient
            for scale in (1., 1e-18, 1e-40):
                with self.subTest(factor=factor, scale=scale):
                    source = SOURCE.replace('torch.sin', 'torch.clone').replace(
                        'y.sum().backward()', f'({scale!r} * y).sum().backward()')
                    record = recorded(source, 'torch.clone')
                    record['gradient_evidence']['events'][0]['inputs'][0]['gradient'].mul_(factor)
                    with patch('mirage.inspector.numerical_gradients.resolve', return_value=WrongIdentity.apply):
                        result = audit_record(record)
                    self.assertEqual(result['status'], 'violation', result)
                    self.assertTrue(result['events'][0]['baseline_verified'])
                    self.assertTrue(all(d['stable_steps'] and d['one_sided_consistent']
                                        for d in result['events'][0]['directions']))

    def test_zero_and_small_linear_derivatives_still_agree(self):
        for factor in (0., 1., 1e-40):
            with self.subTest(factor=factor):
                source = SOURCE.replace('y = probe.call(torch.sin, x)',
                    f'y = probe.call(torch.mul, x, {factor!r})')
                result = audit_record(recorded(source, 'torch.mul'))
                self.assertEqual(result['status'], 'agreement', result)

    def test_unrepresentable_projection_does_not_become_exact_zero(self):
        from mirage.inspector.numerical_gradients import _scalar, AuditUnknown
        output = torch.tensor([1e-200], dtype=torch.float64)
        event = {'outputs': [{'path': '', 'slot': 0,
                             'gradient': torch.tensor([1e-200], dtype=torch.float64)}]}
        with self.assertRaisesRegex(AuditUnknown, 'projection underflow'):
            _scalar(output, event)


if __name__ == '__main__':
    unittest.main()
