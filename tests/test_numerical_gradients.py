"""Generic checker controls; injected bad evidence is never a framework finding."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from mirage.backends.pytorch import PyTorchBackend
from mirage.inspector.numerical_gradients import audit_record
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


if __name__ == '__main__':
    unittest.main()
