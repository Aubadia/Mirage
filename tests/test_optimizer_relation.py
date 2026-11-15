"""Native SGD controls; synthetic record corruption tests oracle sensitivity only."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import torch

from mirage.backends.pytorch import PyTorchBackend
from mirage.rules.catalog import optimizer_gradient_candidates, optimizer_gradient_pair
from mirage.runtime.probes import Recorder
from mirage.runtime.stateful import execute
from mirage.runtime.worker import dispatch, optimizer_followup_config, stateful_replay_evidence


SOURCE = '''import torch

def build(device, seed):
    x = torch.tensor([[0.2, -0.3], [0.4, 0.7]], dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=0.125, momentum=0.9, weight_decay=0.0,
                                dampening=0.0, nesterov=False, maximize=False,
                                foreach=False, fused=False, differentiable=False)
    return {'inputs': {}, 'parameters': {'x': x}, 'optimizer': optimizer}

def run(ctx, probe):
    x = ctx['parameters']['x']
    y = probe.call(torch.sin, x)
    y.sum().backward()
    ctx['optimizer'].step()
    return {'output': y}
'''
CONFIG = {'api': 'torch.sin', 'device': 'cpu', 'requires_backward': True, 'seed': 0, 'predicates': []}


class OptimizerRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = PyTorchBackend()
        cls.left = execute(SOURCE, CONFIG, Recorder())
        candidates = optimizer_gradient_candidates(cls.left)
        if not candidates:
            raise AssertionError('Native fixture must establish an optimizer chain')
        cls.candidate = candidates[0]
        cls.right = execute(SOURCE, optimizer_followup_config(cls.left, cls.candidate), Recorder(), cls.left)

    def relation(self, left=None, right=None):
        return optimizer_gradient_pair(self.left if left is None else left,
            self.right if right is None else right, self.backend, step_index=0, parameter_key='x')

    def test_native_gradient_layout_pair_compares_parameter_and_slot(self):
        result = self.relation()
        self.assertEqual(result['verdict'], 'pass', result)
        self.assertFalse(result['confirmed_defect'])
        self.assertTrue(result['exact_output_comparison']['equal'])
        self.assertGreater(result['max_abs_parameter_update']['source']['x'], 0)
        self.assertIn('momentum_buffer', self.right['optimizer_evidence']['steps'][0]['post']['slots']['x'])

    def test_native_zero_momentum_and_binary_target_are_supported(self):
        source = SOURCE.replace('momentum=0.9', 'momentum=0.0').replace('probe.call(torch.sin, x)', 'probe.call(torch.add, x, 0.25)')
        config = dict(CONFIG, api='torch.add')
        left = execute(source, config, Recorder())
        candidate = optimizer_gradient_candidates(left)[0]
        right = execute(source, optimizer_followup_config(left, candidate), Recorder(), left)
        self.assertEqual(self.relation(left, right)['verdict'], 'pass')

    def test_observed_zero_vjp_retains_chain_and_zero_update_is_only_secondary(self):
        source = SOURCE.replace('probe.call(torch.sin, x)', 'probe.call(torch.mul, x, 0.0)')
        left = execute(source, dict(CONFIG, api='torch.mul'), Recorder())
        candidate = optimizer_gradient_candidates(left)[0]
        right = execute(source, optimizer_followup_config(left, candidate), Recorder(), left)
        result = self.relation(left, right)
        self.assertEqual(result['verdict'], 'pass', result)
        self.assertEqual(result['max_abs_parameter_update']['source']['x'], 0.0)
        self.assertFalse(result['confirmed_defect'])

    def test_scalar_and_empty_gradients_use_attainable_offset_followup(self):
        for value in ('0.2', '[]'):
            with self.subTest(value=value):
                source = SOURCE.replace('[[0.2, -0.3], [0.4, 0.7]]', value)
                left = execute(source, CONFIG, Recorder())
                candidate = optimizer_gradient_candidates(left)[0]
                config = optimizer_followup_config(left, candidate)
                self.assertEqual(config['optimizer_gradient_action']['action'], 'offset')
                right = execute(source, config, Recorder(), left)
                self.assertEqual(self.relation(left, right)['verdict'], 'pass')

    def test_source_owned_representation_action_is_not_replaced_by_followup_attainment(self):
        config = dict(CONFIG, optimizer_gradient_action={'step_index': 0, 'parameter_key': 'x', 'action': 'offset'})
        left = execute(SOURCE, config, Recorder())
        candidate = optimizer_gradient_candidates(left)[0]
        followup = optimizer_followup_config(left, candidate)
        self.assertEqual(followup['optimizer_gradient_action']['action'], 'canonical')
        right = execute(SOURCE, followup, Recorder(), left)
        self.assertEqual(self.relation(left, right)['verdict'], 'pass')
        invalid = deepcopy(left)
        invalid['inspection']['valid'] = False
        self.assertEqual(optimizer_gradient_candidates(invalid), [])
        self.assertEqual(self.relation(invalid, right)['verdict'], 'inapplicable')

    def test_only_selected_representation_goal_is_relaxed(self):
        selected = {'dimension': 'gradient_representation', 'parameter_key': 'x',
                    'checkpoint': self.candidate['pre_checkpoint'], 'field': 'contiguous', 'expected': False}
        companion = dict(selected, dimension='optimizer', field='completed', expected=True,
                         checkpoint=self.candidate['post_checkpoint'])
        other_parameter = dict(selected, parameter_key='other')
        other_step = dict(selected, checkpoint='optimizer:0/step:1/pre_update')
        record = dict(self.left, config=dict(CONFIG, predicates=[selected, companion, other_parameter, other_step]))
        followup = optimizer_followup_config(record, self.candidate)
        self.assertEqual(followup['predicates'], [companion, other_parameter, other_step])
        self.assertEqual(len(record['config']['predicates']), 4)

    def test_pre_state_environment_program_and_registry_mismatches_reject(self):
        def change_state(record, key):
            step = record['optimizer_evidence']['steps'][0]
            if key == 'slots':
                step['pre']['slots']['x']['unexpected'] = torch.ones(1, dtype=torch.float64)
            else:
                step['pre'][key]['x'] = step['pre'][key]['x'] + 1
        changes = [lambda r: change_state(r, 'parameters'), lambda r: change_state(r, 'gradients'),
                   lambda r: change_state(r, 'slots'),
                   lambda r: r['environment'].update(control_difference=True),
                   lambda r: r['graph'].update(fingerprint='different'),
                   lambda r: r['optimizer_evidence']['registry']['groups'][0]['options'].update(lr=0.25)]
        for change in changes:
            with self.subTest(change=change):
                right = deepcopy(self.right)
                change(right)
                self.assertEqual(self.relation(right=right)['verdict'], 'inapplicable')

    def test_missing_domain_provenance_effects_completion_and_observations_reject(self):
        changes = [lambda r: r.update(optimizer_source_certificate=None),
                   lambda r: r['optimizer_evidence']['steps'][0]['domain'].update(status='unknown'),
                   lambda r: r['optimizer_evidence']['steps'][0].update(completed=False),
                   lambda r: r['optimizer_evidence']['steps'][0].update(gradient_provenance=[]),
                   lambda r: r['optimizer_evidence']['steps'][0].update(post={}),
                   lambda r: r['optimizer_evidence']['steps'][0]['mutation'].update(logical_equal=False),
                   lambda r: r.update(target_parameter_bindings=[])]
        for change in changes:
            with self.subTest(change=change):
                right = deepcopy(self.right)
                change(right)
                self.assertEqual(self.relation(right=right)['verdict'], 'inapplicable')

    def test_synthetic_output_corruption_is_sensitivity_not_framework_defect(self):
        for field, slot in [('parameters', None), ('slots', 'momentum_buffer')]:
            right = deepcopy(self.right)
            post = right['optimizer_evidence']['steps'][0]['post']
            if slot is None:
                post[field]['x'] = post[field]['x'] + 0.25
            else:
                post[field]['x'][slot] = post[field]['x'][slot] + 0.25
            result = self.relation(right=right)
            self.assertEqual(result['verdict'], 'suspected_failure')
            self.assertFalse(result['confirmed_defect'])

    def test_nonfinite_outputs_are_unassessed_and_metadata_cannot_create_change(self):
        right = deepcopy(self.right)
        right['optimizer_evidence']['steps'][0]['post']['parameters']['x'][0, 0] = float('nan')
        self.assertIsNone(self.relation(right=right)['verdict'])
        same = deepcopy(self.right)
        same['optimizer_evidence']['steps'][0]['pre']['gradient_metadata'] = deepcopy(
            self.left['optimizer_evidence']['steps'][0]['pre']['gradient_metadata'])
        self.assertEqual(self.relation(right=same)['verdict'], 'inapplicable')

    def test_worker_routes_native_pair_and_replay_preserves_new_evidence(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dispatch({'mode': 'execute_stateful_pair', 'source': SOURCE,
                      'config': dict(CONFIG, numerical_audit=False)}, path)
            summary = json.loads((path / 'summary.json').read_text())
            self.assertEqual(summary['relation'], 'optimizer_gradient_representation')
            self.assertEqual(summary['verdict'], 'pass', summary)
            self.assertTrue(summary['optimizer_coverage']['eligible'])
            self.assertTrue(summary['followup_optimizer_coverage']['eligible'])
            self.assertTrue(summary['target_parameter_bindings'])
        replay = stateful_replay_evidence(self.left)
        self.assertIn('optimizer_evidence', replay)
        self.assertIn('optimizer_source_certificate', replay)
        self.assertIn('target_parameter_bindings', replay)

    def worker_goals(self):
        representation = {'dimension': 'gradient_representation', 'field': 'contiguous', 'expected': False,
            'checkpoint': self.candidate['pre_checkpoint'], 'parameter_key': 'x', 'call_index': 0,
            'operand_path': ['args', 0], 'relation': 'target_gradient_update'}
        completed = dict(representation, dimension='optimizer', field='completed', expected=True,
                         checkpoint=self.candidate['post_checkpoint'])
        return representation, completed

    def test_worker_requires_source_layout_goal_and_preserves_followup_update_goal(self):
        import json
        representation, completed = self.worker_goals()
        config = dict(CONFIG, predicates=[representation, completed], numerical_audit=False,
                      optimizer_gradient_action={'step_index': 0, 'parameter_key': 'x', 'action': 'strided'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dispatch({'mode': 'execute_stateful_pair', 'source': SOURCE, 'config': config}, path)
            summary = json.loads((path / 'summary.json').read_text())
            self.assertEqual(summary['verdict'], 'pass', summary)
            self.assertTrue(summary['source_goal_valid'])
            self.assertTrue(summary['followup_goal_valid'])
            source = torch.load(path / 'source.pt', map_location='cpu', weights_only=True)
            followup = torch.load(path / 'followup.pt', map_location='cpu', weights_only=True)
            self.assertEqual(source['config']['predicates'], [representation, completed])
            self.assertEqual(followup['config']['predicates'], [completed])
            self.assertEqual(source['config']['optimizer_gradient_action']['action'], 'strided')
            self.assertEqual(followup['config']['optimizer_gradient_action']['action'], 'canonical')

    def test_worker_unattained_source_goal_is_invalid_without_followup_rescue(self):
        import json
        representation, completed = self.worker_goals()
        config = dict(CONFIG, predicates=[representation, completed], numerical_audit=False,
                      optimizer_gradient_action={'step_index': 0, 'parameter_key': 'x', 'action': 'canonical'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dispatch({'mode': 'execute_stateful_pair', 'source': SOURCE, 'config': config}, path)
            summary = json.loads((path / 'summary.json').read_text())
            self.assertEqual(summary['verdict'], 'invalid', summary)
            self.assertFalse(summary['inspection']['valid'])
            self.assertFalse(summary['optimizer_coverage']['eligible'])
            self.assertTrue(summary['source_completed'])
            self.assertFalse(summary['followup_completed'])
            self.assertFalse((path / 'followup.pt').exists())


if __name__ == '__main__':
    unittest.main()
