"""Native optimizer lifecycle controls; synthetic mutations are not bug findings."""
from copy import deepcopy
import unittest

import torch

from mirage.backends.pytorch import PyTorchBackend
from mirage.generator.program import ContractError
from mirage.runtime.probes import Recorder
from mirage.runtime.stateful import execute

SOURCE = '''import torch

def build(device, seed):
    p = torch.tensor([[0.3, 0.8], [0.4, 0.6]], dtype=torch.float64, device=device, requires_grad=True)
    other = torch.tensor([[0.2, 0.5], [0.7, 0.9]], dtype=torch.float64, device=device)
    optimizer = torch.optim.SGD([p], lr=0.1, momentum=0.8, foreach=False, fused=False)
    return {'inputs': {'other': other}, 'parameters': {'p': p}, 'optimizer': optimizer}

def run(ctx, probe):
    p = ctx['parameters']['p']
    optimizer = ctx['optimizer']
    y = probe.call(torch.sin, p)
    y.sum().backward()
    optimizer.step()
    return {'output': y, 'gradient': p.grad}
'''
CONFIG = {'api': 'torch.sin', 'device': 'cpu', 'seed': 0, 'requires_backward': True}

MULTI_SOURCE = '''import torch

def build(device, seed):
    p = torch.tensor([[0.3, 0.8], [0.4, 0.6]], dtype=torch.float64, device=device, requires_grad=True)
    q = torch.tensor([[0.2, 0.5], [0.7, 0.9]], dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([{'params': [p], 'lr': 0.1}, {'params': [q], 'lr': 0.2}],
                                lr=0.1, momentum=0.8, foreach=False, fused=False)
    return {'inputs': {}, 'parameters': {'p': p, 'q': q}, 'optimizer': optimizer}

def run(ctx, probe):
    p = ctx['parameters']['p']
    q = ctx['parameters']['q']
    y = probe.call(torch.mul, p, q)
    y.sum().backward()
    ctx['optimizer'].step()
    return {'output': y}
'''


def run(source=SOURCE, *, action=None, api='torch.sin'):
    config = dict(CONFIG, api=api)
    if action is not None:
        config['optimizer_gradient_action'] = {'step_index': 0, 'parameter_key': 'p', 'action': action}
    return execute(source, config, Recorder())


class OptimizerRuntimeTests(unittest.TestCase):
    def test_huge_empty_layout_never_enumerates_coordinate_pools(self):
        from unittest.mock import patch
        from mirage.runtime.optimizer import nonoverlapping

        class EmptyLayout:
            shape = (0, 10**9)

            def numel(self):
                return 0

            def stride(self):
                return (10**9, 1)

        with patch('mirage.runtime.optimizer.product', side_effect=AssertionError('empty layout must not enumerate')) as enumerate_positions:
            self.assertTrue(nonoverlapping(EmptyLayout()))
            enumerate_positions.assert_not_called()

    def test_native_parameter_constructor_is_scoped_to_optimizer_certificate(self):
        for constructor in ('torch.nn.Parameter(p)', 'torch.nn.Parameter(p, requires_grad=True)',
                            'torch.nn.Parameter(data=p, requires_grad=True)'):
            with self.subTest(constructor=constructor):
                source = SOURCE.replace('    other = torch.tensor', '    p = ' + constructor + '\n    other = torch.tensor')
                backend = PyTorchBackend()
                self.assertEqual(backend.optimizer_source_certificate(source), 'explicit_optimizer_gradient_update_v1')
                self.assertIsNone(backend.program_certificate(source))
                self.assertIsNone(backend.numerical_source_certificate(source))
                record = run(source, action='strided')
                self.assertEqual(record['optimizer_evidence']['steps'][0]['gradient_provenance'][0]['status'], 'verified')

    def test_parameter_constructor_requires_native_binding_and_explicit_gradient_flag(self):
        for setup in ('p = torch.nn.Parameter(1)', 'p = torch.nn.Parameter(p, requires_grad=False)',
                      'flag = True\n    p = torch.nn.Parameter(p, requires_grad=flag)',
                      'torch.nn.Parameter = torch.tensor', 'namespace = torch.nn\n    namespace.Parameter = torch.tensor',
                      'p = torch.nn.Parameter(*(p,))'):
            source = SOURCE.replace('    other = torch.tensor', '    ' + setup + '\n    other = torch.tensor')
            self.assertIsNone(PyTorchBackend().optimizer_source_certificate(source), setup)

    def test_native_sgd_complete_chain_and_stable_slot_identity(self):
        record = run()
        self.assertTrue(record['inspection']['valid'], record['inspection'])
        self.assertEqual(record['optimizer_source_certificate'], 'explicit_optimizer_gradient_update_v1')
        self.assertIsNone(record['program_certificate'])
        evidence = record['optimizer_evidence']
        self.assertEqual(evidence['limitations'], [])
        self.assertEqual(evidence['registry']['groups'][0]['parameter_keys'], ['p'])
        self.assertEqual(evidence['registry']['parameters'][0]['parameter_id'], 'parameter:p')
        step, = evidence['steps']
        self.assertTrue(step['completed'])
        self.assertEqual(step['domain']['status'], 'verified')
        origin, = step['gradient_provenance']
        self.assertEqual(origin['status'], 'verified', origin)
        self.assertEqual(origin['operand_path'], ['args', 0])
        self.assertTrue(origin['unchanged_to_pre_step'])
        self.assertEqual(step['pre']['slots'], {'p': {}})
        self.assertEqual(set(step['post']['slots']['p']), {'momentum_buffer'})
        self.assertTrue(torch.equal(step['pre']['gradients']['p'], step['post']['slots']['p']['momentum_buffer']))
        self.assertTrue(step['parameter_changed'])
        self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 1)
        self.assertEqual([c['name'] for c in record['checkpoints']],
                         ['initial', 'optimizer:0/step:0/pre_update', 'optimizer:0/step:0/post_update', 'final'])

    def test_each_case_action_attains_actual_consumed_gradient_representation(self):
        records = {action: run(action=action) for action in ('canonical', 'strided', 'offset')}
        for action, record in records.items():
            step, = record['optimizer_evidence']['steps']
            self.assertEqual(step['mutation']['status'], 'verified', step)
            self.assertTrue(step['mutation']['logical_equal'])
            self.assertTrue(torch.equal(step['mutation']['before_values'], step['pre']['gradients']['p']))
            metadata = step['pre']['gradient_metadata']['p']
            self.assertEqual(metadata['properties']['contiguous'], action != 'strided')
            self.assertEqual(metadata['offset'] > 0, action == 'offset')
            self.assertEqual(step['gradient_provenance'][0]['status'], 'verified')
        reference = records['canonical']['optimizer_evidence']['steps'][0]['post']
        for action in ('strided', 'offset'):
            actual = records[action]['optimizer_evidence']['steps'][0]['post']
            self.assertTrue(torch.equal(reference['parameters']['p'], actual['parameters']['p']))
            self.assertTrue(torch.equal(reference['slots']['p']['momentum_buffer'], actual['slots']['p']['momentum_buffer']))

    def test_binary_and_matrix_targets_keep_parameter_operand_identity(self):
        for api, expression in (('torch.add', "torch.add, p, ctx['inputs']['other']"),
                                ('torch.mm', "torch.mm, p, ctx['inputs']['other']")):
            with self.subTest(api=api):
                source = SOURCE.replace('torch.sin, p', expression)
                record = run(source, action='strided', api=api)
                step, = record['optimizer_evidence']['steps']
                self.assertEqual(step['domain']['status'], 'verified', step)
                self.assertEqual(step['gradient_provenance'][0]['status'], 'verified', step)
                self.assertTrue(step['completed'])

    def test_two_parameters_and_groups_keep_both_chains_and_independent_slots(self):
        source = run(MULTI_SOURCE, api='torch.mul')
        transformed = run(MULTI_SOURCE, api='torch.mul', action='strided')
        for record in (source, transformed):
            with self.subTest(transformed=record is transformed):
                self.assertTrue(record['inspection']['valid'])
                evidence = record['optimizer_evidence']
                self.assertEqual(evidence['limitations'], [])
                groups = evidence['registry']['groups']
                self.assertEqual([(g['group_index'], g['parameter_keys'], g['options']['lr']) for g in groups],
                                 [(0, ['p'], 0.1), (1, ['q'], 0.2)])
                self.assertEqual(evidence['registry']['parameters'], [
                    {'parameter_key': 'p', 'parameter_id': 'parameter:p', 'group_index': 0, 'group_position': 0},
                    {'parameter_key': 'q', 'parameter_id': 'parameter:q', 'group_index': 1, 'group_position': 0}])
                step, = evidence['steps']
                self.assertTrue(step['completed'])
                self.assertEqual(step['domain']['status'], 'verified')
                origins = {item['parameter_key']: item for item in step['gradient_provenance']}
                self.assertEqual(set(origins), {'p', 'q'})
                for index, key in enumerate(('p', 'q')):
                    self.assertEqual(origins[key]['status'], 'verified', origins[key])
                    self.assertEqual(origins[key]['operand_path'], ['args', index])
                    self.assertTrue(origins[key]['unchanged_to_pre_step'])
                    self.assertEqual(step['pre']['slots'][key], {})
                    self.assertEqual(set(step['post']['slots'][key]), {'momentum_buffer'})
                    self.assertTrue(torch.equal(step['post']['slots'][key]['momentum_buffer'], step['pre']['gradients'][key]))
                    self.assertFalse(torch.equal(step['pre']['parameters'][key], step['post']['parameters'][key]))
                self.assertEqual({item['parameter_key'] for item in record['inspection']['optimizer_coverage']['eligible_chains']}, {'p', 'q'})
                # This is one completed optimizer step with two parameter chains.
                self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 1)
        before, after = source['optimizer_evidence']['steps'][0], transformed['optimizer_evidence']['steps'][0]
        self.assertEqual(after['mutation']['parameter_key'], 'p')
        self.assertEqual(after['mutation']['status'], 'verified')
        self.assertTrue(before['pre']['gradient_metadata']['p']['properties']['contiguous'])
        self.assertFalse(after['pre']['gradient_metadata']['p']['properties']['contiguous'])
        self.assertEqual(before['pre']['gradient_metadata']['q'], after['pre']['gradient_metadata']['q'])
        for key in ('p', 'q'):
            self.assertTrue(torch.equal(before['pre']['gradients'][key], after['pre']['gradients'][key]))
            self.assertTrue(torch.equal(before['post']['parameters'][key], after['post']['parameters'][key]))
            self.assertTrue(torch.equal(before['post']['slots'][key]['momentum_buffer'], after['post']['slots'][key]['momentum_buffer']))

    def test_shared_parameter_storage_retains_execution_but_not_update_proof(self):
        source = MULTI_SOURCE.replace(
            'q = torch.tensor([[0.2, 0.5], [0.7, 0.9]], dtype=torch.float64, device=device, requires_grad=True)',
            'q = p.detach().requires_grad_(True)')
        record = run(source, api='torch.mul')
        self.assertTrue(record['completed'])
        self.assertTrue(record['inspection']['valid'])
        self.assertEqual(record['optimizer_source_certificate'], 'explicit_optimizer_gradient_update_v1')
        evidence = record['optimizer_evidence']
        self.assertIn('parameter storage sharing or overlap is unsupported', evidence['limitations'])
        self.assertTrue(evidence['steps'][0]['completed'])
        self.assertEqual(evidence['steps'][0]['domain']['status'], 'unknown')
        self.assertEqual(record['inspection']['optimizer_coverage']['eligible_chains'], [])
        self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 0)

    def test_no_optimizer_still_records_real_target_parameter_binding(self):
        source = SOURCE.replace("'optimizer': optimizer", "'optimizer': None").replace('    optimizer.step()\n', '')
        record = run(source)
        self.assertEqual(record['optimizer_evidence']['steps'], [])
        self.assertEqual(record['optimizer_evidence']['limitations'], [])
        self.assertEqual(record['target_parameter_bindings'], [{'parameter_key': 'p', 'parameter_id': 'parameter:p',
                          'call_index': 0, 'operand_path': ['args', 0]}])
        self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 0)

    def test_unregistered_optimizer_parameter_is_a_contract_error(self):
        source = SOURCE.replace("'parameters': {'p': p}", "'parameters': {}")
        with self.assertRaisesRegex(ContractError, 'Register every optimizer parameter'):
            run(source)

    def test_unknown_optimizer_domain_retains_execution_without_proof(self):
        for change in (lambda s: s.replace('foreach=False', 'foreach=True'),
                       lambda s: s.replace('dtype=torch.float64', 'dtype=torch.float32'),
                       lambda s: s.replace('momentum=0.8', 'momentum=0.8, weight_decay=0.1'),
                       lambda s: s.replace('torch.optim.SGD([p], lr=0.1, momentum=0.8, foreach=False, fused=False)',
                                           'torch.optim.Adam([p], lr=0.1, foreach=False, fused=False)')):
            with self.subTest(source=change(SOURCE)):
                record = run(change(SOURCE))
                self.assertTrue(record['completed'])
                self.assertTrue(record['optimizer_evidence']['limitations'])
                self.assertNotEqual(record['optimizer_evidence']['steps'][0]['domain']['status'], 'verified')
                self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 0)

    def test_unrelated_parameter_loss_does_not_borrow_target_backward(self):
        source = SOURCE.replace('requires_grad=True)', 'requires_grad=True)', 1).replace(
            'device=device)\n    optimizer', 'device=device, requires_grad=True)\n    optimizer')
        source = source.replace('y = probe.call(torch.sin, p)', "y = probe.call(torch.sin, ctx['inputs']['other'])")
        source = source.replace('y.sum().backward()', '(y.sum() + p.sum()).backward()')
        record = run(source)
        step, = record['optimizer_evidence']['steps']
        self.assertEqual(step['domain']['status'], 'verified')
        self.assertTrue(step['completed'])
        self.assertEqual(step['gradient_provenance'][0]['status'], 'unknown')
        self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 0)

    def test_zeroing_and_extra_backward_do_not_establish_chain(self):
        for source in (SOURCE.replace('    optimizer.step()', '    optimizer.zero_grad(set_to_none=True)\n    optimizer.step()'),
                       SOURCE.replace('    optimizer.step()', '    p.sum().backward()\n    optimizer.step()')):
            with self.subTest(source=source):
                record = run(source)
                self.assertEqual(record['optimizer_evidence']['steps'][0]['domain']['status'], 'unknown')
                self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 0)

    def test_manual_gradient_replacement_and_hooks_lack_source_effect_certificate(self):
        for statement in ('p.grad = p.grad.clone()', 'p.grad.zero_()', 'p.data.add_(1)',
                          'p.register_hook(torch.neg)', 'optimizer.register_step_pre_hook(torch.neg)'):
            source = SOURCE.replace('    optimizer.step()', '    ' + statement + '\n    optimizer.step()')
            self.assertIsNone(PyTorchBackend().optimizer_source_certificate(source), statement)
        record = run(SOURCE.replace('    optimizer.step()', '    p.grad = p.grad.clone()\n    optimizer.step()'))
        self.assertEqual(record['optimizer_evidence']['steps'][0]['domain']['status'], 'unknown')

    def test_step_closure_and_dynamic_zero_flags_are_not_certified(self):
        for expression in ('optimizer.step(None)', 'optimizer.step(closure=None)',
                           'optimizer.zero_grad(set_to_none=False)', 'optimizer.zero_grad(**{})'):
            source = SOURCE.replace('optimizer.step()', expression)
            self.assertIsNone(PyTorchBackend().optimizer_source_certificate(source), expression)

    def test_reserved_optimizer_checkpoint_cannot_be_authored(self):
        source = SOURCE.replace('    optimizer.step()', "    probe.mark('optimizer:0/step:0/post_update')\n    optimizer.step()")
        with self.assertRaisesRegex(ContractError, 'reserved'):
            run(source)

    def test_no_parameter_value_change_is_not_a_domain_rejection(self):
        # Tiny but positive lr can round the real parameter update to unchanged.
        record = run(SOURCE.replace('lr=0.1', 'lr=1e-300'))
        step, = record['optimizer_evidence']['steps']
        self.assertTrue(step['completed'])
        self.assertEqual(step['domain']['status'], 'verified')
        self.assertEqual(step['gradient_provenance'][0]['status'], 'verified')
        self.assertFalse(step['parameter_changed'])

    def test_second_step_cannot_extend_the_single_step_domain(self):
        record = run(SOURCE.replace('    optimizer.step()', '    optimizer.step()\n    optimizer.step()'))
        self.assertEqual(len(record['optimizer_evidence']['steps']), 2)
        self.assertEqual(record['optimizer_evidence']['steps'][1]['domain']['status'], 'unknown')
        self.assertIn('only one completed optimizer step is reviewed', record['optimizer_evidence']['limitations'])
        self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 0)

    def test_custom_optimizer_executes_without_native_proof(self):
        source = SOURCE.replace('def build',
            'class CustomOptimizer(torch.nn.Module):\n    def state_dict(self):\n        return {}\n    def step(self):\n        return None\n\ndef build', 1)
        source = source.replace('torch.optim.SGD([p], lr=0.1, momentum=0.8, foreach=False, fused=False)', 'CustomOptimizer()')
        record = run(source)
        self.assertTrue(record['completed'])
        self.assertEqual(record['optimizer_evidence']['steps'], [])
        self.assertIn('only exact native SGD is reviewed', record['optimizer_evidence']['limitations'])

    def test_zero_target_gradient_has_real_provenance_without_an_informative_change(self):
        record = run(SOURCE.replace('y.sum().backward()', '(y.sum() * 0).backward()'))
        self.assertTrue(record['inspection']['valid'])
        step, = record['optimizer_evidence']['steps']
        self.assertTrue(step['completed'])
        self.assertEqual(step['domain']['status'], 'verified')
        self.assertEqual(step['gradient_provenance'][0]['status'], 'verified')
        self.assertFalse(step['gradient_provenance'][0]['informative_nonzero'])
        self.assertFalse(step['parameter_changed'])
        self.assertEqual(record['checkpoints'][-1]['state_facts']['optimizer.target_update_count'], 1)

    def test_invalid_action_configuration_is_diagnosed_before_execution(self):
        for action in ({'step_index': 1, 'parameter_key': 'p', 'action': 'strided'},
                       {'step_index': 0, 'parameter_key': 'p', 'action': 'negative'}):
            with self.subTest(action=action), self.assertRaises(ContractError):
                execute(SOURCE, dict(CONFIG, optimizer_gradient_action=action), Recorder())


if __name__ == '__main__':
    unittest.main()
