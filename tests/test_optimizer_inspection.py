"""Metadata-only optimizer-chain controls; no fixture claims a framework defect."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from mirage.contracts.capabilities import StatePredicate
from mirage.inspector.stateful import inspect_optimizer, inspect_state, optimizer_fact, semantic_coverage


def optimizer_record():
    metadata = {'path': ['args', 0], 'shape': [2], 'strides': [2], 'offset': 0,
        'dtype': 'torch.float64', 'device': 'cpu', 'layout': 'torch.strided',
        'requires_grad': False, 'object_group': 0, 'storage_group': 0,
        'properties': {'contiguous': False, 'negative': False, 'conjugate': False, 'mkldnn': False}}
    state = {'parameters': {'p': [1., 2.]}, 'gradients': {'p': [0.5, 0.25]},
             'gradient_metadata': {'p': metadata}, 'slots': {'p': {}}}
    binding = {'parameter_key': 'p', 'parameter_id': 'parameter:p', 'call_index': 0, 'operand_path': ['args', 0]}
    node = {'call': 0, 'node': 0, 'invocation': 0}
    return {'completed': True, 'api_executed': True, 'config': {'api': 'torch.sin', 'requires_backward': True},
        'target_calls': [{'index': 0, 'metadata': [dict(metadata, requires_grad=True)]}],
        'backward_invocations': [{'index': 0, 'completed': True}],
        'target_backward_nodes': [node], 'target_operand_gradients': [dict(node, path=['args', 0], slot=0)],
        'gradient_evidence': {'version': 'gradient-evidence-1', 'events': [dict(node, status='complete',
            inputs=[{'path': ['args', 0], 'gradient': [0.5, 0.25]}])]},
        'target_parameter_bindings': [binding],
        'optimizer_source_certificate': 'explicit_optimizer_gradient_update_v1',
        'checkpoints': [{'name': 'final', 'state_facts': {}, 'completed_backward_invocations': [0],
                         'evidence_offsets': {'calls': 1, 'nodes': 1, 'operands': 1}}],
        'optimizer_evidence': {'version': 'optimizer-update-1', 'dropped_events': 0, 'limitations': [],
            'accumulations': [{'index': 0, 'parameter_key': 'p', 'parameter_id': 'parameter:p',
                              'invocation': 0, 'started_none': True, 'gradient': [0.5, 0.25]}],
            'registry': {'optimizer_id': 'optimizer:0', 'native_type': 'torch.optim.SGD',
                         'parameters': [{'parameter_key': 'p', 'parameter_id': 'parameter:p',
                                         'group_index': 0, 'group_position': 0}], 'groups': []},
            'steps': [{'index': 0, 'optimizer_id': 'optimizer:0', 'completed': True,
                       'pre_checkpoint': 'optimizer:0/step:0/pre_update',
                       'post_checkpoint': 'optimizer:0/step:0/post_update',
                       'domain': {'status': 'verified'}, 'pre': deepcopy(state), 'post': deepcopy(state),
                       'gradient_provenance': [dict(binding, node=0, invocation=0, status='verified',
                            post_accumulate_event=0, unchanged_to_pre_step=True)],
                       'mutation': {'status': 'verified', 'parameter_key': 'p', 'action': 'strided', 'logical_equal': True}}]}}


def gradient_predicate(**changes):
    return StatePredicate(**dict({'dimension': 'gradient_representation', 'field': 'contiguous',
        'expected': False, 'checkpoint': 'optimizer:0/step:0/pre_update', 'parameter_key': 'p',
        'call_index': 0, 'operand_path': ['args', 0], 'relation': 'target_gradient_update'}, **changes))


class OptimizerInspectionTests(unittest.TestCase):
    def test_gradient_and_completed_update_are_a_scoped_conjunction(self):
        record = optimizer_record()
        gradient = gradient_predicate()
        complete = gradient_predicate(dimension='optimizer', field='completed', expected=True,
                                      checkpoint='optimizer:0/step:0/post_update')
        result = inspect_state(record, [asdict(gradient), asdict(complete)])
        self.assertTrue(result['valid'])
        self.assertTrue(all(p['attained'] for p in result['predicates']))
        coverage = result['optimizer_coverage']
        self.assertEqual(coverage['version'], 'optimizer-chain-1')
        self.assertEqual(coverage['target_update_count'], 1)
        self.assertEqual(len(coverage['eligible_chains']), 1)
        self.assertEqual(coverage['observed_units'], 1)
        self.assertNotIn('parameter_key', coverage['signatures'][0])

    def test_optimizer_units_do_not_rewrite_operand_v2(self):
        record = optimizer_record()
        before = semantic_coverage(record, valid=True)
        changed = deepcopy(record)
        changed.pop('optimizer_evidence')
        record['checkpoints'][0]['state_facts']['optimizer.target_update_count'] = 1
        extra = deepcopy(record['checkpoints'][0])
        extra['name'] = 'optimizer:0/step:0/pre_update'
        record['checkpoints'].insert(0, extra)
        self.assertEqual(before, semantic_coverage(record, valid=True))
        self.assertEqual(before, semantic_coverage(changed, valid=True))
        self.assertEqual(before['version'], 'state-semantic-2')
        unknown = inspect_optimizer(record, valid=False)
        self.assertEqual(unknown['signatures'], [])
        self.assertEqual(unknown['eligible_chains'], [])
        self.assertEqual(len(unknown['witnesses']), 1)

    def test_real_optimizer_observer_does_not_change_old_operand_units(self):
        from test_optimizer_runtime import run
        record = run(action='strided')
        self.assertTrue(record['inspection']['optimizer_coverage']['eligible'])
        self.assertEqual(len(record['inspection']['optimizer_coverage']['eligible_chains']), 1)
        self.assertTrue(any(c['name'].startswith('optimizer:') for c in record['checkpoints']))
        stripped = deepcopy(record)
        stripped['checkpoints'] = [c for c in stripped['checkpoints'] if not c['name'].startswith('optimizer:')]
        for checkpoint in stripped['checkpoints']:
            checkpoint['state_facts'] = {k: v for k, v in checkpoint['state_facts'].items()
                                         if not k.startswith('optimizer.')}
        self.assertEqual(semantic_coverage(record, valid=True), semantic_coverage(stripped, valid=True))

    def test_native_tuple_and_json_list_paths_have_identical_chain_evidence(self):
        import json
        record = optimizer_record()
        record['target_calls'][0]['metadata'][0]['path'] = ('args', 0)
        record['target_parameter_bindings'][0]['operand_path'] = ('args', 0)
        record['target_operand_gradients'][0]['path'] = ('args', 0)
        record['gradient_evidence']['events'][0]['inputs'][0]['path'] = ('args', 0)
        record['optimizer_evidence']['steps'][0]['gradient_provenance'][0]['operand_path'] = ('args', 0)
        actual = inspect_optimizer(record, valid=True)
        restored = inspect_optimizer(json.loads(json.dumps(record)), valid=True)
        self.assertTrue(actual['eligible'])
        self.assertEqual(actual['signatures'], restored['signatures'])
        self.assertEqual(actual['target_update_count'], restored['target_update_count'])
        self.assertTrue(optimizer_fact(record, gradient_predicate()).established)
        for bad_path in (None, [], (), ['args', True], 'args.0'):
            with self.subTest(path=bad_path):
                changed = deepcopy(record)
                changed['target_calls'][0]['metadata'][0]['path'] = bad_path
                self.assertFalse(inspect_optimizer(changed, valid=True)['eligible'])
                changed = deepcopy(record)
                changed['optimizer_evidence']['steps'][0]['gradient_provenance'][0]['operand_path'] = bad_path
                self.assertFalse(inspect_optimizer(changed, valid=True)['eligible'])

    def test_post_values_need_not_change_to_certify_step_execution(self):
        # A missing update can be an output defect; do not invalidate it here.
        record = optimizer_record()
        self.assertEqual(record['optimizer_evidence']['steps'][0]['pre'], record['optimizer_evidence']['steps'][0]['post'])
        self.assertEqual(inspect_optimizer(record, valid=True)['target_update_count'], 1)

    def test_incomplete_wrong_stage_and_cross_backward_chains_are_unknown(self):
        changes = [
            lambda r: r['backward_invocations'][0].update(completed=False),
            lambda r: r['optimizer_evidence']['steps'][0].update(completed=False),
            lambda r: r['optimizer_evidence']['steps'][0].update(index=1),
            lambda r: r['optimizer_evidence']['steps'][0].update(pre_checkpoint='final'),
            lambda r: r['optimizer_evidence']['steps'][0]['gradient_provenance'][0].update(invocation=1),
            lambda r: r['optimizer_evidence']['steps'][0]['gradient_provenance'][0].update(call_index=1),
            lambda r: r.update(target_operand_gradients=[]),
            lambda r: r.update(target_backward_nodes=[]),
            lambda r: r['optimizer_evidence'].update(dropped_events=1),
            lambda r: r['optimizer_evidence']['accumulations'][0].update(started_none=False),
            lambda r: r['optimizer_evidence']['steps'][0]['gradient_provenance'][0].update(unchanged_to_pre_step=False),
            lambda r: r['gradient_evidence']['events'][0].update(status='unknown'),
        ]
        for change in changes:
            record = optimizer_record(); change(record)
            with self.subTest(change=change):
                coverage = inspect_optimizer(record, valid=True)
                self.assertFalse(coverage['eligible'])
                self.assertFalse(coverage['count_verified'])
                self.assertFalse(optimizer_fact(record, gradient_predicate()).established)

    def test_parameter_alias_and_missing_or_wrong_metadata_fail_closed(self):
        changes = [
            lambda r: r['target_parameter_bindings'][0].update(parameter_key='other'),
            lambda r: r['target_parameter_bindings'].append(deepcopy(r['target_parameter_bindings'][0])),
            lambda r: r['optimizer_evidence']['registry']['parameters'].append(deepcopy(r['optimizer_evidence']['registry']['parameters'][0])),
            lambda r: r['optimizer_evidence']['steps'][0]['pre']['gradient_metadata'].update(p=None),
            lambda r: r['optimizer_evidence']['steps'][0]['pre']['gradient_metadata']['p'].update(strides=[]),
            lambda r: r['optimizer_evidence']['steps'][0]['pre']['gradient_metadata']['p'].update(device='cuda:0'),
            lambda r: r['optimizer_evidence']['steps'][0]['gradient_provenance'].append(deepcopy(r['optimizer_evidence']['steps'][0]['gradient_provenance'][0])),
        ]
        for change in changes:
            record = optimizer_record(); change(record)
            with self.subTest(change=change):
                self.assertFalse(inspect_optimizer(record, valid=True)['eligible'])
        self.assertFalse(optimizer_fact(optimizer_record(), gradient_predicate(parameter_key='other')).established)

    def test_source_domain_and_value_preservation_certificates_are_independent(self):
        changes = [lambda r: r.update(optimizer_source_certificate=None),
                   lambda r: r['optimizer_evidence']['steps'][0]['domain'].update(status='unknown'),
                   lambda r: r['optimizer_evidence']['steps'][0]['mutation'].update(logical_equal=False),
                   lambda r: r['optimizer_evidence']['steps'][0]['mutation'].update(action='injected_values')]
        for change in changes:
            record = optimizer_record(); change(record)
            with self.subTest(change=change):
                self.assertFalse(inspect_optimizer(record, valid=True)['eligible'])

    def test_absent_observer_and_forged_mark_cannot_supply_update_count(self):
        predicate = StatePredicate('optimizer', 'target_update_count', 1, 'final')
        record = optimizer_record(); record.pop('optimizer_evidence')
        record['checkpoints'][0]['state_facts']['optimizer.target_update_count'] = 1
        self.assertFalse(optimizer_fact(record, predicate).established)
        self.assertFalse(inspect_state(record, [predicate])['target_state_reached'])
        record['optimizer_evidence'] = {'version': 'optimizer-update-1', 'steps': [], 'limitations': [], 'dropped_events': 0}
        fact = optimizer_fact(record, predicate)
        self.assertTrue(fact.established)
        self.assertEqual(fact.value, 0)
        self.assertFalse(predicate.holds(fact))


if __name__ == '__main__':
    unittest.main()
