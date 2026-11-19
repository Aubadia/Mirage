import unittest
from types import SimpleNamespace

from mirage.generator.state_plan import select_state_goal, state_family_catalog
from mirage.inspector.stateful import training_state_evidence


SPEC = {'api': 'torch.sin', 'metadata': {'documentation': 'A tensor function'},
        'environment': {}, 'capabilities': {}}


class StateGuidanceTests(unittest.TestCase):
    def test_family_catalog_and_environment_admission(self):
        goals = state_family_catalog(SPEC)
        self.assertEqual(len({goal['family'] for goal in goals}), 16)
        self.assertEqual(len({goal['id'] for goal in goals}), len(goals))
        availability = {g['id']: g['availability'] for g in goals}
        self.assertEqual(availability['synchronized_gradient'], 'requires_multi_rank_environment')
        self.assertEqual(availability['captured_target_replay'], 'requires_cuda')
        self.assertEqual(availability['produced_scheduler_step'], 'requires_steppable_target_product')

    def test_verified_prefix_is_reused_and_layout_children_add_no_family_attempt(self):
        entry = dict(id='seed', active=True, configuration={'api': SPEC['api']}, cost={'seconds': 1})
        corpus = SimpleNamespace(entries=[entry])
        row = dict(api=SPEC['api'], configuration={'state_goal': {'id': 'target_backward', 'family': 'autograd'}},
                   inspection={'legal_source': True, 'valid': True, 'training_state_units': ['target_backward']},
                   corpus_admission={'seed_id': 'seed'})
        goal, prefix, decision = select_state_goal(SPEC, [row], corpus, 1)
        self.assertEqual(goal['id'], 'leaf_accumulation')
        self.assertEqual(prefix['id'], 'seed')
        child = dict(row, proposal_origin='framework_representation', representation_candidate_attained=True,
                     followup_inspection=row['inspection'])
        _, _, updated = select_state_goal(SPEC, [row, child], corpus, 2)
        self.assertEqual(decision['family_attempts'], updated['family_attempts'])
        self.assertEqual(decision['reached'], updated['reached'])
        row['inspection']['valid'] = False
        _, prefix, _ = select_state_goal(SPEC, [row], corpus, 3)
        self.assertIsNone(prefix)

    def test_wrong_call_or_unfinished_backward_never_certifies_state_use(self):
        rec = dict(config={'device': 'cpu'}, target_calls=[{'index': 0}],
                   target_backward_nodes=[dict(call=0, node=0, invocation=0)],
                   target_operand_gradients=[dict(call=0, path=['args', 0], invocation=0)],
                   state_operations=[dict(event='used_restored_state', call=1, path=['args', 0],
                                          invocation=0, parameter='w')])
        self.assertNotIn('restored_update', training_state_evidence(rec, {0}, set(), {}))
        rec['state_operations'][0]['call'] = 0
        self.assertNotIn('restored_update', training_state_evidence(rec, set(), set(), {}))
        self.assertIn('restored_update', training_state_evidence(rec, {0}, set(), {}))

    def test_only_identity_linked_capture_scope_and_update_events_count(self):
        record = dict(config={'device': 'cuda'}, target_calls=[dict(index=0, execution_mode='captured',
            completed_replays=1, entry_context={'inference_enabled': True, 'cuda_enabled': True},
            target_rng_consumed=True, registered_resources=[dict(path=['args', 0],
                module_buffer={'module':'m', 'name':'buffer', 'training':False})])],
            context_events=[dict(factory_call=0, balanced=True, autocast_scope=True, decorated_entry=True,
                                 reentrant_same_instance=True, instance_group=0)],
            product_events=[dict(call=0,event='persistent_step_effect',product_kind='optimizer',persistent_state=True),
                            dict(call=0,event='persistent_step_effect',product_kind='scheduler',learning_rate_changed=True)])
        measured = training_state_evidence(record, set(), {0}, {})
        self.assertEqual(set(measured), {'captured_target_replay', 'scope_reentry', 'produced_scheduler_step',
            'produced_optimizer_state', 'target_rng_consumed', 'module_buffer_operand', 'inference_target', 'autocast_target'})
        record['context_events'][0]['factory_call'] = 3
        for event in record['product_events']:
            event['call'] = 3
        record['target_calls'][0]['completed_replays'] = 0
        observed = training_state_evidence(record, set(), {0}, {})
        self.assertFalse({'captured_target_replay','scope_reentry','produced_scheduler_step','produced_optimizer_state'} & observed.keys())

    def test_compound_goal_requires_observed_prerequisites(self):
        corpus = SimpleNamespace(entries=[])
        rows = []
        for goal in state_family_catalog(SPEC):
            if goal['availability'] == 'candidate' and len(goal['prerequisites']) < 2:
                rows.append(dict(api=SPEC['api'], configuration={'state_goal': goal},
                                 inspection={'legal_source':True, 'training_state_units':[]}))
        for _ in range(5):
            goal, _, decision = select_state_goal(SPEC, rows, corpus, len(rows))
            self.assertLess(len(goal['prerequisites']), 2)
            rows.append(dict(api=SPEC['api'], configuration={'state_goal':goal}, inspection={'legal_source':True}))

    def test_target_scoped_torch_observers(self):
        from mirage.runtime.probes import Recorder
        from mirage.runtime.stateful import execute
        build = """def build(device, seed):
    x = torch.tensor([1., 2.], device=device, requires_grad=True)
    y = torch.tensor([3., 4.], device=device)
    return {'inputs': {'x':x, 'y':y}, 'parameters': {'w':x},
            'optimizer':torch.optim.SGD([x], lr=0.1, momentum=0.5),
            'scaler':torch.amp.GradScaler(device)}
"""
        cases = {
            'target_recomputed': "def fn(v): return probe.call(torch.sin, v)\n    out=probe.recompute(fn, x)\n    out.sum().backward()",
            'saved_target_unpacked': "with probe.saved_on_cpu(): out=probe.call(torch.sin, x)\n    out.sum().backward()",
            'restored_update': "x.sum().backward()\n    ctx['optimizer'].step()\n    out=probe.call(torch.sin, x)\n    probe.restore_and_step()",
            'overflow_step_skipped': "out=probe.call(torch.sin, x)\n    probe.scaled_overflow_step()",
            'same_graph_twice': "out=probe.call(torch.sin, x)\n    out.sum().backward(retain_graph=True)\n    out.sum().backward()",
            'distinct_batches': "probe.call(torch.sin, x)\n    out=probe.call(torch.sin, ctx['inputs']['y'])",
        }
        for goal in state_family_catalog(SPEC):
            if goal['id'] not in cases:
                continue
            for positive in (True, False):
                with self.subTest(goal=goal['id'], positive=positive):
                    body = cases[goal['id']] if positive else "out=probe.call(torch.sin, x)"
                    source = build + "\ndef run(ctx, probe):\n    x=ctx['inputs']['x']\n    " + body + "\n    return {'result':out}\n"
                    config = dict(api=SPEC['api'], device='cpu', state_goal=goal, check_claim=goal['check_claim'],
                                  requires_backward=goal['requires_backward'])
                    inspection = execute(source, config, Recorder())['inspection']
                    self.assertTrue(inspection['legal_source'])
                    self.assertEqual(inspection['state_goal']['attained'], positive, inspection['state_goal'])
                    self.assertEqual(goal['id'] in inspection['training_state_units'], positive)
