import unittest
from unittest.mock import patch

from mirage.artifacts.report import check_completion
from mirage.generator.state_plan import state_family_catalog
from mirage.inspector.stateful import training_state_evidence
from mirage.runtime.probes import Recorder
from mirage.runtime.stateful import execute


SPEC = dict(api='unlabelled', capabilities={}, metadata={}, environment={})


def measured(api, build, body, goal, *, claim=None):
    cfg = next((g for g in state_family_catalog(SPEC) if g['id'] == goal),
               dict(id='bootstrap', family='bootstrap', check_claim='callable_use', requires_backward=False))
    source = 'import torch\n' + build + '\ndef run(ctx, probe):\n    ' + body.replace('\n', '\n    ') + '\n'
    return execute(source, dict(api=api, device='cpu', state_goal=cfg, check_claim=claim or cfg['check_claim'],
                                requires_backward=cfg['requires_backward'], predicates=[]), Recorder())


BUILD = """def build(device, seed):
    x = torch.tensor([1., 2.], device=device)
    return {'inputs': {'x': x}, 'parameters': {}}
"""


class EvidenceGuidanceTests(unittest.TestCase):
    def test_truncation_is_local_but_other_failures_remain_visible(self):
        length = dict(attempt=1, status='generation_truncated')
        valid = dict(attempt=2, source_completed=True, api_executed=True, retry_of_attempt=1)
        native = dict(records=[length, valid], stop_reason='llm_time_limit')
        verdict, details = check_completion(native, {}, 1)
        self.assertEqual(verdict[0], 'completed_with_generation_errors')
        self.assertEqual(details, dict(truncated_attempts=[1], retry_attempts=[2]))
        self.assertEqual(check_completion(native, dict(incomplete=True), 1)[0][0], 'partial')
        for fault in [dict(cancellation_uncertain=True), dict(process=dict(outcome='timeout')),
                      dict(status='provider_failed', reason='connection reset')]:
            with self.subTest(fault=fault):
                self.assertEqual(check_completion(dict(native, records=[length, valid, fault]), {}, 1)[0][0], 'failed')

    def test_product_protocol_controls(self):
        from contextlib import ContextDecorator
        from mirage.backends.pytorch import ContextObserver, probe_context_decorator
        from mirage.inspector.stateful import product_use_feedback
        class Scope:
            def __enter__(self): return self
            def __exit__(self, *exc): pass
        class Decorator(Scope, ContextDecorator): pass
        class Fresh(Decorator):
            def _recreate_cm(self): return type(self)()
        for product, expected in ((Decorator(),'same_instance'), (Fresh(),'different_instance'),
                                  (Scope(),'unsupported'), (lambda x:x,'unsupported')):
            self.assertEqual(probe_context_decorator(product), expected)
            if expected == 'unsupported':
                continue
            events = []
            with ContextObserver(type(product), events) as observer:
                observer.register_product(product)
                @product
                def recurse(n):
                    return recurse(n-1) if n else 1
                self.assertEqual(recurse(1), 1)
            group = observer.instances.index(product)
            for e in events:
                e['factory_call'] = 1 if e['instance_group'] == group else None
                e['wrapper_factory_call'] = 1 if e['wrapper_owner_group'] == group else None
            measured = product_use_feedback(dict(target_calls=[dict(index=1)], context_events=events))
            modes = {u['mode'] for u in measured['uses']}
            self.assertEqual('decorated_same_instance' in modes, expected == 'same_instance')
            self.assertEqual('decorated_different_instance' in modes, expected == 'different_instance')
            self.assertFalse(measured['oracle_gaps'])
        factory_spec = dict(SPEC, capabilities={'callable_product':True})
        goals = {g['id']:g['availability'] for g in state_family_catalog(factory_spec, observed=['callable'])}
        self.assertEqual(goals['target_backward'], 'candidate')
        self.assertNotEqual(goals['leaf_accumulation'], 'candidate')

    def test_workflow_retries_each_truncated_proposal_only_once(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from mirage.contracts.programs import RunOptions
        from mirage.generator.provider import IsolatedProvider, ProviderFailure
        from mirage.runtime.api_metadata import specification
        from mirage.workflow import run_stateful
        spec = specification('torch.sin')
        with TemporaryDirectory() as directory, patch.object(IsolatedProvider, 'generate',
                side_effect=ProviderFailure('generation_truncated','Completion ended with length')) as provider:
            options = RunOptions(str(Path(directory)/'run'), 'fixture', strategy='stateful',
                contract_version=4, cases=3, time_limit=90)
            report = run_stateful(options, ('torch.sin',), api_spec=spec, predicates=[], path_policy='mixed')
        rows = report['records']
        self.assertEqual(provider.call_count, 3)
        self.assertEqual([row.get('retry_of_attempt') for row in rows], [None, 0, None])
        self.assertEqual({row['status'] for row in rows}, {'generation_truncated'})

    def test_real_factory_without_protocol_text(self):
        for invoked in (False, True):
            body = "f=probe.call(torch.nn.ReLU)\n"
            body += "return {'y':probe.call(f, ctx['inputs']['x'])}" if invoked else "return {'x':ctx['inputs']['x']}"
            obs = measured('torch.nn.ReLU', BUILD, body, 'bootstrap')['inspection']
            self.assertIn('callable', obs['observed_products'])
            self.assertEqual(obs['valid'], invoked)
            if not invoked:
                from types import SimpleNamespace
                from mirage.generator.state_plan import select_state_goal
                row = dict(api=SPEC['api'], configuration={'check_claim':'forward'}, inspection=obs)
                goal, _, _ = select_state_goal(SPEC, [row], SimpleNamespace(entries=[]), 1)
                self.assertEqual(goal['check_claim'], 'callable_use')

    def test_context_without_decorator_and_opaque_alias(self):
        import torch
        def factory():
            return torch.no_grad()
        with patch.object(torch, 'mirage_test_context', factory, create=True):
            for nested in (False, True):
                body = "c=probe.call(torch.mirage_test_context)\nwith c:\n"
                body += "    with c:\n        x=ctx['inputs']['x']" if nested else "    x=ctx['inputs']['x']"
                record = measured('torch.mirage_test_context', BUILD, body+"\nreturn {'x':x}", 'scope_reentry')
                self.assertFalse(record['context_contract'])
                self.assertEqual(record['inspection']['valid'], nested)
                goals = state_family_catalog(SPEC, observed=record['observed_products'])
                self.assertEqual(next(g['availability'] for g in goals if g['id']=='scope_reentry'), 'candidate')

    def test_rng_with_explicit_generator(self):
        for stochastic in (False, True):
            api = 'torch.bernoulli' if stochastic else 'torch.add'
            body = ('g=torch.Generator().manual_seed(123)\ny=probe.call(torch.bernoulli, '
                    "ctx['inputs']['x'] * 0.25, generator=g)") if stochastic else (
                    "g=torch.Generator().manual_seed(123)\ny=probe.call(torch.add, ctx['inputs']['x'], 1)")
            record = measured(api, BUILD, body+"\nreturn {'y':y}", 'target_rng_consumed')
            self.assertEqual(record['inspection']['valid'], stochastic)
            if stochastic:
                self.assertIn(['kwargs', 'generator'], record['target_calls'][0]['rng_sources_advanced'])

    def test_all_catalog_units_require_linked_observations(self):
        import torch
        path = ['args', 0]
        calls = [dict(index=i, inputs={'args':[torch.tensor([i])]}, registered_resources=[
            dict(path=path, input_key=i, module_buffer={'module':'m','name':'b'})],
            entry_context=dict(cpu_enabled=True, inference_enabled=True), recompute_frame=(0,i),
            target_rng_consumed=True, backward_invocation=i) for i in range(2)]
        calls[0].update(execution_mode='compiled', evidence='executed FX graph')
        calls[1].update(execution_mode='captured', completed_replays=1)
        record = dict(config={'device':'cpu'}, target_calls=calls,
            target_backward_nodes=[dict(call=0, node=0, invocation=i) for i in range(2)],
            target_operand_gradients=[dict(call=i, invocation=i, path=path) for i in range(2)],
            accumulations=[dict(first=0, second=1, leaf_calls=[(0,path),(1,path)])],
            saved_tensor_events=[dict(call=0, invocation=0, source_device='cpu', stored_device='cpu', restored_device='cpu')],
            state_operations=[dict(event=e, call=0, parameter='p', invocation=0, path=path, world_size=2)
                              for e in ('synchronized_gradient','skipped_scaled_step','used_restored_state')],
            context_events=[dict(factory_call=0, balanced=True, reentrant_same_instance=True, instance_group=0)],
            product_events=[dict(call=0, event='persistent_step_effect', product_kind='optimizer', persistent_state=True),
                            dict(call=1, event='persistent_step_effect', product_kind='scheduler', learning_rate_changed=True)])
        optimizer = dict(witnesses=[dict(call_index=0, completed=True, post_checkpoint='post_update', parameter_key='p')])
        goals = {g['id'] for g in state_family_catalog(SPEC)}
        self.assertEqual(set(training_state_evidence(record, {0,1}, {1}, optimizer)), goals)
        negative = dict(record, target_calls=[dict(index=8), dict(index=9)])
        self.assertFalse(training_state_evidence(negative, {0,1}, {1}, optimizer))
        # Keeping both states on distinct calls must not certify the conjunction.
        calls[0]['entry_context']['cpu_enabled'] = False
        self.assertNotIn('autocast_compiled_target', training_state_evidence(record, {0,1}, {1}, optimizer))
        record['state_operations'][0]['world_size'] = 1
        self.assertNotIn('synchronized_gradient', training_state_evidence(record, {0,1}, {1}, optimizer))

def distributed_control():
    import torch.distributed as dist
    dist.init_process_group('gloo')
    try:
        build = '''def build(device, seed):
    p = torch.tensor([1., 2.], device=device, requires_grad=True)
    return {'inputs': {'x':p}, 'parameters': {'w':p}, 'optimizer':torch.optim.SGD([p], lr=0.1)}
'''
        body = "y=probe.call(torch.sin, ctx['inputs']['x'])\nprobe.synchronize_gradient()\nreturn {'y':y}"
        obs = measured('torch.sin', build, body, 'synchronized_gradient')['inspection']
        assert obs['valid'], obs['state_goal']
        print('verified synchronized_gradient on rank', dist.get_rank(), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    distributed_control()
