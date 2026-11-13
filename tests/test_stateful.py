"""Stateful source owns every training operation; observations cannot replace it."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import torch
from mirage.generator.program import analyze_stateful, stateful_request, ContractError, EDGE_CASE_PROMPT
from mirage.runtime.stateful import execute
from mirage.runtime.probes import Recorder
from mirage.backends.pytorch import PyTorchBackend

SOURCE = '''
import torch

def build(device, seed):
    x = torch.tensor([0.25, 0.75], dtype=torch.float64, device=device, requires_grad=True)
    return {"inputs": {"x": x}, "parameters": {"x": x}}

def run(ctx, probe):
    x = ctx['inputs']['x']
    y = probe.call(torch.sin, x)
    y.sum().backward()
    probe.mark('after_backward', {'gradient': x.grad})
    return {'output': y, 'gradient': x.grad}
'''
CONFIG = {'api': 'torch.sin', 'device': 'cpu', 'seed': 0, 'writable': False}


class StatefulProgramTests(unittest.TestCase):
    def test_input_leaf_gradients_and_portable_extra_state_are_saved(self):
        source=SOURCE.replace('"parameters": {"x": x}', '"parameters": {}, "buffer": torch.tensor([3.])')
        result=execute(source,CONFIG,Recorder())
        self.assertTrue(result['final_training_state']['input_gradients'])
        self.assertEqual(result['initial']['extra_state']['buffer'].tolist(),[3.])
        with self.assertRaisesRegex(ContractError,'initial values'):
            execute(source.replace('[3.]','[4.]'),CONFIG,Recorder(),result)

    def test_optimizer_events_and_observation_copies_have_distinct_provenance(self):
        source=SOURCE.replace('"parameters": {"x": x}', '"parameters": {"x": x}, "optimizer": torch.optim.SGD([x], lr=0.01)')
        source=source.replace("    probe.mark('after_backward'", "    ctx['optimizer'].step()\n    ctx['optimizer'].zero_grad()\n    unused = y.detach().clone()\n    probe.mark('after_backward'")
        record=execute(source,CONFIG,Recorder())
        self.assertIn('optimizer_step',record['transition_signature'])
        self.assertIn('zero_grad',record['transition_signature'])
        self.assertIn('detach',record['observed_actions'])
        self.assertNotIn('detach',record['transition_signature'])

    def test_delimiter_normalization_does_not_rewrite_training_statements(self):
        from mirage.generator.repair import normalize_source_delimiter
        source,repair=normalize_source_delimiter(SOURCE.rstrip()+'\n}\n')
        self.assertEqual(source.strip(),SOURCE.strip())
        self.assertEqual(repair['training_statements_added'],0)
        with self.assertRaises(SyntaxError): normalize_source_delimiter(SOURCE+'\n}\n}\n')
        with self.assertRaises(SyntaxError): normalize_source_delimiter('def run(:\n    pass\n}')
        with self.assertRaisesRegex(ContractError,'omit unused'):
            execute(SOURCE.replace('"parameters": {"x": x}', '"parameters": {"x": x}, "optimizer": {}'),CONFIG,Recorder())

    def test_unused_registered_optimizer_does_not_remove_purity_certificate(self):
        source=SOURCE.replace('"parameters": {"x": x}', '"parameters": {"x": x}, "optimizer": torch.optim.SGD([x], lr=0.01)')
        self.assertIsNotNone(PyTorchBackend().program_certificate(source))
        updated=source.replace("    y.sum().backward()", "    y.sum().backward()\n    ctx['optimizer'].step()")
        self.assertIsNone(PyTorchBackend().program_certificate(updated))

    def test_direct_same_instance_context_reentry_is_not_a_certified_defect(self):
        from mirage.rules.catalog import context_restoration
        source="""import torch

def build(device,seed): return {'inputs':{},'parameters':{}}
def run(ctx,probe):
    scope=probe.call(torch.no_grad)
    with scope:
        with scope: pass
    return {}
"""
        record=execute(source,dict(api='torch.no_grad',device='cpu',seed=0,requires_backward=False),Recorder())
        self.assertTrue(any(e['reentrant_same_instance'] for e in record['context_events']))
        self.assertEqual(context_restoration(record,PyTorchBackend())['verdict'],'inapplicable')

    def test_same_verdict_is_insufficient_replay_evidence(self):
        from copy import deepcopy
        from mirage.runtime.worker import stateful_replay_evidence
        original=execute(SOURCE,CONFIG,Recorder());changed=deepcopy(original)
        changed['observations']['output'][0]+=1
        self.assertEqual(original['verdict'],changed['verdict'])
        self.assertFalse(PyTorchBackend().compare(stateful_replay_evidence(original),
                         stateful_replay_evidence(changed),'exact')['equal'])

    def test_literal_prompt_and_raw_documentation(self):
        raw = 'Example: torch.randn(2)'
        metadata = {'api': 'torch.sin', 'documentation': raw}
        base = stateful_request(metadata, CONFIG, edges=False)
        edge = stateful_request(metadata, CONFIG)
        self.assertEqual(edge.prompt, base.prompt + '\n\n' + EDGE_CASE_PROMPT)
        self.assertEqual(edge.context['api_context']['documentation'], raw)
        self.assertEqual(metadata, {'api': 'torch.sin', 'documentation': raw})
        self.assertIn('not instructions', edge.context['api_context']['documentation_annotation'])
        self.assertNotIn('issue', edge.prompt)

    def test_alias_module_and_seed_are_not_automatic_rejections(self):
        source = SOURCE.replace('import torch', 'import torch\nfrom torch import nn')
        source = source.replace('    x = torch.tensor', '    class M(nn.Module):\n        def __init__(self):\n            super().__init__()\n    torch.manual_seed(seed)\n    x = torch.tensor', 1)
        graph = analyze_stateful(source)
        self.assertTrue(graph['seed_calls'])
        self.assertEqual(graph['determinism'], 'runtime_evidence_required')
        with self.assertRaises(ContractError): analyze_stateful(source.replace('torch.manual_seed(seed)', 'torch.randn(2)'))

    def test_signatures_json_and_local_functions(self):
        decoded = json.loads(json.dumps({'source': SOURCE}))
        self.assertEqual(analyze_stateful(decoded['source'])['version'], 'stateful-1')
        with self.assertRaises(ContractError): analyze_stateful(SOURCE.replace('def run(ctx, probe)', 'def forward(ctx)'))
        with self.assertRaises(SyntaxError): analyze_stateful(SOURCE + '\n}')
        source = SOURCE.replace('    return {"inputs"', '    def helper(x): return x\n    return {"checkpoint_data": {"forward": helper}, "inputs"', 1)
        with self.assertRaisesRegex(ValueError, 'Unsupported checkpoint data'):
            execute(source, CONFIG, Recorder())
        with self.assertRaises(ContractError): execute(SOURCE.replace('return {"inputs": {"x": x}, "parameters": {"x": x}}', 'return (x,)'), CONFIG, Recorder())

    def test_training_operations_are_source_owned(self):
        full = execute(SOURCE, CONFIG, Recorder())
        no_backward = execute(SOURCE.replace('    y.sum().backward()\n', ''), CONFIG, Recorder())
        self.assertTrue(full['backward_executed'])
        self.assertFalse(no_backward['backward_executed'])
        self.assertIsNone(no_backward['observations']['gradient'])
        self.assertEqual(full['correctness'], 'unassessed')
        self.assertIsNone(full['verdict'])

    def test_two_histories_reach_accumulated_gradients(self):
        retained = SOURCE.replace('    y.sum().backward()', '    loss = y.sum()\n    loss.backward(retain_graph=True)\n    loss.backward()')
        recomputed = SOURCE.replace('    y.sum().backward()', "    y.sum().backward()\n    z = probe.call(torch.sin, x)\n    z.sum().backward()")
        a, b = [execute(s, CONFIG, Recorder()) for s in (retained, recomputed)]
        self.assertEqual(len(a['target_calls']), 1)
        self.assertEqual(len(b['target_calls']), 2)
        self.assertEqual(len(a['target_gradients']), 2)
        torch.testing.assert_close(a['observations']['gradient'], b['observations']['gradient'])
        torch.testing.assert_close(a['observations']['gradient'], 2*torch.cos(torch.tensor([.25, .75], dtype=torch.float64)))

    def test_bound_tensor_method_and_missing_other(self):
        source = SOURCE.replace('probe.call(torch.sin, x)', 'probe.call(x.add, 2)')
        result = execute(source, dict(CONFIG, api='Tensor.add'), Recorder())
        self.assertTrue(result['api_executed'])
        with self.assertRaises((TypeError, ContractError)):
            execute(source.replace('probe.call(x.add, 2)', 'probe.call(x.add)'), dict(CONFIG, api='Tensor.add'), Recorder())

    def test_actual_operand_mutation_and_wrong_api(self):
        config = dict(CONFIG, mutation={'path': ['args', 0], 'action': 'strided'})
        result = execute(SOURCE, config, Recorder())
        self.assertEqual(result['target_calls'][0]['metadata'][0]['strides'], (2,))
        self.assertTrue(result['mutations'])
        with self.assertRaises(ContractError): execute(SOURCE.replace('probe.call(torch.sin', 'probe.call(torch.cos'), CONFIG, Recorder())

    def test_observable_snapshot_roundtrip(self):
        result = execute(SOURCE, CONFIG, Recorder())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.pt'
            torch.save(result, path)
            restored = torch.load(path, weights_only=True)
            torch.testing.assert_close(restored['observations']['gradient'], result['observations']['gradient'])

    def test_unknown_exception_not_repaired_to_pass(self):
        from mirage.generator.provider import Response
        from mirage.generator.repair import generate, Diagnostic, DiagnosticKind
        class Provider:
            calls = 0
            def generate(self, request, deadline=None):
                self.calls += 1
                return Response(json.dumps({'source': SOURCE}))
        provider = Provider()
        result = generate(provider, stateful_request({'documentation': ''}, CONFIG),
                          lambda content: Diagnostic(DiagnosticKind.UNKNOWN_EXCEPTION, 'unexplained target error'))
        self.assertEqual(provider.calls, 1)
        self.assertEqual(result.status, 'unknown_exception')


class StatefulInspectionTests(unittest.TestCase):
    def test_generated_values_cannot_claim_state_attainment(self):
        from mirage.inspector.stateful import inspect_state
        result = execute(SOURCE, CONFIG, Recorder())
        predicate = {'dimension': 'gradient', 'field': 'backward_count', 'expected': 2, 'checkpoint': 'final'}
        result['checkpoints'][-1]['values']['gradient.backward_count'] = 2
        check = inspect_state(result, [predicate])
        self.assertFalse(check['target_state_reached'])
        self.assertEqual(check['predicates'][0]['observed'], 1)
        self.assertEqual(check['correctness'], 'unassessed')

    def test_actual_small_input_and_underflow_have_different_coverage(self):
        backend = PyTorchBackend()
        tiny = torch.tensor([1e-30], dtype=torch.float32)
        rounded = torch.tensor([1e-50], dtype=torch.float32)
        self.assertTrue(backend.input_properties(tiny)['small_finite'])
        self.assertFalse(backend.input_properties(rounded)['small_finite'])
        self.assertTrue(backend.input_properties(rounded)['zero'])
        overflow = torch.tensor([1e40], dtype=torch.float32)
        self.assertFalse(backend.input_properties(overflow)['finite'])
        self.assertFalse(backend.input_properties(overflow)['large_finite'])

    def test_target_coverage_does_not_come_from_requests(self):
        result = execute(SOURCE, dict(CONFIG, input_categories=['small_finite']), Recorder())
        self.assertEqual(result['input_coverage']['requested'], ['small_finite'])
        self.assertNotIn('small_finite', result['input_coverage']['target_observed'])
        self.assertEqual(result['input_coverage']['relation_eligible'], {})

    def test_transition_counts_and_ambiguous_checkpoints(self):
        from mirage.inspector.stateful import inspect_state
        source = SOURCE.replace('    y.sum().backward()', '    y.sum().backward(retain_graph=True)\n    y.sum().backward()')
        result = execute(source, CONFIG, Recorder())
        predicate = {'dimension': 'gradient', 'field': 'backward_count', 'expected': 2, 'checkpoint': 'final'}
        self.assertTrue(inspect_state(result, [predicate])['target_state_reached'])
        result['checkpoints'].append(result['checkpoints'][-1])
        self.assertFalse(inspect_state(result, [predicate])['target_state_reached'])

    def test_missing_compile_evidence_is_unknown(self):
        from mirage.inspector.stateful import inspect_state
        result = execute(SOURCE, dict(CONFIG, backend='inductor'), Recorder())
        check = inspect_state(result, [{'dimension': 'execution_mode', 'field': 'compiled_execution',
                                        'expected': True, 'checkpoint': 'final'}])
        self.assertFalse(check['target_state_reached'])
        self.assertEqual(check['predicates'][0]['evidence_status'], 'unknown')

    def test_alias_observables_and_exclusive_failure_funnel(self):
        from mirage.inspector.stateful import failure_record
        source = SOURCE.replace("return {'output': y, 'gradient': x.grad}", "return {'base': y, 'view': y.view(-1)}")
        result = execute(source, CONFIG, Recorder())
        metadata = result['checkpoints'][-1]['observable_metadata']
        self.assertEqual(metadata[0]['storage_group'], metadata[1]['storage_group'])
        failure = failure_record('static_contract', 'alias spelling', ['RNG initialization'])
        self.assertEqual(failure['first_blocking_stage'], 'static_contract')
        self.assertEqual(len(failure['secondary_diagnostics']), 1)


    def test_instrumentation_is_excluded_from_training_signature(self):
        result = execute(SOURCE, CONFIG, Recorder())
        self.assertEqual(result['transition_signature'], ['target_enter', 'target_return', 'backward'])
        observed = SOURCE.replace("    probe.mark('after_backward', {'gradient': x.grad})", "    probe.mark('after_backward', {'gradient': x.grad})\n    probe.mark('extra', x)")
        self.assertEqual(execute(observed, CONFIG, Recorder())['transition_signature'], result['transition_signature'])


    def test_execution_record_does_not_alias_caller_configuration(self):
        config = dict(CONFIG, input_categories=['ordinary'])
        result = execute(SOURCE, config, Recorder())
        result['config']['api'] = 'torch.cos'
        result['config']['input_categories'].append('zero')
        self.assertEqual(config['api'], 'torch.sin')
        self.assertEqual(config['input_categories'], ['ordinary'])



class SourceFixtureProvider:
    """Only integration plumbing; never counted as model generation evidence."""
    def generate(self, request, deadline=None):
        from mirage.generator.provider import Response
        return Response(json.dumps({'source': SOURCE}), model='authored-fixture')


def stateful_fixture_provider():
    return SourceFixtureProvider()


class StatefulWorkflowTests(unittest.TestCase):
    def test_real_isolated_workflow_uses_new_contract_and_schema(self):
        from mirage.contracts.programs import RunOptions
        from mirage.workflow import run
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'PYTHONPATH': str(Path(__file__).parent)+os.pathsep+os.environ.get('PYTHONPATH','')}):
            options = RunOptions(str(Path(directory)/'run'), 'test_stateful:stateful_fixture_provider',
                                 cases=1, time_limit=120, worker_timeout=60, strategy='stateful', contract_version=4)
            result = run(options, ('torch.sin',))
            self.assertEqual(len(result['records']), 1, result)
            row = result['records'][0]
            self.assertEqual(row['status'], 'executed', row)
            self.assertEqual(row['verdict'], 'pass', row)
            self.assertTrue(row['api_executed'])
            self.assertTrue(row['backward_executed'])
            self.assertTrue((Path(row['directory'])/'execution/source.pt').exists())
            self.assertTrue((Path(row['directory'])/'execution/followup.pt').exists())
            request = json.loads((Path(row['directory'])/'request.json').read_text())
            self.assertEqual(request['context']['program_kind'], 'stateful-1')
            self.assertEqual(request['context']['contract'], 'training-program')
            from mirage.workflow import replay_stateful
            replay = replay_stateful(Path(row['directory'])/'stateful-manifest.json', timeout=60)
            self.assertEqual(replay['provider_calls'], 0)
            self.assertTrue(replay['matches_original_verdict'], replay)
            self.assertTrue(replay['replay_valid'], replay)




class CompilerEvidenceTests(unittest.TestCase):
    def test_failed_graph_or_unrelated_eager_backward_cannot_establish_compiled_target_state(self):
        goal=[dict(dimension='execution_mode',field='compiled_execution',expected=True,checkpoint='final')]
        failing=SOURCE.replace("    y = probe.call(torch.sin, x)", """    def backend(graph,examples):
        def fail(*values): raise RuntimeError('synthetic backend failure')
        return fail
    compiled = torch.compile(lambda value: torch.sin(value), backend=backend, fullgraph=True)
    try: compiled(x)
    except RuntimeError: pass
    y = probe.call(torch.sin, x)""")
        record=execute(failing,dict(CONFIG,predicates=goal),Recorder())
        self.assertFalse(record['compiled_graphs'])
        self.assertFalse(record['inspection']['valid'])
        detached=SOURCE.replace("    y = probe.call(torch.sin, x)", """    compiled = torch.compile(lambda value: torch.sin(value).detach()+value, backend='eager', fullgraph=True)
    compiled(x).sum().backward()
    y = probe.call(torch.sin, x)""")
        record=execute(detached,dict(CONFIG,predicates=goal),Recorder())
        self.assertTrue(record['backward_executed'])
        self.assertFalse(record['inspection']['required_backward_executed'])
        self.assertFalse(record['inspection']['valid'])

    def test_compiler_identity_observations_do_not_acquire_equivalence_certificate(self):
        from mirage.backends.pytorch import execution_source_certificate
        source=SOURCE.replace("    y = probe.call(torch.sin, x)", "    def operation(value): return torch.sin(value)\n    compiled = torch.compile(operation,backend='eager',fullgraph=True)\n    y = compiled(x)")
        self.assertTrue(execution_source_certificate(source))
        self.assertFalse(execution_source_certificate(source.replace("return {'output': y", "return {'identity':id(compiled), 'output': y")))
        self.assertFalse(execution_source_certificate(source.replace("return {'output': y", "return {'history':str(y.grad_fn), 'output': y")))

    def test_unknown_backend_cannot_certify_target_semantics(self):
        source=SOURCE.replace("    y = probe.call(torch.sin, x)", """    def backend(graph,examples):
        return lambda *values: (values[0]*0,)
    compiled = torch.compile(lambda value: torch.sin(value),backend=backend,fullgraph=True)
    y = compiled(x)""")
        record=execute(source,CONFIG,Recorder())
        self.assertTrue(record['compiled_graphs'])
        self.assertFalse(record['api_executed'])
        self.assertNotIn('execution_mode.compiled_execution',record['checkpoints'][-1]['state_facts'])

    def test_source_owned_compilation_has_executed_graph_and_backward_evidence(self):
        source = SOURCE.replace("    y = probe.call(torch.sin, x)", "    def operation(value): return torch.sin(value)\n    compiled = torch.compile(operation, backend='eager', fullgraph=True)\n    y = compiled(x)")
        result = execute(source, CONFIG, Recorder())
        self.assertTrue(result['api_executed'])
        self.assertTrue(result['backward_executed'])
        self.assertTrue(result['compiled_graphs'])
        self.assertTrue(result['target_calls'][0]['target_operands_observed'])
        self.assertTrue(result['checkpoints'][-1]['state_facts']['execution_mode.compiled_execution'])
        self.assertTrue(result['inspection']['valid'])
        self.assertNotIn('compiler_reference', result['config'])

    def test_missing_compile_call_cannot_be_supplied_by_configuration(self):
        result = execute(SOURCE, dict(CONFIG, backend='inductor', compiler_reference=True), Recorder())
        self.assertEqual(result['compiled_graphs'], [])

    def test_unrelated_compiled_output_does_not_prove_target_backward(self):
        source = SOURCE.replace("    y = probe.call(torch.sin, x)", "    def operation(value): return torch.sin(value).detach(), value.square()\n    compiled = torch.compile(operation, backend='eager', fullgraph=True)\n    target_output, y = compiled(x)")
        result = execute(source, CONFIG, Recorder())
        self.assertTrue(result['api_executed'])
        self.assertFalse(result['backward_executed'])
        self.assertFalse(result['inspection']['valid'])


    def test_compiled_reference_pair_preserves_source_owned_history(self):
        from mirage.rules.catalog import execution_mode_pair
        source = SOURCE.replace("    y = probe.call(torch.sin, x)", "    def operation(value): return torch.sin(value)\n    compiled = torch.compile(operation, backend='aot_eager', fullgraph=True)\n    y = compiled(x)")
        left = execute(source, dict(CONFIG, compiler_reference=True), Recorder())
        right = execute(source, CONFIG, Recorder())
        result = execution_mode_pair(left, right, PyTorchBackend())
        self.assertEqual(result['verdict'], 'pass', result)
        self.assertEqual(left['transition_signature'], right['transition_signature'])
        right['observations']['gradient'] *= -1
        self.assertEqual(execution_mode_pair(left, right, PyTorchBackend())['verdict'], 'suspected_failure')



class CapturedEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').environ.get('MIRAGE_TEST_CUDA') == '1', 'Slurm CUDA validation only')
    def test_factory_capture_and_authored_accumulation_have_runtime_evidence(self):
        from mirage.rules.catalog import captured_execution_pair
        source = """
import torch

def build(device, seed):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.,3.],device=device))
        def forward(self,x):
            return x*self.weight
    model = Model()
    return {'inputs':{'a':torch.tensor([1.,2.],device=device), 'b':torch.tensor([3.,5.],device=device)},
            'parameters':{'weight':model.weight}, 'modules':{'model':model}}

def run(ctx, probe):
    model = ctx['modules']['model']
    a,b = ctx['inputs']['a'],ctx['inputs']['b']
    captured = probe.call(torch.cuda.make_graphed_callables,model,(torch.ones_like(a),))
    model.weight.grad = None
    first = captured(a)
    first.sum().backward()
    second = captured(b)
    second.sum().backward()
    return {'output':second,'gradient':model.weight.grad}
"""
        config = {'api':'torch.cuda.make_graphed_callables','device':'cuda','requires_backward':True}
        actual = execute(source,config,Recorder())
        reference = execute(source,dict(config,capture_reference=True),Recorder())
        self.assertTrue(actual['api_executed'])
        self.assertFalse(reference['api_executed'])
        self.assertTrue(actual['capture_replays'])
        self.assertEqual(actual['checkpoints'][-1]['state_facts']['gradient.backward_count'],2)
        self.assertTrue(actual['backward_executed'])
        self.assertEqual(reference['reference_substitutions'],1)
        expected = torch.tensor([4.,7.])
        torch.testing.assert_close(reference['observations']['gradient'],expected)
        difference = not torch.equal(actual['observations']['gradient'],expected)
        verdict = captured_execution_pair(actual,reference,PyTorchBackend())
        self.assertEqual(verdict['verdict'],'suspected_failure' if difference else 'pass',verdict)



class SnapshotContractTests(unittest.TestCase):
    def test_reserved_checkpoints_are_diagnosed_before_execution(self):
        with self.assertRaisesRegex(ContractError,'reserved'):
            analyze_stateful(SOURCE.replace("probe.mark('after_backward'", "probe.mark('final'"))

    def test_inference_tensor_unknown_version_does_not_reject_legal_forward(self):
        source = """
import torch
def build(device,seed):
    with torch.inference_mode():
        x=torch.ones(2,device=device)
    return {'inputs':{'x':x},'parameters':{}}
def run(ctx,probe):
    return {'output':probe.call(torch.sin,ctx['inputs']['x'])}
"""
        record=execute(source,dict(CONFIG,requires_backward=False),Recorder())
        self.assertTrue(record['inspection']['valid'])
