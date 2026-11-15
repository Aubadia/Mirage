"""Semantic relation controls use independent labels and preserve unknown cases."""
import unittest
import torch
from mirage.backends.pytorch import PyTorchBackend
from mirage.contracts.capabilities import Evidence, EvidenceStatus
from mirage.rules.catalog import RELATIONS, select, evaluate, representation_pair
from mirage.runtime.stateful import execute
from mirage.runtime.probes import Recorder
from test_stateful import SOURCE, CONFIG


class RelationCatalogTests(unittest.TestCase):
    def setUp(self): self.backend = PyTorchBackend()

    def evidence(self, name):
        relation = RELATIONS[name]
        return {key: Evidence(EvidenceStatus.VERIFIED, 'authored independent fixture', True, 'comparison')
                for key in relation.preconditions + relation.input_relations}

    def test_unknown_and_llm_certificates_do_not_enable_relations(self):
        self.assertEqual(select({}), [])
        facts = self.evidence('representation')
        facts['deterministic'] = Evidence(EvidenceStatus.VERIFIED, 'generated claim', True, 'comparison', 'llm')
        out = evaluate('representation', facts, torch.ones(2), torch.ones(2), self.backend.compare)
        self.assertEqual(out['verdict'], 'inapplicable')
        self.assertIn('deterministic', out['missing'])

    def test_same_oracle_across_value_apis(self):
        # These are synthetic corruption tests, not claims about PyTorch defects.
        facts = self.evidence('representation')
        x = torch.tensor([.25, .75], dtype=torch.float64)
        for fn in (torch.sin, torch.square, torch.sum):
            expected = fn(x)
            out = evaluate('representation', facts, expected, expected.clone(), self.backend.compare)
            self.assertEqual(out['verdict'], 'pass')
            bad = evaluate('representation', facts, expected, -expected, self.backend.compare)
            self.assertEqual(bad['verdict'], 'suspected_failure')
            self.assertFalse(bad['confirmed_defect'])

    def test_tiny_scale_is_not_erased_by_absolute_tolerance(self):
        a = torch.tensor([1e-30], dtype=torch.float64)
        self.assertFalse(self.backend.compare(a, -a, 'dtype_scaled_v1')['equal'])
        self.assertTrue(self.backend.compare(a, a*(1+1e-15), 'dtype_scaled_v1')['equal'])
        self.assertFalse(self.backend.compare(a, torch.zeros_like(a), 'dtype_scaled_v1')['equal'])

    def test_extreme_finite_nonfinite_and_unknown_policy(self):
        a = torch.tensor([1e300], dtype=torch.float64)
        self.assertFalse(self.backend.compare(a, -a, 'dtype_scaled_v1')['equal'])
        self.assertIsNone(self.backend.compare(torch.tensor(float('inf')), torch.tensor(float('inf')), 'exact')['equal'])
        self.assertIsNone(self.backend.compare(a, a, 'missing_policy')['equal'])

    def test_unsupported_relations_and_failed_certificates(self):
        out = evaluate('unknown', {}, 1, 2, self.backend.compare)
        self.assertIsNone(out['verdict'])
        for name in RELATIONS:
            out = evaluate(name, {}, 1, 2, self.backend.compare)
            self.assertEqual(out['verdict'], 'inapplicable')
        facts = self.evidence('differentiation')
        facts['reference_stable'] = Evidence(EvidenceStatus.REJECTED, 'control disagreement')
        self.assertEqual(evaluate('differentiation', facts, 1, 2, self.backend.compare)['verdict'], 'inapplicable')

    def test_state_attainment_does_not_establish_relation(self):
        source = SOURCE.replace("    y = probe.call", "    def unused_helper(x): return x\n    y = probe.call")
        left = execute(source, CONFIG, Recorder())
        right = execute(source, dict(CONFIG, mutation={'path': ['args',0], 'action':'strided'}), Recorder())
        self.assertTrue(left['inspection']['valid'])
        self.assertEqual(representation_pair(left, right, self.backend)['verdict'], 'inapplicable')

    def test_nondiscriminating_values_are_not_rejected(self):
        facts = self.evidence('representation')
        x = torch.eye(2)
        self.assertEqual(evaluate('representation', facts, x, x, self.backend.compare)['verdict'], 'pass')


    def test_real_representation_pair_uses_reviewed_certificate(self):
        for api in ('torch.sin', 'torch.square'):
            source = SOURCE.replace('torch.sin', api)
            config = dict(CONFIG, api=api)
            left = execute(source, config, Recorder())
            right = execute(source, dict(config, mutation={'path': ['args', 0], 'action': 'strided'}), Recorder())
            result = representation_pair(left, right, self.backend)
            self.assertEqual(result['verdict'], 'pass', result)
            right['target_calls'][0]['output'] *= -1
            result = representation_pair(left, right, self.backend)
            self.assertEqual(result['verdict'], 'suspected_failure', result)

    def test_metadata_inspection_and_unknown_effects_are_not_certified(self):
        self.assertIsNotNone(self.backend.program_certificate(SOURCE))
        for operation in ('x.stride()', 'x.is_contiguous()', 'x.grad_fn', 'x.data'):
            source = SOURCE.replace("    y = probe.call", f"    metadata = {operation}\n    y = probe.call")
            self.assertIsNone(self.backend.program_certificate(source))


    def test_ill_conditioned_reference_is_not_a_target_failure(self):
        source = SOURCE.replace('torch.sin', 'torch.linalg.slogdet')
        # The supported family guard is independently exercised with recorded inputs.
        left = execute(SOURCE, CONFIG, Recorder())
        right = execute(SOURCE, dict(CONFIG, mutation={'path':['args',0], 'action':'strided'}), Recorder())
        for record in (left,right):
            record['config']['api'] = 'torch.linalg.slogdet'
            record['target_calls'][0]['inputs'] = {'args': (torch.diag(torch.tensor([1., 1e-12])),), 'kwargs': {}}
        result = representation_pair(left, right, self.backend)
        self.assertEqual(result['verdict'], 'inapplicable')
        self.assertIn('numerical_domain', result['missing'])

    def test_independent_broadcast_audit_ignores_production_labels(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1]/'benchmarks/pytorch_correctness/evaluate.py'
        spec = importlib.util.spec_from_file_location('stateful_audit', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        a = torch.tensor([[1.], [2.]], dtype=torch.float64)
        b = torch.tensor([3., 4., 5.], dtype=torch.float64)
        expected = torch.tensor([[4.,5.,6.],[5.,6.,7.]], dtype=torch.float64)
        record = {'config':{'api':'torch.add'}, 'verdict':'suspected_failure',
                  'target_calls':[{'inputs':{'args':(a,b),'kwargs':{}}, 'output':expected}]}
        self.assertEqual(module.audit_stateful_arithmetic(record)['label'], 'negative')
        record['verdict']='pass';record['target_calls'][0]['output']=-expected
        self.assertEqual(module.audit_stateful_arithmetic(record)['label'], 'positive')
        record['config']['api']='torch.unknown'
        self.assertEqual(module.audit_stateful_arithmetic(record)['label'], 'unknown')


    def test_audit_rejects_integer_overflow_domain(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1]/'benchmarks/pytorch_correctness/evaluate.py'
        spec = importlib.util.spec_from_file_location('stateful_audit_integer', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        x = torch.tensor([127], dtype=torch.int8)
        record = {'config': {'api':'torch.add'}, 'target_calls': [
            {'inputs': {'args': (x,x), 'kwargs': {}}, 'output': torch.tensor([-2], dtype=torch.int8)}]}
        self.assertEqual(module.audit_stateful_arithmetic(record)['label'], 'unknown')


    def test_pure_bounded_helpers_have_no_overloaded_effect_certificate(self):
        literal = SOURCE.replace('    x = torch.tensor', '    size = min(4, max(1, 2))\n    x = torch.tensor', 1)
        self.assertIsNotNone(self.backend.program_certificate(literal))
        overloaded = SOURCE.replace('    y = probe.call', '    size = min(x)\n    y = probe.call')
        self.assertIsNone(self.backend.program_certificate(overloaded))


    def test_context_observer_checks_original_scopes_without_inserting_them(self):
        from mirage.rules.catalog import context_restoration
        source = """
import torch

def build(device, seed):
    return {'inputs':{}, 'parameters':{}}

def run(ctx, probe):
    mode = probe.call(torch.no_grad)
    with mode:
        probe.mark('inside')
    return {}
"""
        config = {'api':'torch.no_grad', 'device':'cpu', 'requires_backward':False}
        result = execute(source, config, Recorder())
        self.assertEqual(len(result['context_events']), 1)
        self.assertEqual(context_restoration(result, self.backend)['verdict'], 'pass')
        no_scope = source.replace("    with mode:\n        probe.mark('inside')", "    probe.mark('outside')")
        result = execute(no_scope, config, Recorder())
        self.assertEqual(len(result['context_events']), 0)
        self.assertEqual(context_restoration(result, self.backend)['verdict'], 'inapplicable')

    def test_recursive_context_regression_is_source_owned_and_cleanup_is_separate(self):
        from mirage.rules.catalog import context_restoration
        source = """
import torch

def build(device, seed):
    return {'inputs':{}, 'parameters':{}}

def run(ctx, probe):
    mode = probe.call(torch.autocast, 'cpu', dtype=torch.bfloat16)
    @mode
    def nested(depth):
        if depth:
            nested(depth - 1)
    nested(1)
    return {}
"""
        config = {'api':'torch.autocast', 'device':'cpu', 'requires_backward':False}
        initial = torch.is_autocast_enabled('cpu')
        result = execute(source, config, Recorder())
        self.assertEqual(len(result['context_events']), 2)
        # This integration assertion follows the observed invariant; its label is
        # not fixed to a particular installed version having the historical bug.
        violated = any(e['before'] != e['after'] for e in result['context_events'])
        self.assertEqual(context_restoration(result, self.backend)['verdict'], 'suspected_failure' if violated else 'pass')
        self.assertEqual(torch.is_autocast_enabled('cpu'), initial)


    def test_local_input_dictionary_construction_is_not_an_external_effect(self):
        source = SOURCE.replace('    return {"inputs"', '    payload = {}\n    payload["x"] = x\n    return {"inputs"', 1)
        self.assertIsNotNone(self.backend.program_certificate(source))
        changed = SOURCE.replace('    y = probe.call', '    ctx["external"] = x\n    y = probe.call')
        self.assertIsNone(self.backend.program_certificate(changed))


    def test_reduction_admission_has_runtime_positive_and_negative_controls(self):
        for api in ('torch.sum', 'torch.mean', 'torch.std'):
            with self.subTest(api=api):
                source = SOURCE.replace('torch.sin', api)
                config = dict(CONFIG, api=api)
                left = execute(source, config, Recorder())
                right = execute(source, dict(config, mutation={'path':['args',0], 'action':'strided'}), Recorder())
                result = representation_pair(left, right, self.backend)
                self.assertEqual(result['verdict'], 'pass', result)
                right['target_calls'][0]['output'] *= -1
                self.assertEqual(representation_pair(left, right, self.backend)['verdict'], 'suspected_failure')
        for operation, values in (('sum', [1., -1.]), ('mean', [1., -1.]), ('std', [1., 1.])):
            call = {'inputs':{'args':(torch.tensor(values, dtype=torch.float64),), 'kwargs':{}}}
            self.assertFalse(self.backend.reduction_numerical_domain(operation, call))

    def test_reduction_guards_keep_width_dtype_dimension_and_effect_limits(self):
        def call(value, **kwargs):
            return {'inputs':{'args':(value,), 'kwargs':kwargs}}
        self.assertTrue(self.backend.reduction_numerical_domain('sum', call(torch.ones(2, 3), dim=1)))
        self.assertFalse(self.backend.reduction_numerical_domain('sum', call(torch.ones(32))))
        self.assertFalse(self.backend.reduction_numerical_domain('sum', call(torch.ones(2), dim=4)))
        self.assertFalse(self.backend.reduction_numerical_domain('sum', call(torch.ones(2), dtype=torch.float64)))
        self.assertFalse(self.backend.reduction_numerical_domain('mean', call(torch.tensor([float('inf')]))))
        self.assertFalse(self.backend.reduction_numerical_domain('sum', call(torch.tensor([1e-44, 1e-44]))))
        self.assertFalse(self.backend.reduction_numerical_domain('sum', call(torch.full((2,), torch.finfo(torch.float32).max))))
        self.assertFalse(self.backend.reduction_numerical_domain('std', call(torch.tensor([1., 2.]), correction=2)))
        effectful = SOURCE.replace('torch.sin', 'torch.sum').replace('    y.sum().backward()', '    y.sum().backward()\n    x.zero_()')
        self.assertIsNone(self.backend.program_certificate(effectful))

    def test_rectangular_multiplication_does_not_require_square_operands(self):
        source = """
import torch

def build(device, seed):
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64, device=device, requires_grad=True)
    b = torch.tensor([[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]], dtype=torch.float64, device=device)
    return {'inputs':{'x':x, 'b':b}, 'parameters':{'x':x}}

def run(ctx, probe):
    x = ctx['inputs']['x']
    y = probe.call(torch.matmul, x, ctx['inputs']['b'])
    y.sum().backward()
    return {'output':y, 'gradient':x.grad}
"""
        config = dict(CONFIG, api='torch.matmul')
        left = execute(source, config, Recorder())
        right = execute(source, dict(config, mutation={'path':['args',0], 'action':'offset'}), Recorder())
        result = representation_pair(left, right, self.backend)
        self.assertEqual(result['verdict'], 'pass', result)
        right['target_calls'][0]['output'] *= -1
        self.assertEqual(representation_pair(left, right, self.backend)['verdict'], 'suspected_failure')
        cancelled = {'inputs':{'args':(torch.ones(1, 2), torch.tensor([[1., 2.], [-1., -2.]])), 'kwargs':{}}}
        self.assertFalse(self.backend.multiplication_numerical_domain(cancelled))
        wide = {'inputs':{'args':(torch.ones(2, 32), torch.ones(32, 3)), 'kwargs':{}}}
        self.assertFalse(self.backend.multiplication_numerical_domain(wide))

    def test_solve_conditioning_applies_to_coefficient_matrix_not_rhs(self):
        left = execute(SOURCE, CONFIG, Recorder())
        right = execute(SOURCE, dict(CONFIG, mutation={'path':['args',0], 'action':'strided'}), Recorder())
        matrix = torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64)
        lu, pivots = torch.linalg.lu_factor(matrix)
        # Singular RHS columns are legal; the coefficient matrix is invertible.
        rhs = torch.ones(2, 2, dtype=torch.float64)
        for record in (left, right):
            record['config']['api'] = 'torch.linalg.lu_solve'
            record['target_calls'][0]['inputs'] = {'args':(lu, pivots, rhs), 'kwargs':{}}
        self.assertTrue(self.backend.representation_evidence(left, right)['numerical_domain'].established)
        bad_lu, bad_pivots = torch.linalg.lu_factor(torch.diag(torch.tensor([1., 1e-12], dtype=torch.float64)))
        for record in (left, right):
            record['target_calls'][0]['inputs'] = {'args':(bad_lu, bad_pivots, rhs), 'kwargs':{}}
        self.assertFalse(self.backend.representation_evidence(left, right)['numerical_domain'].established)
