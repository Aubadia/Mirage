import json
from pathlib import Path
import tempfile
import unittest

from mirage.runtime.input_constraints import (evaluate_shape_guard, make_tensor, plan_guard_probes,
    refine_domains, refine_trace_constraints, trace_input_observations)


class GuidedInputTests(unittest.TestCase):
    def test_bounded_interpreter_handles_dimensions_while_rejecting_unknown_expressions(self):
        schema = 'native.overload'
        guard = dict(overload=schema, argument_map={'a':'self', 'b':'other'})
        operands = {'self':(3, 2), 'other':(2, 1)}
        def evaluate(expr, overload=schema):
            return evaluate_shape_guard(dict(guard, expression=expr), overload, operands)['value']
        self.assertTrue(evaluate('len(a) == 3 and len(a.shape) == 2 and ((a.size(-1) == b.shape[0]) & (a.ndim == b.dim()))'))
        self.assertFalse(evaluate('not (a.size() in [(3,2), (2,1)]) or a.shape[1] == 7'))
        self.assertIsNone(evaluate('a.shape[-3] > 0'))
        self.assertIsNone(evaluate('unbound.size(0) > 0'))
        self.assertIsNone(evaluate('a.add_(1)'))
        self.assertIsNone(evaluate('a.ndim == 2', 'unmatched'))

    def test_guard_guided_probes_include_inside_and_complement_without_hiding_ordinary(self):
        schema = 'test.bound'
        guard = dict(expression='a.size(-1) == b.size(0)', overload=schema, argument_map={'a':'a','b':'b'})
        plans = plan_guard_probes([guard], schema, ['a','b'])
        self.assertEqual(plans[0][0], {'a': (2,2), 'b': (2,2)})
        self.assertEqual({evaluate_shape_guard(guard, schema, dims)['value'] for dims, _ in plans}, {True,False})
        singleton = dict(guard, expression='a.size(-1) == 1')
        self.assertTrue(any(dims['a'][-1:] == (1,) for dims, _ in plan_guard_probes([singleton],schema,['a','b'])))

    def test_complementary_observations_revise_one_error_hypothesis_and_keep_backend_scope(self):
        def obs(accepted, shape, i, device='cpu', error=None, schema='overload-A'):
            rec = dict(returned=accepted, operands=[dict(path=['args',0], shape=shape,
                       dtype='float32', device=device)], attempt=i, evidence='fixture-' + str(i), schema=schema)
            if error:
                rec.update(exception='RuntimeError', diagnostic=error)
            return rec
        msg = 'expected a square matrix'
        rows = [obs(False,[2,3],0,error=msg), obs(True,[2,2],1),
                obs(False,[3,2],2,error='unsupported dtype'),
                obs(True,[3,2],3,device='cuda'), obs(True,[3,2],4,schema='overload-B')]
        select = lambda rows: next(r for r in refine_trace_constraints({'api':'example'},rows)
                                   if r['proposal']=='last_two_extents_equal')
        self.assertEqual(select(rows)['status'], 'outside_changed_error_only')
        rows += [obs(True,[2,3],5), obs(True,[3,2],6)]
        refined = select(rows)
        self.assertEqual(refined['status'],'overrestricted')
        self.assertEqual(refined['latest_counterexample']['attempt'],6)
        self.assertEqual(refined['counts']['outside_accepted'],2)

    def test_online_native_guard_binds_named_and_positional_operands_only_in_its_overload(self):
        guard = dict(overload='T', expression='x.shape == y.shape', source_sha256='test', line=7,
                     diagnostic='input sizes should match', argument_map={'x':'self','y':'other'})
        spec = dict(api='torch.fn', constraints=dict(validation_guards=[guard], bindings=[
            dict(schema='T', operands=[dict(name='self'),dict(name='other')])]))
        operands = [dict(path=['args',0],shape=[1,3],device='cpu'),dict(path=['kwargs','other'],shape=[2,3],device='cpu')]
        rows = [dict(schema=None, returned=True, operands=operands,attempt=3)]
        self.assertEqual(refine_trace_constraints(spec,rows),[])
        rows.append(dict(schema='T', returned=True, operands=operands, attempt=4))
        rule, = refine_trace_constraints(spec,rows)
        self.assertEqual(rule['status'],'overrestricted')
        self.assertEqual(rule['latest_counterexample']['attempt'],4)
        self.assertEqual(rule['counts'],{'outside_accepted':1})
    def test_constructor_preserves_geometry_and_distinguishes_coordinates(self):
        import torch
        a = make_tensor((2, 4, 4), dtype=torch.float64, kind="invertible", requires_grad=True)
        self.assertTrue(a.is_leaf and a.requires_grad)
        self.assertFalse(torch.equal(a, a.mT))
        self.assertTrue(bool((torch.linalg.svdvals(a) > 0.5).all()))
        i = make_tensor((4, 4), kind="index", index_bound=3)
        self.assertTrue(bool((i >= 0).all() and (i < 3).all()))
        self.assertFalse(torch.equal(make_tensor((3,), sample=0), make_tensor((3,), sample=1)))
        self.assertEqual(make_tensor((0, 2)).shape, (0, 2))
        with self.assertRaises(ValueError):
            make_tensor((64, 65))

    def test_version_guard_and_complement_refute_an_overrestriction(self):
        import torch
        from mirage.runtime.api_metadata import specification
        spec = specification("torch.sin", probe_device="cpu")
        constraint = spec["constraints"]
        self.assertEqual(constraint["framework_version"], str(torch.__version__))
        self.assertEqual(constraint["probe_status"], "probed")
        from mirage.runtime.input_constraints import choose_input_recipe, make_operand
        recipe = choose_input_recipe(constraint, 1)
        value = make_operand(recipe["operands"][0], device="cpu", recipe=recipe)
        self.assertEqual(list(value.shape), recipe["shape"])
        self.assertEqual(str(value.dtype), recipe["dtype"])
        with self.assertRaises(ValueError):
            make_operand("missing", recipe=recipe, device="cpu")
        for name in ("only_square_matrices", "only_float32"):
            rule = next(r for r in constraint["refinements"] if r["proposal"] == name)
            self.assertEqual(rule["status"], "overrestricted")
            self.assertTrue(rule["counterexamples"])
        obs = [dict(shape=[2, 2], dtype="torch.float32", returned=True)]
        self.assertTrue(all(r["status"] == "unresolved" for r in refine_domains(obs)))

    def test_trace_only_attributes_exceptions_at_target_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            enter = dict(event="api_enter", call=0, metadata=[dict(path=["args", 0], shape=[3, 2],
                                                                    dtype="torch.float32", device="cpu")])
            fail = dict(event="target_exception", call=0, exception="ValueError", diagnostic="observed")
            path.write_text("\n".join(map(json.dumps, [enter, fail, dict(event="api_exit", call=1)])) + "\n")
            self.assertEqual(len(trace_input_observations(path)), 1)
            self.assertFalse(trace_input_observations(path)[0]["returned"])
