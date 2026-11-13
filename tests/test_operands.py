"""General mutation contracts across operands, APIs and differentiation paths."""
from collections import namedtuple
from dataclasses import replace
import unittest
import torch
from mirage.backends.pytorch import PyTorchBackend
from mirage.contracts.capabilities import MutationObligations, EvidenceStatus
from mirage.runtime.operands import discover, candidates, apply, compositions


class OperandTests(unittest.TestCase):
    def setUp(self):
        self.backend = PyTorchBackend()
        self.read = MutationObligations(writable=False)

    def plan(self, tree, action, path=None, obligations=None):
        return next(p for p in candidates(tree, self.backend, obligations or self.read)
                    if p.action == action and (path is None or p.path == path))

    def test_nested_namedtuple_and_repeated_identity(self):
        Pair = namedtuple("Pair", "a b")
        x = torch.tensor([1., 2.])
        tree = {"args": (Pair(x, x),), "kwargs": {"out": [x]}}
        self.assertEqual(len(discover(tree, self.backend)), 3)
        changed, fact = apply(tree, self.plan(tree, "strided"), self.backend, self.read)
        self.assertTrue(fact.established)
        self.assertIsInstance(changed['args'][0], Pair)
        self.assertIs(changed['args'][0].a, changed['kwargs']['out'][0])
        self.assertEqual(len(candidates(tree, self.backend, self.read)), 8)

    def test_aliases_and_identity_are_not_silently_broken(self):
        x = torch.arange(4.)
        plans = candidates([x, x[1:]], self.backend, self.read)
        self.assertTrue(all(not p.eligibility.established for p in plans))
        identity = replace(self.read, preserve_identity=True)
        self.assertTrue(all(not p.eligibility.established for p in candidates([x], self.backend, identity)))

    def test_empty_allocations_do_not_alias(self):
        ds = discover([torch.zeros(0), torch.zeros(0)], self.backend)
        self.assertNotEqual(ds[0][0].storage_group, ds[1][0].storage_group)

    def test_unknown_effects_and_unsafe_parameter_rejected(self):
        self.assertFalse(self.plan([torch.ones(2)], "clone", obligations=MutationObligations()).eligibility.established)
        self.assertFalse(self.plan([torch.nn.Parameter(torch.ones(2))], "clone").eligibility.established)

    def test_dtype_and_shape_eligibility(self):
        for dtype in (torch.float32, torch.int64, torch.complex64):
            tree = [torch.ones(2, dtype=dtype)]
            self.assertEqual(self.plan(tree, "conjugate").eligibility.established, dtype == torch.complex64)
            self.assertEqual(self.plan(tree, "negative").eligibility.established, dtype == torch.float32)
            self.assertFalse(self.plan(tree, "singleton_stride").eligibility.established)
        self.assertFalse(self.plan([torch.tensor([1., 2.])], "expanded").eligibility.established)

    def test_gradients_remain_connected_across_apis(self):
        # Independent expected derivatives exercise graph preservation, not only grad_fn metadata.
        for operation, derivative in ((torch.square, lambda x: 2*x),
                                      (torch.sin, torch.cos)):
            for action in ("clone", "strided", "offset", "negative", "singleton_stride"):
                x = torch.tensor([[0.25, 0.75]], dtype=torch.float64, requires_grad=True)
                tree = {"kwargs": {"input": x}}
                changed, fact = apply(tree, self.plan(tree, action), self.backend, self.read)
                y = changed['kwargs']['input']
                gradient, = torch.autograd.grad(operation(y).sum(), x)
                torch.testing.assert_close(gradient, derivative(x), rtol=1e-12, atol=0)
                self.assertTrue(fact.established)

    def test_complex_conjugate_preserves_logical_value(self):
        x = torch.tensor([1+2j, 3-4j], requires_grad=True)
        changed, _ = apply([x], self.plan([x], "conjugate"), self.backend, self.read)
        self.assertTrue(changed[0].is_conj())
        g, = torch.autograd.grad(changed[0].abs().square().sum(), x)
        torch.testing.assert_close(g, 2*x)

    def test_expansion_preserves_values_but_not_independent_gradients(self):
        x = torch.ones(2, 3)
        changed, _ = apply([x], self.plan([x], "expanded"), self.backend, self.read)
        self.assertTrue(torch.equal(changed[0], x))
        self.assertIn(0, changed[0].stride())
        x.requires_grad_()
        self.assertFalse(self.plan([x], "expanded").eligibility.established)

    def test_revalidation_rejects_changed_values(self):
        x = torch.ones(2)
        plan = self.plan([x], "expanded")
        x[1] = 2
        with self.assertRaises(ValueError): apply([x], plan, self.backend, self.read)

    def test_bounded_multi_operand_plans(self):
        plans = candidates([torch.ones(2), torch.ones(3)], self.backend, self.read)
        groups = list(compositions(plans))
        self.assertTrue(any(len(g) == 2 for g in groups))
        self.assertTrue(all(len({p.path for p in g}) == len(g) <= 2 for g in groups))
        with self.assertRaises(ValueError): list(compositions(plans, 3))

    def test_csr_values_and_keyword_writable_output(self):
        out = torch.sparse_csr_tensor(torch.tensor([0, 1, 2]), torch.tensor([0, 1]),
                                      torch.tensor([2., 3.]), size=(2, 2))
        tree = {"kwargs": {"out": out}}
        write = MutationObligations(writable=True)
        changed, fact = apply(tree, self.plan(tree, "values_strided", obligations=write), self.backend, write)
        y = changed['kwargs']['out']
        self.assertEqual(y.values().stride(), (2,))
        self.assertTrue(torch.equal(y.to_dense(), out.to_dense()))
        self.assertTrue(torch.equal(y.crow_indices(), out.crow_indices()))
        self.assertTrue(fact.established)
        x = torch.ones(2, requires_grad=True)
        self.assertFalse(self.plan([x], "clone", obligations=write).eligibility.established)

    def test_mkldnn_logical_adapter_and_write_rejection(self):
        x = torch.tensor([[1., 2.], [3., 4.]]).to_mkldnn()
        changed, fact = apply([x], self.plan([x], "logical_materialize"), self.backend, self.read)
        self.assertFalse(changed[0].is_mkldnn)
        self.assertTrue(torch.equal(x.to_dense(), changed[0]))
        self.assertTrue(fact.established)
        self.assertFalse(self.plan([x], "logical_materialize", obligations=MutationObligations(writable=True)).eligibility.established)

    def test_cycles_and_unsupported_layouts(self):
        tree = []; tree.append(tree)
        with self.assertRaises(ValueError): discover(tree, self.backend)
        x = torch.eye(2).to_sparse()
        self.assertEqual(candidates([x], self.backend, self.read), [])


    def test_csr_values_alias_and_overlapping_writes_rejected(self):
        values = torch.tensor([2., 3.])
        sparse = torch.sparse_csr_tensor(torch.tensor([0, 1, 2]), torch.tensor([0, 1]), values,
                                         size=(2, 2), check_invariants=True)
        self.assertTrue(all(not p.eligibility.established for p in candidates([sparse, values], self.backend, self.read)))
        x = torch.ones(1).expand(3)
        write = MutationObligations(writable=True)
        self.assertFalse(self.plan([x], "clone", obligations=write).eligibility.established)


    def test_installed_schemas_supply_operand_write_roles(self):
        tree = {'args': (torch.ones(2), torch.ones(2)), 'kwargs': {}}
        roles = self.backend.operand_obligations(torch.Tensor.copy_, tree)
        self.assertTrue(roles[('args',0)].writable)
        self.assertFalse(roles[('args',1)].writable)
        plans = candidates(tree, self.backend, roles)
        self.assertTrue(any(p.eligibility.established for p in plans))
        unknown = self.backend.operand_obligations(lambda x:x, {'args':(torch.ones(2),),'kwargs':{}})
        self.assertIsNone(unknown[('args',0)].writable)


    def test_native_binding_diagnoses_missing_receiver_without_api_recipes(self):
        x=torch.ones(2)
        self.assertEqual(self.backend.bind_native(torch.Tensor.copy_, (x,), {}).status, EvidenceStatus.REJECTED)
        self.assertTrue(self.backend.bind_native(torch.Tensor.copy_, (x,x), {}).established)
        self.assertEqual(self.backend.bind_native(torch.sin, (), {'input':x}).status, EvidenceStatus.UNKNOWN)
