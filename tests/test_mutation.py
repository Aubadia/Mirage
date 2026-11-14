"""Static edit boundaries and runtime validation are independent obligations."""
import ast
from copy import deepcopy
import unittest

from mirage.generator.mutation import plan_mutation, apply_mutation
from mirage.generator.program import ContractError, stateful_request
from test_stateful import SOURCE, CONFIG

TAIL = '''y.sum().backward(retain_graph=True)
y.sum().backward()
return {'output': y, 'gradient': x.grad}
'''


class MutationContractTests(unittest.TestCase):
    def test_only_declared_region_changes(self):
        contract = plan_mutation(SOURCE, 'parent', 0)
        child, edit = apply_mutation(SOURCE, TAIL, contract)
        before, after = ast.parse(SOURCE), ast.parse(child)
        self.assertEqual(ast.dump(before.body[1]), ast.dump(after.body[1]))
        self.assertEqual(ast.dump(before.body[2].body[1]), ast.dump(after.body[2].body[1]))
        self.assertTrue(edit['outside_region_unchanged'])
        self.assertEqual(edit['runtime_obligations'], 'not established by static editing')

    def test_full_program_cannot_escape_the_region(self):
        contract = plan_mutation(SOURCE, 'parent', 0)
        with self.assertRaises(ContractError):
            apply_mutation(SOURCE, SOURCE, contract)
        with self.assertRaises(ContractError):
            apply_mutation(SOURCE, 'import torch\n' + TAIL, contract)

    def test_wrong_parent_and_protected_target_deletion_are_rejected(self):
        contract = plan_mutation(SOURCE, 'parent', 0)
        with self.assertRaisesRegex(ContractError, 'parent'):
            apply_mutation(SOURCE + '\n', TAIL, contract)
        contract['editable_region']['start'] = 1
        with self.assertRaisesRegex(ContractError, 'protected'):
            apply_mutation(SOURCE, TAIL, contract)

    def test_noop_and_edit_budgets(self):
        contract = plan_mutation(SOURCE, 'parent', 0)
        with self.assertRaisesRegex(ContractError, 'no semantic'):
            apply_mutation(SOURCE, '\n'.join(contract['original_statements']), contract)
        for snippet in ('pass\n' * 9, '#' * 4097, 'value = [' + ','.join('1' for _ in range(130)) + ']'):
            with self.assertRaisesRegex(ContractError, 'budget'):
                apply_mutation(SOURCE, snippet, contract)

    def test_setup_cannot_return_before_or_add_an_earlier_target(self):
        contract = plan_mutation(SOURCE, 'parent', 0, kind='operand_setup')
        for snippet in ("return {'output': x}", 'z = probe.call(torch.sin, x)', "raise RuntimeError('skip')"):
            with self.assertRaisesRegex(ContractError, 'protected target'):
                apply_mutation(SOURCE, snippet, contract)
        child, _ = apply_mutation(SOURCE, 'x = x.clone()', contract)
        self.assertIn('x = x.clone()', child)

    def test_initialization_preserves_run_and_declares_allowed_difference(self):
        contract = plan_mutation(SOURCE, 'parent', 0, kind='initialization')
        child, _ = apply_mutation(SOURCE,
            'x = torch.tensor([0., 0.75], dtype=torch.float64, device=device, requires_grad=True)', contract)
        self.assertEqual(ast.dump(ast.parse(SOURCE).body[2]), ast.dump(ast.parse(child).body[2]))
        self.assertIn('initialization', contract['allowed_semantic_changes'][0])
        self.assertIn('lineage only', contract['parent_child_relation'])

    def test_unsupported_nested_target_uses_no_local_contract(self):
        nested = SOURCE.replace('    y = probe.call(torch.sin, x)', '    with torch.enable_grad():\n        y = probe.call(torch.sin, x)')
        self.assertIsNone(plan_mutation(nested, 'parent', 0))

    def test_prompt_preserves_existing_provider_source_transport(self):
        contract = plan_mutation(SOURCE, 'parent', 0)
        request = stateful_request({}, CONFIG, canonical=SOURCE, mutation=contract)
        self.assertEqual(request.context['contract'], 'training-program')
        self.assertIn('replacement Python statements', request.prompt)
        self.assertNotIn('Rewrite it as', request.prompt)
        self.assertNotIn('a complete Python source string', request.prompt)
        self.assertEqual(request.context['mutation_contract'], contract)
        contract['budget']['characters'] = 1
        self.assertEqual(request.context['mutation_contract']['budget']['characters'], 4096)

    def test_assembled_local_child_reaches_new_runtime_state(self):
        from mirage.runtime.stateful import execute
        from mirage.runtime.probes import Recorder
        contract = plan_mutation(SOURCE, 'parent', 0)
        child, _ = apply_mutation(SOURCE, TAIL, contract)
        result = execute(child, dict(CONFIG, predicates=[{'dimension':'graph_lifecycle',
                         'field':'target_backward_node_count', 'expected':2, 'checkpoint':'final'}]), Recorder())
        self.assertTrue(result['inspection']['valid'])
        self.assertEqual(len(result['target_backward_nodes']), 2)
        self.assertEqual(result['correctness'], 'unassessed')
