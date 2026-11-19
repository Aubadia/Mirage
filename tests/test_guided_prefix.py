from copy import deepcopy
import unittest

from mirage.artifacts.corpus import guided_prefix_witness
from mirage.generator.program import ContractError
from mirage.generator.repair import supports_guided_prefix, validate_guided_prefix


PARENT = """def build(device, seed):
    w = torch.tensor([1., 2.], device=device, requires_grad=True)
    return {'inputs': {}, 'parameters': {'w': w}}

def run(ctx, probe):
    w = ctx['parameters']['w']
    out = probe.call(torch.sin, w)
    out.sum().backward()
    return {'result': out}
"""
CHILD = PARENT.replace("    return {'result': out}", """    second = probe.call(torch.sin, w)
    second.sum().backward()
    return {'result': out}""")
GOAL = {'id': 'leaf_accumulation'}


class GuidedPrefixTests(unittest.TestCase):
    def test_append_preserves_whole_program_and_early_bindings(self):
        self.assertTrue(supports_guided_prefix(PARENT, GOAL))
        validate_guided_prefix(PARENT, "# explanatory comment\n" + CHILD)
        for candidate in (PARENT,
                          CHILD.replace("[1., 2.]", "[1., 3.]"),
                          CHILD.replace("out.sum()", "out.mean()", 1),
                          CHILD.replace("out.sum().backward()", "out.sum().backward(retain_graph=True)"),
                          CHILD.replace("return {'result': out}", "return {'result': second}"),
                          CHILD.replace("    second = ", "    torch = None\n    second = ")):
            with self.subTest(candidate=candidate), self.assertRaises(ContractError):
                validate_guided_prefix(PARENT, candidate)

    def test_scope_rewrites_and_deferred_target_cannot_be_prefix(self):
        for goal in ('autocast_target', 'same_graph_twice', 'saved_target_unpacked', 'target_recomputed'):
            self.assertFalse(supports_guided_prefix(PARENT, {'id': goal}))
        for original in (PARENT.replace("return {'result': out}", "return {'result': probe.call(torch.sin, w)}"),
                         PARENT.replace("    out.sum()", "    if w.numel(): return {}\n    out.sum()")):
            self.assertFalse(supports_guided_prefix(original, GOAL))

    def test_real_target_and_previous_backward_must_replay_before_suffix_credit(self):
        import random
        from mirage.backends.pytorch import PyTorchBackend
        from mirage.generator.state_plan import state_family_catalog
        from mirage.runtime.probes import Recorder
        from mirage.runtime.stateful import execute

        spec = {'api': 'torch.sin', 'metadata': {}, 'capabilities': {}}
        goals = {g['id']: g for g in state_family_catalog(spec)}
        config = {'api': 'torch.sin', 'device': 'cpu', 'seed': 42, 'requires_backward': True,
                  'state_goal': goals['target_backward'], 'check_claim': 'backward'}
        parent = execute(PARENT, config, Recorder())
        random.random()  # A subsequent worker can start with a different process RNG.
        child = execute(CHILD, dict(config, state_goal=goals['leaf_accumulation'], check_claim='accumulation'),
                        Recorder(), parent, restore_prefix_rng=True)
        backend = PyTorchBackend()
        witness = guided_prefix_witness(parent, child, backend)
        self.assertTrue(witness['verified'], witness)
        self.assertEqual(witness['parent_target_calls'], 1)
        self.assertEqual(len(child['target_calls']), 2)

        for field in ('initial_rng', 'initial_metadata', 'target_calls', 'target_operand_gradients'):
            with self.subTest(field=field):
                changed = deepcopy(child)
                if field == 'initial_rng':
                    changed[field]['cpu'][0] ^= 1
                elif field == 'initial_metadata':
                    changed[field][0]['offset'] = 11
                elif field == 'target_calls':
                    changed[field][0]['output'] *= 2
                else:
                    changed[field].append(deepcopy(changed[field][0]))
                self.assertFalse(guided_prefix_witness(parent, changed, backend)['verified'])
