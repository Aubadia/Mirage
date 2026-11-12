import ast
import unittest

from mirage.contracts.apis import API_IDS, PROFILES
from mirage.generator.validation import UnsupportedProgram, validate_source
from mirage.rules.registry import RULES, applicable_rules
from mirage.rules.tolerances import policy


class RuleTests(unittest.TestCase):
    def test_mutation_and_randomness_constraints(self):
        self.assertEqual(applicable_rules(API_IDS[6], "eager"), ())
        self.assertEqual(applicable_rules(API_IDS[7], "inductor"), ("mutation",))
        self.assertEqual(applicable_rules(API_IDS[5], "inductor"), ("checkpoint",))
        self.assertIn("mutation", RULES["mutation"].observables)

    def test_program_purity_is_checked(self):
        for api, profile in PROFILES.items():
            ast.parse(profile.source)
            validate_source("# Explanatory comment\n" + profile.source, api)
            with self.assertRaises(UnsupportedProgram):
                validate_source(profile.source + "\nprint('unexpected side effect')\n", api)
        with self.assertRaises(UnsupportedProgram):
            validate_source(PROFILES[API_IDS[0]].source.replace("torch.matmul(x, weight)", "x"), API_IDS[0])

    def test_tolerances_are_contextual_and_discrete_state_is_exact(self):
        a = policy(API_IDS[0], "fp32", "eager", "layout")
        b = policy(API_IDS[0], "bf16", "inductor", "layout")
        self.assertLess(a["observables"]["output"]["rtol"], b["observables"]["output"]["rtol"])
        self.assertEqual(a["observables"]["scaler"], {"atol": 0, "rtol": 0})
        self.assertEqual(a["nonfinite"], "reject")

    def test_canonical_can_select_legal_operator_options(self):
        validate_source(PROFILES[API_IDS[5]].source.replace("p=0.25", "p=0.5"), API_IDS[5])
        validate_source(PROFILES[API_IDS[4]].source.replace("dim=-1", "dim=0"), API_IDS[4])
        validate_source(PROFILES[API_IDS[3]].source.replace("gelu(x)", "gelu(x, approximate='tanh')"), API_IDS[3])
        with self.assertRaises(UnsupportedProgram):
            validate_source(PROFILES[API_IDS[5]].source.replace("p=0.25", "p=3"), API_IDS[5])
