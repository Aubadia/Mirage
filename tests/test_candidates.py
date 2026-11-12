from dataclasses import replace
import unittest

from mirage.contracts.apis import API_IDS
from mirage.contracts.programs import CaseConfig, RunOptions
from mirage.contracts.states import StateTarget
from mirage.sampler.candidates import candidates, pair_configs


class CandidateTests(unittest.TestCase):
    def test_all_apis_retained_and_stochastic_layout_not_assumed(self):
        universe = candidates(API_IDS, "eager", "fp32", 3)
        self.assertEqual(len(universe), len(set(universe)))
        self.assertEqual({item.api for item in universe}, set(API_IDS))
        dropout = [item for item in universe if item.api == API_IDS[5]]
        self.assertEqual({item.mr for item in dropout}, {None, "checkpoint"})
        self.assertTrue(all(item.mr is None for item in universe if item.api.startswith("Tensor.")))

    def test_mutation_pair_preserves_representation(self):
        item = next(c for c in candidates((API_IDS[6],), "aot_eager", "fp32", 3)
                    if c.mr == "mutation" and c.state.startswith("shared:"))
        source, followup = pair_configs(item, CaseConfig(API_IDS[6]))
        self.assertEqual(source.backend, "eager")
        self.assertEqual(followup.backend, "aot_eager")
        self.assertEqual(source.target.representation, followup.target.representation)

    def test_target_and_budget_validation(self):
        with self.assertRaises(ValueError):
            CaseConfig(API_IDS[0], updates=1, target=StateTarget(iteration=2))
        for change in ({"time_limit": float("nan")}, {"epsilon": 2}, {"replay_failures": -1}):
            with self.assertRaises(ValueError):
                RunOptions("/tmp/output", "reference", **change)
        base = CaseConfig(API_IDS[0])
        for change in ({"seed": True}, {"size": 1}, {"device": "invented"}):
            with self.assertRaises(ValueError):
                replace(base, **change)
