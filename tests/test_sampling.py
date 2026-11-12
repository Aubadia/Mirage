from dataclasses import replace
import unittest

from mirage.contracts.models import Candidate, Evidence, Execution, Experiment, Inspection, Outcome, Verdict
from mirage.sampler.coverage import Coverage
from mirage.sampler.policy import Sampler


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.a = Candidate("torch.matmul", "offset", "before_api", "slice", "layout")
        self.b = replace(self.a, state="strided", transition="transpose")
        self.experiment = Experiment("e", self.a, 0)
        self.evidence = Evidence("before_api", True, True, True, True, ("trace.json",))
        self.source = Execution("s", "e", "source", Outcome.COMPLETED, self.evidence, 0.1)
        self.followup = Execution("f", "e", "followup", Outcome.COMPLETED, self.evidence, 0.2)
        self.pass_result = Inspection("e", ("s", "f"), Verdict.PASS, True, True, True, True, True)

    def test_distribution_and_reproducibility(self):
        counts = {self.a: 9}
        first = Sampler((self.a, self.b), seed=7)
        second = Sampler((self.a, self.b), seed=7)
        probabilities = first.probabilities(counts)
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertLess(probabilities[0], probabilities[1])
        self.assertGreaterEqual(probabilities[0], 0.05)
        self.assertEqual([first.sample(counts) for _ in range(50)],
                         [second.sample(counts) for _ in range(50)])
        self.assertEqual(Sampler((self.a, self.b), seed=0, epsilon=1).probabilities(counts), (0.5, 0.5))

    def test_invalid_candidate_universe_and_counts(self):
        for universe in ((), (self.a, self.a)):
            with self.assertRaises(ValueError):
                Sampler(universe, seed=0)
        for count in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                Sampler((self.a,), seed=0).probabilities({self.a: count})

    def test_pass_and_failure_both_count_verified_coverage(self):
        for verdict, output in ((Verdict.PASS, True), (Verdict.SUSPECTED_FAILURE, False)):
            coverage = Coverage((self.a, self.b))
            result = replace(self.pass_result, verdict=verdict, output_relation=output)
            coverage.update(self.experiment, (self.source, self.followup), result)
            self.assertEqual(coverage.counts[self.a], 1)
            self.assertEqual(coverage.report()["combinations"]["denominator"], 2)
            self.assertEqual(coverage.report()["api_mr"]["covered"], 1)
            with self.assertRaises(ValueError):
                coverage.update(self.experiment, (self.source, self.followup), result)

    def test_unverified_state_does_not_count(self):
        coverage = Coverage((self.a,))
        record = replace(self.followup, evidence=Evidence("before_api"))
        coverage.update(self.experiment, (record,))
        self.assertEqual(coverage.report()["state"]["covered"], 0)
        self.assertEqual(coverage.report()["verdicts"], {"unassessed": 1})

    def test_inapplicable_retains_state_but_not_api_mr(self):
        coverage = Coverage((self.a,))
        result = Inspection("e", ("s", "f"), Verdict.INAPPLICABLE, True, False)
        coverage.update(self.experiment, (self.source, self.followup), result)
        self.assertEqual(coverage.report()["state"]["covered"], 1)
        self.assertEqual(coverage.report()["api_mr"]["covered"], 0)
        self.assertEqual(coverage.counts[self.a], 0)

    def test_source_need_not_reach_followup_predicate(self):
        source = replace(self.source, evidence=replace(self.evidence, target_state_reached=False))
        coverage = Coverage((self.a,))
        coverage.update(self.experiment, (source, self.followup), self.pass_result)
        self.assertEqual(coverage.counts[self.a], 1)

    def test_reject_contradictory_inspection_without_partial_update(self):
        coverage = Coverage((self.a,))
        for followup in (replace(self.followup, outcome=Outcome.CRASH),
                         replace(self.followup, evidence=Evidence("before_api")),
                         replace(self.followup, experiment_id="other")):
            with self.assertRaises(ValueError):
                coverage.update(self.experiment, (self.source, followup), self.pass_result)
            self.assertEqual(coverage.report()["experiments"], 0)

    def test_coverage_only_never_receives_correctness_pass(self):
        candidate = replace(self.a, mr=None)
        coverage = Coverage((candidate,))
        experiment = replace(self.experiment, candidate=candidate)
        with self.assertRaises(ValueError):
            coverage.update(experiment, (self.source, self.followup), self.pass_result)
        coverage.update(experiment, (self.followup,))
        self.assertEqual(coverage.report()["verdicts"], {"unassessed": 1})
        self.assertIsNone(coverage.report()["api_mr"]["fraction"])
