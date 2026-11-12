import unittest

from mirage.contracts.apis import API_IDS, select_apis
from mirage.contracts.models import Candidate, Evidence, Experiment, Inspection, Verdict


class ContractTests(unittest.TestCase):
    def test_api_subset(self):
        self.assertEqual(select_apis(), API_IDS)
        self.assertEqual(select_apis([API_IDS[0], API_IDS[0]]), (API_IDS[0],))
        for subset in ([], ["invented"]):
            with self.assertRaises(ValueError):
                select_apis(subset)

    def test_missing_evidence_is_not_reachability(self):
        experiment = Experiment("e", Candidate(API_IDS[0], "offset", "before_api", "slice", "layout"), 1)
        complete = Evidence("before_api", True, True, True, True, ("trace.json",))
        self.assertTrue(complete.verifies(experiment))
        self.assertFalse(Evidence("before_api", True, True, True, True).verifies(experiment))
        self.assertFalse(Evidence("after_api", True, True, True, True, ("trace",)).verifies(experiment))
        self.assertFalse(Evidence("before_api").verifies(experiment))

    def test_output_verdicts_require_all_conditions(self):
        for verdict, output in ((Verdict.PASS, True), (Verdict.SUSPECTED_FAILURE, False)):
            result = Inspection("e", ("a", "b"), verdict, True, True, True, True, output)
            self.assertTrue(result.applicable)
            with self.assertRaises(ValueError):
                Inspection("e", ("a", "b"), verdict, True, None, True, True, output)
            with self.assertRaises(ValueError):
                Inspection("e", ("a", "b"), verdict, True, True, True, True, not output)

    def test_unknown_is_distinct_from_invalid_and_inapplicable(self):
        self.assertIsNone(Inspection("e", ("a",), None).verdict)
        Inspection("e", ("a",), Verdict.INVALID, False)
        Inspection("e", ("a", "b"), Verdict.INAPPLICABLE, True, False)
        with self.assertRaises(ValueError):
            Inspection("e", ("a",), Verdict.INVALID)
        with self.assertRaises(ValueError):
            Inspection("e", ("a",), Verdict.INAPPLICABLE, True)

    def test_strict_checks_and_ids(self):
        with self.assertRaises(ValueError):
            Evidence("before_api", api_executed=1)
        with self.assertRaises(ValueError):
            Inspection("e", ("a", "a"), None)
        with self.assertRaises(ValueError):
            Candidate(API_IDS[0], "offset", "before", "slice", "layout", precision="float8")
