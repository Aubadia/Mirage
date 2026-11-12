from copy import deepcopy
from dataclasses import replace
import unittest

from mirage.contracts.models import Outcome, Verdict
from mirage.inspector.comparison import differences
from mirage.inspector.oracle import inspect
from test_torch_integration import HAS_TORCH, execute_pair


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class InspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arguments = execute_pair()

    def test_output_violation_is_suspected_not_confirmed(self):
        experiment, executions, configs, original = self.arguments
        records = deepcopy(original)
        records[1]["observations"][0]["gradient"] += 10
        result, detail = inspect(experiment, executions, configs, records)
        self.assertEqual(result.verdict, Verdict.SUSPECTED_FAILURE)
        self.assertFalse(detail["confirmed_defect"])

    def test_initial_values_and_checkpoint_alignment_gate_comparison(self):
        experiment, executions, configs, original = self.arguments
        for field in ("initial", "alignment"):
            records = deepcopy(original)
            if field == "initial":
                records[1]["initial"]["parameter"] += 1
            else:
                records[1]["observations"][0]["successful_updates"] += 1
            result, detail = inspect(experiment, executions, configs, records)
            self.assertEqual(result.verdict, Verdict.INAPPLICABLE)
            self.assertIsNone(result.output_relation)
            self.assertNotIn("differences", detail)

    def test_invalid_and_unknown_execution_are_distinct(self):
        experiment, executions, configs, records = self.arguments
        invalid = replace(executions[1], evidence=replace(executions[1].evidence, target_state_reached=False))
        result, _ = inspect(experiment, (executions[0], invalid), configs, records)
        self.assertEqual(result.verdict, Verdict.INVALID)
        crashed = replace(executions[1], outcome=Outcome.CRASH)
        result, _ = inspect(experiment, (executions[0], crashed), configs, records)
        self.assertIsNone(result.verdict)

    def test_modified_computation_is_inapplicable_not_framework_failure(self):
        experiment, executions, configs, original = self.arguments
        records = deepcopy(original)
        records[1]["program_ast"] = "a different profiled computation"
        result, _ = inspect(experiment, executions, configs, records)
        self.assertEqual(result.verdict, Verdict.INAPPLICABLE)

    def test_saved_tolerance_cannot_be_silently_relaxed(self):
        with self.assertRaises(ValueError):
            inspect(*self.arguments, saved_policy={"version": 1, "atol": 100})

    def test_nonfinite_shape_dtype_and_discrete_values(self):
        import torch

        self.assertTrue(differences(torch.tensor([float("nan")]), torch.tensor([float("nan")])))
        self.assertTrue(differences(torch.ones(2), torch.ones(3)))
        self.assertTrue(differences(torch.ones(2), torch.ones(2, dtype=torch.float64)))
        self.assertTrue(differences(torch.tensor(1), torch.tensor(2), {"atol": 100, "rtol": 100}))
        self.assertFalse(differences(torch.tensor([1.0]), torch.tensor([1.00001]), {"atol": 1e-4, "rtol": 0}))
