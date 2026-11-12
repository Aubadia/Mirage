"""Real execution and negative-oracle tests for the general program contract."""

from pathlib import Path
import tempfile
import unittest

from mirage.generator.program import analyze, ContractError
from mirage.runtime.api_metadata import describe
from mirage.runtime.program import initialize, run
from mirage.runtime.probes import Recorder
from mirage.rules.conditional import inspect


SOURCE = '''
import torch

def build(device, seed):
    torch.manual_seed(seed)
    x = torch.randn(5, 3, device=device)
    w = torch.nn.Parameter(torch.randn(3, 2, device=device))
    return {"inputs": {"x": x}, "parameters": {"w": w},
            "optimizer": torch.optim.SGD([w], lr=0.01), "batch_axes": {"x": 0}}

def forward(ctx):
    return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])

def sample_loss(output):
    return output.square().mean(dim=-1)

def loss(output):
    return output.square().mean()
'''


def configuration(**kwargs):
    return {"api": "torch.matmul", "device": "cpu", "seed": 42,
            "backend": "eager", "precision": "fp32", "representation": "contiguous",
            "input_name": "x", "gradient_state": "none", "optimizer_steps": 0,
            "graph_mode": "fresh", "checkpoint": False, "saved_tensors": False,
            "microbatches": 1, "accumulation_source": False, **kwargs}


class GeneralProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.initial = self.root / "initial.pt"
        initialize(SOURCE, configuration(), self.initial)
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, **kwargs):
        self.counter += 1
        return run(SOURCE, configuration(**kwargs), self.initial,
                   Recorder(self.root / f"trace-{self.counter}.jsonl"))

    def test_signature_and_open_api_resolution(self):
        info = describe("torch.sin")
        self.assertIsNotNone(info["signature"])
        self.assertIn("torch_version", info)

    def test_api_outside_acceptance_list_executes(self):
        source = SOURCE.replace('torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'torch.sin(ctx["parameters"]["w"])')
        result = run(source, configuration(api="torch.sin"), self.initial,
                     Recorder(self.root / "sin.jsonl"))
        self.assertTrue(result["api_executed"])
        self.assertTrue(result["backward_executed"])

    def test_explicit_forward_only_target(self):
        source = SOURCE.replace('return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'return {"shape": ctx["inputs"]["x"].size(), "value": torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])}')
        source = source.replace('return output.square().mean()', 'return output["value"].square().mean()')
        config = configuration(api="Tensor.size", requires_backward=False)
        left = run(source, config, self.initial, Recorder(self.root / "size-source.jsonl"))
        right = run(source, dict(config, representation="strided"), self.initial,
                    Recorder(self.root / "size-followup.jsonl"))
        self.assertTrue(left["api_executed"])
        self.assertFalse(left["backward_executed"])
        self.assertEqual(inspect("layout", left, right)["verdict"], "pass")

    def test_lifecycle_contract(self):
        self.assertEqual(analyze(SOURCE)["version"], 3)
        with self.assertRaises(ContractError):
            analyze("def forward(x): return x")

    def test_layout(self):
        left, right = self.execute(), self.execute(representation="strided")
        self.assertTrue(right["api_executed"])
        self.assertTrue(right["backward_executed"])
        self.assertEqual(inspect("layout", left, right)["verdict"], "pass")

    def test_storage(self):
        self.assertEqual(inspect("storage", self.execute(), self.execute(representation="clone"))["verdict"], "pass")

    def test_checkpoint(self):
        self.assertEqual(inspect("checkpoint", self.execute(), self.execute(checkpoint=True))["verdict"], "pass")

    def test_saved_tensor_roundtrip(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        self.assertGreater(right["saved_tensors"]["unpacked"], 0)
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "pass")

    def test_uneven_microbatches(self):
        left = self.execute(accumulation_source=True)
        right = self.execute(microbatches=2)
        self.assertTrue(left["batch_separable"])
        self.assertTrue(right["batch_separable"])
        self.assertNotEqual(left["backwards"], right["backwards"])
        self.assertEqual(inspect("accumulation", left, right)["verdict"], "pass")

    def test_history_and_retained_graph(self):
        result = self.execute(optimizer_steps=2, gradient_state="accumulated", graph_mode="retained")
        self.assertEqual(result["updates"], 3)
        self.assertEqual(result["backwards"], 5)

    def test_corrupted_output_is_suspected(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["observations"]["parameters"]["w"] += 10
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "suspected_failure")

    def test_missing_evidence_is_not_pass(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["api_executed"] = False
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "invalid")

    def test_unequal_initial_state_is_inapplicable(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["initial"]["inputs"]["x"] += 1
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "inapplicable")

    def test_missing_saved_tensors_is_inapplicable(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["saved_tensors"]["unpacked"] = 0
        self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "inapplicable")

    def test_nonfinite_is_unassessed(self):
        left, right = self.execute(), self.execute(saved_tensors=True)
        right["observations"]["parameters"]["w"][0, 0] = float("inf")
        self.assertIsNone(inspect("saved_tensor", left, right)["verdict"])

    def test_layout_must_reach_target_operands(self):
        source = SOURCE.replace('ctx["inputs"]["x"], ctx["parameters"]',
                                'ctx["inputs"]["x"].contiguous(), ctx["parameters"]')
        result = run(source, configuration(representation="strided", input_name="x"), self.initial,
                     Recorder(self.root / "materialized.jsonl"))
        self.assertFalse(result["target_state_reached"])

    def test_compiled_target_and_backward(self):
        result = self.execute(backend="compile_eager")
        self.assertTrue(result["api_executed"])
        self.assertTrue(result["backward_executed"])

    def test_compiled_detach_does_not_fake_target_backward(self):
        source = SOURCE.replace('return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"]).detach() + ctx["parameters"]["w"].sum()')
        result = run(source, configuration(backend="compile_eager"), self.initial,
                     Recorder(self.root / "compiled-detach.jsonl"))
        self.assertTrue(result["api_executed"])
        self.assertFalse(result["backward_executed"])

    def test_compiled_unrelated_output_does_not_prove_target_backward(self):
        source = SOURCE.replace('return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'return (torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"]), ctx["parameters"]["w"] * 2)')
        source = source.replace('return output.square().mean()', 'return output[1].square().mean()')
        result = run(source, configuration(backend="compile_eager"), self.initial,
                     Recorder(self.root / "compiled-unrelated.jsonl"))
        self.assertTrue(result["api_executed"])
        self.assertFalse(result["backward_executed"])

    def test_initial_input_gradients_survive_restore(self):
        import torch
        from mirage.generator.program import load
        from mirage.runtime.program import restore

        source = SOURCE.replace('    w = torch.nn.Parameter',
                                '    x.requires_grad_()\n    x.grad = torch.ones_like(x)\n    w = torch.nn.Parameter')
        initialize(source, configuration(), self.initial)
        restored = restore(load(source), configuration(), self.initial)
        self.assertTrue(torch.equal(restored["inputs"]["x"].grad, torch.ones(5, 3)))

    def test_inplace_keyword_is_observed_as_mutation(self):
        source = SOURCE.replace('torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'torch.nn.functional.relu(ctx["parameters"]["w"] * 1.0, inplace=True)')
        result = run(source, configuration(api="torch.nn.functional.relu"), self.initial,
                     Recorder(self.root / "inplace-keyword.jsonl"))
        self.assertFalse(result["pure"])
        self.assertIn("relu", result["mutated_operations"])

    def test_mutation_rule(self):
        source = SOURCE.replace('torch.randn(5, 3', 'torch.randn(3, 2')
        source = source.replace('return torch.matmul(ctx["inputs"]["x"], ctx["parameters"]["w"])',
                                'value = ctx["parameters"]["w"] * 1.0\n    return value.add_(ctx["inputs"]["x"])')
        config = configuration(api="Tensor.add_")
        initial = self.root / "mutation.pt"
        initialize(source, config, initial)
        left = run(source, config, initial, Recorder(self.root / "mutation-eager.jsonl"))
        right = run(source, dict(config, backend="compile_eager"), initial,
                    Recorder(self.root / "mutation-compiled.jsonl"))
        self.assertEqual(inspect("mutation", left, right)["verdict"], "pass")

    def test_plan_order_and_actual_history(self):
        from mirage.contracts.exploration import validate_history

        config = configuration(optimizer_steps=1, gradient_state="zero")
        with self.assertRaises(ContractError):
            validate_history(["step", "zero:zero"], config)
        actions = ["zero:none", "forward", "backward:retain", "backward", "step", "zero:zero"]
        validate_history(actions, config)
        result = self.execute(optimizer_steps=1, gradient_state="zero", history=actions)
        self.assertTrue(result["target_state_reached"])
        self.assertEqual(result["updates"], 2)

    def test_cross_sample_matmul_is_not_certified(self):
        from mirage.runtime.mr_actions import accumulation_supported
        from mirage.generator.program import load

        source = SOURCE.replace('ctx["parameters"]["w"]', 'ctx["inputs"]["x"]')
        ctx = load(SOURCE)["build"]("cpu", 42)
        self.assertFalse(accumulation_supported(source, ctx, []))

    def test_signature_import_alias_is_legal(self):
        self.assertEqual(analyze("import torch.nn.functional as F\n" + SOURCE)["version"], 3)

    def test_module_parameters_are_rebound_after_restore(self):
        import torch
        source = '''
import torch
def build(device, seed):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2).to(device)
    return {"inputs": {"x": torch.randn(5, 3, device=device)},
            "parameters": dict(model.named_parameters()), "modules": {"model": model},
            "optimizer": torch.optim.Adam(model.parameters()), "batch_axes": {"x": 0}}
def forward(ctx):
    return ctx["modules"]["model"](ctx["inputs"]["x"])
def loss(output):
    return output.square().mean()
'''
        config = configuration(api="torch.nn.functional.linear")
        initial = self.root / "module.pt"
        initialize(source, config, initial)
        result = run(source, config, initial, Recorder(self.root / "module.jsonl"))
        self.assertTrue(result["backward_executed"])
        self.assertFalse(torch.equal(result["initial"]["state"]["parameters"]["weight"],
                                     result["observations"]["parameters"]["weight"]))

    def test_cuda_precision_and_inductor(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        config = configuration(device="cuda")
        initial = self.root / "cuda.pt"
        # Keep this equivalence fixture away from GradScaler overflow; nonfinite
        # observations have a separate unassessed-policy test.
        source = SOURCE.replace("torch.randn(", "0.05 * torch.randn(")
        initialize(source, config, initial)
        for precision in ("fp32", "fp16", "bf16"):
            with self.subTest(precision=precision):
                left = run(source, dict(config, precision=precision), initial,
                           Recorder(self.root / f"{precision}-source.jsonl"))
                right = run(source, dict(config, precision=precision, saved_tensors=True), initial,
                            Recorder(self.root / f"{precision}-followup.jsonl"))
                self.assertEqual(inspect("saved_tensor", left, right)["verdict"], "pass")
        for backend in ("aot_eager", "inductor"):
            with self.subTest(backend=backend):
                result = run(source, dict(config, backend=backend), initial,
                             Recorder(self.root / f"{backend}.jsonl"))
                self.assertTrue(result["api_executed"])
                self.assertTrue(result["backward_executed"])

    def test_accumulation_rejects_different_loss_normalization(self):
        source = SOURCE.replace('return output.square().mean()', 'return output.square().sum()')
        left = run(source, configuration(accumulation_source=True), self.initial,
                   Recorder(self.root / "wrong-loss-source.jsonl"))
        right = run(source, configuration(microbatches=2), self.initial,
                    Recorder(self.root / "wrong-loss-followup.jsonl"))
        self.assertEqual(inspect("accumulation", left, right)["verdict"], "inapplicable")


if __name__ == "__main__":
    unittest.main()
