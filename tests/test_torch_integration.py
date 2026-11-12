import importlib.util
import os
import unittest

from mirage.contracts.apis import API_IDS, PROFILES
from mirage.contracts.models import Candidate, Experiment, Verdict
from mirage.contracts.programs import CaseConfig, Program
from mirage.contracts.states import StateTarget
from mirage.inspector.oracle import inspect
from mirage.inspector.validation import assess
from mirage.runtime.harness import execute
from mirage.runtime.probes import Recorder
from mirage.runtime.process import ProcessResult
from mirage.runtime.state import make_initial
from mirage.sampler.candidates import pair_configs


HAS_TORCH = importlib.util.find_spec("torch") is not None


def execute_pair(api="torch.matmul", mr="layout", representation="strided", backend="eager",
                 precision="fp32", device="cpu", warmup="none"):
    base = CaseConfig(api, device=device, backend=backend, precision=precision, updates=2)
    target = StateTarget(representation, mr == "checkpoint", 1, "after_update", warmup)
    candidate = Candidate(api, target.key, target.checkpoint, "integration", mr, backend, precision)
    experiment = Experiment("integration", candidate, 0)
    configs = pair_configs(candidate, base)
    initial = make_initial(base)
    records = tuple(execute(Program(PROFILES[api].source, config.actions), config, initial, Recorder())
                    for config in configs)
    process = ProcessResult("completed", 0, 0.1, "/tmp/integration")
    executions = tuple(assess(experiment, role, role, config, process, record)
                       for role, config, record in zip(("source", "followup"), configs, records))
    return experiment, executions, configs, records


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchIntegrationTests(unittest.TestCase):
    def test_all_api_forward_backward_and_state_paths(self):
        for api in API_IDS:
            with self.subTest(api=api):
                experiment, executions, configs, records = execute_pair(api, mr=None, representation="offset")
                self.assertTrue(all(e.evidence.verifies(experiment) for e in executions))
                self.assertTrue(all(r["observations"][0]["actual_output_dtype"] == "torch.float32" for r in records))

    def test_layout_storage_and_checkpoint_families(self):
        for mr, representation in (("layout", "strided"), ("layout", "offset"),
                                   ("storage", "clone"), ("storage", "shared"),
                                   ("checkpoint", "contiguous")):
            with self.subTest(mr=mr, representation=representation):
                result, detail = inspect(*execute_pair(mr=mr, representation=representation))
                self.assertEqual(result.verdict, Verdict.PASS, detail)

    def test_dropout_checkpoint_rng_is_restored(self):
        arguments = execute_pair(API_IDS[5], "checkpoint", "contiguous")
        result, detail = inspect(*arguments)
        self.assertEqual(result.verdict, Verdict.PASS, detail)
        self.assertTrue(any(event["event"] == "checkpoint_enter" and event["recompute"]
                            for event in arguments[3][1]["events"]))

    def test_matmul_gradient_against_independent_scalar_formula(self):
        arguments = execute_pair()
        observed = arguments[3][1]["observations"][0]
        x = observed["inputs"]["x"].tolist()
        weight = observed["inputs"]["weight"].tolist()
        actual = observed["gradient"].tolist()
        count = len(x)
        output = [[sum(x[i][k] * weight[k][j] for k in range(count))
                   for j in range(count)] for i in range(count)]
        for i in range(count):
            for k in range(count):
                expected = 2 * sum(output[i][j] * weight[k][j] for j in range(count)) / count ** 2
                self.assertAlmostEqual(actual[i][k], expected, places=6)

    @unittest.skipUnless(os.environ.get("MIRAGE_TEST_COMPILE") == "1", "Set MIRAGE_TEST_COMPILE=1 for compilation tests")
    def test_compile_backends_and_warmup_restoration(self):
        for backend in ("compile_eager", "aot_eager", "inductor"):
            with self.subTest(backend=backend):
                result, detail = inspect(*execute_pair(backend=backend, warmup="alternate"))
                self.assertEqual(result.verdict, Verdict.PASS, detail)
                result, detail = inspect(*execute_pair(API_IDS[6], "mutation", "shared", backend))
                self.assertEqual(result.verdict, Verdict.PASS, detail)
                result, detail = inspect(*execute_pair(API_IDS[5], "checkpoint", "contiguous", backend))
                self.assertEqual(result.verdict, Verdict.PASS, detail)

    @unittest.skipUnless(os.environ.get("MIRAGE_TEST_CUDA") == "1", "Set MIRAGE_TEST_CUDA=1 for CUDA tests")
    def test_cuda_precision_modes(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        backends = ("eager", "compile_eager", "aot_eager", "inductor") if os.environ.get("MIRAGE_TEST_COMPILE") == "1" else ("eager",)
        for precision in ("fp32", "fp16", "bf16"):
            for backend in backends:
                with self.subTest(precision=precision, backend=backend):
                    result, detail = inspect(*execute_pair(precision=precision, backend=backend, device="cuda"))
                    self.assertEqual(result.verdict, Verdict.PASS, detail)
