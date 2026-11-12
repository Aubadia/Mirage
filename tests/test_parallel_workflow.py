"""Real PyTorch execution with a deterministic HTTP service, never a real LLM."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mirage.artifacts.bundle import verify
from mirage.artifacts.replay import replay
from mirage.contracts.programs import RunOptions
from mirage.workflow import run
from test_http_provider import completion, stub_service


HAS_TORCH = importlib.util.find_spec("torch") is not None
ENABLED = HAS_TORCH and os.environ.get("MIRAGE_TEST_E2E") == "1"


def program_response(payload):
    context = json.loads(payload["messages"][1]["content"].split("\nContext:\n", 1)[1])
    result = completion(payload)
    result["choices"][0]["message"]["content"] = json.dumps({
        "source": context["previous_program"] or context["grammar_example"],
        "actions": context["actions"],
    })
    return 200, result


@unittest.skipUnless(ENABLED, "Set MIRAGE_TEST_E2E=1 with PyTorch")
class ParallelWorkflowTests(unittest.TestCase):
    def test_single_and_parallel_http_cases_preserve_initial_state_and_replay(self):
        import torch

        device = "cuda" if os.environ.get("MIRAGE_TEST_CUDA") == "1" else "cpu"
        initial_states = []
        with tempfile.TemporaryDirectory() as temporary, stub_service(program_response) as (url, requests):
            with patch.dict(os.environ, {"MIRAGE_VLLM_BASE_URL": url}):
                for workers in (1, 2):
                    output = Path(temporary) / f"workers-{workers}"
                    report = run(RunOptions(str(output), "vllm", cases=2, workers=workers,
                                            generation_concurrency=2, updates=2, device=device,
                                            worker_timeout=60, time_limit=240), ("torch.matmul",))
                    self.assertEqual(report["stop_reason"], "case_limit")
                    self.assertEqual(report["attempted_experiments"], 2)
                    self.assertEqual(report["coverage"]["experiments"], 2)
                    self.assertEqual(report["coverage"]["cost"]["generation"]["attempts"], 8)
                    states = {}
                    applicable_manifests = []
                    for summary in report["experiments"]:
                        manifest = Path(summary["artifact"])
                        saved = json.loads(manifest.read_text())["payload"]
                        self.assertEqual(summary["status"], "executed", saved)
                        self.assertTrue(all(e["outcome"] == "completed" for e in saved["executions"]), saved)
                        self.assertTrue(all(e["evidence"]["target_state_reached"] for e in saved["executions"]), saved)
                        if saved["experiment"]["candidate"]["mr"] is None:
                            self.assertIsNone(summary["verdict"])
                            self.assertEqual(saved["comparison"]["correctness"], "unassessed")
                        else:
                            self.assertEqual(summary["verdict"], "pass", saved)
                            applicable_manifests.append(manifest)
                        bundle = manifest.parent / saved["bundle"]
                        verify(bundle, saved["files"])
                        states[saved["index"]] = torch.load(bundle / saved["initial"], weights_only=True)
                    initial_states.append(states)
                before = len(requests)
                self.assertTrue(applicable_manifests)
                reproduced = replay(applicable_manifests[0], timeout=60)
                self.assertEqual(reproduced["provider_calls"], 0)
                self.assertEqual(reproduced["verdict"], "pass")
                self.assertEqual(reproduced["environment_changes"], [])
                self.assertEqual(len(requests), before)
            for index in (0, 1):
                for field in ("parameter", "bias"):
                    self.assertTrue(torch.equal(initial_states[0][index][field], initial_states[1][index][field]))
                self.assertTrue(torch.equal(initial_states[0][index]["rng"]["cpu"],
                                            initial_states[1][index]["rng"]["cpu"]))
                if device == "cuda":
                    self.assertTrue(torch.equal(initial_states[0][index]["rng"]["cuda"],
                                                initial_states[1][index]["rng"]["cuda"]))
            self.assertFalse(torch.equal(initial_states[1][0]["parameter"], initial_states[1][1]["parameter"]))

    def test_service_timeout_preserves_attempt_and_stops_dispatch(self):
        def slow_response(payload):
            time.sleep(0.4)
            return program_response(payload)

        with tempfile.TemporaryDirectory() as temporary, stub_service(slow_response) as (url, requests):
            with patch.dict(os.environ, {"MIRAGE_VLLM_BASE_URL": url, "MIRAGE_HTTP_TIMEOUT": "0.05"}):
                report = run(RunOptions(temporary, "vllm", cases=4, workers=2,
                                        generation_concurrency=1, updates=2,
                                        worker_timeout=60, time_limit=120), ("torch.matmul",))
            self.assertEqual(report["stop_reason"], "provider_cancellation_unconfirmed")
            self.assertEqual(report["attempted_experiments"], 2)
            self.assertEqual(len(requests), 1)
            self.assertTrue(all(item["verdict"] is None for item in report["experiments"]))
            self.assertTrue(any(item["status"] == "generation_provider_timeout" for item in report["experiments"]))
            for item in report["experiments"]:
                self.assertTrue(Path(item["artifact"]).exists())
