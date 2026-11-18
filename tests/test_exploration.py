"""Exercise canonical reuse, open API names, declared targets, and saved engines."""

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from mirage.contracts.apis import select_apis, ACCEPTANCE_APIS
from mirage.contracts.exploration import targets, validate_history
from mirage.contracts.programs import RunOptions
from mirage.generator.program import ContractError
from mirage.workflow import base_config, run, replay


class ExplorationTests(unittest.TestCase):
    def test_multi_api_entry_keeps_seeds_independent_of_completion_order(self):
        from mirage.runtime.multi_api import run_multi_api
        with tempfile.TemporaryDirectory() as temporary:
            options = RunOptions(temporary, 'fixture', seed=7, workers=2, active_api_limit=2)
            with patch('mirage.runtime.multi_api.run_sessions', return_value={}) as sessions:
                run_multi_api(options, ('torch.sin', 'torch.cos'))
            jobs = sessions.call_args.args[0]
            self.assertEqual([job.options.seed for job in jobs], [7, 1016])
            self.assertEqual([job.api for job in jobs], ['torch.sin', 'torch.cos'])
            self.assertEqual(len({job.options.output for job in jobs}), 2)

    def test_api_entry_is_open(self):
        self.assertEqual(len(ACCEPTANCE_APIS), 16)
        self.assertEqual(select_apis(["torch.sin", "torch.linalg.svd"]), ("torch.sin", "torch.linalg.svd"))
        with self.assertRaises(ValueError):
            select_apis(["not_torch"])

    def test_target_pool_is_not_an_mr_whitelist(self):
        config = base_config("torch.sin", RunOptions("unused", "unused"), 1)
        candidates = targets(config, {".0.x": {"dtype": "torch.float32"}}, None)
        self.assertTrue(any(rule is None and c["graph_mode"] == "retained" for rule, c in candidates))
        self.assertEqual(len(candidates), 12)

    def test_history_rejects_step_with_live_graph(self):
        config = base_config("torch.sin", RunOptions("unused", "unused"), 1)
        config.update(optimizer_steps=1)
        with self.assertRaises(ContractError):
            validate_history(["forward", "backward:retain", "step", "zero:none"], config)

    def test_controller_reuses_canonical_and_replays_without_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            test_root = str(Path(__file__).parent)
            environment = {"PYTHONPATH": test_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
            with patch.dict(os.environ, environment):
                options = RunOptions(temporary, "fixture_provider:create", cases=4, canonicals_per_api=1,
                                     targets_per_canonical=2, paths_per_target=2, worker_timeout=90,
                                     time_limit=300, seed=0)
                report = run(options, ("torch.matmul",))
            self.assertEqual(report["coverage"]["api"]["covered"], 1, report["canonicals"])
            self.assertEqual(len(report["canonicals"]), 1)
            self.assertEqual(len(report["experiments"]), 4)
            self.assertEqual(report["generation"]["calls"], 5)
            self.assertGreater(report["candidate_metrics"]["registry"]["admitted"], 0)
            self.assertGreater(report["candidate_metrics"]["registry"]["rejected"], 0)
            self.assertEqual(report["candidate_metrics"]["sampled_path_attempts"], 4)
            case = next(item for item in report["experiments"] if item["verdict"] == "pass")
            with patch.dict(os.environ, {"MIRAGE_VLLM_BASE_URL": "http://127.0.0.1:1/v1"}):
                reproduced = replay(case["manifest"], timeout=90)
            self.assertEqual(reproduced["provider_calls"], 0)
            self.assertEqual(reproduced["verdict"], "pass")
            self.assertEqual(reproduced["environment_changes"], [])
