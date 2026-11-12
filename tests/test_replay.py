from pathlib import Path
import json
import os
import tempfile
import unittest

from mirage.artifacts.bundle import inventory, verify
from mirage.artifacts.replay import replay
from mirage.contracts.programs import RunOptions
from mirage.workflow import run
from test_torch_integration import HAS_TORCH


class BundleTests(unittest.TestCase):
    def test_missing_or_changed_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "program.py"
            payload.write_text("original\n")
            files = inventory(root)
            verify(root, files)
            payload.write_text("modified\n")
            with self.assertRaises(ValueError):
                verify(root, files)
            payload.unlink()
            with self.assertRaises(ValueError):
                verify(root, files)


@unittest.skipUnless(HAS_TORCH and os.environ.get("MIRAGE_TEST_E2E") == "1", "Set MIRAGE_TEST_E2E=1 with PyTorch")
class ReplayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.report = run(RunOptions(cls.directory.name, "reference", cases=1, updates=2,
                                   seed=0, worker_timeout=60, time_limit=240), ("torch.matmul",))
        cls.manifest = Path(cls.report["experiments"][0]["artifact"])

    def test_batch_replay_and_minimization_preserve_original(self):
        self.assertEqual(self.report["experiments"][0]["verdict"], "pass")
        original = self.manifest.read_bytes()
        repeated = replay(self.manifest, timeout=60)
        self.assertEqual(repeated["provider_calls"], 0)
        self.assertEqual(repeated["verdict"], "pass")
        self.assertEqual(repeated["environment_changes"], [])
        reduced = replay(self.manifest, timeout=60, size=2, updates=1, iteration=0)
        self.assertEqual(reduced["verdict"], "pass")
        child = json.loads(Path(reduced["artifact"]).read_text())
        self.assertEqual(child["parent_id"], self.manifest.stem)
        self.assertEqual(self.manifest.read_bytes(), original)
