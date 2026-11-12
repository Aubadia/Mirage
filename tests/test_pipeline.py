import json
from pathlib import Path
import tempfile
import time
import unittest

from mirage.contracts.programs import CaseConfig
from mirage.generator.pipeline import generate_stage
from mirage.generator.provider import ReferenceProvider, Stage
from mirage.generator.repair import Diagnostic, DiagnosticKind


class PipelineTests(unittest.TestCase):
    def test_every_stage_preserves_program_and_runtime_feedback(self):
        config = CaseConfig("torch.matmul")
        calls = []
        previous = None
        with tempfile.TemporaryDirectory() as temporary:
            def validate(program, actual_config, directory):
                calls.append((program, actual_config))
                return Diagnostic(DiagnosticKind.ACCEPTED), "observed"

            for stage in Stage:
                previous, result, observed = generate_stage(
                    ReferenceProvider(), stage, config, Path(temporary) / stage.value,
                    time.monotonic() + 10, validate, previous)
                self.assertEqual(result.status, "accepted")
                self.assertEqual(observed, "observed")
                self.assertTrue((Path(temporary) / stage.value / "program.py").exists())
            self.assertEqual(len(calls), 4)

    def test_unknown_execution_exception_never_triggers_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            program, result, observed = generate_stage(
                ReferenceProvider(), Stage.CANONICAL, CaseConfig("torch.matmul"),
                Path(temporary) / "stage", time.monotonic() + 10,
                lambda *_: (Diagnostic(DiagnosticKind.UNKNOWN_EXCEPTION, "unexplained"), "partial trace"))
            self.assertIsNone(program)
            self.assertEqual(len(result.attempts), 1)
            self.assertEqual(observed, "partial trace")
            saved = json.loads((Path(temporary) / "stage" / "attempt-01.json").read_text())
            self.assertEqual(saved["diagnostic"]["kind"], "unknown_exception")
