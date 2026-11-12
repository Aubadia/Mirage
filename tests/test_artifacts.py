from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from mirage.artifacts.store import ArtifactStore
from mirage.cli import main
from mirage.contracts.models import Candidate, Experiment


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ArtifactStore(self.directory.name)

    def test_round_trip_and_derivation_preserve_original(self):
        experiment = Experiment("e", Candidate("torch.matmul", "offset", "before", "slice", "layout"), 9)
        original = self.store.write("original", "experiment", {"experiment": experiment})
        before = original.read_bytes()
        self.store.write("child", "experiment", {"program": "modified"}, parent_id="original")
        self.assertEqual(self.store.read("child")["parent_id"], "original")
        self.assertEqual(self.store.read("original")["payload"]["experiment"]["seed"], 9)
        with self.assertRaises(FileExistsError):
            self.store.write("original", "experiment", {})
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(list(Path(self.directory.name).glob(".pending-*")), [])

    def test_missing_parent_and_nonfinite_data_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.store.write("child", "experiment", {}, parent_id="missing")
        with self.assertRaises(ValueError):
            self.store.write("bad", "execution", {"loss": float("nan")})
        self.assertEqual(self.store.list_manifests(), [])

    def test_corrupt_version_id_and_duplicate_keys_rejected(self):
        path = self.store.write("a", "generation", {})
        original = self.store.read("a")
        for field, value in (("schema_version", 99), ("artifact_id", "other"), ("created_at", "bad")):
            changed = dict(original, **{field: value})
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.store.read("a")
        path.write_text('{"schema_version": 1, "schema_version": 1}')
        with self.assertRaises(ValueError):
            self.store.read("a")

    def test_identifier_validation(self):
        for value in ("../outside", "", "a/b"):
            with self.assertRaises(ValueError):
                self.store.write(value, "experiment", {})

    def test_cli_inspect_report_and_execution_validation(self):
        path = self.store.write("a", "experiment", {"note": "測試"})
        for arguments, expected_key in ((["inspect", str(path)], "payload"),
                                        (["report", self.directory.name], "artifact_count")):
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(arguments), 0)
            result = json.loads(output.getvalue())
            self.assertIn(expected_key, result)
            if arguments[0] == "report":
                self.assertIsNone(result["coverage"])

        for arguments in (["run"], ["replay", str(path)], ["run", "--apis", "unknown"]):
            output, errors = StringIO(), StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(main(arguments), 2)
            self.assertEqual(output.getvalue(), "")
            self.assertTrue(errors.getvalue())

    def test_cli_malformed_manifest_is_an_error(self):
        path = Path(self.directory.name) / "bad.json"
        path.write_text("{")
        with redirect_stderr(StringIO()):
            self.assertEqual(main(["inspect", str(path)]), 2)
