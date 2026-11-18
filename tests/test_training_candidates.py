"""Candidate soundness controls; runtime tests are dispatched through Slurm."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from mirage.contracts.candidates import CapabilitySnapshot, path_key, tensor_paths
from mirage.contracts.capabilities import Evidence, EvidenceStatus
from mirage.contracts.exploration import validate_history
from mirage.sampler.candidates import registry, legal_paths, unmet_requirements
from mirage.runtime.mr_actions import semantic_evidence


def evidence(value, provenance="adapter"):
    return Evidence(EvidenceStatus.VERIFIED if value else EvidenceStatus.REJECTED,
                    "fixture runtime fact", value, "canonical_execution", provenance)


class FakeBackend:
    """No torch objects or operator identities enter the shared sampler."""
    name = "fake"

    def training_capabilities(self):
        facts = {key: evidence(True) for key in ("backward", "gradients", "accumulated_gradients", "optimizer", "microbatch", "mutation")}
        subject = {"path": ["nested", 0], "facts": {key: vars(evidence(True)) for key in
            ("target_operand", "alias_preserved", "clone", "offset", "strided", "shared")}}
        return CapabilitySnapshot(self.name, "canonical_execution", {"program_sha256": "fixture"}, (subject,), facts)


class CandidateContractTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(api="arbitrary_framework.operation", backend="eager", seed=0,
                           optimizer_steps=0, gradient_state="none", representation="contiguous")
        self.snapshot = FakeBackend().training_capabilities()

    def test_framework_neutral_roundtrip_and_unknown_llm_evidence(self):
        snapshot = CapabilitySnapshot.read(json.loads(json.dumps(self.snapshot.payload())))
        self.assertTrue(all(c.status == "admitted" for c in registry(self.config, snapshot)))
        facts = dict(snapshot.facts, optimizer=Evidence(EvidenceStatus.VERIFIED,
                     "model claim", True, "canonical_execution", "llm"))
        candidates = registry(self.config, replace(snapshot, facts=facts))
        self.assertTrue(all(c.status == "unresolved" for c in candidates if c.configuration["optimizer_steps"]))
        self.assertIn("optimizer", unmet_requirements(candidates))

    def test_absent_optimizer_and_partner_are_never_admitted(self):
        facts = dict(self.snapshot.facts, optimizer=evidence(False))
        subjects = tuple(dict(s, facts=dict(s["facts"], shared=vars(evidence(False)))) for s in self.snapshot.subjects)
        candidates = registry(self.config, replace(self.snapshot, facts=facts, subjects=subjects))
        excluded = [c for c in candidates if c.configuration["optimizer_steps"] or c.configuration["representation"] == "shared"]
        self.assertEqual(len(excluded), 3)
        self.assertTrue(all(c.status == "rejected" for c in excluded))
        self.assertTrue(any(c.status == "admitted" for c in candidates))

    def test_all_offered_paths_are_complete_and_diverse(self):
        for steps in range(3):
            for gradients in ("none", "zero", "accumulated"):
                config = dict(self.config, optimizer_steps=steps, gradient_state=gradients)
                paths = legal_paths(config)
                self.assertGreaterEqual(len(paths), 2)
                for path in paths:
                    self.assertEqual(validate_history(path, config), path)
                self.assertNotIn(("forward", "backward:retain", "zero:none"), paths)
                with self.assertRaises(ValueError):
                    validate_history(("forward", "backward:retain", "zero:none"), config)
        self.assertEqual(legal_paths(dict(self.config, optimizer_steps=1), optimizer=False), ())

    def test_typed_paths_do_not_merge_nested_or_integer_coordinates(self):
        token = object()
        paths = list(tensor_paths({"a.b": token, "a": [token], 0: token, "0": token}, lambda v: v is token))
        self.assertEqual(len({path_key(p) for p, _ in paths}), 4)
        self.assertEqual(len(paths), 4)

    def test_unknown_semantics_are_not_promoted_by_absent_writes(self):
        self.assertEqual(semantic_evidence(False, False, [])["representation_independence"]["status"], "unknown")
        self.assertEqual(semantic_evidence(False, False, ["write"])["representation_independence"]["status"], "rejected")
        self.assertEqual(semantic_evidence(True, True, [])["batch_separability"]["status"], "verified")


class CampaignPreflightTests(unittest.TestCase):
    def test_fresh_campaign_persists_request_before_dispatch(self):
        import argparse
        import importlib.util
        import os
        import signal
        from unittest.mock import patch
        from test_http_provider import stub_service

        path = Path(__file__).resolve().parents[1] / "benchmarks/training_candidates.py"
        spec = importlib.util.spec_from_file_location("training_candidate_campaign", path)
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        protocol = {"code": str(path.parents[1]), "model": "fixture", "max_model_len": 16384,
                    "mtp_tokens": 3, "model_revision": "fixture", "targets": []}
        actions = ["zero:none", "forward", "backward", "step", "zero:none"]
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                session = root / "session"
                (session / "serving").mkdir(parents=True)
                (session / "serving/environment.json").write_text(json.dumps(
                    dict(model="fixture", max_model_len=16384, mtp_tokens=3, declared_revision="fixture")))
                output = root / "campaign"
                output.mkdir()

                def respond(payload):
                    self.assertEqual(len(list((output / "path-schema-preflight").glob("http-*.json"))), 1)
                    return 200, {"model": "fixture", "choices": [{"message": {
                        "content": json.dumps({"actions": actions})}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

                with stub_service(respond) as (url, requests), patch.dict(os.environ, {
                    "MIRAGE_SESSION_DIR": str(session), "MIRAGE_VLLM_BASE_URL": url,
                    "MIRAGE_VLLM_MODEL": "fixture", "SLURM_JOB_ID": "fixture", "SLURM_STEP_ID": "fixture"}), \
                        patch.object(campaign, "verify", return_value=protocol):
                    self.assertEqual(campaign.campaign(argparse.Namespace(protocol=output / "protocol.json")), 0)
                self.assertEqual(len(requests), 1)
                artifact = json.loads(next((output / "path-schema-preflight").glob("http-*.json")).read_text())
                self.assertEqual(artifact["http_status"], 200)
                self.assertEqual(json.loads((output / "path-schema-preflight.json").read_text())["actions"], actions)
                self.assertEqual(json.loads((output / "campaign-completion.json").read_text())["runs"], [])
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


class CandidateRuntimeTests(unittest.TestCase):
    def test_repeated_input_identity_records_each_coordinate_once(self):
        import torch
        from mirage.runtime.program import initialize, run
        from mirage.runtime.probes import Recorder
        from test_program import configuration
        source = '''
import torch
def build(device, seed):
    x = torch.ones(3, device=device, requires_grad=True)
    return {"inputs": {"a": x, "b": x}, "parameters": {}, "optimizer": None, "batch_axes": {}}
def forward(ctx):
    return torch.add(ctx["inputs"]["a"], ctx["inputs"]["b"])
def loss(output):
    return output.sum()
'''
        config = configuration(api="torch.add", input_name="a")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize(source, config, root / "initial.pt")
            record = run(source, config, root / "initial.pt", Recorder(root / "trace.jsonl"))
            for name in ("a", "b"):
                torch.testing.assert_close(record["observations"]["gradients"]["inputs"][name], torch.full((3,), 2.0))

    def test_capability_record_is_loadable_without_pickle_globals(self):
        import io
        import torch
        buffer = io.BytesIO()
        torch.save({"capabilities": FakeBackend().training_capabilities().payload()}, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        self.assertTrue(CapabilitySnapshot.read(restored["capabilities"]).facts["optimizer"].established)

    def test_nested_input_gradient_coordinate_and_optimizer_link(self):
        import torch
        from mirage.runtime.program import initialize, run
        from mirage.runtime.probes import Recorder
        from test_program import SOURCE, configuration
        source = SOURCE.replace('"x": x', '"nested": [x.requires_grad_()]').replace('ctx["inputs"]["x"]', 'ctx["inputs"]["nested"][0]').replace('{"x": 0}', '{}')
        config = configuration(input_name=path_key(("nested", 0)), input_path=["nested", 0])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.pt"
            initialize(source, config, initial)
            canonical = run(source, config, initial, Recorder(root / "canonical.jsonl"))
            caps = CapabilitySnapshot.read(canonical["capabilities"])
            self.assertTrue(caps.facts["optimizer"].established)
            candidate = next(c for c in registry(config, caps) if c.configuration["representation"] == "strided" and c.status == "admitted")
            followup = run(source, dict(candidate.configuration, candidate_id=candidate.identifier,
                candidate_prerequisites=candidate.prerequisites, history=list(legal_paths(candidate.configuration)[0])),
                initial, Recorder(root / "followup.jsonl"))
            self.assertTrue(followup["target_state_reached"])
            key = path_key(("nested", 0))
            self.assertIn(key, followup["observations"]["gradients"]["inputs"])
            torch.testing.assert_close(canonical["observations"]["gradients"]["inputs"][key],
                                       followup["observations"]["gradients"]["inputs"][key])

    def test_alias_topology_and_missing_optimizer_capabilities(self):
        import torch
        from mirage.backends.pytorch import PyTorchBackend
        x = torch.randn(4, requires_grad=True)
        ctx = dict(inputs={"x": x, "same": x, "view": x.detach().view(2, 2)},
                   parameters={}, optimizer=None, batch_axes={})
        caps = PyTorchBackend().training_capabilities(ctx, checkpoint="canonical_execution",
            provenance={"precision": "fp32"}, observed={"backward_executed": True, "input_names": ["x", "same", "view"]})
        self.assertFalse(caps.facts["optimizer"].established)
        self.assertEqual(caps.subjects[0]["object_group"], caps.subjects[1]["object_group"])
        self.assertEqual(caps.subjects[0]["storage_group"], caps.subjects[2]["storage_group"])
        self.assertTrue(all(c.status != "admitted" for c in registry(
            dict(api="unseen.operation", backend="eager", optimizer_steps=0, gradient_state="none"), caps)
            if "representation" in c.goals))

    def test_dynamic_partner_recheck_rejects_stale_admission(self):
        import torch
        from mirage.runtime.program import transform
        from mirage.generator.program import ContractError
        ctx = {"inputs": {"a": [torch.ones(3)], "b": torch.ones(3)}}
        transform(ctx, "shared", path_key(("a", 0)), ("a", 0))
        self.assertEqual(ctx["inputs"]["a"][0].untyped_storage().data_ptr(), ctx["inputs"]["b"].untyped_storage().data_ptr())
        ctx["inputs"]["b"] = torch.zeros(3)
        with self.assertRaises(ContractError):
            transform(ctx, "shared", path_key(("a", 0)), ("a", 0))


class SavedTrainingEvidenceTests(unittest.TestCase):
    def test_saved_missing_capabilities_and_invalid_endings(self):
        from mirage.runtime.program import run
        from mirage.runtime.probes import Recorder
        root = next((p for p in Path(__file__).resolve().parents
                     if (p / "result/training-slogdet/20261117-500s/protocol.json").is_file()), None)
        if root is None:
            self.skipTest("Saved historical campaign is unavailable")
        prior = root / "result/training-slogdet/20261117-500s/cases"
        report = json.loads(next(prior.glob("*-report.json")).read_text())["payload"]
        caps_by_id = {}
        with tempfile.TemporaryDirectory() as temporary:
            for row in report["canonicals"]:
                canonical = prior / row["id"] / "canonical-01"
                config = json.loads((canonical / "validation/request.json").read_text())["config"]
                config["device"] = "cpu"
                source = (canonical / "program.py").read_text()
                record = run(source, config, canonical / "initialize/initial.pt",
                             Recorder(Path(temporary) / (row["id"] + ".jsonl")))
                caps_by_id[row["id"]] = registry(config, CapabilitySnapshot.read(record["capabilities"]))
                self.assertEqual(record["mr_evidence"]["representation_independence"]["status"], "unknown")
            errors = json.loads((prior.parent / "failure-details.json").read_text())["failed_executions"]
            blocked_optimizer = blocked_shared = 0
            for error in errors:
                payload = json.loads((prior / error["case"]).read_text())["payload"]
                candidates = caps_by_id[payload["canonical"]]
                if "requires an optimizer" in error["error"]:
                    excluded = [c for c in candidates if c.configuration["optimizer_steps"]]
                    blocked_optimizer += 1
                elif "input partner" in error["error"]:
                    excluded = [c for c in candidates if c.configuration["representation"] == "shared"]
                    blocked_shared += 1
                else:
                    self.fail("Unrecognized historical failure")
                self.assertTrue(excluded)
                self.assertTrue(all(c.status != "admitted" for c in excluded))
            blocked_plans = 0
            for row in report["experiments"]:
                if row["status"] != "plan_repair_exhausted":
                    continue
                generation = json.loads((prior / row["id"] / "generation.json").read_text())
                attempt = generation["attempts"][-1]
                config = attempt["request"]["context"]["configuration"]
                actions = json.loads(attempt["response"]["content"])["actions"]
                self.assertNotIn(tuple(actions), legal_paths(config, optimizer=False))
                with self.assertRaises(ValueError):
                    validate_history(actions, config)
                blocked_plans += 1
            self.assertEqual((blocked_optimizer, blocked_shared, blocked_plans), (12, 6, 8))
