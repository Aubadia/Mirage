"""Core contracts reject missing and generated-only evidence."""
import ast
from dataclasses import asdict
from pathlib import Path
import unittest
from mirage.backends.base import require_capability
from mirage.contracts.capabilities import (
    Evidence, EvidenceStatus, StatePredicate, RelationSpec, InputCoverage, Verdict,
    INITIAL_PREDICATES, STATE_DIMENSIONS,
)


class ContractTests(unittest.TestCase):
    def test_unknown_and_llm_claims_are_not_observed(self):
        predicate = StatePredicate("gradient", "backward_count", 2, "final")
        self.assertFalse(predicate.holds(Evidence()))
        for provenance in ("llm", "inspector"):
            fact = Evidence(EvidenceStatus.VERIFIED, "test", 2, "final", provenance)
            self.assertEqual(predicate.holds(fact), provenance == "inspector")
        self.assertFalse(predicate.holds(Evidence(EvidenceStatus.VERIFIED, "test", 2, "early")))

    def test_bad_records_fail_closed(self):
        with self.assertRaises(ValueError): Evidence(EvidenceStatus.VERIFIED)
        with self.assertRaises(TypeError): Evidence("verified")
        with self.assertRaises(ValueError): InputCoverage(("issue-specific-input",))
        with self.assertRaises(ValueError): StatePredicate("unknown", "x", 0, "final")
        self.assertEqual(set(p.dimension for p in INITIAL_PREDICATES), set(STATE_DIMENSIONS))
        self.assertEqual(len(set(Verdict)), 5)

    def test_relation_requires_all_evidence_at_checkpoint(self):
        relation = RelationSpec("logical", 1, ("legal",), ("same_inputs",),
                                ("output",), ("strides",), "target", "exact")
        yes = Evidence(EvidenceStatus.VERIFIED, "measured", True, "target")
        self.assertEqual(relation.missing({"legal": yes}), ("same_inputs",))
        self.assertFalse(relation.missing({"legal": yes, "same_inputs": yes}))
        wrong = Evidence(EvidenceStatus.VERIFIED, "measured", True, "other")
        self.assertEqual(relation.missing({"legal": yes, "same_inputs": wrong}), ("same_inputs",))

    def test_fake_backend_uses_same_evidence_flow(self):
        class Fake:
            name = "fake"
            def capability(self, name): return Evidence(EvidenceStatus.UNSUPPORTED, name)
        record = require_capability(Fake(), "opaque_layout")
        self.assertFalse(record.established)
        self.assertEqual(asdict(record)["status"], "unsupported")

    def test_core_has_no_framework_import(self):
        root = Path(__file__).resolve().parents[1] / "src/mirage"
        for relative in ("contracts/capabilities.py", "backends/base.py"):
            tree = ast.parse((root / relative).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(x.name.startswith(("torch", "tensorflow")) for x in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith(("torch", "tensorflow")))


class ProgramContractTests(unittest.TestCase):
    def test_method_binding(self):
        from mirage.contracts.programs import bind_call
        class Operand:
            def method(self, other): pass
        obj = Operand()
        self.assertTrue(bind_call(obj.method, (1,), {}).established)
        self.assertTrue(bind_call(Operand.method, (obj, 1), {}).established)
        self.assertEqual(bind_call(Operand.method, (1,), {}).status, EvidenceStatus.REJECTED)
        self.assertEqual(bind_call(obj.method, (), {}).status, EvidenceStatus.REJECTED)

    def test_checkpoint_payload_excludes_code(self):
        from mirage.contracts.programs import validate_checkpoint_data
        validate_checkpoint_data({"parameters": [1, 2], "step": 3}, lambda x: False)
        with self.assertRaises(ValueError):
            validate_checkpoint_data({"forward": lambda x: x}, lambda x: False)

    def test_seed_is_not_sampling_or_determinism_proof(self):
        from mirage.contracts.programs import initialization_call_kind
        self.assertEqual(initialization_call_kind("torch.manual_seed"), "seed_setting")
        self.assertEqual(initialization_call_kind("torch.randn"), "sampling")
        self.assertEqual(initialization_call_kind("torch.nn.Linear"), "unresolved")

    def test_version_and_schedule(self):
        from mirage.contracts.programs import RunOptions
        from mirage.contracts.exploration import stateful_input_profile
        self.assertEqual(RunOptions("x", "fake", strategy="stateful", contract_version=4).strategy,
                         "stateful")
        with self.assertRaises(ValueError): RunOptions("x", "fake", strategy="stateful")
        a = [stateful_input_profile(i).requested for i in range(10)]
        self.assertEqual(len(set(a)), 10)
        self.assertEqual(a[0], stateful_input_profile(10).requested)


class StateSamplingTests(__import__('unittest').TestCase):
    def test_observed_capabilities_not_api_names_choose_goals(self):
        from mirage.contracts.exploration import stateful_state_predicates
        self.assertEqual(stateful_state_predicates({},0),[])
        a={'backward_executed':True,'api':'unregistered.first'}
        b={'backward_executed':True,'api':'unregistered.second'}
        self.assertEqual(stateful_state_predicates(a,0),stateful_state_predicates(b,0))
        self.assertEqual(stateful_state_predicates(a,2)[0]['dimension'],'execution_mode')
        self.assertEqual(stateful_state_predicates({'transition_signature':['context_exit']},0)[0]['dimension'],'context')


class JointStateSamplingTests(__import__('unittest').TestCase):
    def test_joint_goals_do_not_prescribe_actions(self):
        from mirage.contracts.exploration import stateful_diversity_predicates
        core=[dict(dimension='gradient',field='backward_count',expected=2,checkpoint='final')]
        a=stateful_diversity_predicates(core,0);b=stateful_diversity_predicates(core,1)
        self.assertEqual(a[0],b[0]);self.assertEqual(len(core),1)
        self.assertEqual((a[1]['expected'],b[1]['expected']),(1,2))
        self.assertEqual(stateful_diversity_predicates([],0),[])
        self.assertEqual(stateful_diversity_predicates(a,1),a)
        self.assertNotIn('actions',str(a))


class CoverageFeedbackTests(__import__('unittest').TestCase):
    def test_coverage_feedback_cannot_repair_suspected_failures(self):
        from mirage.inspector.stateful import input_category_feedback
        outcome={'inspection':{'valid':True},'verdict':'pass','input_coverage':{
            'requested':['small_finite'],'target_observed':{'ordinary':[]},
            'operand_measurements':[{'call':0,'path':['args',0],'facts':{'small_threshold':1e-19}}]}}
        self.assertEqual(input_category_feedback(outcome)['missing_categories'],['small_finite'])
        self.assertEqual(outcome['verdict'],'pass')
        self.assertIsNone(input_category_feedback(dict(outcome,verdict='suspected_failure')))
        self.assertIsNone(input_category_feedback(dict(outcome,inspection={'valid':False})))
