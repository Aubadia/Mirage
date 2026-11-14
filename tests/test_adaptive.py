"""Evidence/cost scheduling and common metrics for stateful search treatments."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mirage.sampler.policy import FeedbackSampler
from mirage.artifacts.report import summarize_search, verified_search_units
from mirage.contracts.capabilities import SEMANTIC_COVERAGE_VERSION


def record(unit='a', *, valid=True, seconds=2, verdict='pass'):
    return {'api':'torch.sin', 'executable':True, 'api_executed':True,
            'mutation_obligations_valid':True, 'started_seconds':0, 'finished_seconds':seconds,
            'verdict':verdict, 'relation':'representation', 'transition_signature':['target', 'backward'],
            'generation_usage':{'input_tokens':10, 'output_tokens':20},
            'inspection':{'valid':valid, 'semantic_coverage':{'eligible':valid,
                'version':SEMANTIC_COVERAGE_VERSION, 'signatures':[{'api':'torch.sin','unit':unit}]}}}


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.a = {'api':'torch.sin', 'parent':'a', 'kind':'run_tail'}
        self.b = {'api':'torch.sin', 'parent':'b', 'kind':'initialization'}

    def test_valid_novel_and_cheaper_candidates_gain_probability(self):
        sampler = FeedbackSampler(seed=0)
        for i in range(4):
            sampler.observe(self.a, record(str(i), seconds=1))
            sampler.observe(self.b, record(valid=False, seconds=20))
        _, decision = sampler.choose([self.a,self.b])
        self.assertGreater(decision['candidates'][0]['probability'], decision['candidates'][1]['probability'])
        self.assertGreaterEqual(decision['candidates'][1]['probability'], .05)
        self.assertAlmostEqual(sum(c['probability'] for c in decision['candidates']), 1.)
        self.assertEqual(decision['candidate_count'], 2)

    def test_failed_preservation_and_unverified_state_receive_no_gain(self):
        sampler = FeedbackSampler()
        row = record(); row['mutation_obligations_valid'] = False
        feedback = sampler.observe(self.a, row)
        self.assertEqual(feedback['bounded_gain'], 0)
        self.assertEqual(feedback['measured_seconds'], 2)
        self.assertEqual(sampler.observe(self.a, record(valid=False))['new_units'], 0)
        self.assertEqual(sampler.observe(self.a, record())['new_units'], 1)

    def test_duplicate_units_do_not_repeat_novelty_reward(self):
        sampler = FeedbackSampler()
        self.assertEqual(sampler.observe(self.a, record())['bounded_gain'], 2)
        self.assertEqual(sampler.observe(self.a, record())['bounded_gain'], 0)
        row = record(); row['relation'] = 'execution_mode'
        self.assertEqual(sampler.observe(self.a, row)['bounded_gain'], 1)

    def test_transition_and_semantic_feedback_use_distinct_denominators(self):
        transition = FeedbackSampler(coverage_basis='transition')
        semantic = FeedbackSampler(coverage_basis='semantic')
        for sampler in (transition, semantic): sampler.observe(self.a, record('a'))
        self.assertEqual(transition.observe(self.a, record('b'))['new_units'], 0)
        self.assertEqual(semantic.observe(self.a, record('b'))['new_units'], 1)

    def test_coverage_only_is_valid_without_an_applicable_mr(self):
        sampler = FeedbackSampler()
        feedback = sampler.observe(self.a, record(verdict='inapplicable'))
        self.assertTrue(feedback['verified'])
        self.assertEqual(feedback['bounded_gain'], 1)
        self.assertEqual(feedback['new_applicable_units'], 0)

    def test_identical_seed_and_feedback_reproduce_choices(self):
        left, right = FeedbackSampler(seed=17), FeedbackSampler(seed=17)
        for sampler in (left, right): sampler.observe(self.a, record())
        self.assertEqual([left.choose([self.a,self.b])[0] for _ in range(10)],
                         [right.choose([self.a,self.b])[0] for _ in range(10)])

    def test_invalid_resources_and_empty_candidates_are_rejected(self):
        for epsilon in (-1, float('nan'), 2):
            with self.assertRaises(ValueError): FeedbackSampler(epsilon=epsilon)
        with self.assertRaises(ValueError): FeedbackSampler().choose([])
        with self.assertRaises(ValueError): FeedbackSampler().observe(self.a, record(seconds=-1))

    def test_family_evidence_transfers_to_new_parents(self):
        sampler = FeedbackSampler()
        for i in range(4): sampler.observe(self.a, record(str(i), seconds=1))
        new = dict(self.a, parent='new')
        _, decision = sampler.choose([new,self.b])
        self.assertGreater(decision['candidates'][0]['score'], decision['candidates'][1]['score'])

    def test_report_counts_common_units_without_claiming_defects(self):
        rows = [record('a'), record('b', verdict='inapplicable'), record('c', valid=False)]
        metrics = summarize_search({'records':rows,'elapsed_seconds':12})
        self.assertEqual(metrics['runtime_valid'], 2)
        self.assertEqual(metrics['semantic_units'], 2)
        self.assertEqual(metrics['transition_units'], 1)
        self.assertEqual(metrics['coverage_only_cases'], 1)
        self.assertEqual(metrics['applicable_comparisons'], 1)
        self.assertIsNone(metrics['confirmed_defects'])
        rows[0]['generation_usage'] = None
        self.assertIsNone(summarize_search({'records':rows})['semantic_units_per_1000_tokens'])

    def test_numerical_report_does_not_invent_mr_applicability(self):
        row = record(verdict='suspected_failure')
        row['mr_verdict'] = 'inapplicable'
        row['numerical_violation_members'] = ['followup']
        row['gradient_audits'] = {
            'source': {'status': 'agreement', 'cost': {'seconds': 0.1}},
            'followup': {'status': 'violation', 'cost': {'seconds': 0.2}}}
        metrics = summarize_search({'records': [row], 'elapsed_seconds': 1})
        self.assertEqual(metrics['applicable_comparisons'], 0)
        self.assertEqual(metrics['numerically_reported_cases'], 1)
        self.assertAlmostEqual(metrics['numerical_audit_seconds'], 0.3)
        self.assertEqual(metrics['numerical_audit_statuses'], {'agreement': 1, 'violation': 1})
        self.assertIsNone(metrics['confirmed_defects'])

    def test_corpus_transition_ablation_and_explicit_selection(self):
        from mirage.artifacts.corpus import Corpus
        from test_corpus import fixture_bundle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = Corpus(root, coverage_basis='transition')
            first = corpus.admit(*fixture_bundle(root, 0))
            self.assertEqual(first['reason'], 'new_transition_coverage')
            second = corpus.admit(*fixture_bundle(root, 1, coverage='other', seconds=20))
            self.assertFalse(second['admitted'])
            self.assertEqual(corpus.select('torch.sin', seed_id=first['seed_id'])['id'], first['seed_id'])
            with self.assertRaises(ValueError): corpus.select('torch.sin', seed_id='absent')
            with self.assertRaises(ValueError): corpus.select('torch.cos', seed_id=first['seed_id'])
            self.assertEqual(Corpus(root).coverage_basis, 'transition')
            with self.assertRaises(ValueError): Corpus(root, coverage_basis='semantic')

    def test_adaptive_and_full_workflows_keep_per_case_decisions(self):
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'PYTHONPATH':str(Path(__file__).parent)+os.pathsep+os.environ.get('PYTHONPATH','')}):
            for mode, basis in (('local','semantic'), ('full','transition')):
                options = RunOptions(str(Path(tmp)/mode), 'test_corpus:local_fixture_provider', cases=3,
                                     time_limit=300, worker_timeout=60, strategy='stateful', contract_version=4)
                run = run_stateful(options, ('torch.sin',), predicates=[], coverage_refinement=False,
                                   generation_mode=mode, coverage_basis=basis)
                self.assertEqual(len(run['records']), 3)
                self.assertEqual(run['search_policy'], 'adaptive')
                self.assertTrue(all('search_feedback' in r and 'selection' in r for r in run['records']))
                self.assertTrue(all(r['selection']['candidate_count'] > 0 for r in run['records']))
                if mode == 'full': self.assertTrue(all(r['generation_mode'] == 'full' for r in run['records']))
                self.assertGreater(run['search_metrics']['runtime_valid'], 0)


class GoalFeedbackTests(unittest.TestCase):
    def test_goal_families_have_separate_cost_and_validity_feedback(self):
        sampler = FeedbackSampler()
        base = {'api':'torch.sin','parent':'seed','kind':'planned'}
        good, bad = dict(base, goal='gradient.backward_count'), dict(base, goal='representation.negative')
        for index in range(3):
            sampler.observe(good, record(str(index), seconds=1))
            sampler.observe(bad, record(valid=False, seconds=15))
        _, selection = sampler.choose([good,bad])
        self.assertGreater(selection['candidates'][0]['probability'], selection['candidates'][1]['probability'])

    def test_planning_usage_is_included_in_total_cost(self):
        row = record()
        row['planning_usage'] = {'input_tokens':100,'output_tokens':50}
        self.assertEqual(summarize_search({'records':[row]})['known_tokens'], 180)
