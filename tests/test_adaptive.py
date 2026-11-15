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

    def test_goal_feedback_transfers_without_merging_exact_candidates(self):
        sampler = FeedbackSampler()
        known = dict(self.a, kind='planned', goal='representation.layout:first', feedback_family='representation.layout')
        for index in range(4):
            sampler.observe(known, record(str(index), seconds=1))
        unseen = dict(known, parent='new', goal='representation.layout:other')
        unrelated = dict(unseen, goal='context.grad:third', feedback_family='context.grad')
        _, decision = sampler.choose([unseen, unrelated])
        self.assertGreater(decision['candidates'][0]['score'], decision['candidates'][1]['score'])
        self.assertEqual(decision['candidate_count'], 2)
        self.assertNotIn(('torch.sin', 'new', 'planned', unseen['goal']), sampler.parents)

    def test_state_repair_progress_requires_new_attained_obligations(self):
        from mirage.workflow import state_repair_progress, state_repair_stalled
        first = {'predicates': [
            {'field': 'extent', 'expected': 3, 'observed': 2, 'attained': False},
            {'field': 'shared_storage', 'expected': True, 'observed': False, 'attained': False}]}
        before = state_repair_progress(first)
        changed_numbers = deepcopy(first)
        changed_numbers['predicates'][0]['observed'] = 8
        self.assertTrue(state_repair_stalled(before, state_repair_progress(changed_numbers)))
        changed_numbers['predicates'][1].update(observed=True, attained=True)
        self.assertFalse(state_repair_stalled(before, state_repair_progress(changed_numbers)))
        changed_goal = deepcopy(first)
        changed_goal['predicates'][0]['expected'] = 8
        self.assertFalse(state_repair_stalled(before, state_repair_progress(changed_goal)))
        self.assertFalse(state_repair_stalled(None, before))
        self.assertIsNone(state_repair_progress({'predicates': []}))

    def test_repair_payload_equality_preserves_actual_source_differences(self):
        from mirage.workflow import same_repair_payload
        self.assertTrue(same_repair_payload('{', '{'))
        self.assertTrue(same_repair_payload('{"source":"x=1"}', '{ "source": "x=1" }'))
        self.assertFalse(same_repair_payload(None, '{}'))
        self.assertFalse(same_repair_payload('{"source":"x=1"}', '{"source":"x=2"}'))

    def test_report_counts_common_units_without_claiming_defects(self):
        rows = [record('a'), record('b', verdict='inapplicable'), record('c', valid=False)]
        metrics = summarize_search({'records':rows,'elapsed_seconds':12})
        self.assertEqual(metrics['runtime_valid'], 2)
        self.assertEqual(metrics['semantic_units'], 2)
        self.assertEqual(metrics['transition_units'], 1)
        self.assertEqual(metrics['coverage_only_cases'], 1)
        self.assertEqual(metrics['applicable_comparisons'], 1)
        self.assertEqual(metrics['applicable_comparisons_per_second'], 1 / 12)
        self.assertEqual(metrics['goal_attained_applicable_cases'], 0)
        self.assertIsNone(metrics['confirmed_defects'])
        rows[0]['generation_usage'] = None
        self.assertIsNone(summarize_search({'records':rows})['semantic_units_per_1000_tokens'])

    def test_optimizer_feedback_is_distinct_and_report_keeps_old_units_separate(self):
        from mirage.contracts.capabilities import OPTIMIZER_COVERAGE_VERSION
        first, second, invalid = record(), record(), record(valid=False)
        chain = {'version': OPTIMIZER_COVERAGE_VERSION, 'eligible': True,
                 'signatures': [{'version': OPTIMIZER_COVERAGE_VERSION, 'api': 'torch.sin',
                                 'parameter_ordinal': 0, 'gradient_contiguous': False}]}
        for row in (second, invalid):
            row['inspection']['optimizer_coverage'] = deepcopy(chain)
            row['configuration'] = {'predicates': [
                {'dimension': 'gradient_representation', 'field': 'contiguous'},
                {'dimension': 'optimizer', 'field': 'completed'}]}
            row['inspection']['target_state_reached'] = True
            row['state_plan'] = {'family': 'gradient_representation'}
            row['relation'] = 'optimizer_gradient_representation'
        sampler = FeedbackSampler()
        self.assertEqual(sampler.observe(self.a, first)['new_units'], 1)
        self.assertEqual(sampler.observe(self.a, second)['new_units'], 1)
        self.assertEqual(sampler.observe(self.a, invalid)['new_units'], 0)
        metrics = summarize_search({'records': [first, second, invalid], 'elapsed_seconds': 10})
        self.assertEqual(metrics['semantic_units'], 1)
        optimizer = metrics['optimizer_chain']
        self.assertEqual(optimizer['source_units'], 1)
        self.assertEqual(optimizer['source_linked_by_origin'], {'planned': 1})
        self.assertEqual(optimizer['gradient_goal_attempts'], 2)
        self.assertEqual(optimizer['gradient_update_conjunction_attained_cases'], 1)
        self.assertEqual(optimizer['unmeasured_source_cases'], 1)

    def test_followup_optimizer_chain_never_credits_source_attainment(self):
        from mirage.contracts.capabilities import OPTIMIZER_COVERAGE_VERSION
        row = record()
        row['followup_optimizer_coverage'] = {
            'version': OPTIMIZER_COVERAGE_VERSION, 'eligible': True,
            'signatures': [{'version': OPTIMIZER_COVERAGE_VERSION, 'parameter_ordinal': 0}]}
        optimizer = summarize_search({'records': [row]})['optimizer_chain']
        self.assertEqual(optimizer['source_linked_cases'], 0)
        self.assertEqual(optimizer['followup_linked_cases'], 1)
        self.assertEqual(optimizer['unmeasured_source_cases'], 1)

    def test_optimizer_unknown_count_is_distinct_from_verified_zero(self):
        from mirage.contracts.capabilities import OPTIMIZER_COVERAGE_VERSION
        absent, unknown, zero = record(), record(), record()
        for row, verified in ((unknown, False), (zero, True)):
            row['inspection']['optimizer_coverage'] = {
                'version': OPTIMIZER_COVERAGE_VERSION, 'count_verified': verified,
                'target_update_count': 0, 'eligible': False}
        optimizer = summarize_search({'records': [absent, unknown, zero]})['optimizer_chain']
        self.assertEqual(optimizer['unmeasured_source_cases'], 1)
        self.assertEqual(optimizer['measured_count_unknown_cases'], 1)
        self.assertEqual(optimizer['measured_verified_zero_count_cases'], 1)
        self.assertEqual(optimizer['source_linked_cases'], 0)

    def test_joint_coverage_feedback_and_report_use_verified_units(self):
        first, second, invalid = record(), record(), record(valid=False)
        unary = {'version': SEMANTIC_COVERAGE_VERSION, 'unit_kind': 'unary', 'api': 'torch.sin', 'unit': 'a'}
        pair = {'version': SEMANTIC_COVERAGE_VERSION, 'unit_kind': 'joint', 'api': 'torch.sin',
                'extent_order': 'lt', 'shared_storage': False}
        for row, signatures in ((first, [unary]), (second, [unary, pair]), (invalid, [pair])):
            row['inspection']['semantic_coverage']['signatures'] = deepcopy(signatures)
        for row in (second, invalid):
            row['configuration'] = {'predicates': [{'dimension': 'tensor_relation', 'field': 'extent_order'}]}
            row['inspection']['target_state_reached'] = True
        second['inspection']['semantic_coverage']['measurement_counts'] = {'joint_candidate_units': 1}
        sampler = FeedbackSampler()
        self.assertEqual(sampler.observe(self.a, first)['new_units'], 1)
        self.assertEqual(sampler.observe(self.a, second)['new_units'], 1)
        self.assertEqual(sampler.observe(self.a, second)['new_units'], 0)
        self.assertEqual(sampler.observe(self.b, invalid)['new_units'], 0)
        metrics = summarize_search({'records': [first, second, invalid], 'elapsed_seconds': 10})
        self.assertEqual(metrics['semantic_units'], 2)
        self.assertEqual((metrics['unary_semantic_units'], metrics['joint_semantic_units']), (1, 1))
        self.assertEqual(metrics['joint_measurement_counts'], {'joint_candidate_units': 1})
        self.assertEqual(metrics['relational_goal_attempts'], 2)
        self.assertEqual(metrics['relational_goal_attained_cases'], 1)
        self.assertEqual(metrics['relational_goal_attainment_rate'], .5)
        self.assertEqual(metrics['joint_units_per_second'], .1)
        self.assertIsNone(metrics['joint_coverage_denominator'])

    def test_legacy_coverage_is_identified_and_not_mixed_into_v2_feedback(self):
        current, old = record(), record('old')
        old['inspection']['semantic_coverage']['version'] = 'state-semantic-1'
        self.assertFalse(verified_search_units(old))
        metrics = summarize_search({'records': [current, old], 'elapsed_seconds': 1})
        self.assertEqual(metrics['runtime_valid'], 1)
        self.assertEqual(metrics['observed_coverage_versions'], {SEMANTIC_COVERAGE_VERSION: 1, 'state-semantic-1': 1})
        self.assertEqual(metrics['semantic_coverage_version'], SEMANTIC_COVERAGE_VERSION)
        self.assertIsNone(metrics['relational_goal_attainment_rate'])

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


def optimizer_configuration_fixture_provider():
    """Authored controller test: any unnecessary source regeneration fails loudly."""
    from mirage.generator.provider import Response, Stage
    from test_optimizer_runtime import SOURCE
    class Provider:
        def generate(self, request, *, deadline=None):
            if request.stage == Stage.STATE_GOAL:
                goal = request.context['goal_catalog'][0]
                return Response(json.dumps({'plans': [{'status': 'planned', 'goal_ids': [goal['id']],
                    'actions': ['Apply the configured gradient representation and preserve the native update'],
                    'preconditions': ['Verified target-linked native gradient update'], 'regions': ['run_tail'],
                    'rationale': 'Source-owned backward and step already exist', 'blockers': []}],
                    'examined_goal_ids': [goal['id']]}))
            if request.context.get('canonical_source'):
                raise AssertionError('Configuration-only reuse must not dispatch program generation')
            return Response(json.dumps({'source': SOURCE}))
    return Provider()


class OptimizerConfigurationTests(unittest.TestCase):
    def test_reuse_guard_requires_matching_verified_companion_and_inputs(self):
        from dataclasses import asdict
        from mirage.contracts.capabilities import OPTIMIZER_COVERAGE_VERSION, StatePredicate
        from mirage.inspector.stateful import inspect_optimizer
        from mirage.workflow import optimizer_configuration_reusable
        from test_optimizer_inspection import optimizer_record, gradient_predicate
        coverage = inspect_optimizer(optimizer_record(), valid=True)
        gradient = asdict(gradient_predicate(expected=True))
        companion = asdict(StatePredicate('optimizer', 'completed', True,
            'optimizer:0/step:0/post_update', parameter_key='p', call_index=0,
            operand_path=['args', 0], relation='target_gradient_update'))
        parent = {'configuration': {'api': 'torch.sin', 'device': 'cpu', 'seed': 0, 'requires_backward': True},
                  'optimizer_coverage_version': OPTIMIZER_COVERAGE_VERSION,
                  'optimizer_coverage': coverage['signatures'],
                  'evidence': {'optimizer_witnesses': coverage['witnesses'], 'optimizer_target_update_count': 1}}
        config = dict(parent['configuration'], predicates=[gradient, companion],
                      optimizer_gradient_action={'step_index': 0, 'parameter_key': 'p', 'action': 'canonical'})
        self.assertTrue(optimizer_configuration_reusable(parent, config))
        mixed_paths = deepcopy(config)
        mixed_paths['predicates'][1]['operand_path'] = ('args', 0)
        self.assertTrue(optimizer_configuration_reusable(parent, mixed_paths))
        for changed in (dict(config, seed=1), dict(config, predicates=[gradient]),
                        dict(config, optimizer_gradient_action={'step_index': 0, 'parameter_key': 'p', 'action': 'offset'})):
            self.assertFalse(optimizer_configuration_reusable(parent, changed))
        unknown = deepcopy(parent)
        unknown['evidence']['optimizer_witnesses'][0]['relations']['target_gradient_update']['status'] = 'unknown'
        self.assertFalse(optimizer_configuration_reusable(unknown, config))
        self.assertFalse(optimizer_configuration_reusable(dict(parent, optimizer_coverage=[]), config))

    def test_real_controller_reuses_source_and_revalidates_layout_update_conjunction(self):
        from dataclasses import asdict
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        from test_optimizer_inspection import gradient_predicate
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', '')}):
            options = RunOptions(str(Path(tmp) / 'run'), 'test_adaptive:optimizer_configuration_fixture_provider',
                                 cases=2, time_limit=300, worker_timeout=60, seed=0,
                                 strategy='stateful', contract_version=4)
            run = run_stateful(options, ('torch.sin',),
                               predicates=[asdict(gradient_predicate())], coverage_refinement=False,
                               search_policy='round_robin')
            self.assertEqual(len(run['records']), 2, run)
            first, second = run['records']
            self.assertEqual(run['provider_calls'], 2)
            self.assertTrue(second.get('source_reused'), second)
            self.assertFalse(second['program_generation_dispatched'])
            self.assertEqual(second['generation_usage'], {'input_tokens': 0, 'output_tokens': 0})
            self.assertEqual(second['generation_mode'], 'configuration')
            self.assertTrue(second['inspection']['valid'], second['inspection'])
            self.assertEqual(second['optimizer_certificate_diagnostic']['status'], 'verified')
            self.assertEqual(second['relation'], 'optimizer_gradient_representation')
            self.assertEqual(second['verdict'], 'pass', second)
            self.assertEqual(len(second['inspection']['predicates']), 2)
            self.assertTrue(all(p['attained'] for p in second['inspection']['predicates']))
            self.assertEqual((Path(first['directory']) / 'program.py').read_bytes(),
                             (Path(second['directory']) / 'program.py').read_bytes())


def library_configuration_fixture_provider():
    from mirage.generator.provider import Response, Stage
    from test_optimizer_runtime import SOURCE
    class Provider:
        def generate(self, request, *, deadline=None):
            if request.stage == Stage.STATE_GOAL or request.context.get('canonical_source'):
                raise AssertionError('Known configuration must not dispatch either provider stage')
            return Response(json.dumps({'source': SOURCE}))
    return Provider()


def failed_library_planning_fixture_provider():
    from mirage.generator.provider import Response, Stage
    from test_optimizer_runtime import SOURCE
    class Provider:
        def generate(self, request, *, deadline=None):
            if request.stage == Stage.STATE_GOAL:
                return Response('{malformed planning reply')
            if request.context.get('canonical_source'):
                raise AssertionError('Known configuration must reuse source after failed LLM planning')
            return Response(json.dumps({'source': SOURCE}))
    return Provider()


def unresolved_library_planning_fixture_provider():
    from mirage.generator.provider import Response, Stage
    from test_optimizer_runtime import SOURCE
    class Provider:
        def generate(self, request, *, deadline=None):
            if request.stage == Stage.STATE_GOAL:
                identifier = request.context['goal_catalog'][0]['id']
                return Response(json.dumps({'plans': [{'status': 'unresolved', 'goal_ids': [identifier],
                    'actions': [], 'preconditions': [], 'regions': [],
                    'rationale': 'Authored planner abstains from editing the source',
                    'blockers': ['No source edit is proposed by this planner']}],
                    'examined_goal_ids': [identifier]}))
            if request.context.get('canonical_source'):
                raise AssertionError('Known action must not need source generation after unresolved planning')
            return Response(json.dumps({'source': SOURCE}))
    return Provider()


class LibraryConfigurationWorkflowTests(unittest.TestCase):
    def parent(self):
        from mirage.contracts.capabilities import OPTIMIZER_COVERAGE_VERSION
        from mirage.inspector.stateful import inspect_optimizer
        from test_optimizer_inspection import optimizer_record
        coverage = inspect_optimizer(optimizer_record(), valid=True)
        return {'id': 'parent', 'source_sha256': 'static-fixture-source-hash',
                'configuration': {'api': 'torch.sin', 'device': 'cpu', 'seed': 0, 'requires_backward': True},
                'optimizer_coverage_version': OPTIMIZER_COVERAGE_VERSION,
                'optimizer_coverage': coverage['signatures'],
                'evidence': {'optimizer_witnesses': coverage['witnesses'], 'optimizer_target_update_count': 1}}

    def test_pool_caches_rejected_metadata_and_invalidates_changed_evidence(self):
        from mirage.generator.state_plan import goal_catalog, library_configuration_plans
        from mirage.workflow import library_configuration_pool
        parent, cache = self.parent(), {}
        parent['optimizer_coverage'] = []
        catalog = goal_catalog(parent)
        with patch('mirage.generator.state_plan.library_configuration_plans', wraps=library_configuration_plans) as planning:
            first = library_configuration_pool(parent, catalog, cache)
            second = library_configuration_pool(parent, catalog, cache)
            self.assertIs(first, second)
            self.assertEqual(planning.call_count, 1)
            self.assertEqual(first['accepted'], [])
            self.assertTrue(first['rejected'])
            parent['optimizer_coverage'] = self.parent()['optimizer_coverage']
            third = library_configuration_pool(parent, catalog, cache)
            self.assertEqual(planning.call_count, 2)
            self.assertTrue(third['accepted'])
            self.assertNotEqual(first['cache_key'], third['cache_key'])

    def test_enumeration_does_not_consume_goals_and_selected_ids_filter_only_their_plan(self):
        from mirage.generator.state_plan import goal_catalog
        from mirage.workflow import library_configuration_pool, optimizer_configuration_reusable
        parent, attempted = self.parent(), set()
        pool = library_configuration_pool(parent, goal_catalog(parent), {})
        self.assertGreaterEqual(len(pool['accepted']), 2)
        self.assertEqual(attempted, set())
        chosen = pool['accepted'][0]
        attempted.update(chosen['goal_ids'])
        remaining = [p for p in pool['accepted'] if not set(p['goal_ids']) & attempted]
        self.assertEqual(len(remaining), len(pool['accepted']) - 1)
        config = dict(parent['configuration'], predicates=chosen['predicates'],
                      optimizer_gradient_action=chosen['optimizer_gradient_action'])
        self.assertTrue(optimizer_configuration_reusable(parent, config))
        self.assertFalse(optimizer_configuration_reusable(parent, dict(config, seed=1)))
        self.assertFalse(optimizer_configuration_reusable(parent, dict(config, predicates=config['predicates'][:1])))

    def test_valid_unresolved_ids_do_not_consume_library_actions_but_selected_ids_do(self):
        from mirage.generator.state_plan import goal_catalog, validate_plans
        from mirage.workflow import library_configuration_pool, available_library_plans
        parent, cache = self.parent(), {}
        catalog = goal_catalog(parent)
        pool = library_configuration_pool(parent, catalog, cache)
        identifier = pool['accepted'][0]['goal_ids'][0]
        accepted, rejected = validate_plans(json.dumps({'plans': [{
            'status': 'unresolved', 'goal_ids': [identifier], 'actions': [], 'regions': [],
            'preconditions': [], 'blockers': ['Planner has no source edit'], 'rationale': 'Abstain'}],
            'examined_goal_ids': [identifier]}), catalog)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]['status'], 'unresolved')
        examined = set(rejected[0]['examined_goal_ids'])
        selected = set()
        self.assertNotIn(identifier, [g['id'] for g in catalog if g['id'] not in examined])
        self.assertTrue(any(identifier in p['goal_ids'] for p in available_library_plans(pool, selected)))
        selected.add(identifier)
        self.assertFalse(any(identifier in p['goal_ids'] for p in available_library_plans(pool, selected)))
        self.assertEqual(library_configuration_pool(parent, catalog, cache), pool)

    def test_report_separates_known_proposals_from_provider_calls_and_runtime_coverage(self):
        row = record(valid=False)
        row.update(proposal_origin='library_known_action', planning_dispatched=False,
                   program_generation_dispatched=False, source_reused=True,
                   generation_usage={'input_tokens': 0, 'output_tokens': 0},
                   library_configuration_outcome='not_verified')
        report = summarize_search({'records': [row], 'library_planning': {
            'accepted_proposals': 2, 'rejected_proposals': 1, 'metadata_seconds': .001}})
        self.assertEqual(report['planning_calls'], 0)
        self.assertEqual(report['program_generation_calls'], 0)
        self.assertEqual(report['library_configuration_cases'], 1)
        self.assertEqual(report['runtime_valid'], 0)
        self.assertEqual(report['known_tokens'], 0)
        self.assertEqual(report['library_configuration_outcomes'], {'not_verified': 1})
        self.assertEqual(report['library_planning']['accepted_proposals'], 2)

    def run_library(self, tmp, *, failed_planning=False, unresolved_planning=False, reject_child=False):
        from dataclasses import asdict
        from mirage.workflow import run_stateful, run_worker
        from mirage.contracts.programs import RunOptions
        from test_optimizer_inspection import gradient_predicate
        original_choose = FeedbackSampler.choose
        outer, worker_calls = [], []
        def choose(sampler, candidates, *, reason='adaptive'):
            if candidates and 'plan_index' not in candidates[0]:
                outer.append(deepcopy(candidates))
                desired = ('planned' if failed_planning and len(outer) in (2, 3)
                           or unresolved_planning and len(outer) == 2 else 'configuration')
                selected = [c for c in candidates if c['kind'] == desired]
                if selected:
                    result, info = original_choose(sampler, selected, reason='authored_branch_control')
                    info['unfiltered_candidates'] = deepcopy(candidates)
                    return result, info
            return original_choose(sampler, candidates, reason=reason)
        def worker(request, directory, timeout, *args, **kwargs):
            result = run_worker(request, directory, timeout, *args, **kwargs)
            if request.get('mode') == 'execute_stateful_pair':
                worker_calls.append(request)
                if reject_child and len(worker_calls) == 2:
                    summary = Path(directory) / 'summary.json'
                    outcome = json.loads(summary.read_text())
                    outcome['inspection'].update(valid=False, target_state_reached=False)
                    for check in outcome['inspection']['predicates']:
                        check['attained'] = False
                    outcome.update(verdict='invalid', mr_verdict='inapplicable')
                    summary.write_text(json.dumps(outcome))
            return result
        provider = ('failed_library_planning_fixture_provider' if failed_planning else
                    'unresolved_library_planning_fixture_provider' if unresolved_planning else
                    'library_configuration_fixture_provider')
        with patch.dict(os.environ, {'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', '')}), \
                patch.object(FeedbackSampler, 'choose', choose), patch('mirage.workflow.run_worker', worker):
            options = RunOptions(str(Path(tmp) / 'run'), 'test_adaptive:' + provider,
                cases=4 if failed_planning else 3 if unresolved_planning else 2, time_limit=600, worker_timeout=60,
                seed=0, strategy='stateful', contract_version=4)
            return run_stateful(options, ('torch.sin',), predicates=[asdict(gradient_predicate())],
                                coverage_refinement=False), outer

    def test_real_library_branch_reuses_source_without_either_model_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, outer = self.run_library(tmp)
            parent, child = result['records']
            self.assertEqual({c['kind'] for c in outer[1]}, {'fresh', 'planned', 'configuration'})
            self.assertEqual(sum(c['kind'] == 'configuration' for c in outer[1]), 1)
            self.assertEqual(result['provider_calls'], 1)
            self.assertEqual(child['candidate']['kind'], 'configuration')
            self.assertEqual(child['proposal_origin'], 'library_known_action')
            self.assertNotIn('planning_usage', child)
            self.assertFalse(child['program_generation_dispatched'])
            self.assertTrue(child['mutation_obligations_valid'])
            self.assertEqual(child['library_configuration_outcome'], 'verified')
            self.assertTrue(all(p['attained'] for p in child['inspection']['predicates']))
            self.assertEqual({p['dimension'] for p in child['inspection']['predicates']},
                             {'gradient_representation', 'optimizer'})
            self.assertEqual((Path(parent['directory']) / 'program.py').read_bytes(),
                             (Path(child['directory']) / 'program.py').read_bytes())

    def test_two_failed_llm_plans_do_not_block_known_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, outer = self.run_library(tmp, failed_planning=True)
            self.assertEqual([r.get('planning_budget_failures') for r in result['records'][1:3]], [1, 2])
            self.assertEqual({c['kind'] for c in outer[3]}, {'fresh', 'configuration'})
            child = result['records'][3]
            self.assertEqual(child['candidate']['kind'], 'configuration')
            self.assertEqual(child['library_configuration_outcome'], 'verified')
            self.assertEqual(result['provider_calls'], 3)
            self.assertEqual(result['search_metrics']['planning_calls'], 2)

    def test_valid_unresolved_plan_does_not_remove_the_unselected_library_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, outer = self.run_library(tmp, unresolved_planning=True)
            self.assertEqual(len(result['records']), 3, result)
            _, unresolved, child = result['records']
            self.assertEqual(unresolved['status'], 'planning_unresolved')
            self.assertEqual({c['kind'] for c in outer[2]}, {'fresh', 'configuration'})
            self.assertEqual(child['state_plan']['goal_ids'],
                             unresolved['planning_unresolved'][0]['examined_goal_ids'])
            self.assertEqual(child['candidate']['kind'], 'configuration')
            self.assertEqual(child['library_configuration_outcome'], 'verified')
            self.assertEqual(result['provider_calls'], 2)
            self.assertNotIn('planning_usage', child)
            self.assertFalse(child['program_generation_dispatched'])

    def test_unverified_configuration_keeps_evidence_without_model_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self.run_library(tmp, reject_child=True)
            child = result['records'][1]
            self.assertEqual(result['provider_calls'], 1)
            self.assertEqual(child['library_configuration_outcome'], 'not_verified')
            self.assertEqual(child['repair_stopped'], 'library_configuration_evidence_retained')
            self.assertFalse(child['inspection']['valid'])
            pools = json.loads((Path(tmp) / 'run/library-proposals.json').read_text())
            self.assertEqual(len(pools[0]['execution_rejections']), 1)
            self.assertEqual(pools[0]['execution_rejections'][0]['reachability'], 'unassessed')
            self.assertTrue((Path(child['directory']) / 'execution/summary.json').exists())


class ConfigurationBudgetTests(unittest.TestCase):
    def evidence(self, seconds=30.):
        parent = {'id': 'p1', 'source_sha256': 'source', 'configuration': {'seed': 0}}
        row = {'corpus_admission': {'admitted': True, 'seed_id': 'p1'},
               'configuration': {'seed': 0}, 'inspection': {'valid': True},
               'mutation_obligations_valid': True,
               'process': {'outcome': 'completed', 'returncode': 0, 'elapsed_seconds': seconds}}
        return parent, row

    def test_known_configuration_uses_own_worker_and_preserves_general_quota(self):
        from mirage.workflow import configuration_dispatch_budget, affordable_candidates
        parent, row = self.evidence()
        unrelated = deepcopy(row)
        unrelated['corpus_admission']['seed_id'] = 'other'
        unrelated['process']['elapsed_seconds'] = 1000.
        budget = configuration_dispatch_budget(parent, [row, unrelated], 100.)
        self.assertEqual(budget['reserve_seconds'], 50.)
        self.assertEqual(budget['cost_basis'], 'verified_parent_worker')
        library = {'kind': 'configuration', 'dispatch_budget': budget}
        fresh, planned = {'kind': 'fresh'}, {'kind': 'planned'}
        self.assertEqual(affordable_candidates([fresh, planned, library], 60., 100.), [library])
        # The controller passes the quota-filtered list; it cannot replace a fresh quota.
        self.assertEqual(affordable_candidates([fresh], 60., 100.), [])
        self.assertEqual(affordable_candidates([library], 50., 100.), [])
        self.assertEqual(affordable_candidates([fresh, planned, library], 101., 100.), [fresh, planned, library])

    def test_unknown_invalid_or_unbound_timing_retains_whole_round_reserve(self):
        from mirage.workflow import configuration_dispatch_budget
        parent, original = self.evidence()
        variants = []
        for value in (None, True, float('nan'), float('inf'), 0, -1, '30'):
            row = deepcopy(original); row['process']['elapsed_seconds'] = value; variants.append(row)
        for location, key, value in ((('process',), 'outcome', 'timeout'),
                (('process',), 'returncode', 1), (('process',), 'returncode', False),
                (('configuration',), 'seed', 1), (('inspection',), 'valid', False),
                (('corpus_admission',), 'admitted', False), ((), 'mutation_obligations_valid', False)):
            row = deepcopy(original); target = row
            for item in location: target = target[item]
            target[key] = value; variants.append(row)
        for rows in ([row] for row in variants):
            result = configuration_dispatch_budget(parent, rows, 100.)
            self.assertEqual(result['reserve_seconds'], 100.)
            self.assertEqual(result['cost_basis'], 'whole_round_fallback')
        self.assertEqual(configuration_dispatch_budget(parent, [], 100.)['reserve_seconds'], 100.)

    def test_slowest_matching_measurement_and_existing_floor_are_retained(self):
        from mirage.workflow import configuration_dispatch_budget
        parent, first = self.evidence(20.)
        _, second = self.evidence(40.)
        self.assertEqual(configuration_dispatch_budget(parent, [first, second], 100.)['reserve_seconds'], 65.)
        _, short = self.evidence(.1)
        self.assertEqual(configuration_dispatch_budget(parent, [short], 100.)['reserve_seconds'], 10.)


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


def unresolved_fixture_provider():
    from mirage.generator.provider import Response
    from test_stateful import SOURCE
    class Provider:
        def generate(self, request, *, deadline=None):
            source = SOURCE.replace('y.sum().backward()',
                                    "raise RuntimeError('authored unresolved execution')")
            return Response(json.dumps({'source': source}))
    return Provider()


def contract_fixture_provider():
    from mirage.generator.provider import Response
    from test_stateful import SOURCE
    class Provider:
        def generate(self, request, *, deadline=None):
            source = SOURCE if request.previous_content else SOURCE.replace(
                '"parameters": {"x": x}', '"parameters": {"x": x+0}')
            return Response(json.dumps({'source': source}))
    return Provider()


def unchanged_repair_fixture_provider():
    from mirage.generator.provider import Response
    from test_stateful import SOURCE
    class Provider:
        def generate(self, request, *, deadline=None):
            source = SOURCE.replace('"parameters": {"x": x}', '"parameters": {"x": x+0}')
            if request.previous_content and os.environ.get('MIRAGE_TEST_COSMETIC_REPAIR') == '1':
                source += '\n# Cosmetic change leaves the rejected AST unchanged.\n'
            return Response(json.dumps({'source': source}))
    return Provider()


def callable_fixture_provider():
    from mirage.generator.provider import Response
    source = '''import torch
def operation(x):
    return x.cos() + x.square()
def build(device, seed):
    x = torch.tensor([0.25, 0.75], device=device, requires_grad=True)
    return {'inputs': {'x': x}, 'parameters': {'x': x}}
def run(ctx, probe):
    x = ctx['inputs']['x']
    compiled = probe.call(torch.compile, operation, backend='eager', fullgraph=True)
    y = probe.call(compiled, x)
    y.sum().backward()
    return {'output': y, 'gradient': x.grad}
'''
    class Provider:
        def generate(self, request, *, deadline=None):
            if request.context.get('mutation_contract'):
                replacement = ("y.sum().backward()\n" if os.environ.get('MIRAGE_TEST_RETAIN_CALLABLE') == '1' else '')
                replacement += "z = compiled(x)\nz.sum().backward()\nreturn {'output': z, 'gradient': x.grad}"
                return Response(json.dumps({'source': replacement}))
            return Response(json.dumps({'source': source}))
    return Provider()


class StatefulTriageTests(unittest.TestCase):
    def test_new_callable_backward_cannot_replace_protected_parent_invocation(self):
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as tmp:
            for retain in (False, True):
                with self.subTest(retain=retain), patch.dict(os.environ, {
                        'PYTHONPATH':str(Path(__file__).parent)+os.pathsep+os.environ.get('PYTHONPATH',''),
                        'MIRAGE_TEST_RETAIN_CALLABLE':str(int(retain))}):
                    options = RunOptions(str(Path(tmp)/str(retain)), 'test_adaptive:callable_fixture_provider',
                                         cases=2, time_limit=180, worker_timeout=40,
                                         strategy='stateful', contract_version=4)
                    run = run_stateful(options, ('torch.compile',), predicates=[],
                                       coverage_refinement=False, search_policy='round_robin')
                    parent, child = run['records']
                    self.assertTrue(parent['inspection']['valid'])
                    self.assertTrue(child['inspection']['valid'])
                    self.assertTrue(child['backward_executed'])
                    self.assertEqual(child['mutation_obligations_valid'], retain)
                    self.assertEqual(child['corpus_admission']['admitted'], retain)
                    request = json.loads((Path(child['directory'])/'request.json').read_text())
                    obligations = request['context']['mutation_contract']
                    self.assertEqual(obligations['required_backward_calls'], [])
                    self.assertTrue(obligations['required_callable_backward_calls'])

    def test_exception_names_and_generated_contract_errors_are_not_legality_evidence(self):
        from mirage.runtime.worker import runtime_diagnostic
        from mirage.generator.program import ContractError
        for exception in (RuntimeError, TypeError, ValueError, ContractError):
            try:
                raise exception('authored test exception')
            except Exception as error:
                diagnosis = runtime_diagnostic(error)
            self.assertFalse(diagnosis['repair_allowed'])
            self.assertEqual(diagnosis['kind'], 'unknown_exception')

    def test_legacy_contract_prefix_cannot_authorize_repair(self):
        from mirage.runtime.process import ProcessResult
        from mirage.workflow import stateful_runtime_diagnostic, rerun_stateful_exception
        with tempfile.TemporaryDirectory() as tmp:
            process = ProcessResult('exception', 1, 0.1, tmp, 'ContractError: legacy text')
            self.assertFalse(stateful_runtime_diagnostic(process)['repair_allowed'])
            with patch('mirage.workflow.run_worker') as worker:
                result = rerun_stateful_exception(process, Path(tmp)/'repeat', 0)
            worker.assert_not_called()
            self.assertEqual(result['status'], 'not_run')

    def test_replay_preserves_original_numerical_audit_budget(self):
        from mirage.workflow import replay_stateful
        from mirage.runtime.process import ProcessResult
        from mirage.artifacts.bundle import inventory
        from test_corpus import fixture_bundle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest_path = fixture_bundle(root, 0)
            directory = manifest_path.parent
            (directory/'execution').mkdir()
            (directory/'execution/request.json').write_text(json.dumps({'audit_budget_seconds': 2.5}))
            manifest = json.loads(manifest_path.read_text())
            manifest['expected_verdict'] = 'pass'
            manifest['files'] = {k:v for k,v in inventory(directory).items() if k != manifest_path.name}
            manifest_path.write_text(json.dumps(manifest))
            def fake_worker(request, destination, timeout, **kwargs):
                self.assertEqual(request['audit_budget_seconds'], 2.5)
                destination.mkdir()
                (destination/'summary.json').write_text(json.dumps({
                    'verdict':'pass', 'replay_source_evidence_equal':True,
                    'replay_followup_evidence_equal':True}))
                return ProcessResult('completed', 0, .1, str(destination))
            with patch('mirage.workflow.run_worker', side_effect=fake_worker):
                self.assertTrue(replay_stateful(manifest_path)['replay_valid'])

    def test_unknown_runtime_exception_is_retained_and_bounded_rerun_is_not_repair(self):
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'PYTHONPATH':str(Path(__file__).parent)+os.pathsep+os.environ.get('PYTHONPATH','')}):
            options = RunOptions(str(Path(tmp)/'run'), 'test_adaptive:unresolved_fixture_provider',
                                 # Two cases plus one rerun must fit the dispatch reserve.
                                 cases=2, time_limit=600, worker_timeout=30,
                                 strategy='stateful', contract_version=4, replay_failures=1)
            run = run_stateful(options, ('torch.sin',), predicates=[], coverage_refinement=False)
            self.assertEqual(run['provider_calls'], 2)
            self.assertEqual(run['corpus']['active_seeds'], 0)
            self.assertTrue(all(r['quarantined'] and r['repair_index'] == 0 for r in run['records']))
            repeats = [r['exception_rerun'] for r in run['records'] if 'exception_rerun' in r]
            self.assertEqual(len(repeats), 1)
            self.assertTrue(repeats[0]['same_exception'])
            self.assertEqual(repeats[0]['provider_calls'], 0)
            self.assertEqual(repeats[0]['initial_state_equivalence'], 'unassessed')
            self.assertEqual(repeats[0]['correctness'], 'unassessed')

    def test_observed_parameter_contract_violation_can_be_repaired(self):
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'PYTHONPATH':str(Path(__file__).parent)+os.pathsep+os.environ.get('PYTHONPATH','')}):
            options = RunOptions(str(Path(tmp)/'run'), 'test_adaptive:contract_fixture_provider',
                                 cases=2, time_limit=180, worker_timeout=30,
                                 strategy='stateful', contract_version=4)
            run = run_stateful(options, ('torch.sin',), predicates=[], coverage_refinement=False)
            first, second = run['records']
            self.assertTrue(first['runtime_diagnostic']['repair_allowed'])
            self.assertEqual(first['first_blocking_stage'], 'runtime_contract')
            self.assertNotIn('exception_rerun', first)
            self.assertEqual(second['repair_index'], 1)
            self.assertTrue(second['inspection']['valid'])

    def test_unchanged_repair_stops_before_repeating_execution(self):
        from mirage.workflow import run_stateful
        from mirage.runtime.process import run_worker
        from mirage.contracts.programs import RunOptions
        for cosmetic in (False, True):
            with self.subTest(cosmetic=cosmetic), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                    'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', ''),
                    'MIRAGE_TEST_COSMETIC_REPAIR': str(int(cosmetic))}), \
                    patch('mirage.workflow.run_worker', wraps=run_worker) as worker:
                options = RunOptions(str(Path(tmp) / 'run'), 'test_adaptive:unchanged_repair_fixture_provider',
                                     cases=2, time_limit=180, worker_timeout=30,
                                     strategy='stateful', contract_version=4)
                report = run_stateful(options, ('torch.sin',), predicates=[], coverage_refinement=False)
                first, second = report['records']
                self.assertTrue(first['runtime_diagnostic']['repair_allowed'])
                self.assertEqual(second['status'], 'repair_unchanged')
                self.assertEqual(second['duplicate_repair_kind'], 'ast' if cosmetic else 'response')
                self.assertEqual(second['repair_evidence_reference'], first['directory'])
                self.assertNotIn('process', second)
                self.assertIsNone(second['verdict'])
                self.assertEqual(sum(c.args[0]['mode'] == 'execute_stateful_pair' for c in worker.call_args_list), 1)

    def test_suspect_routes_to_replay_without_changing_immutable_evidence(self):
        from mirage.workflow import run_stateful
        from mirage.runtime.process import run_worker
        from mirage.contracts.programs import RunOptions
        from mirage.artifacts.bundle import verify
        def authored_suspect(request, directory, timeout, **kwargs):
            process = run_worker(request, directory, timeout, **kwargs)
            if request['mode'] == 'execute_stateful_pair' and process.outcome == 'completed':
                path = directory/'summary.json'
                summary = json.loads(path.read_text())
                summary.update(verdict='suspected_failure', reason='authored replay routing control')
                path.write_text(json.dumps(summary))
            return process
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'PYTHONPATH':str(Path(__file__).parent)+os.pathsep+os.environ.get('PYTHONPATH','')}), \
                patch('mirage.workflow.run_worker', side_effect=authored_suspect), \
                patch('mirage.workflow.replay_stateful', return_value={
                    'replay_valid':True, 'provider_calls':0}) as repeat:
            options = RunOptions(str(Path(tmp)/'run'), 'test_corpus:local_fixture_provider',
                                 cases=1, time_limit=180, worker_timeout=30,
                                 strategy='stateful', contract_version=4, replay_failures=1)
            run = run_stateful(options, ('torch.sin',), predicates=[], coverage_refinement=False)
            row = run['records'][0]
            repeat.assert_called_once()
            self.assertEqual(run['provider_calls'], 1)
            self.assertEqual(row['verdict'], 'suspected_failure')
            self.assertEqual(len(row['automatic_replays']), 1)
            self.assertFalse(row['corpus_admission']['admitted'])
            directory = Path(row['directory'])
            manifest = json.loads((directory/'stateful-manifest.json').read_text())
            verify(directory, manifest['files'])
            self.assertNotIn('automatic_replays', json.loads((directory/'outcome.json').read_text()))


class RepresentationTraversalTests(unittest.TestCase):
    def fixture(self, root, number, *, actions=('clone', 'strided', 'offset')):
        source = f'# source {number}\n'
        directory = root / str(number)
        (directory / 'execution').mkdir(parents=True)
        (directory / 'program.py').write_text(source)
        (directory / 'execution/source.pt').write_bytes(b'opaque saved initialization')
        row = record()
        row.update(directory=str(directory), configuration={'api': 'torch.sin', 'seed': 0,
                   'attempt': number, 'predicates': []}, process={'elapsed_seconds': 2},
                   mutation_candidates=[{'path': ['args', 0], 'action': action, 'call_index': 0,
                       'eligibility': {'status': 'verified', 'reason': 'observed legal'}} for action in actions])
        return source, row

    def test_round_robin_sources_enumerate_each_action_without_llm_or_recursion(self):
        from mirage.sampler.representations import RepresentationTraversal
        with tempfile.TemporaryDirectory() as tmp:
            queue = RepresentationTraversal()
            fixtures = [self.fixture(Path(tmp), i) for i in range(2)]
            identifiers = [queue.register(*fixture) for fixture in fixtures]
            selected = []
            for _ in range(6):
                origin, candidate, source, _ = queue.choose(100, 20)
                selected.append((origin['source_id'], candidate['plan']['action']))
                row = dict(fixtures[0][1], representation_candidate_attained=True,
                           proposal_origin='framework_representation')
                queue.complete(origin['source_id'], candidate['id'], row)
                self.assertIsNone(queue.register(source, row))
            self.assertEqual(selected, [(i, action) for action in ('clone', 'strided', 'offset') for i in identifiers])
            self.assertIsNone(queue.choose(100, 20))
            self.assertTrue(all(c['status'] == 'reached' for s in queue.sources for c in s['candidates']))

    def test_new_sources_cannot_starve_older_sources_remaining_actions(self):
        from mirage.sampler.representations import RepresentationTraversal
        with tempfile.TemporaryDirectory() as tmp:
            queue = RepresentationTraversal()
            original = self.fixture(Path(tmp), 0)
            oldest = queue.register(*original)
            oldest_dispatches = []
            # Each served turn adds another source: catalog growth must not
            # keep pushing the older source's remaining actions out of reach.
            for turn in range(8):
                origin, candidate, _, _ = queue.choose(100, 20)
                if origin['source_id'] == oldest:
                    oldest_dispatches.append((turn, candidate['plan']['action']))
                row = dict(original[1], representation_candidate_attained=True)
                queue.complete(origin['source_id'], candidate['id'], row)
                queue.register(*self.fixture(Path(tmp), turn + 1))
            self.assertEqual(oldest_dispatches, [(0, 'clone'), (1, 'strided'), (3, 'offset')])
            self.assertTrue(all(c['status'] == 'reached' for c in queue.sources[0]['candidates']))

    def test_unaffordable_source_retains_turn_before_later_arrivals(self):
        from mirage.sampler.representations import RepresentationTraversal
        with tempfile.TemporaryDirectory() as tmp:
            queue = RepresentationTraversal()
            expensive_source, expensive = self.fixture(Path(tmp), 0)
            expensive['process']['elapsed_seconds'] = 100
            oldest = queue.register(expensive_source, expensive)
            cheap = queue.register(*self.fixture(Path(tmp), 1))
            origin, _, _, _ = queue.choose(20, 20)
            self.assertEqual(origin['source_id'], cheap)
            queue.register(*self.fixture(Path(tmp), 2))
            origin, candidate, _, _ = queue.choose(1000, 20)
            self.assertEqual(origin['source_id'], oldest)
            self.assertEqual(candidate['plan']['action'], 'clone')

    def test_coverage_only_sources_keep_ineligible_and_budget_pending_candidates(self):
        from mirage.sampler.representations import RepresentationTraversal
        with tempfile.TemporaryDirectory() as tmp:
            queue = RepresentationTraversal()
            source, row = self.fixture(Path(tmp), 0)
            row.update(verdict='inapplicable', corpus_admission={'admitted': False})
            row['mutation_candidates'][0]['eligibility']['status'] = 'rejected'
            queue.register(source, row)
            self.assertIsNone(queue.choose(9, 20))
            states = [c['status'] for c in queue.sources[0]['candidates']]
            self.assertEqual(states, ['ineligible', 'pending', 'pending'])
            origin, candidate, _, _ = queue.choose(100, 20)
            self.assertEqual(candidate['plan']['action'], 'strided')
            row['representation_candidate_attained'] = False
            self.assertEqual(queue.complete(origin['source_id'], candidate['id'], row), 'not_verified')

    def test_same_program_config_dedup_ignores_scheduling_but_not_state(self):
        from mirage.sampler.representations import RepresentationTraversal
        with tempfile.TemporaryDirectory() as tmp:
            queue = RepresentationTraversal()
            source, row = self.fixture(Path(tmp), 0)
            identity = queue.register(source, row)
            row['configuration'].update(attempt=5, proposal_index=7, canonical_reference=True, oracle_seen_units=['x'])
            self.assertEqual(queue.register(source, row), identity)
            row['configuration']['predicates'] = [{'dimension': 'gradient', 'field': 'backward_count', 'expected': 2}]
            self.assertNotEqual(queue.register(source, row), identity)
            self.assertEqual(len(queue.sources), 2)

    def test_observed_initial_followup_skips_only_exact_verified_action(self):
        from mirage.sampler.representations import RepresentationTraversal
        with tempfile.TemporaryDirectory() as tmp:
            queue = RepresentationTraversal()
            source, row = self.fixture(Path(tmp), 0)
            plan = {'path': ['args', 0], 'action': 'clone', 'call_index': 0}
            row.update(verdict='inapplicable', mutation=plan, followup_completed=True,
                       followup_inspection={'valid': True},
                       followup_mutations=[dict(plan, status='verified')])
            queue.register(source, row)
            candidate = queue.sources[0]['candidates'][0]
            self.assertEqual(candidate['status'], 'previously_reached')
            self.assertEqual(candidate['evidence']['member'], 'followup')
            self.assertNotIn('seconds', candidate)
            _, following, _, _ = queue.choose(100, 20)
            self.assertEqual(following['plan']['action'], 'strided')

    def test_observed_action_requires_actual_exact_evidence_and_preserved_paths(self):
        from mirage.sampler.representations import previously_reached
        plan = {'path': ['args', 0], 'action': 'clone', 'call_index': 0}
        witness = {'call': 0, 'checkpoint': 'final', 'relations': {
            'target_backward_node': {'status': 'verified'},
            'callable_backward_node': {'status': 'verified'}}}
        row = {'mutation': plan, 'mutation_obligations_valid': True, 'followup_completed': True,
               'inspection': {'valid': True, 'semantic_coverage': {'witnesses': [witness]}},
               'followup_inspection': {'valid': True, 'semantic_coverage': {'witnesses': [deepcopy(witness)]}},
               'followup_mutations': [dict(plan, status='verified')]}
        self.assertIsNotNone(previously_reached(plan, row))
        variants = []
        for key, value in [('followup_completed', False), ('followup_mutations', []),
                           ('mutation_obligations_valid', False), ('mutation', {'action': 'canonicalize_actual_operands'})]:
            value_row = deepcopy(row); value_row[key] = value; variants.append(value_row)
        for key, value in [('path', ['args', 1]), ('call_index', 1), ('action', 'offset'), ('status', 'unknown')]:
            value_row = deepcopy(row); value_row['followup_mutations'][0][key] = value; variants.append(value_row)
        for relation in ('target_backward_node', 'callable_backward_node'):
            value_row = deepcopy(row)
            value_row['followup_inspection']['semantic_coverage']['witnesses'][0]['relations'][relation]['status'] = 'unknown'
            variants.append(value_row)
        invalid = deepcopy(row); invalid['followup_inspection']['valid'] = False; variants.append(invalid)
        duplicate = deepcopy(row); duplicate['followup_mutations'] *= 2; variants.append(duplicate)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertIsNone(previously_reached(plan, variant))

    def test_saved_source_and_initialization_are_verified_before_dispatch(self):
        from mirage.sampler.representations import RepresentationTraversal
        for changed in ('program.py', 'execution/source.pt'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                queue = RepresentationTraversal()
                source, row = self.fixture(Path(tmp), 0)
                queue.register(source, row)
                (Path(row['directory']) / changed).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    queue.choose(100, 20)

    def test_llm_goals_retain_semantic_and_training_dimensions(self):
        from mirage.sampler.representations import program_goal_catalog
        names = ['representation', 'tensor_relation', 'data', 'gradient', 'gradient_representation', 'optimizer']
        goals = [{'predicate': {'dimension': n}} for n in names]
        self.assertEqual([g['predicate']['dimension'] for g in program_goal_catalog(goals)], names[2:])

    def test_controller_traversal_reuses_source_without_provider_call(self):
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', '')}):
            options = RunOptions(str(Path(tmp) / 'run'), 'test_corpus:local_fixture_provider',
                                 cases=2, time_limit=180, worker_timeout=30,
                                 strategy='stateful', contract_version=4, replay_failures=0)
            report = run_stateful(options, ('torch.sin',), predicates=[], coverage_refinement=False,
                                  representation_traversal=True)
            self.assertEqual(report['provider_calls'], 1)
            first, child = report['records']
            self.assertEqual(child['proposal_origin'], 'framework_representation')
            self.assertFalse(child['program_generation_dispatched'])
            self.assertTrue(child['representation_candidate_attained'])
            self.assertTrue(child['followup_inspection']['valid'])
            self.assertEqual(child['representation_traversal']['status'], 'reached')
            self.assertEqual((Path(first['directory']) / 'program.py').read_text(),
                             (Path(child['directory']) / 'program.py').read_text())
            self.assertEqual(len(report['representation_traversal']['sources']), 1)
            candidates = report['representation_traversal']['sources'][0]['candidates']
            self.assertEqual(candidates[0]['status'], 'previously_reached')
            self.assertEqual(candidates[0]['plan']['action'], 'clone')
            self.assertEqual(child['representation_traversal']['plan']['action'], 'strided')
            self.assertTrue(any(m.get('action') == 'clone' and m.get('status') == 'verified'
                                for m in first['followup_mutations']))
            worker_request = json.loads((Path(child['directory']) / 'execution/request.json').read_text())
            self.assertEqual(worker_request['initial_reference'], str(Path(first['directory']) / 'execution/source.pt'))
            self.assertEqual(worker_request['reference_scope'], 'initialization')
            self.assertTrue(child['initialization_reference_verified'])
            self.assertNotIn('replay_followup_evidence_equal', child)
