"""Goal feasibility, immutable edit framing and runtime operand evidence."""
import ast
from dataclasses import asdict
import json
import unittest

from mirage.generator.state_plan import goal_catalog, validate_plans, planning_request, goal_family
from mirage.generator.mutation import plan_regions, apply_regions
from mirage.generator.program import ContractError
from mirage.contracts.capabilities import StatePredicate
from test_stateful import SOURCE, CONFIG


class OptimizerStatePlanTests(unittest.TestCase):
    def entry(self):
        from test_optimizer_inspection import optimizer_record
        from mirage.inspector.stateful import inspect_optimizer
        coverage = inspect_optimizer(optimizer_record())
        return {'configuration': {'api': 'torch.sin', 'requires_backward': True},
                'evidence': {'witnesses': [], 'optimizer_witnesses': coverage['witnesses'],
                             'optimizer_target_update_count': 1}}

    def test_bootstrap_needs_no_invented_parameter_and_requires_full_edit(self):
        entry = {'configuration': {'api': 'torch.sin', 'requires_backward': True}, 'evidence': {}}
        catalog = goal_catalog(entry)
        goal = next(g for g in catalog if g['predicate']['field'] == 'target_update_count')
        self.assertIsNone(goal['predicate']['parameter_key'])
        self.assertEqual(goal['required_regions'], ['full'])
        self.assertIsNone(goal['observed'])
        self.assertEqual(goal_family(goal), 'training_state')
        entry['configuration']['requires_backward'] = False
        self.assertFalse(goal_catalog(entry))

    def test_gradient_goal_carries_completed_companion_even_when_parent_completed(self):
        catalog = goal_catalog(self.entry())
        goal = next(g for g in catalog if g['predicate']['field'] == 'contiguous')
        self.assertEqual(goal_family(goal), 'tensor_state')
        self.assertTrue(goal['predicate']['expected'])
        self.assertEqual(goal['required_companions'][0]['field'], 'completed')
        self.assertEqual(goal['required_companions'][0]['parameter_key'], 'p')
        content = json.dumps({'plans': [{'goal_ids': [goal['id']], 'actions': ['Preserve target backward and perform the registered native step'],
            'preconditions': ['registered parameter identity'], 'regions': ['run_tail'], 'rationale': 'gradient layout'}]})
        accepted, rejected = validate_plans(content, catalog, planning_family='tensor_state')
        self.assertFalse(rejected)
        self.assertEqual({p['dimension'] for p in accepted[0]['predicates']}, {'optimizer', 'gradient_representation'})
        self.assertEqual(accepted[0]['optimizer_gradient_action'],
                         {'step_index': 0, 'parameter_key': 'p', 'action': 'canonical'})

    def test_action_conjunctions_match_available_representations(self):
        from mirage.generator.state_plan import optimizer_action_for_predicates
        from test_optimizer_inspection import gradient_predicate
        def predicates(contiguous, offset):
            return [asdict(gradient_predicate(expected=contiguous)),
                    asdict(gradient_predicate(field='offset_nonzero', expected=offset))]
        for contiguous, offset, action in ((True, False, 'canonical'), (True, True, 'offset'), (False, False, 'strided')):
            self.assertEqual(optimizer_action_for_predicates(predicates(contiguous, offset))['action'], action)
        with self.assertRaises(ValueError): optimizer_action_for_predicates(predicates(False, True))
        with self.assertRaises(ValueError): optimizer_action_for_predicates(
            [asdict(gradient_predicate()), asdict(gradient_predicate(parameter_key='other'))])

    def test_explicit_gradient_goals_keep_parameter_selector_and_companions(self):
        from test_optimizer_inspection import gradient_predicate
        raw = asdict(gradient_predicate(expected=True))
        catalog = goal_catalog(self.entry(), [raw])
        request = planning_request({}, SOURCE, self.entry(), catalog, CONFIG, planning_family='tensor_state')
        self.assertEqual(request.context['measured_optimizer_witnesses'][0]['parameter_key'], 'p')
        self.assertIn('required_companions remain mandatory', request.prompt)
        self.assertIn('not the forward operand', request.prompt)
        self.assertEqual(request.context['goal_catalog'][0]['source_case_action']['action'], 'canonical')

    def test_scalar_gradient_does_not_offer_impossible_noncontiguity(self):
        entry = self.entry()
        metadata = entry['evidence']['optimizer_witnesses'][0]['metadata']
        metadata.update(shape=[], strides=[])
        metadata['properties']['contiguous'] = True
        goals = goal_catalog(entry)
        self.assertFalse(any(g['predicate']['field'] == 'contiguous' for g in goals))
        self.assertTrue(any(g['predicate']['field'] == 'offset_nonzero' for g in goals))


class LibraryConfigurationPlanTests(unittest.TestCase):
    def catalog(self):
        return goal_catalog(OptimizerStatePlanTests().entry())

    def test_known_action_has_exact_companion_and_never_claims_attainment(self):
        from mirage.generator.state_plan import library_configuration_plans
        catalog = self.catalog()
        plans, rejected = library_configuration_plans(catalog)
        self.assertTrue(plans)
        self.assertFalse(rejected)
        self.assertEqual({p['goal_ids'][0] for p in plans},
                         {g['id'] for g in catalog if 'source_case_action' in g})
        for plan in plans:
            self.assertEqual(plan['status'], 'planned_not_verified')
            self.assertEqual(plan['proposal_origin'], 'library_known_action')
            self.assertTrue(plan['configuration_only'])
            self.assertEqual(len(plan['predicates']), 2)
            self.assertEqual(plan['predicates'][1]['dimension'], 'optimizer')
            self.assertTrue(plan['predicates'][1]['expected'])
            self.assertEqual(plan['predicates'][1]['parameter_key'], plan['predicates'][0]['parameter_key'])

    def test_missing_wrong_companion_unknown_action_and_duplicate_ids_fail_closed(self):
        from copy import deepcopy
        from mirage.generator.state_plan import library_configuration_plans
        original = next(g for g in self.catalog() if 'source_case_action' in g)
        for kind in ('missing', 'path', 'checkpoint', 'action', 'regions'):
            goal = deepcopy(original)
            if kind == 'missing': goal.pop('required_companions')
            elif kind == 'path': goal['required_companions'][0]['operand_path'] = ['args', 1]
            elif kind == 'checkpoint': goal['required_companions'][0]['checkpoint'] = 'final'
            elif kind == 'action': goal['source_case_action']['action'] = 'invented'
            else: goal['required_regions'] = ['full']
            plans, rejected = library_configuration_plans([goal])
            self.assertEqual(plans, [], kind)
            self.assertEqual(len(rejected), 1, kind)
        plans, rejected = library_configuration_plans([original, deepcopy(original)])
        self.assertEqual(plans, [])
        self.assertTrue(rejected)

    def test_bootstrap_and_unrelated_goals_remain_for_general_planning(self):
        from mirage.generator.state_plan import library_configuration_plans
        catalog = goal_catalog({'configuration': {'requires_backward': True}, 'evidence': {}})
        self.assertTrue(any(g['predicate']['field'] == 'target_update_count' for g in catalog))
        before = json.dumps(catalog, sort_keys=True)
        self.assertEqual(library_configuration_plans(catalog), ([], []))
        self.assertEqual(json.dumps(catalog, sort_keys=True), before)


class StatePlanTests(unittest.TestCase):
    def entry(self):
        return {'configuration': CONFIG, 'evidence': {'witnesses': [{'checkpoint':'final', 'call':0,
            'operand_path':['args',0], 'metadata':{'properties':{'contiguous':True, 'negative':False},
                'offset':0, 'dtype':'torch.float64'}, 'state_facts':{'gradient.backward_count':1}}]}}

    def plan(self, goal, regions):
        return {'plans':[{'goal_ids':[goal['id']], 'actions':['retain graph before first backward; run twice'],
                         'preconditions':['target supports backward'], 'regions':regions, 'rationale':'new state'}]}

    def test_backward_goal_cannot_be_served_by_initialization(self):
        catalog = goal_catalog(self.entry())
        goal = next(g for g in catalog if g['predicate']['dimension'] == 'gradient')
        good, bad = validate_plans(json.dumps(self.plan(goal, ['initialization'])), catalog)
        self.assertEqual(good, [])
        self.assertIn('mismatch', bad[0]['reason'])
        good, bad = validate_plans(json.dumps(self.plan(goal, ['run_tail'])), catalog)
        self.assertEqual(good[0]['predicates'][0]['expected'], 2)
        self.assertEqual(bad, [])

    def test_already_attained_is_filtered_unknown_not_claimed_reachable(self):
        goal = {'dimension':'gradient','field':'backward_count','expected':1,'checkpoint':'final'}
        self.assertEqual(goal_catalog(self.entry(), [goal]), [])
        goal['field'] = 'unobserved'
        catalog = goal_catalog(self.entry(), [goal])
        self.assertIsNone(catalog[0]['observed'])
        self.assertIn('runtime_validation', catalog[0]['feasibility'])

    def test_representation_goals_preserve_attributable_operand_backward(self):
        entry = self.entry()
        entry['evidence']['witnesses'][0]['relations'] = {'operand_gradient_contribution':{'status':'verified'}}
        goals = [g for g in goal_catalog(entry) if g['predicate']['dimension']=='representation']
        self.assertTrue(goals)
        self.assertTrue(all(g['predicate']['relation']=='operand_gradient_contribution' for g in goals))

    def test_automatic_counts_respect_multiple_coverage_bucket(self):
        entry = self.entry()
        entry['evidence']['witnesses'][0]['state_facts']['gradient.backward_count'] = 2
        self.assertFalse(any(g['predicate']['field'] == 'backward_count' for g in goal_catalog(entry)))

    def test_planning_context_is_not_generated_code_or_runtime_proof(self):
        entry = self.entry()
        request = planning_request({'documentation':'spec'}, SOURCE, entry, goal_catalog(entry), CONFIG)
        self.assertEqual(request.stage.value, 'state_goal')
        self.assertEqual(request.context['api_context']['documentation'], 'spec')
        self.assertIn('starts at build()', request.prompt)
        self.assertIn('missing MR proof, not an invalid program', request.prompt)
        self.assertIn('never delete a hard state', request.prompt)

    def test_multi_region_preimages_and_frame(self):
        contract = plan_regions(SOURCE, 'parent', 0, ['operand_setup','run_tail'])
        patches = [{'region_id':r['region_id'], 'preimage_sha256':r['preimage_sha256'],
                    'source': 'x = x.clone()' if r['kind']=='operand_setup' else
                        "y.sum().backward(retain_graph=True)\ny.sum().backward()\nreturn {'y':y}"} for r in contract['regions']]
        child, edit = apply_regions(SOURCE, patches, contract)
        self.assertTrue(edit['outside_region_unchanged'])
        self.assertEqual(ast.dump(ast.parse(SOURCE).body[1]), ast.dump(ast.parse(child).body[1]))
        patches[0]['preimage_sha256'] = 'wrong'
        with self.assertRaisesRegex(ContractError, 'preimage'): apply_regions(SOURCE, patches, contract)

    def test_operand_goal_requires_exact_path_and_runtime_relation(self):
        from mirage.runtime.stateful import execute
        from mirage.runtime.probes import Recorder
        from mirage.inspector.stateful import inspect_state
        record = execute(SOURCE, CONFIG, Recorder())
        witness = next(w for w in record['inspection']['semantic_coverage']['witnesses'] if w['checkpoint']=='final' and w['metadata'])
        predicate = asdict(StatePredicate('representation','contiguous',True,'final',
            witness['call'], tuple(witness['operand_path']), 'operand_gradient_contribution'))
        self.assertTrue(inspect_state(record, [predicate])['valid'])
        predicate['operand_path'] = ['absent']
        failed = inspect_state(record, [predicate])
        self.assertFalse(failed['valid'])
        self.assertEqual(failed['predicates'][0]['evidence_status'], 'unknown')

    def test_installed_doc_reference_closure(self):
        from mirage.runtime.api_metadata import describe
        value = describe('torch.Tensor.scatter_add')
        reference = next(r for r in value['documentation_references'] if r['api']=='torch.Tensor.scatter_add_')
        self.assertIn('index', reference['documentation'])
        self.assertEqual(len(reference['sha256']), 64)
        self.assertTrue(value['torch_version'])

    def test_workflow_plans_before_patch_and_counts_both_calls(self):
        import os
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            'PYTHONPATH':str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH','')}):
            report = run_stateful(RunOptions(str(Path(temporary)/'run'), 'test_state_plan:fixture_provider',
                cases=2, time_limit=180, worker_timeout=60, strategy='stateful', contract_version=4),
                ('torch.sin',), predicates=[{'dimension':'gradient', 'field':'backward_count',
                    'expected':2, 'checkpoint':'final'}], search_policy='round_robin', coverage_refinement=False)
            self.assertEqual(report['provider_calls'], 3)
            child = report['records'][1]
            self.assertTrue(child['inspection']['valid'], child.get('reason'))
            self.assertEqual(child['generation_mode'], 'local')
            self.assertEqual(child['state_plan']['status'], 'planned_not_verified')
            self.assertEqual(child['inspection']['predicates'][0]['observed'], 2)
            self.assertEqual(report['search_metrics']['planning_calls'], 1)

    def test_two_malformed_planning_replies_exhaust_parent_budget_then_generate_fresh(self):
        import os
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            'PYTHONPATH':str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH','')}):
            report = run_stateful(RunOptions(str(Path(temporary)/'run'), 'test_state_plan:malformed_fixture_provider',
                cases=4, time_limit=180, worker_timeout=60, strategy='stateful', contract_version=4),
                ('torch.sin',), predicates=[{'dimension':'gradient', 'field':'backward_count',
                    'expected':2, 'checkpoint':'final'}], search_policy='round_robin', coverage_refinement=False)
            parent, first, second, fresh = report['records']
            self.assertTrue(parent['inspection']['valid'])
            self.assertEqual([first['first_blocking_stage'], second['first_blocking_stage']], ['state_plan'] * 2)
            self.assertEqual([first['status'], second['status']], ['planning_rejected'] * 2)
            self.assertEqual([first['planning_budget_failures'], second['planning_budget_failures']], [1, 2])
            self.assertEqual(first['parent_seed_id'], second['parent_seed_id'])
            self.assertNotIn('program_generation_dispatched', first)
            self.assertNotIn('program_generation_dispatched', second)
            self.assertIsNone(fresh['parent_seed_id'])
            self.assertTrue(fresh['program_generation_dispatched'])
            self.assertTrue(fresh['inspection']['valid'])
            self.assertEqual(report['search_metrics']['planning_calls'], 2)

    def test_unresolved_plan_is_not_classified_as_an_invalid_program(self):
        import os
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', '')}):
            report = run_stateful(RunOptions(str(Path(temporary) / 'run'), 'test_state_plan:unresolved_fixture_provider',
                cases=2, time_limit=180, worker_timeout=60, strategy='stateful', contract_version=4),
                ('torch.sin',), predicates=[{'dimension': 'gradient', 'field': 'backward_count',
                    'expected': 2, 'checkpoint': 'final'}], search_policy='round_robin', coverage_refinement=False)
            row = report['records'][1]
            self.assertEqual(row['status'], 'planning_unresolved')
            self.assertEqual(row['planning_outcome'], 'unresolved')
            self.assertNotIn('program_generation_dispatched', row)
            self.assertIsNone(row['verdict'])
            self.assertEqual(report['search_metrics']['planning_outcomes'], {'unresolved': 1})

    def test_representation_plan_survives_json_corpus_roundtrip(self):
        import os
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            'PYTHONPATH':str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH','')}):
            report = run_stateful(RunOptions(str(Path(temporary)/'run'), 'test_state_plan:representation_fixture_provider',
                cases=2, time_limit=180, worker_timeout=60, strategy='stateful', contract_version=4),
                ('torch.sin',), predicates=[{'dimension':'representation','field':'offset_nonzero', 'expected':True,
                    'checkpoint':'final','call_index':0,'operand_path':['args',0]}],
                search_policy='round_robin', coverage_refinement=False)
            child = report['records'][1]
            self.assertTrue(child['inspection']['valid'], child.get('reason'))
            self.assertTrue(child['corpus_admission']['admitted'])
            self.assertIsInstance(child['configuration']['predicates'][0]['operand_path'], list)

    def test_mixed_policy_does_not_add_hidden_backward_obligation(self):
        import os
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from mirage.workflow import run_stateful
        from mirage.contracts.programs import RunOptions
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            'PYTHONPATH':str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH','')}):
            report = run_stateful(RunOptions(str(Path(temporary)/'run'), 'test_state_plan:mixed_fixture_provider',
                cases=2, time_limit=180, worker_timeout=60, strategy='stateful', contract_version=4),
                ('torch.sin',), predicates=[{'dimension':'graph_lifecycle', 'field':'target_boundary_count',
                    'expected':2, 'checkpoint':'final'}], search_policy='round_robin', path_policy='mixed', coverage_refinement=False)
            parent, child = report['records']
            self.assertFalse(parent['configuration']['requires_backward'])
            self.assertFalse(parent['backward_executed'])
            self.assertFalse(child['configuration']['requires_backward'])
            self.assertTrue(child['inspection']['valid'], child.get('reason'))
            self.assertEqual(child['inspection']['predicates'][0]['observed'], 2)


class RelationalPlanTests(unittest.TestCase):
    def entry(self, count=4, proposal=0):
        witnesses = []
        for index in range(count):
            witnesses.append({'checkpoint':'final', 'call':0, 'operand_path':['args', index],
                'metadata':{'shape':[2 + index, 3], 'strides':[3, 1], 'storage_group':index, 'layout':'torch.strided',
                    'offset':0, 'dtype':'torch.float64',
                    'properties':{'contiguous':True, 'negative':False, 'floating':True, 'mkldnn':False}},
                'relations':{'operand_gradient_contribution':{'status':'verified'}},
                'state_facts':{'gradient.backward_count':1}})
        return {'configuration':dict(CONFIG, proposal_index=proposal), 'evidence':{'witnesses':witnesses}}

    def plan(self, predicates):
        catalog = [{'id':str(index), 'predicate':asdict(predicate), 'required_regions':['operand_setup']}
                   for index, predicate in enumerate(predicates)]
        value = {'plans':[{'goal_ids':[goal['id'] for goal in catalog],
            'actions':['prepare the selected operands legally'], 'preconditions':['API domains hold'],
            'regions':['operand_setup'], 'rationale':'joint target state'}]}
        return validate_plans(json.dumps(value), catalog)

    def test_catalog_bounds_rotation_and_both_operand_gradient_obligations(self):
        from mirage.generator.state_plan import CATALOG_LIMITS
        seen = set()
        for proposal in range(8):
            entry = self.entry(6, proposal)
            catalog = goal_catalog(entry)
            self.assertEqual(catalog, goal_catalog(entry))
            self.assertLessEqual(len(catalog), CATALOG_LIMITS['goals'])
            relational = [g for g in catalog if g['predicate']['dimension']=='tensor_relation']
            self.assertTrue(relational)
            for goal in relational:
                p = goal['predicate']
                self.assertEqual(p['relation'], 'operand_gradient_contribution')
                self.assertEqual(p['peer_relation'], 'operand_gradient_contribution')
                self.assertLess(p['operand_path'][1], 4)
                self.assertLess(p['peer_operand_path'][1], 4)
                self.assertNotEqual(goal['observed'], p['expected'])
                seen.add((tuple(p['operand_path']), tuple(p['peer_operand_path']), p['axis'], p['peer_axis']))
        self.assertGreater(len(seen), 4)

    def test_single_operand_still_has_selected_axis_goals(self):
        catalog = goal_catalog(self.entry(1))
        selected = [g['predicate'] for g in catalog if g['predicate'].get('axis') is not None]
        self.assertTrue(selected)
        self.assertEqual({p['axis'] for p in selected}, {0, 1})
        self.assertEqual({p['field'] for p in selected}, {'zero_stride', 'singleton_stride'})
        self.assertFalse(any(g['predicate']['dimension']=='tensor_relation' for g in catalog))

    def test_planning_rounds_expose_each_family_without_mandatory_relation_priority(self):
        entry = self.entry(2)
        catalog = goal_catalog(entry)
        for family in ('training_state', 'tensor_state', 'tensor_relation'):
            with self.subTest(family=family):
                request = planning_request({'documentation':'spec'}, SOURCE, entry, catalog, CONFIG,
                                           planning_family=family)
                expected = [g['id'] for g in catalog if goal_family(g) == family]
                self.assertTrue(expected)
                self.assertEqual(request.context['requested_goal_ids'], expected)
                self.assertEqual(request.context['planning_family'], family)
                self.assertEqual(request.context['goal_families'][family], expected)
                self.assertNotIn('exploration_goal_ids', request.context)
                self.assertIn('single-goal plan is sufficient', request.prompt)
                self.assertIn('not proof of global unreachability', request.prompt)
                goal = next(g for g in catalog if goal_family(g) == family)
                accepted, rejected = validate_plans(json.dumps(StatePlanTests().plan(
                    goal, goal['required_regions'])), catalog, planning_family=family)
                self.assertEqual(len(accepted), 1)
                self.assertFalse(rejected)
                other = next(g for g in catalog if goal_family(g) != family)
                accepted, rejected = validate_plans(json.dumps(StatePlanTests().plan(
                    other, other['required_regions'])), catalog, planning_family=family)
                self.assertFalse(accepted)
                self.assertEqual(rejected[0]['examined_goal_ids'], [])
        axis = next(g for g in catalog if g['predicate']['dimension']=='representation'
                    and g['predicate'].get('axis') is not None)
        self.assertEqual(goal_family(axis), 'tensor_state')

    def test_catalog_does_not_join_different_calls_or_unknown_storage(self):
        entry = self.entry(2)
        entry['evidence']['witnesses'][1]['call'] = 1
        self.assertFalse(any(g['predicate']['dimension']=='tensor_relation' for g in goal_catalog(entry)))
        entry = self.entry(2)
        for w in entry['evidence']['witnesses']:
            w['metadata']['storage_group'] = None
        catalog = goal_catalog(entry)
        self.assertFalse(any(g['predicate']['field']=='shared_storage' for g in catalog))
        self.assertTrue(any(g['predicate']['field']=='extent_order' for g in catalog))

    def test_explicit_relation_attainment_needs_both_witness_paths(self):
        entry = self.entry(2)
        predicate = asdict(StatePredicate('tensor_relation', 'extent_order', 'lt', 'final',
            call_index=0, operand_path=['args',0], relation='operand_gradient_contribution', axis=0,
            peer_operand_path=['args',1], peer_axis=0, peer_relation='operand_gradient_contribution'))
        self.assertEqual(goal_catalog(entry, [predicate]), [])
        entry['evidence']['witnesses'][1]['relations']['operand_gradient_contribution']['status'] = 'unknown'
        unresolved = goal_catalog(entry, [predicate])
        self.assertIsNone(unresolved[0]['observed'])
        self.assertIn('runtime_validation', unresolved[0]['feasibility'])
        old = {'dimension':'representation', 'field':'zero_stride', 'expected':False,
               'checkpoint':'final', 'call_index':0, 'operand_path':['args',0]}
        self.assertEqual(goal_catalog(entry, [old]), [])

    def test_axes_and_peers_are_distinct_but_same_selector_disagreement_is_rejected(self):
        common = dict(dimension='tensor_relation', field='extent_order', checkpoint='final',
                      call_index=0, operand_path=['args',0], axis=0)
        a = StatePredicate(expected='lt', peer_operand_path=['args',1], peer_axis=0, **common)
        b = StatePredicate(expected='gt', peer_operand_path=['args',2], peer_axis=0, **common)
        c = StatePredicate(expected='eq', peer_operand_path=['args',1], peer_axis=1, **common)
        accepted, rejected = self.plan([a, b, c])
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)
        contradictory = StatePredicate(expected='gt', peer_operand_path=['args',1], peer_axis=0,
                                       relation='target_backward_node', **common)
        accepted, rejected = self.plan([a, contradictory])
        self.assertFalse(accepted)
        self.assertIn('Conflicting', rejected[0]['reason'])
        accepted, rejected = self.plan([a, a])
        self.assertFalse(accepted)
        self.assertIn('Duplicate', rejected[0]['reason'])

    def test_reversed_pair_conflicts_preserve_path_specific_obligations(self):
        common = dict(dimension='tensor_relation', checkpoint='final', call_index=0)
        forward = dict(operand_path=['args',0], peer_operand_path=['args',1],
                       relation='operand_gradient_contribution', peer_relation='target_backward_node')
        reverse = dict(operand_path=['args',1], peer_operand_path=['args',0],
                       relation='target_backward_node', peer_relation='operand_gradient_contribution')
        for field, primary, contradictory, consistent, axes, reverse_axes in (
                ('shared_storage', True, False, True, {}, {}),
                ('extent_order', 'lt', 'lt', 'gt', {'axis':0, 'peer_axis':1}, {'axis':1, 'peer_axis':0})):
            with self.subTest(field=field):
                a = StatePredicate(field=field, expected=primary, **common, **forward, **axes)
                b = StatePredicate(field=field, expected=contradictory, **common, **reverse, **reverse_axes)
                accepted, rejected = self.plan([a, b])
                self.assertFalse(accepted)
                self.assertIn('Conflicting', rejected[0]['reason'])
                b = StatePredicate(field=field, expected=consistent, **common, **reverse, **reverse_axes)
                accepted, rejected = self.plan([a, b])
                self.assertEqual(len(accepted), 1)
                self.assertFalse(rejected)
                self.assertEqual(accepted[0]['predicates'], [asdict(a), asdict(b)])

    def test_zero_stride_and_singleton_on_same_axis_cannot_be_conjoined(self):
        common = dict(dimension='representation', checkpoint='final', call_index=0,
                      operand_path=['args',0], axis=0)
        accepted, rejected = self.plan([
            StatePredicate(field='zero_stride', expected=True, **common),
            StatePredicate(field='singleton_stride', expected='other',
                           **dict(common, operand_path=('args',0)))])
        self.assertFalse(accepted)
        self.assertIn('Conflicting', rejected[0]['reason'])

    def test_requested_peer_witnesses_survive_prompt_cutoff_and_full_region_is_legal(self):
        entry = self.entry(10)
        predicate = asdict(StatePredicate('tensor_relation', 'shared_storage', True, 'final',
            call_index=0, operand_path=['args',8], peer_operand_path=['args',9]))
        catalog = goal_catalog(entry, [predicate])
        request = planning_request({'documentation':'spec'}, SOURCE, entry, catalog, CONFIG)
        self.assertEqual({tuple(w['operand_path']) for w in request.context['measured_parent_witnesses']},
                         {('args',8), ('args',9)})
        self.assertIn('not an MR pair', request.prompt)
        value = StatePlanTests().plan(catalog[0], ['full'])
        accepted, rejected = validate_plans(json.dumps(value), catalog)
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)

    def test_relational_guidance_survives_local_region_prompt_replacement(self):
        from mirage.generator.program import stateful_request
        predicate = asdict(StatePredicate('representation', 'zero_stride', True, 'final',
                                          call_index=0, operand_path=['args',0], axis=0))
        mutation = plan_regions(SOURCE, 'parent', 0, ['operand_setup'])
        request = stateful_request({'documentation':'spec'}, CONFIG, canonical=SOURCE,
                                   predicates=[predicate], mutation=mutation, edges=False)
        self.assertEqual(request.context['contract'], 'region-patches')
        for text in ('TENSOR RELATION GOALS', 'peer_relation', 'SAME call_index', 'not an MR pair'):
            self.assertIn(text, request.prompt)
        self.assertEqual(request.context['prompt_version'], 'stateful-10-local-base')
        self.assertIn('missing MR proof, not an invalid program', request.prompt)
        self.assertIn('never delete a hard state', request.prompt)


class StructuredPlanProtocolTests(unittest.TestCase):
    def setUp(self):
        self.entry = StatePlanTests().entry()
        self.catalog = goal_catalog(self.entry)
        self.goal = next(g for g in self.catalog if g['predicate']['dimension']=='gradient')

    def proposal(self, status='planned'):
        plan = StatePlanTests().plan(self.goal, ['run_tail'])['plans'][0]
        plan.update(status=status, blockers=[])
        if status == 'unresolved':
            plan.update(actions=[], regions=[], blockers=['The legal graph retention ordering is not established.'])
        return {'plans':[plan], 'examined_goal_ids':[self.goal['id']]}

    def test_planned_and_legacy_replies_remain_provisional(self):
        for value in (self.proposal(), StatePlanTests().plan(self.goal, ['run_tail'])):
            accepted, rejected = validate_plans(json.dumps(value), self.catalog, planning_family='training_state')
            self.assertEqual(accepted[0]['status'], 'planned_not_verified')
            self.assertEqual(accepted[0]['examined_goal_ids'], [self.goal['id']])
            self.assertFalse(rejected)
        self.assertEqual(validate_plans('{"plans":[]}', self.catalog), ([], []))
        self.assertEqual(validate_plans('{"plans":[],"examined_goal_ids":[]}', self.catalog), ([], []))

    def test_unresolved_consumes_only_explicitly_examined_ids_without_reachability_claim(self):
        accepted, rejected = validate_plans(json.dumps(self.proposal('unresolved')), self.catalog)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['status'], 'unresolved')
        self.assertEqual(rejected[0]['examined_goal_ids'], [self.goal['id']])
        self.assertEqual(len(rejected[0]['blockers']), 1)
        self.assertIn('unassessed', rejected[0]['reason'])

    def test_malformed_status_actions_or_examination_does_not_consume_catalog(self):
        for change in ({'actions':['do something']}, {'regions':['run_tail']}, {'blockers':[]}, {'status':'unreachable'}):
            value = self.proposal('unresolved')
            value['plans'][0].update(change)
            accepted, rejected = validate_plans(json.dumps(value), self.catalog)
            self.assertFalse(accepted)
            self.assertEqual(rejected[0]['examined_goal_ids'], [])
        for examined in ([], ['unknown'], [self.goal['id'], self.goal['id']]):
            value = self.proposal()
            value['examined_goal_ids'] = examined
            with self.assertRaises(ContractError):
                validate_plans(json.dumps(value), self.catalog)
        with self.assertRaises(ContractError):
            validate_plans(json.dumps({'plans':[], 'examined_goal_ids':[self.goal['id']]}), self.catalog)

    def test_family_round_allows_compatible_cross_family_conjunction(self):
        unary = next(g for g in self.catalog if goal_family(g)=='tensor_state')
        value = self.proposal()
        value['plans'][0]['goal_ids'].append(unary['id'])
        value['plans'][0]['regions'].append('operand_setup')
        value['examined_goal_ids'].append(unary['id'])
        for family in ('training_state', 'tensor_state'):
            accepted, rejected = validate_plans(json.dumps(value), self.catalog, planning_family=family)
            self.assertEqual(len(accepted[0]['predicates']), 2)
            self.assertFalse(rejected)
        with self.assertRaises(ValueError):
            validate_plans(json.dumps(value), self.catalog, planning_family='unknown')

    def test_unscoped_training_goals_do_not_require_static_operand_bindings(self):
        source = SOURCE.replace('    y = probe.call(torch.sin, x)',
                                '    with torch.enable_grad():\n        y = probe.call(torch.sin, x)')
        request = planning_request({}, source, self.entry, [self.goal], CONFIG, planning_family='training_state')
        self.assertEqual(request.context['source_operand_bindings']['status'], 'unresolved')
        self.assertEqual(request.context['requested_goal_ids'], [self.goal['id']])
        self.assertIn('For goals selecting operand_path or peer_operand_path', request.prompt)
        self.assertIn('paths do not require tensor bindings', request.prompt)
        self.assertEqual(request.context['plan_protocol'], 'state-goal-plan-4')
        accepted, rejected = validate_plans(json.dumps(self.proposal()), [self.goal], planning_family='training_state')
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)

    def test_planned_shape_still_requires_actions_regions_and_no_blockers_after_generation(self):
        for change in ({'actions':[]}, {'regions':[]}, {'blockers':['ordering is unknown']}):
            value = self.proposal()
            value['plans'][0].update(change)
            accepted, rejected = validate_plans(json.dumps(value), self.catalog)
            self.assertFalse(accepted)
            self.assertEqual(rejected[0]['examined_goal_ids'], [])

    def test_known_coverage_context_balances_kinds_and_preserves_omission_counts(self):
        self.entry['coverage'] = [json.dumps({'unit_kind':kind, 'ordinal':i})
            for kind in ('joint', 'unary') for i in range(9)] + ['malformed']
        request = planning_request({}, SOURCE, self.entry, self.catalog, CONFIG)
        kinds = [json.loads(unit)['unit_kind'] for unit in request.context['known_coverage']]
        self.assertEqual(kinds.count('unary'), 4)
        self.assertEqual(kinds.count('joint'), 4)
        self.assertEqual(request.context['known_coverage_counts']['unary'], {'total':9,'shown':4,'omitted':5})
        self.assertEqual(request.context['known_coverage_counts']['unknown'], {'total':1,'shown':0,'omitted':1})
        self.assertEqual(len(self.entry['coverage']), 19)
        self.assertEqual(request.context['source_operand_bindings']['bindings'][0]['expression'], 'x')


class PlanFixtureProvider:
    goal_field = 'backward_count'

    def generate(self, request, deadline=None):
        from mirage.generator.provider import Response, Usage
        if request.context['contract'] == 'state-goal-plan':
            goal = next(g for g in request.context['goal_catalog'] if g['predicate']['field']==self.goal_field)
            value = StatePlanTests().plan(goal, goal['required_regions'])
        elif request.context['contract'] == 'region-patches':
            region = request.context['mutation_contract']['regions'][0]
            value = {'patches':[{'region_id':region['region_id'], 'preimage_sha256':region['preimage_sha256'],
                'source':"x = torch.cat((x[:1], x))[1:]" if region['kind']=='operand_setup' else
                    "y.sum().backward(retain_graph=True)\ny.sum().backward()\nreturn {'y':y, 'gradient':x.grad}"}]}
        else:
            value = {'source':SOURCE}
        return Response(json.dumps(value), Usage(10, 20), model='authored-plan-fixture')


def fixture_provider():
    return PlanFixtureProvider()


def representation_fixture_provider():
    provider = PlanFixtureProvider()
    provider.goal_field = 'offset_nonzero'
    return provider


class MixedFixtureProvider(PlanFixtureProvider):
    goal_field = 'target_boundary_count'

    def generate(self, request, deadline=None):
        from mirage.generator.provider import Response, Usage
        if request.context['contract'] == 'training-program':
            value = {'source': SOURCE.replace('    y.sum().backward()\n', '')}
        elif request.context['contract'] == 'region-patches':
            region = request.context['mutation_contract']['regions'][0]
            value = {'patches':[{'region_id':region['region_id'], 'preimage_sha256':region['preimage_sha256'],
                                'source':"z = probe.call(torch.sin, x)\nreturn {'y': y, 'z': z}"}]}
        else:
            return super().generate(request, deadline)
        return Response(json.dumps(value), Usage(10, 20), model='authored-mixed-fixture')


def mixed_fixture_provider():
    return MixedFixtureProvider()


class MalformedPlanFixtureProvider(PlanFixtureProvider):
    def generate(self, request, deadline=None):
        if request.context['contract'] == 'state-goal-plan':
            from mirage.generator.provider import Response, Usage
            return Response('{', Usage(10, 1), model='authored-malformed-plan-fixture')
        return super().generate(request, deadline)


def malformed_fixture_provider():
    return MalformedPlanFixtureProvider()


class UnresolvedPlanFixtureProvider(PlanFixtureProvider):
    def generate(self, request, deadline=None):
        if request.context['contract'] == 'state-goal-plan':
            from mirage.generator.provider import Response, Usage
            goal = request.context['requested_goal_ids'][0]
            value = {'plans': [{'goal_ids': [goal], 'actions': [], 'preconditions': [], 'regions': [],
                      'rationale': 'Legal action ordering has not been established.', 'status': 'unresolved',
                      'blockers': ['No justified transition sequence.']}], 'examined_goal_ids': [goal]}
            return Response(json.dumps(value), Usage(10, 20), model='authored-unresolved-plan-fixture')
        return super().generate(request, deadline)


def unresolved_fixture_provider():
    return UnresolvedPlanFixtureProvider()
