"""Goal feasibility, immutable edit framing and runtime operand evidence."""
import ast
from dataclasses import asdict
import json
import unittest

from mirage.generator.state_plan import goal_catalog, validate_plans, planning_request
from mirage.generator.mutation import plan_regions, apply_regions
from mirage.generator.program import ContractError
from mirage.contracts.capabilities import StatePredicate
from test_stateful import SOURCE, CONFIG


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
                ('torch.sin',), search_policy='round_robin', coverage_refinement=False)
            self.assertEqual(report['provider_calls'], 3)
            child = report['records'][1]
            self.assertTrue(child['inspection']['valid'], child.get('reason'))
            self.assertEqual(child['generation_mode'], 'local')
            self.assertEqual(child['state_plan']['status'], 'planned_not_verified')
            self.assertEqual(child['inspection']['predicates'][0]['observed'], 2)
            self.assertEqual(report['search_metrics']['planning_calls'], 1)

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
                ('torch.sin',), search_policy='round_robin', path_policy='mixed', coverage_refinement=False)
            parent, child = report['records']
            self.assertFalse(parent['configuration']['requires_backward'])
            self.assertFalse(parent['backward_executed'])
            self.assertFalse(child['configuration']['requires_backward'])
            self.assertTrue(child['inspection']['valid'], child.get('reason'))
            self.assertEqual(child['inspection']['predicates'][0]['observed'], 2)


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
