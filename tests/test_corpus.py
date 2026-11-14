"""Corpus admission controls and isolated authored-provider protocol validation."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mirage.artifacts.bundle import inventory
from mirage.artifacts.corpus import Corpus
from mirage.contracts.capabilities import SEMANTIC_COVERAGE_VERSION
from mirage.generator.provider import Response
from test_stateful import SOURCE
from test_mutation import TAIL


def fixture_bundle(root, number, *, source=SOURCE, coverage='base', verdict='pass', seconds=10):
    """Synthetic trusted-inspector records for index policy tests, not MR evidence."""
    directory = root / f'attempt-{number}'
    directory.mkdir()
    (directory/'program.py').write_text(source)
    (directory/'source.pt').write_bytes(b'authored initialization reference fixture')
    engine = root/'engine'
    engine.mkdir(exist_ok=True)
    config = {'api':'torch.sin', 'device':'cpu'}
    manifest = {'configuration':config, 'source':'program.py', 'initial_reference':'source.pt',
                'files':inventory(directory), 'engine':str(engine), 'engine_inventory':inventory(engine)}
    path = directory/'stateful-manifest.json'
    path.write_text(json.dumps(manifest))
    row = {'api':'torch.sin', 'configuration':config, 'verdict':verdict,
           'correctness':'assessed' if verdict == 'pass' else 'unassessed', 'relation':'representation',
           'backward_executed':True, 'transition_signature':['target_return', 'backward'],
           'started_seconds':0, 'finished_seconds':seconds,
           'inspection':{'valid':True, 'semantic_coverage':{'version':SEMANTIC_COVERAGE_VERSION,
                         'eligible':True, 'signatures':[{'api':'torch.sin', 'unit':coverage}]}}}
    return source, row, path


class CorpusTests(unittest.TestCase):
    def test_different_coverage_keeps_parent_and_round_robins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = Corpus(root)
            first = corpus.admit(*fixture_bundle(root, 0))
            second = corpus.admit(*fixture_bundle(root, 1, coverage='negative'), parent_id=first['seed_id'], edit={'kind':'fixture'})
            self.assertTrue(second['admitted'])
            self.assertEqual(corpus.entries[1]['parent_id'], first['seed_id'])
            self.assertEqual([corpus.select('torch.sin')['id'] for _ in range(3)],
                             [first['seed_id'], second['seed_id'], first['seed_id']])
            loaded = Corpus(root)
            self.assertEqual(loaded.select('torch.sin')['id'], second['seed_id'])
            self.assertTrue(all(entry['active'] for entry in corpus.entries))

    def test_unknown_invalid_or_failed_obligations_do_not_enter_corpus(self):
        for field in ('valid', 'eligible', 'version', 'mutation'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); corpus = Corpus(root)
                source, row, manifest = fixture_bundle(root, 0)
                if field == 'valid': row['inspection']['valid'] = False
                elif field == 'mutation': row['mutation_obligations_valid'] = False
                else: row['inspection']['semantic_coverage'][field] = False
                self.assertFalse(corpus.admit(source, row, manifest)['admitted'])
                self.assertIsNone(corpus.select('torch.sin'))

    def test_coverage_only_seed_remains_unassessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = Corpus(root)
            self.assertTrue(corpus.admit(*fixture_bundle(root, 0, verdict='inapplicable'))['admitted'])
            seed = corpus.select('torch.sin')
            self.assertFalse(seed['mr_applicable'])
            self.assertEqual(seed['correctness'], 'unassessed')

    def test_suspected_failures_are_persisted_but_never_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = Corpus(root)
            result = corpus.admit(*fixture_bundle(root, 0, verdict='suspected_failure'))
            self.assertEqual(result['reason'], 'quarantined_suspected_failure')
            self.assertIsNone(Corpus(root).select('torch.sin'))
            self.assertEqual(len(Corpus(root).quarantine), 1)

    def test_duplicate_and_cheaper_seed_preserve_old_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = Corpus(root)
            first = corpus.admit(*fixture_bundle(root, 0))
            self.assertEqual(corpus.admit(*fixture_bundle(root, 1, seconds=20))['reason'], 'duplicate_coverage')
            self.assertEqual(corpus.admit(*fixture_bundle(root, 2, seconds=5))['reason'], 'lower_observed_cost')
            self.assertFalse(corpus.entries[0]['active'])
            self.assertEqual(corpus.entries[0]['id'], first['seed_id'])
            self.assertEqual(corpus.source(corpus.entries[0]), SOURCE)

    def test_alternative_structures_are_bounded_and_relations_are_revalidated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = Corpus(root)
            corpus.admit(*fixture_bundle(root, 0))
            alternate = SOURCE.replace('y.sum().backward()', '(y.sum() + 0).backward()')
            third = SOURCE.replace('y.sum().backward()', '(y.sum() * 1).backward()')
            self.assertEqual(corpus.admit(*fixture_bundle(root, 1, source=alternate))['reason'], 'alternative_program_structure')
            self.assertFalse(corpus.admit(*fixture_bundle(root, 2, source=third))['admitted'])
            source, row, manifest = fixture_bundle(root, 3, source=third)
            row['relation'] = 'execution_mode'
            self.assertEqual(corpus.admit(source, row, manifest)['reason'], 'new_applicable_relation')

    def test_tampered_source_or_missing_initialization_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = Corpus(root)
            source, row, manifest = fixture_bundle(root, 0)
            corpus.admit(source, row, manifest)
            (manifest.parent/'program.py').write_text(SOURCE + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                corpus.select('torch.sin')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); corpus = Corpus(root)
            source, row, manifest = fixture_bundle(root, 0)
            (manifest.parent/'source.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'initialization'):
                corpus.admit(source, row, manifest)


class LocalFixtureProvider:
    """Authored edits exercise protocol; not model quality or discovery evidence."""
    def generate(self, request, deadline=None):
        contract = request.context.get('mutation_contract')
        if contract is None:
            source = SOURCE
        elif contract['kind'] == 'run_tail':
            source = TAIL
        elif contract['kind'] == 'operand_setup':
            source = 'x = x.clone()'
        else:
            source = 'x = torch.tensor([0., 0.75], dtype=torch.float64, device=device, requires_grad=True)'
        return Response(json.dumps({'source':source}), model='authored-local-fixture')


def local_fixture_provider():
    return LocalFixtureProvider()


class CorpusWorkflowTests(unittest.TestCase):
    def options(self, root, cases):
        from mirage.contracts.programs import RunOptions
        return RunOptions(str(root/'run'), 'test_corpus:local_fixture_provider', cases=cases,
                          time_limit=600, worker_timeout=60, strategy='stateful', contract_version=4)

    def provider_environment(self):
        return patch.dict(os.environ, {'PYTHONPATH':str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH','')})

    def test_local_children_keep_parent_and_independent_replay(self):
        from mirage.workflow import run_stateful, replay_stateful
        import torch
        with tempfile.TemporaryDirectory() as tmp, self.provider_environment():
            root = Path(tmp)
            result = run_stateful(self.options(root, 5), ('torch.sin',), predicates=[], coverage_refinement=False, search_policy='round_robin')
            rows = result['records']
            self.assertEqual(len(rows), 5)
            self.assertEqual([r['generation_reason'] for r in rows],
                ['periodic_fresh_generation','local_mutation','local_mutation','local_mutation','periodic_fresh_generation'])
            self.assertTrue(all(r['inspection']['valid'] and r['mutation_obligations_valid'] for r in rows), rows)
            self.assertGreaterEqual(len(result['corpus']['entries']), 4)
            self.assertTrue(all(r['corpus_admission']['admitted'] for r in rows[:4]))
            first_directory, child_directory = Path(rows[0]['directory']), Path(rows[3]['directory'])
            self.assertEqual((first_directory/'program.py').read_text(), SOURCE)
            first = torch.load(first_directory/'execution/source.pt', weights_only=True)
            child = torch.load(child_directory/'execution/source.pt', weights_only=True)
            self.assertFalse(torch.equal(first['initial']['inputs']['x'], child['initial']['inputs']['x']))
            for directory in (first_directory, child_directory):
                replay = replay_stateful(directory/'stateful-manifest.json', timeout=60)
                self.assertTrue(replay['replay_valid'], replay)
                self.assertEqual(replay['provider_calls'], 0)
            corpus = Corpus(root/'run')
            for entry in corpus.entries:
                self.assertTrue(corpus.source(entry))
            self.assertTrue(all(e['manifest_sha256'] and e['initial_reference'] for e in corpus.entries))

    def test_suspected_violation_never_enters_repairs_even_with_missing_state(self):
        import mirage.workflow as workflow
        original = workflow.run_worker
        def injected(request, directory, *args, **kwargs):
            process = original(request, directory, *args, **kwargs)
            if request['mode'] == 'execute_stateful_pair' and process.outcome == 'completed':
                path = Path(directory)/'summary.json'
                outcome = json.loads(path.read_text())
                outcome.update(verdict='suspected_failure', correctness='assessed')
                outcome['inspection']['valid'] = False
                path.write_text(json.dumps(outcome))
            return process
        with tempfile.TemporaryDirectory() as tmp, self.provider_environment(), patch.object(workflow, 'run_worker', injected):
            result = workflow.run_stateful(self.options(Path(tmp), 2), ('torch.sin',), predicates=[], coverage_refinement=False, search_policy='round_robin')
            self.assertEqual(len(result['corpus']['quarantine']), 2)
            self.assertEqual(result['corpus']['entries'], [])
            for row in result['records']:
                request = json.loads((Path(row['directory'])/'request.json').read_text())
                self.assertIsNone(request['previous_content'])
                self.assertEqual(row['repair_index'], 0)

    def test_repair_retains_original_parent_and_edit_contract(self):
        from mirage.workflow import run_stateful
        with tempfile.TemporaryDirectory() as tmp, self.provider_environment():
            options = replace(self.options(Path(tmp), 3), provider='test_corpus:repair_fixture_provider')
            result = run_stateful(options, ('torch.sin',), predicates=[], coverage_refinement=False, search_policy='round_robin')
            first, rejected, repaired = result['records']
            self.assertEqual(rejected['status'], 'invalid_program')
            self.assertEqual(repaired['repair_index'], 1)
            self.assertTrue(repaired['corpus_admission']['admitted'])
            requests = [json.loads((Path(r['directory'])/'request.json').read_text()) for r in (rejected, repaired)]
            self.assertEqual(requests[0]['context']['mutation_contract'], requests[1]['context']['mutation_contract'])
            self.assertEqual(requests[1]['context']['canonical_source'], SOURCE)
            self.assertEqual(repaired['parent_seed_id'], first['corpus_admission']['seed_id'])

    def test_new_backward_call_cannot_replace_protected_parent_backward(self):
        from mirage.workflow import run_stateful
        with tempfile.TemporaryDirectory() as tmp, self.provider_environment():
            options = replace(self.options(Path(tmp), 2), provider='test_corpus:lost_backward_fixture_provider')
            result = run_stateful(options, ('torch.sin',), predicates=[], coverage_refinement=False, search_policy='round_robin')
            child = result['records'][1]
            self.assertTrue(child['inspection']['valid'])
            self.assertTrue(child['backward_executed'])
            self.assertFalse(child['mutation_obligations_valid'])
            self.assertFalse(child['corpus_admission']['admitted'])
            self.assertEqual(len(result['corpus']['entries']), 1)

    def test_input_refinement_is_a_new_local_child_not_full_regeneration(self):
        from mirage.workflow import run_stateful
        from mirage.contracts.capabilities import InputCoverage
        with tempfile.TemporaryDirectory() as tmp, self.provider_environment(), patch(
                'mirage.contracts.exploration.stateful_input_profile', return_value=InputCoverage(('zero',))):
            result = run_stateful(self.options(Path(tmp), 2), ('torch.sin',), predicates=[], search_policy='round_robin')
            first, child = result['records']
            self.assertTrue(first['coverage_refinement_requested'])
            self.assertEqual(child['generation_reason'], 'local_input_refinement')
            self.assertEqual(child['parent_seed_id'], first['corpus_admission']['seed_id'])
            self.assertTrue(child['corpus_admission']['admitted'])
            self.assertIn('zero', child['input_coverage']['target_observed'])


class RepairFixtureProvider(LocalFixtureProvider):
    def generate(self, request, deadline=None):
        if request.context.get('mutation_contract') and request.previous_content is None:
            return Response(json.dumps({'source':SOURCE}), model='authored-invalid-local-fixture')
        return super().generate(request, deadline=deadline)


def repair_fixture_provider():
    return RepairFixtureProvider()


class LostBackwardFixtureProvider(LocalFixtureProvider):
    def generate(self, request, deadline=None):
        if request.context.get('mutation_contract'):
            return Response(json.dumps({'source':"z = probe.call(torch.sin, x)\nz.sum().backward()\nreturn {'output': z, 'gradient': x.grad}"}),
                            model='authored-lost-backward-fixture')
        return super().generate(request, deadline=deadline)


def lost_backward_fixture_provider():
    return LostBackwardFixtureProvider()
