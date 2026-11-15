"""Pipeline budget, feedback causality, bounded overlap, and drain contracts."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mirage.generator.provider import Request, Response, Stage, Usage
from mirage.runtime.pipeline import BudgetedProvider, ExecutionPipeline, GenerationBudgetExhausted
from mirage.sampler.representations import RepresentationTraversal


def parent(root, number=0, actions=('clone', 'strided', 'offset', 'negative', 'canonical')):
    directory = root / f'parent-{number}'
    (directory / 'execution').mkdir(parents=True)
    source = f'# immutable source {number}\n'
    (directory / 'program.py').write_text(source)
    (directory / 'execution/source.pt').write_bytes(b'initialization')
    row = {'directory': str(directory), 'configuration': {'api': 'torch.linalg.slogdet', 'seed': 0},
           'inspection': {'valid': True}, 'mutation_obligations_valid': True,
           'process': {'elapsed_seconds': 29}, 'verdict': 'inapplicable',
           'mutation_candidates': [{'path': ['args', 0], 'call_index': 0, 'action': a,
               'eligibility': {'status': 'verified', 'reason': 'runtime candidate'}} for a in actions]}
    return source, row


class FakeProvider:
    def __init__(self, hook=None):
        self.calls = 0
        self.request_timings = []
        self.requests = []
        self.hook = hook

    def generate(self, request, *, deadline):
        self.calls += 1
        self.requests.append(request)
        start = time.time()
        if self.hook:
            self.hook()
        finish = time.time()
        self.request_timings.append({'elapsed_seconds': .25, 'started_epoch': start,
            'finished_epoch': finish, 'basis': 'provider_request_end_to_end'})
        return Response('{}', Usage(1, 1))


class PipelineTests(unittest.TestCase):
    def pipeline(self, root, budget=1):
        return ExecutionPipeline(root, 'torch.linalg.slogdet', budget, 120, replay_failures=0)

    def test_only_model_requests_spend_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.pipeline(Path(tmp), .5)
            provider = FakeProvider()
            b = BudgetedProvider(provider, p)
            request = Request(Stage.CANONICAL, 'generate')
            p.worker_events.append({'started': time.time()-29, 'finished': time.time(), 'seconds': 29})
            b.generate(request)
            self.assertEqual(p.model_seconds, .25)
            b.generate(request)
            with self.assertRaises(GenerationBudgetExhausted):
                b.generate(request)
            self.assertEqual(provider.calls, 2)
            self.assertEqual(p.timing()['framework_worker_seconds'], 29)
            p.close()

    def test_feedback_is_a_snapshot_and_next_call_receives_completed_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.pipeline(Path(tmp))
            provider = FakeProvider(lambda: p.observe({'directory': 'finished', 'verdict': 'pass'}))
            b = BudgetedProvider(provider, p)
            request = Request(Stage.CANONICAL, 'generate')
            b.generate(request)
            b.generate(request)
            snapshots = [r.context['feedback_snapshot'] for r in provider.requests]
            self.assertEqual([x['version'] for x in snapshots], [0, 1])
            self.assertEqual(snapshots[0]['completed_cases'], [])
            self.assertEqual(len(snapshots[1]['completed_cases']), 1)
            self.assertNotIn('feedback_snapshot', request.context)
            p.close()

    def test_failed_call_is_charged(self):
        class Fails(FakeProvider):
            def generate(self, request, *, deadline):
                super().generate(request, deadline=deadline)
                raise RuntimeError('provider failed')
        with tempfile.TemporaryDirectory() as tmp:
            p = self.pipeline(Path(tmp))
            with self.assertRaises(RuntimeError):
                BudgetedProvider(Fails(), p).generate(Request(Stage.CANONICAL, 'generate'))
            self.assertEqual(p.model_seconds, .25)
            self.assertEqual(p.model_events[0]['outcome'], 'RuntimeError')
            p.close()

    def test_model_and_worker_actually_overlap(self):
        from mirage.runtime.process import ProcessResult
        entered, leave = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            p = self.pipeline(Path(tmp))
            def worker(request, directory, timeout, **kw):
                entered.set()
                self.assertTrue(leave.wait(5))
                return ProcessResult('completed', 0, .1, str(directory))
            with patch('mirage.runtime.pipeline.run_worker', worker):
                thread = threading.Thread(target=lambda: p.worker({'mode': 'test'}, Path(tmp)/'worker', 10))
                thread.start()
                self.assertTrue(entered.wait(5))
                def model():
                    self.assertTrue(thread.is_alive())
                    time.sleep(.01)
                BudgetedProvider(FakeProvider(model), p).generate(Request(Stage.CANONICAL, 'generate'))
                leave.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())
            self.assertGreater(p.timing()['overlap_seconds'], 0)
            p.close()

    def test_exhausted_model_budget_still_drains_negative_and_other_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = self.pipeline(root)
            source, row = parent(root)
            row['verdict'] = 'suspected_failure'
            p.register(source, row)
            p.model_seconds = p.budget
            actions = []
            def execute(origin, candidate, source):
                actions.append(candidate['plan']['action'])
                return {'directory': str(root/candidate['id']), 'started_seconds': 0, 'finished_seconds': 1,
                        'representation_candidate_attained': True, 'mutation_obligations_valid': True,
                        'representation_traversal': {}}
            p.execute_representation = execute
            p.start()
            p.finish()
            self.assertEqual(actions, ['clone', 'strided', 'offset', 'negative', 'canonical'])
            self.assertTrue(p.summary()['dispatch_complete'])
            self.assertTrue(p.summary()['all_eligible_attained'])
            p.close()

    def test_backpressure_bounds_pending_sources_without_charging_model_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.pipeline(Path(tmp))
            for i in range(2):
                p.register(*parent(Path(tmp), i))
            ready = threading.Event()
            thread = threading.Thread(target=lambda: (p.wait_capacity(), ready.set()))
            thread.start()
            self.assertFalse(ready.wait(.02))
            with p.condition:
                for c in p.traversal.sources[0]['candidates']:
                    c['status'] = 'reached'
                p.condition.notify_all()
            self.assertTrue(ready.wait(5))
            thread.join()
            self.assertEqual(p.model_seconds, 0)
            p.close()

    def test_resume_requeues_only_interrupted_work_and_checks_parent_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = self.pipeline(root)
            p.register(*parent(root))
            a, c, _, _ = p.traversal.choose(1000, 0)
            p.traversal.complete(a['source_id'], c['id'], {'directory': 'done',
                'started_seconds': 0, 'finished_seconds': 1, 'representation_candidate_attained': True,
                'mutation_obligations_valid': True})
            p.traversal.choose(1000, 0)
            p.close(interrupted=True)
            resumed = ExecutionPipeline(root, p.api, 1, 120, resume=True)
            self.assertEqual([c['status'] for c in resumed.traversal.sources[0]['candidates']],
                             ['reached', 'pending', 'pending', 'pending', 'pending'])
            (root/'parent-0/program.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                resumed.traversal.choose(1000, 0)
            resumed.close()

    def test_runtime_unverified_is_terminal_but_not_attained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.pipeline(Path(tmp))
            p.register(*parent(Path(tmp), actions=('negative',)))
            origin, candidate, _, _ = p.traversal.choose(1000, 0)
            p.traversal.complete(origin['source_id'], candidate['id'], {
                'directory': 'failed', 'started_seconds': 0, 'finished_seconds': 1,
                'representation_candidate_attained': False, 'reason': 'state not reached'})
            self.assertTrue(p.summary()['dispatch_complete'])
            self.assertFalse(p.summary()['all_eligible_attained'])
            p.close()

    def test_multi_api_interruption_cannot_report_partial_panel_complete(self):
        from mirage.contracts.programs import RunOptions
        from mirage.workflow import run_stateful
        with tempfile.TemporaryDirectory() as tmp:
            options = RunOptions(str(Path(tmp)/'panel'), 'unused', generation_pipeline=True,
                                 strategy='stateful', contract_version=4)
            first = {'strategy': 'stateful', 'records': [], 'provider_calls': 1,
                     'elapsed_seconds': 1, 'pipeline_complete': True,
                     'stop_reason': 'provider_cancellation_unconfirmed'}
            with patch('mirage.workflow._run_stateful', return_value=first) as run:
                report = run_stateful(options, ('torch.add', 'torch.mul'), representation_traversal=True)
            self.assertEqual(run.call_count, 1)
            self.assertFalse(report['pipeline_complete'])
            self.assertEqual(report['stop_reason'], 'pipeline_incomplete')
            self.assertEqual(report['completed_api_runs'], 1)
            self.assertEqual(report['expected_apis'], ['torch.add', 'torch.mul'])


SOURCE = '''import torch

def build(device, seed):
    x = torch.tensor([[2.0, 0.5], [0.25, 3.0]], dtype=torch.float64, device=device, requires_grad=True)
    return {'inputs': {'x': x}, 'parameters': {}}

def run(ctx, probe):
    sign, value = probe.call(torch.linalg.slogdet, ctx['inputs']['x'])
    value.backward()
    return {'value': value}
'''


class FixtureProvider:
    def generate(self, request, *, deadline):
        return Response(json.dumps({'source': SOURCE}), Usage(1, 1))


def create():
    return FixtureProvider()


class PipelineRuntimeTests(unittest.TestCase):
    def test_controller_drains_after_source_case_limit(self):
        from mirage.contracts.programs import RunOptions
        from mirage.workflow import run_stateful
        with tempfile.TemporaryDirectory() as tmp:
            options = RunOptions(str(Path(tmp)/'run'), 'test_pipeline:create', cases=1,
                time_limit=30, worker_timeout=120, strategy='stateful', contract_version=4,
                generation_pipeline=True, replay_failures=0)
            with patch.dict(os.environ, {'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', '')}):
                report = run_stateful(options, ('torch.linalg.slogdet',), predicates=[],
                    coverage_refinement=False, representation_traversal=True, oracle_improvements=True)
            self.assertTrue(report['pipeline_complete'])
            self.assertEqual(report['provider_calls'], 1)
            candidates = [c for s in report['representation_traversal']['sources'] for c in s['candidates']]
            self.assertTrue(candidates, report['records'])
            self.assertTrue(all(c['status'] not in ('pending', 'dispatched') for c in candidates))
            negative = [c for c in candidates if c['plan']['action'] == 'negative']
            self.assertEqual(len(negative), 1)
            self.assertEqual(negative[0]['status'], 'reached')
            self.assertGreater(report['pipeline_timing']['framework_worker_seconds'],
                               report['pipeline_timing']['llm_request_seconds'])

            from mirage.runtime.pipeline import resume_pipeline
            with patch('mirage.generator.provider.IsolatedProvider.generate', side_effect=AssertionError('resume called model')):
                resumed = resume_pipeline(Path(tmp)/'run')
            self.assertTrue(resumed['pipeline_complete'])
            self.assertEqual(len(resumed['records']), len(report['records']))
            self.assertEqual(resumed['pipeline_timing']['llm_request_seconds'],
                             report['pipeline_timing']['llm_request_seconds'])
