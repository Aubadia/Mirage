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
        return ExecutionPipeline(root, 'torch.linalg.slogdet', budget, 120, replay_failures=0, workers=1)

    def test_eight_and_32_slots_shared_by_sources_representations_and_replays(self):
        for workers in (8, 32):
            with self.subTest(workers=workers):
                self._check_shared_slots(workers)

    def _check_shared_slots(self, workers):
        from concurrent.futures import ThreadPoolExecutor
        entered, release = threading.Event(), threading.Event()
        lock = threading.Lock()
        active = peak = 0
        modes = []
        def work(mode):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                modes.append(mode)
                if active == workers:
                    entered.set()
            try:
                if not release.wait(10):
                    raise TimeoutError('parallel dispatch did not fill configured slots')
                time.sleep(.01)
            finally:
                with lock:
                    active -= 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = ExecutionPipeline(root, 'torch.linalg.slogdet', 1, 10, workers=workers)
            sources = (workers + 4) // 5
            for i in range(sources):
                source, row = parent(root, i)
                row['verdict'] = 'suspected_failure'
                p.register(source, row)
            def execute(origin, candidate, source):
                directory = root / origin['source_id'] / candidate['id']
                p.worker({'mode': 'representation'}, directory, 10)
                p.operation(work, 'replay')
                return {'directory': str(directory), 'started_seconds': 0, 'finished_seconds': 1,
                    'verdict': 'suspected_failure', 'representation_candidate_attained': True,
                    'mutation_obligations_valid': True, 'representation_traversal': {}}
            p.execute_representation = execute
            with patch('mirage.runtime.pipeline.run_worker', side_effect=lambda request, *a, **kw: work(request['mode'])):
                p.start()
                try:
                    self.assertTrue(entered.wait(10))
                    with ThreadPoolExecutor(2) as callers:
                        source = callers.submit(p.worker, {'mode': 'source'}, root/'source', 10)
                        replay = callers.submit(p.operation, work, 'source_replay')
                        self.assertFalse(source.done())
                        self.assertFalse(replay.done())
                        release.set()
                        source.result(timeout=10)
                        replay.result(timeout=10)
                    p.finish()
                finally:
                    release.set()
                    p.close(interrupted=True)
            self.assertEqual(peak, workers)
            self.assertEqual(modes.count('representation'), sources * 5)
            self.assertEqual(modes.count('replay'), sources * 5)
            self.assertIn('source', modes)
            self.assertIn('source_replay', modes)
            self.assertEqual(len(p.rows), sources * 5)
            self.assertTrue(p.summary()['all_eligible_attained'])
            self.assertEqual(p.timing()['framework_peak_workers'], workers)

    def test_parallel_interval_union_does_not_double_count_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.pipeline(Path(tmp))
            p.session_started = 1
            p.model_events = [{'request_started': 2, 'request_finished': 7,
                               'started': 2, 'finished': 7, 'seconds': 5}]
            p.worker_events = [{'started': a, 'finished': b, 'seconds': b-a}
                               for a,b in ((1,5), (3,8), (9,10))]
            with patch('mirage.runtime.pipeline.time.time', return_value=11):
                timing = p.timing()
            self.assertEqual(timing['framework_worker_seconds'], 10)
            self.assertEqual(timing['framework_busy_seconds'], 8)
            self.assertEqual(timing['overlap_seconds'], 5)
            self.assertEqual(timing['controller_and_unattributed_seconds'], 2)
            self.assertEqual(timing['framework_peak_workers'], 2)
            p.close()

    def test_cancel_all_inflight_then_resume_preserves_limit_and_backpressure(self):
        entered = threading.Barrier(4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = ExecutionPipeline(root, 'torch.linalg.slogdet', 1, 10, workers=3,
                                  generation_overlap=False)
            p.register(*parent(root))
            p.queue_seconds = 7.5
            def execute(*args):
                entered.wait(timeout=10)
                p.cancel.wait(10)
                return {}
            p.execute_representation = execute
            p.start()
            entered.wait(timeout=10)
            p.close(interrupted=True)
            self.assertFalse(p.thread.is_alive())
            self.assertEqual(p.summary()['status_counts']['dispatched'], 3)
            resumed = ExecutionPipeline(root, p.api, 1, 10, resume=True)
            self.assertEqual(resumed.workers, 3)
            self.assertFalse(resumed.generation_overlap)
            self.assertEqual(resumed.timing()['generation_backpressure_seconds'], 7.5)
            self.assertEqual(resumed.summary()['status_counts'], {'pending': 5})
            resumed.close()
            state = json.loads((root/'pipeline-state.json').read_text())
            self.assertEqual(state['traversal'], json.loads((root/'representation-traversal.json').read_text()))
            for key in ('workers', 'generation_overlap', 'queue_seconds', 'backpressure_measurement_complete'):
                state.pop(key)
            (root/'pipeline-state.json').write_text(json.dumps(state))
            legacy = ExecutionPipeline(root, p.api, 1, 10, resume=True)
            self.assertEqual(legacy.workers, 1)
            self.assertTrue(legacy.generation_overlap)
            self.assertIsNone(legacy.timing()['generation_backpressure_seconds'])
            self.assertFalse(legacy.timing()['backpressure_measurement_complete'])
            legacy.close()

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

    def test_alternating_provider_waits_for_full_parallel_drain_and_feedback(self):
        from concurrent.futures import ThreadPoolExecutor
        import queue
        entered = threading.Barrier(3)
        release = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = ExecutionPipeline(root, 'torch.linalg.slogdet', 1, 10, workers=2,
                                  generation_overlap=False)
            p.register(*parent(root))
            count = 0
            lock = threading.Lock()
            def work(mode):
                nonlocal count
                with lock:
                    count += 1
                    first = count <= 2
                if first:
                    entered.wait(timeout=5)
                if not release.wait(5):
                    raise TimeoutError('test did not release framework work')
                time.sleep(.01)
            def execute(origin, candidate, source):
                directory = root/candidate['id']
                p.operation(work, 'representation')
                p.operation(work, 'replay')
                return {'directory': str(directory), 'started_seconds': 0, 'finished_seconds': 1,
                        'representation_candidate_attained': True, 'mutation_obligations_valid': True,
                        'representation_traversal': {}}
            def consume():
                while True:
                    try:
                        row = p.completed.get_nowait()
                    except queue.Empty:
                        return
                    p.observe(row)
            p.execute_representation = execute
            p.consume = consume
            provider = FakeProvider(lambda: time.sleep(.01))
            p.start()
            try:
                entered.wait(timeout=5)
                with ThreadPoolExecutor(1) as callers:
                    call = callers.submit(BudgetedProvider(provider, p).generate,
                                          Request(Stage.CANONICAL, 'generate'))
                    try:
                        time.sleep(.03)
                        self.assertFalse(call.done())
                        self.assertEqual(provider.calls, 0)
                    finally:
                        release.set()
                    call.result(timeout=10)
                p.finish()
                self.assertEqual(count, 10)
                self.assertEqual(p.pending_sources(), 0)
                self.assertEqual(provider.requests[0].context['feedback_snapshot']['version'], 5)
                self.assertEqual(p.timing()['overlap_seconds'], 0)
                self.assertEqual(p.timing()['framework_peak_workers'], 2)
                self.assertEqual(p.model_seconds, .25)
                self.assertFalse(p.timing()['generation_overlap'])
            finally:
                release.set()
                p.close(interrupted=True)

    def test_alternating_wait_cancellation_does_not_start_or_charge_model(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = ExecutionPipeline(root, 'torch.linalg.slogdet', 1, 10, generation_overlap=False)
            p.register(*parent(root))
            provider = FakeProvider()
            try:
                with ThreadPoolExecutor(1) as callers:
                    call = callers.submit(BudgetedProvider(provider, p).generate,
                                          Request(Stage.CANONICAL, 'generate'))
                    try:
                        time.sleep(.03)
                        self.assertFalse(call.done())
                    finally:
                        with p.condition:
                            p.cancel.set()
                            p.condition.notify_all()
                    with self.assertRaises(InterruptedError):
                        call.result(timeout=5)
                self.assertEqual(provider.calls, 0)
                self.assertEqual(p.model_events, [])
                self.assertEqual(p.model_seconds, 0)
            finally:
                p.close(interrupted=True)

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
    def test_alternating_controller_publishes_first_drain_before_second_prompt(self):
        from mirage.contracts.programs import RunOptions
        from mirage.workflow import run_stateful
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            options = RunOptions(str(root), 'test_pipeline:create', cases=2,
                time_limit=120, worker_timeout=120, strategy='stateful', contract_version=4,
                generation_pipeline=True, generation_overlap=False, workers=2, replay_failures=0)
            with patch.dict(os.environ, {'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', '')}):
                report = run_stateful(options, ('torch.linalg.slogdet',), predicates=[],
                    search_policy='round_robin', generation_mode='full', coverage_refinement=False,
                    representation_traversal=True, oracle_improvements=True)
            self.assertEqual(report['provider_calls'], 2)
            self.assertTrue(report['pipeline_complete'])
            state = json.loads((root/'pipeline-state.json').read_text())
            second = json.loads((root/'provider-002/request.json').read_text())['request']
            snapshot = second['context']['feedback_snapshot']
            self.assertTrue(any('representations' in r['directory'] for r in snapshot['completed_cases']))
            first_source = state['source_registrations'][0]['source_id']
            first_events = [event for event in state['worker_events']
                            if f'/representations/{first_source}/' in event['directory']]
            self.assertTrue(first_events)
            second_model_start = state['model_events'][1]['started']
            self.assertTrue(all(event['finished'] <= second_model_start for event in first_events))
            self.assertEqual(report['pipeline_timing']['overlap_seconds'], 0)
            self.assertFalse(report['pipeline_timing']['generation_overlap'])
            self.assertEqual(report['pipeline_timing']['framework_worker_limit'], 2)

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
            self.assertEqual(report['pipeline_timing']['framework_worker_limit'], 8)
            self.assertGreater(report['pipeline_timing']['framework_peak_workers'], 1)
            self.assertLessEqual(report['pipeline_timing']['framework_peak_workers'], 8)

            from mirage.runtime.pipeline import resume_pipeline
            with patch('mirage.generator.provider.IsolatedProvider.generate', side_effect=AssertionError('resume called model')):
                resumed = resume_pipeline(Path(tmp)/'run')
            self.assertTrue(resumed['pipeline_complete'])
            self.assertEqual(len(resumed['records']), len(report['records']))
            self.assertEqual(resumed['pipeline_timing']['llm_request_seconds'],
                             report['pipeline_timing']['llm_request_seconds'])
