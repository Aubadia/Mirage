"""Adversarial resource scheduling and independent strategy execution controls."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mirage.contracts.programs import RunOptions
from mirage.contracts.scheduling import APIJob, SchedulingOptions
from mirage.runtime.multi_api import run_sessions
from mirage.runtime.scheduler import (
    SharedScheduler, SchedulingCancelled, SessionScope, current_scope,
    generation_slot, session_scope,
)


def wait_until(predicate, seconds=5):
    end = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError('Timed out waiting for scheduling evidence')
        time.sleep(.005)


class SchedulerTests(unittest.TestCase):
    def test_capacity_contract_rejects_invalid_limits(self):
        for changes in ({'workers': 0}, {'workers': True}, {'active_api_limit': 0},
                        {'per_api_worker_limit': 3}, {'per_api_generation_limit': 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SchedulingOptions(**(dict(workers=2) | changes))

    def test_fifo_within_session_and_round_robin_between_ready_sessions(self):
        scheduler = SharedScheduler(SchedulingOptions(1))
        entered = []
        def task(api, name):
            with scheduler.lease(api, 'execution', label=name):
                entered.append(name)
        with ThreadPoolExecutor(3) as pool:
            with scheduler.lease('held', 'execution'):
                futures = []
                for api, name in [('a', 'a0'), ('b', 'b0'), ('a', 'a1')]:
                    futures.append(pool.submit(task, api, name))
                    wait_until(lambda: len(scheduler.snapshot()['events']) == len(futures) + 1)
            for future in futures:
                future.result(timeout=5)
        self.assertEqual(entered, ['a0', 'b0', 'a1'])
        self.assertEqual(scheduler.snapshot()['active']['execution'], 0)

    def test_idle_capacity_is_borrowed_and_per_api_cap_is_enforced(self):
        for cap, expected in [(3, 3), (1, 1)]:
            scheduler = SharedScheduler(SchedulingOptions(3, per_api_worker_limit=cap))
            release = threading.Event()
            entered = []
            def task():
                with scheduler.lease('only-ready-api', 'execution'):
                    entered.append(1)
                    release.wait(5)
            with ThreadPoolExecutor(3) as pool:
                futures = [pool.submit(task) for _ in range(3)]
                try:
                    wait_until(lambda: len(entered) == expected)
                    self.assertEqual(scheduler.snapshot()['active']['execution'], expected)
                finally:
                    release.set()
                for future in futures:
                    future.result(timeout=5)

    def test_generation_does_not_occupy_execution_capacity(self):
        scheduler = SharedScheduler(SchedulingOptions(1, 1))
        with scheduler.lease('a', 'execution'):
            with scheduler.lease('b', 'generation', deadline=time.monotonic() + 1):
                self.assertEqual(scheduler.snapshot()['active'], {'execution': 1, 'generation': 1})

    def test_cuda_serializes_without_blocking_independent_cpu_work(self):
        scheduler = SharedScheduler(SchedulingOptions(3))
        with ThreadPoolExecutor(2) as pool:
            with scheduler.lease('a', 'execution', resource='cuda'):
                def cuda():
                    with scheduler.lease('b', 'execution', resource='cuda'):
                        return 'cuda'
                pending = pool.submit(cuda)
                wait_until(lambda: len(scheduler.snapshot()['events']) == 2)
                with scheduler.lease('c', 'execution', resource='cpu'):
                    self.assertFalse(pending.done())
                    self.assertEqual(scheduler.snapshot()['cuda_active'], 1)
            self.assertEqual(pending.result(timeout=5), 'cuda')

    def test_expired_queue_entry_never_launches_or_leaks_a_slot(self):
        scheduler = SharedScheduler(SchedulingOptions(1))
        with ThreadPoolExecutor(1) as pool:
            with scheduler.lease('held', 'execution'):
                def expired():
                    with scheduler.lease('a', 'execution', deadline=time.monotonic() + .05):
                        self.fail('Expired work must not execute')
                future = pool.submit(expired)
                with self.assertRaises(SchedulingCancelled):
                    future.result(timeout=5)
            with scheduler.lease('after', 'execution'):
                self.assertEqual(scheduler.snapshot()['active']['execution'], 1)
        self.assertEqual(scheduler.snapshot()['events'][1]['state'], 'cancelled')

    def test_local_cancellation_does_not_abort_other_api(self):
        scheduler = SharedScheduler(SchedulingOptions(1))
        cancel = threading.Event()
        with ThreadPoolExecutor(1) as pool:
            with scheduler.lease('held', 'execution'):
                def local():
                    with scheduler.lease('a', 'execution', cancel=cancel):
                        self.fail('Cancelled work must not execute')
                future = pool.submit(local)
                wait_until(lambda: len(scheduler.snapshot()['events']) == 2)
                cancel.set()
                with self.assertRaises(SchedulingCancelled):
                    future.result(timeout=5)
            with scheduler.lease('b', 'execution'):
                self.assertFalse(scheduler.cancel.is_set())

    def test_shared_abort_cancels_all_registered_sessions(self):
        scheduler = SharedScheduler(SchedulingOptions(1))
        cancellations = [threading.Event(), threading.Event()]
        for i, cancel in enumerate(cancellations):
            with session_scope(SessionScope(scheduler, str(i), 'cpu', cancel)):
                pass
        scheduler.abort('unconfirmed shared request')
        self.assertTrue(all(cancel.is_set() for cancel in cancellations))
        with self.assertRaises(SchedulingCancelled):
            with scheduler.lease('next', 'generation'):
                self.fail('Aborted shared endpoint must not accept requests')

    def test_nested_generation_gate_uses_one_lease(self):
        scheduler = SharedScheduler(SchedulingOptions(1, 1))
        with session_scope(SessionScope(scheduler, 'a', 'cpu', threading.Event())):
            with generation_slot():
                with generation_slot():
                    self.assertEqual(scheduler.snapshot()['active']['generation'], 1)
        self.assertEqual(len(scheduler.snapshot()['events']), 1)

    def test_process_queue_respects_session_wall_deadline(self):
        from mirage.runtime.process import run_worker
        scheduler = SharedScheduler(SchedulingOptions(1))
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(1) as pool:
            with scheduler.lease('held', 'execution'):
                def expire():
                    scope = SessionScope(scheduler, 'a', 'cpu', threading.Event(), time.monotonic() + .05)
                    with session_scope(scope):
                        return run_worker({'mode': 'should_not_launch'}, Path(tmp) / 'task', 10)
                with patch('mirage.runtime.process.subprocess.Popen') as launch:
                    result = pool.submit(expire).result(timeout=5)
                    launch.assert_not_called()
                self.assertEqual(result.outcome, 'timeout')
                self.assertGreater(result.resources['scheduling']['queue_seconds'], .02)

    def test_inherited_replay_cancellation_survives_shared_dispatch(self):
        from mirage.runtime.process import cancellation_scope, run_worker
        scheduler = SharedScheduler(SchedulingOptions(1))
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as tmp:
            scope = SessionScope(scheduler, 'a', 'cpu', threading.Event())
            with session_scope(scope), cancellation_scope(cancel), patch('mirage.runtime.process.subprocess.Popen') as launch:
                result = run_worker({'mode': 'cancelled-replay'}, Path(tmp) / 'task', 10)
            launch.assert_not_called()
            self.assertEqual(result.outcome, 'timeout')
            self.assertFalse(scope.cancel.is_set())
            self.assertFalse(scheduler.cancel.is_set())

    def test_model_budget_does_not_charge_wait_for_shared_generation(self):
        from mirage.generator.provider import Request, Response, Stage, Usage
        from mirage.runtime.pipeline import BudgetedProvider, ExecutionPipeline
        scheduler = SharedScheduler(SchedulingOptions(1, 1))
        class Provider:
            calls = 0
            request_timings = []
            def generate(self, request, *, deadline):
                self.calls += 1
                self.request_timings.append(dict(elapsed_seconds=.01, basis='fixture'))
                return Response('{}', Usage(1, 1))
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(1) as pool:
            scope = SessionScope(scheduler, 'a', 'cpu', threading.Event())
            with session_scope(scope):
                pipeline = ExecutionPipeline(Path(tmp), 'generic.operation', .05, 1, workers=1)
                provider = BudgetedProvider(Provider(), pipeline)
                try:
                    with scheduler.lease('held', 'generation'):
                        future = pool.submit(copy_context().run, provider.generate, Request(Stage.CANONICAL, 'fixture', {}))
                        wait_until(lambda: len(scheduler.snapshot()['events']) == 2)
                        time.sleep(.08)
                    future.result(timeout=5)
                    self.assertAlmostEqual(pipeline.model_seconds, .01)
                    self.assertGreater(scheduler.snapshot()['events'][1]['queue_seconds'], .05)
                finally:
                    pipeline.close()

    def test_provider_uncertainty_retains_only_its_slot(self):
        from mirage.generator.provider import IsolatedProvider, ProviderFailure, Request, Stage
        scheduler = SharedScheduler(SchedulingOptions(1))
        with tempfile.TemporaryDirectory() as tmp:
            provider = IsolatedProvider('vllm', Path(tmp), 1)
            with session_scope(SessionScope(scheduler, 'a', 'cpu', threading.Event())):
                with patch.object(provider, '_timed_generate', side_effect=ProviderFailure('timeout', 'unknown', cancellation_uncertain=True)):
                    with self.assertRaises(ProviderFailure):
                        provider.generate(Request(Stage.CANONICAL, '', {}), deadline=time.monotonic() + 1)
        self.assertFalse(scheduler.cancel.is_set())
        self.assertEqual(scheduler.snapshot()['active']['generation'], 1)


class SessionTests(unittest.TestCase):
    def test_benchmark_audit_requires_promised_members_and_retains_trial_cost(self):
        import sys
        with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1]/'benchmarks'), *sys.path]):
            from stateful_check_e2e import score_session, write
            from pytorch_correctness import evaluate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root/'execution/attempt-0000'
            write(root/'summary.json', dict(status='completed', terminal=True))
            write(case/'outcome.json', dict(attempt=0, source_completed=True, followup_completed=True,
                api_executed=True, inspection={'valid':False,'legal_source':True}))
            for n, elapsed in enumerate((1.,2.)):
                stage = root/'differential/attempt-0000/source'/f'trial-{n}'/'0-cpu'
                write(stage/'request.json', {})
                write(stage/'process.json',dict(elapsed_seconds=elapsed))
            session = dict(directory=str(root), session_id='test', state='completed')
            target = dict(api='generic', id='generic', device='cpu')
            # Only the source was saved, although both the source and followup were promised.
            (case/'execution').mkdir()
            (case/'execution/source.pt').touch()
            with patch('mirage.artifacts.bundle.load_tensor_record', return_value={}), \
                 patch.object(evaluate,'stateful_independent_audit',return_value={'label':'negative'}) as audited, \
                 patch.object(evaluate,'stateful_attribution',return_value={'structural_trigger':False,'benchmark_reproduction':False}):
                result = score_session(session,target,time.monotonic()+5)
            self.assertEqual(audited.call_count, 1)
            self.assertFalse(result['attribution_complete'])
            self.assertEqual(result['case_counts']['legal_state_unmet_members'], 1)
            self.assertEqual(result['evidence_errors'][0]['file'], str(case/'execution/followup.pt'))
            self.assertEqual(result['differential_worker_seconds'], 3.)
            self.assertEqual(result['differential_all_trials']['recorded_processes'], 2)

    def test_fusion_startup_requires_successful_requests_target_returns_and_overlap(self):
        import sys
        with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1]/'benchmarks'), *sys.path]):
            from stateful_check_e2e import MODEL, REVISION, started, write
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'service-ready.json', {'health_http_status':200, 'model_list_verified':True})
            write(root/'serving/environment.json', {'model':MODEL,'declared_revision':REVISION})
            for index in range(2):
                session = root/'campaign'/f'api-{index}'
                execution = session/'execution'
                write(session.parent/f'check-start-{index}.json',
                      {'api':f'api{index}','directory':str(session),'started_epoch':100+index})
                write(session/'session.json', {'finished_epoch':130+index})
                write(execution/'provider-001/http-good.json', {'http_status':200,'request':{'model':MODEL}})
                write(execution/'provider-001/response.json', {'model':MODEL})
                write(execution/'provider-001/process.json', {'outcome':'completed'})
                trace = execution/'attempt-0000/execution/source-trace.jsonl'
                trace.parent.mkdir(parents=True)
                trace.write_text('{"event":"api_exit"}\n' if index==0 else '{"event":"api_enter"}\n')
            self.assertFalse(started(root)['started'])
            trace.write_text('{"event":"api_exit"}\n')
            self.assertTrue(started(root)['started'])
            write(session.parent/'api-0/session.json', {'finished_epoch':101})
            self.assertFalse(started(root)['started'])

    def test_name_checks_have_independent_model_budgets_and_overlapping_pools(self):
        together = threading.Barrier(2, timeout=5)
        def check(api, *, output, session_options):
            together.wait()
            scope = current_scope()
            self.assertEqual(scope.identifier, api)
            self.assertTrue(session_options.generation_pipeline)
            self.assertEqual(session_options.time_limit, 180)
            Path(output).mkdir(parents=True)
            Path(output, 'summary.json').write_text('{}')
            return dict(terminal=True, exit_status=0, status='completed', heavy_native_data=True)
        with tempfile.TemporaryDirectory() as tmp:
            jobs = []
            for name in ('a','b'):
                options = RunOptions(str(Path(tmp) / name), 'fixture', strategy='stateful', contract_version=4,
                                     generation_pipeline=True, time_limit=180)
                jobs.append(APIJob(name, name, options, check=True))
            with patch('mirage.workflow.check_api', side_effect=check) as run:
                result = run_sessions(jobs, SchedulingOptions(2, active_api_limit=2), Path(tmp)/'campaign')
            self.assertEqual(run.call_count, 2)
            self.assertEqual(result['stop_reason'], 'completed')
            for row in result['sessions']:
                self.assertEqual(row['budget_scope'], 'model_request_seconds_then_drain')
                self.assertTrue(Path(row['report']['path']).is_file())
                self.assertNotIn('heavy_native_data', row['report'])

    def test_projection_device_and_declared_mixed_topology_share_cuda_lane(self):
        from mirage.runtime.process import ProcessResult, run_worker
        def finished(request, directory, timeout, *, cancel):
            directory.mkdir()
            return ProcessResult('completed', 0, 0., str(directory))
        scheduler = SharedScheduler(SchedulingOptions(2))
        with tempfile.TemporaryDirectory() as tmp:
            scope = SessionScope(scheduler, 'cpu-source', 'cpu', threading.Event())
            with session_scope(scope), patch('mirage.runtime.process._run_worker', side_effect=finished):
                run_worker(dict(mode='device_member', device='cuda'), Path(tmp)/'gpu', 5)
                run_worker(dict(mode='device_member', device='cpu', config={'device':'cuda'}), Path(tmp)/'mixed', 5)
            self.assertEqual([r['resource'] for r in scheduler.events], ['cuda','cuda'])

    def test_benchmark_scatter_trigger_uses_destination_width(self):
        import sys
        with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1]/'benchmarks'), *sys.path]):
            from pytorch_correctness.evaluate import stateful_mechanism
        def measured(width):
            return {'target_calls': [{'metadata': [
                dict(path=['args',0], shape=[3,width], strides=[width,1]),
                dict(path=['args',2], shape=[2,16], strides=[1,0]),
                dict(path=['args',3], shape=[2,16], strides=[16,1]),
            ]}]}
        self.assertTrue(stateful_mechanism('203078', measured(32)))
        self.assertFalse(stateful_mechanism('203078', measured(16)))

    def test_sessions_keep_seed_feedback_and_case_limits_and_reallocate_on_completion(self):
        seen = []
        lock = threading.Lock()
        def run(options, apis):
            scope = current_scope()
            with scope.scheduler.lease(scope.identifier, 'execution'):
                with lock:
                    seen.append((apis, options.seed, options.cases, options.active_api_limit, scope.identifier))
            return dict(stop_reason='completed', local_feedback=[apis[0]], canonicals=[dict(status='validated')])
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [APIJob(str(i), f'framework.op{i}', RunOptions(str(Path(tmp) / str(i)), 'fixture', seed=10+i, cases=2))
                    for i in range(4)]
            with patch('mirage.workflow.run', side_effect=run):
                report = run_sessions(jobs, SchedulingOptions(2, active_api_limit=2), Path(tmp) / 'summary')
            self.assertEqual(report['completed_api_runs'], 4)
            self.assertEqual(sorted(item[1] for item in seen), [10, 11, 12, 13])
            self.assertTrue(all(item[2:4] == (2, 1) for item in seen))
            self.assertEqual([r['local_feedback'] for r in report['per_api']], [[j.api] for j in jobs])
            self.assertEqual(report['stop_reason'], 'completed')
            self.assertTrue(all(value == 0 for value in report['scheduling']['active'].values()))
            self.assertEqual(json.loads((Path(tmp) / 'summary/multi-api.json').read_text()), report)

    def test_session_specific_limits_are_not_silently_raised(self):
        seen = []
        def run(options, apis):
            seen.append((options.workers, options.generation_concurrency))
            return {'stop_reason': 'completed'}
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [APIJob(str(i), str(i), RunOptions(str(Path(tmp) / str(i)), 'fixture',
                    workers=i + 1, generation_concurrency=i + 1)) for i in range(2)]
            with patch('mirage.workflow.run', side_effect=run):
                run_sessions(jobs, SchedulingOptions(4, 4, 2), Path(tmp) / 'summary')
        self.assertEqual(sorted(seen), [(1, 1), (2, 2)])

    def test_ordinary_api_failure_does_not_cancel_other_api(self):
        def run(options, apis):
            if apis == ('bad.operation',):
                raise ValueError('local controller failure')
            return dict(stop_reason='case_limit', records=[])
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [APIJob(str(i), api, RunOptions(str(Path(tmp) / str(i)), 'fixture'))
                    for i, api in enumerate(('bad.operation', 'good.operation'))]
            with patch('mirage.workflow.run', side_effect=run):
                result = run_sessions(jobs, SchedulingOptions(1, active_api_limit=2), Path(tmp) / 'summary')
            self.assertEqual([row['state'] for row in result['sessions']], ['failed', 'completed'])
            self.assertFalse(result['scheduling']['cancelled'])
            self.assertEqual(result['stop_reason'], 'completed_with_failures')

    def test_single_active_api_has_no_overlapping_session_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [APIJob(str(i), f'framework.op{i}', RunOptions(str(Path(tmp) / str(i)), 'fixture')) for i in range(3)]
            with patch('mirage.workflow.run', return_value={'stop_reason': 'completed'}):
                report = run_sessions(jobs, SchedulingOptions(2, active_api_limit=1), Path(tmp) / 'summary')
            rows = report['sessions']
            self.assertTrue(all(a['finished_epoch'] <= b['started_epoch'] for a, b in zip(rows, rows[1:])))

    def test_output_collisions_and_unsupported_strategy_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = RunOptions(str(Path(tmp) / 'same'), 'fixture')
            jobs = [APIJob('a', 'a', options), APIJob('b', 'b', options)]
            with self.assertRaises(ValueError):
                run_sessions(jobs, SchedulingOptions(1), Path(tmp) / 'summary')
            with self.assertRaises(ValueError):
                APIJob('p', 'p', replace(options, strategy='properties'))

    def test_shared_abort_preserves_not_started_sessions(self):
        def run(options, apis):
            current_scope().scheduler.abort('service request unresolved')
            return dict(stop_reason='provider_cancellation_unconfirmed')
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [APIJob(str(i), str(i), RunOptions(str(Path(tmp) / str(i)), 'fixture')) for i in range(3)]
            with patch('mirage.workflow.run', side_effect=run):
                report = run_sessions(jobs, SchedulingOptions(1), Path(tmp) / 'summary')
            self.assertEqual(report['stop_reason'], 'interrupted')
            self.assertEqual([r['state'] for r in report['sessions']], ['failed', 'not_started', 'not_started'])


class CampaignAdapterTests(unittest.TestCase):
    def load(self, name):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / 'benchmarks' / (name + '.py')
        spec = importlib.util.spec_from_file_location('test_' + name + '_driver', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_existing_campaigns_preserve_devices_and_stateful_treatment(self):
        import sys
        for strategy in ('training', 'stateful'):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                driver = self.load('training_candidates' if strategy == 'training' else 'e2e_latency')
                protocol = dict(targets=[dict(api='torch.sin', device='cpu'), dict(api='torch.cos', device='cuda')],
                    workers=2, generation_concurrency=2, active_api_limit=2, api_concurrency=2,
                    worker_timeout=10, cases=1, seed=0, replay_failures=0, canonicals_per_api=1,
                    targets_per_canonical=1, paths_per_target=1, wall_seconds_per_api=500,
                    model_budget_seconds=180, engine=str(Path(__file__).resolve().parents[1] / 'src'),
                    generation_overlap=False, path_policy='mixed', search_policy='round_robin',
                    generation_mode='full', coverage_basis='semantic', coverage_refinement=False,
                    representation_traversal=True, oracle_improvements=True)
                captured = []
                def session_runner(jobs, limits, output, **kwargs):
                    captured.extend(jobs)
                    return dict(stop_reason='completed', sessions=[dict(directory=job.options.output,
                        state='completed', report={'stop_reason':'case_limit'}) for job in jobs])
                with patch('mirage.runtime.multi_api.run_sessions', side_effect=session_runner), patch.object(sys, 'path', list(sys.path)):
                    if strategy == 'stateful':
                        with patch.object(driver, 'remaining', return_value=3600):
                            driver.parallel_campaign(protocol, root)
                    else:
                        driver.parallel_campaign(protocol, root)
                self.assertEqual([job.options.device for job in captured], ['cpu', 'cuda'])
                self.assertTrue(all(job.options.strategy == strategy for job in captured))
                if strategy == 'stateful':
                    self.assertTrue(all(job.stateful_options['path_policy'] == 'mixed' for job in captured))
                    self.assertTrue(all(not job.options.generation_overlap for job in captured))
                    self.assertTrue(all(job.options.time_limit == 180 for job in captured))
                else:
                    self.assertTrue(all(job.options.time_limit == 500 for job in captured))


TRAINING_SOURCE = '''import torch

def build(device, seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    x = torch.randn(3, 2, device=device)
    w = torch.nn.Parameter(torch.randn(2, 2, device=device))
    return {'inputs': {'x': x}, 'parameters': {'w': w}, 'batch_axes': {'x': 0},
            'optimizer': torch.optim.SGD([w], lr=0.01)}

def forward(ctx):
    return TARGET(ctx['inputs']['x'], ctx['parameters']['w'])

def loss(output):
    return output.square().mean()
'''
STATEFUL_SOURCE = '''import torch

def build(device, seed):
    x = torch.tensor([[0.25, 0.5], [0.75, 1.0]], device=device, requires_grad=True)
    y = torch.tensor([[1.0, 0.25], [0.5, 1.0]], device=device)
    return {'inputs': {'x': x, 'y': y}, 'parameters': {}}

def run(ctx, probe):
    value = probe.call(TARGET, ctx['inputs']['x'], ctx['inputs']['y'])
    value.sum().backward()
    return {'value': value}
'''


class FixtureProvider:
    def generate(self, request, *, deadline):
        from mirage.generator.provider import Response, Usage
        context = request.context
        if context['contract'] == 'state-plan':
            value = {'actions': context['offered_paths'][0]}
        else:
            api = context.get('configuration', {}).get('api') or context['api_context']['api']
            source = STATEFUL_SOURCE if context.get('program_kind') == 'stateful-1' else TRAINING_SOURCE
            value = {'source': source.replace('TARGET', api)}
        return Response(json.dumps(value), Usage(1, 1), 'deterministic-test-fixture')


def create():
    return FixtureProvider()


@unittest.skipUnless(os.environ.get('MIRAGE_TEST_E2E') == '1', 'Slurm integration opt-in')
class StrategyRuntimeTests(unittest.TestCase):
    def jobs(self, root, strategy, *, device='cpu', alternating=False):
        return [APIJob(f'api-{i}', api,
            RunOptions(str(root / f'api-{i}'), 'test_multi_api_scheduling:create', device=device,
                strategy=strategy, contract_version=4 if strategy == 'stateful' else 3,
                workers=2, cases=2 if alternating else 1, canonicals_per_api=1,
                targets_per_canonical=1, paths_per_target=1, replay_failures=0,
                time_limit=180, worker_timeout=60, generation_pipeline=strategy == 'stateful',
                generation_overlap=not alternating),
            dict(predicates=[], search_policy='round_robin', generation_mode='full',
                 coverage_refinement=False, representation_traversal=True, oracle_improvements=True)
                 if strategy == 'stateful' else {})
            for i, api in enumerate(('torch.matmul', 'torch.mm'))]

    def run_jobs(self, jobs, root, active):
        with patch.dict(os.environ, {'PYTHONPATH': str(Path(__file__).parent) + os.pathsep + os.environ.get('PYTHONPATH', '')}):
            return run_sessions(jobs, SchedulingOptions(2, 2, active), root / 'campaign')

    def test_training_fixed_semantics_match_serial_and_parallel(self):
        import torch
        from mirage.artifacts.bundle import load_tensor_record
        def assert_equal(a, b, path='value'):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b), path)
            elif isinstance(a, dict):
                self.assertEqual(set(a), set(b), path)
                for key in a: assert_equal(a[key], b[key], f'{path}.{key}')
            elif isinstance(a, (tuple, list)):
                self.assertEqual(len(a), len(b), path)
                for index, (x, y) in enumerate(zip(a, b)):
                    assert_equal(x, y, f'{path}[{index}]')
            else:
                self.assertEqual(a, b, path)
        with tempfile.TemporaryDirectory() as tmp:
            arms = []
            for active in (1, 2):
                root = Path(tmp) / str(active)
                report = self.run_jobs(self.jobs(root, 'training'), root, active)
                self.assertEqual(report['completed_api_runs'], 2, report['sessions'])
                observations = []
                for session in report['sessions']:
                    case = session['report']['experiments'][0]
                    self.assertTrue(case['verified'], case)
                    bundle = Path(case['manifest']).parent / case['id']
                    records = sorted(bundle.glob('*/result.pt'))
                    self.assertTrue(records, case)
                    # Compare captured initial values/RNG as well as final observables.
                    observations.append({'initial': load_tensor_record(bundle / 'initial.pt'),
                        'final': [load_tensor_record(path)['observations'] for path in records]})
                arms.append(observations)
            assert_equal(*arms)

    def test_stateful_alternating_feedback_and_full_drain_are_per_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_jobs(self.jobs(root, 'stateful', alternating=True), root, 2)
            self.assertEqual(result['completed_api_runs'], 2, result['sessions'])
            for session in result['sessions']:
                report = session['report']
                self.assertTrue(report['pipeline_complete'])
                directory = Path(session['directory'])
                request = json.loads((directory / 'provider-002/request.json').read_text())['request']
                feedback = request['context']['feedback_snapshot']['completed_cases']
                self.assertTrue(feedback)
                self.assertTrue(all(row['api'] == session['api'] for row in feedback))
                self.assertTrue(all(Path(row['directory']).is_relative_to(directory) for row in feedback))
                state = json.loads((directory / 'pipeline-state.json').read_text())
                second_start = state['model_events'][1]['started']
                first_source = state['source_registrations'][0]['source_id']
                first_events = [event for event in state['worker_events'] if f'/representations/{first_source}/' in event['directory']]
                self.assertTrue(first_events)
                self.assertTrue(all(event['finished'] <= second_start for event in first_events))

    def test_stateful_shared_sessions_resume_without_new_generation(self):
        from mirage.runtime.pipeline import resume_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_jobs(self.jobs(root, 'stateful'), root, 2)
            self.assertEqual(result['completed_api_runs'], 2, result['sessions'])
            for session in result['sessions']:
                self.assertTrue(session['report']['pipeline_complete'])
                with patch('mirage.generator.provider.IsolatedProvider.generate', side_effect=AssertionError('Resume must not generate')):
                    resumed = resume_pipeline(Path(session['directory']))
                self.assertTrue(resumed['pipeline_complete'])
                self.assertFalse(resumed['generation_resumed'])
                self.assertEqual(len(resumed['records']), len(session['report']['records']))

    @unittest.skipUnless(os.environ.get('MIRAGE_TEST_CUDA') == '1', 'CUDA opt-in')
    def test_two_cuda_training_sessions_share_one_test_gpu(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('No allocated CUDA device')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.run_jobs(self.jobs(root, 'training', device='cuda'), root, 2)
            self.assertEqual(report['completed_api_runs'], 2, report['sessions'])
            cuda = [event for event in report['scheduling']['events'] if event['resource'] == 'cuda' and 'granted' in event]
            self.assertTrue(cuda)
            cuda.sort(key=lambda event: event['granted'])
            self.assertTrue(all(a['finished'] <= b['granted'] for a, b in zip(cuda, cuda[1:])))
