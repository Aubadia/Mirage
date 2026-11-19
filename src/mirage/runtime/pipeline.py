"""Bounded model/execution overlap with a durable representation drain."""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from contextvars import copy_context
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import queue
import threading
import time

from mirage.artifacts.bundle import inventory, verify
from mirage.runtime.process import run_worker
from mirage.sampler.representations import RepresentationTraversal


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class GenerationBudgetExhausted(TimeoutError):
    pass


class BudgetedProvider:
    def __init__(self, provider, pipeline):
        self.provider, self.pipeline = provider, pipeline

    @property
    def calls(self):
        return self.provider.calls

    def generate(self, request, *, deadline=None):
        from mirage.runtime.scheduler import generation_slot
        # Queue for shared generation capacity before starting the model timer.
        # Capacity/feedback waits must not hold a scarce generation lease.
        self.pipeline.wait_capacity()
        with generation_slot(label=self.pipeline.root / f'provider-{self.calls + 1:03d}'):
            return self._generate(request, deadline=deadline)

    def _generate(self, request, *, deadline=None):
        p = self.pipeline
        p.consume()
        remaining = p.budget - p.model_seconds
        with p.condition:
            history = [event for event in p.model_events
                if event.get('outcome') == 'completed' and event.get('finished') is not None
                and event.get('stage') == request.stage.value
                and event.get('contract') == request.context.get('contract')]
            # Avoid dispatching a request that cannot fit its already observed cost.
            # This estimate is not a guarantee; real timeout uncertainty stays unchanged.
            reserve = max((max(event.get('seconds', 0.), event['finished'] - event['started'])
                           for event in history), default=0.)
            if remaining <= 0.01 or remaining < reserve:
                p.generation_budget_stop = {
                    'stage': request.stage.value, 'contract': request.context.get('contract'),
                    'remaining_model_seconds': max(0., remaining), 'estimated_request_seconds': reserve,
                    'history_samples': len(history), 'request_dispatched': False,
                    'basis': 'maximum observed completed same-stage/contract request plus wrapper cost'}
                p.persist()
                raise GenerationBudgetExhausted('Remaining model budget cannot admit the next request')
        context = deepcopy(request.context)
        context['feedback_snapshot'] = p.feedback_snapshot()
        request = replace(request, context=context)
        begin = time.time()
        event = {'api': p.api, 'started': begin, 'finished': None,
                 'feedback_version': p.feedback_version, 'stage': request.stage.value,
                 'contract': context.get('contract'), 'retry_of_attempt': context.get('truncated_attempt'),
                 'outcome': 'running'}
        if context.get('oracle_precondition'):
            event['oracle_precondition'] = context['oracle_precondition']
        with p.condition:
            p.model_events.append(event)
            p.persist()
        try:
            response = self.provider.generate(request, deadline=time.monotonic() + remaining)
            event['outcome'] = 'completed'
            return response
        except BaseException as error:
            event['outcome'] = getattr(error, 'category', type(error).__name__)
            event.update(getattr(error, 'details', {}))
            if 'truncated_attempt' in context:
                event['retry_of_attempt'] = context['truncated_attempt']
            raise
        finally:
            end = time.time()
            timing = self.provider.request_timings[-1] if self.provider.request_timings else {}
            # A killed provider may not have flushed its inner timer. Charge the
            # conservative outer duration and explicitly identify this fallback.
            measured = timing.get('elapsed_seconds', end - begin)
            with p.condition:
                event.update(finished=end, seconds=measured,
                             request_started=timing.get('started_epoch', begin),
                             request_finished=timing.get('finished_epoch', end),
                             basis=timing.get('basis', 'provider_process_upper_bound'))
                p.model_seconds += measured
                p.persist()


class ExecutionPipeline:
    def __init__(self, root, api, budget, worker_timeout, *, replay_failures=1, workers=8,
                 generation_overlap=True, resume=False):
        self.root, self.api = Path(root), api
        self.budget, self.worker_timeout = budget, worker_timeout
        self.replay_failures = replay_failures
        self.condition = threading.Condition(threading.RLock())
        from mirage.runtime.scheduler import cancellation_event
        self.cancel = cancellation_event()
        self.workers = workers
        self.generation_overlap = generation_overlap
        self.completed = queue.Queue()
        self.traversal = RepresentationTraversal(include_suspects=True)
        self.model_events, self.worker_events, self.rows = [], [], []
        self.model_seconds = self.queue_seconds = 0.
        self.generation_budget_stop = None
        self.backpressure_measurement_complete = True
        self.feedback_version, self.feedback_rows = 0, []
        self.consume = lambda: None
        self.error = None
        self.closed = False
        self.started = time.time()
        self.session_started = self.started
        self.prior_session_seconds = 0.
        self.source_registrations = []
        if resume:
            state = json.loads((self.root / 'pipeline-state.json').read_text())
            self.traversal = RepresentationTraversal.restore(state['traversal'])
            self.model_events = state['model_events']
            self.worker_events = state['worker_events']
            self.model_seconds = state['model_seconds']
            self.generation_budget_stop = state.get('generation_budget_stop')
            # Old ledgers used one slot and did not persist backpressure time.
            self.workers = state.get('workers', 1)
            self.generation_overlap = state.get('generation_overlap', True)
            self.queue_seconds = state.get('queue_seconds', 0.)
            self.backpressure_measurement_complete = state.get(
                'backpressure_measurement_complete', 'queue_seconds' in state)
            self.source_registrations = state.get('source_registrations', [])
            self.rows = [json.loads((Path(r['directory'])/'outcome.json').read_text())
                         if set(r) == {'directory'} else r for r in state.get('representation_rows', [])]
            self.feedback_version = state.get('feedback_version', 0)
            self.feedback_rows = state.get('feedback_rows', [])
            self.started = state['started_epoch']
            self.prior_session_seconds = state.get('prior_session_seconds', 0.) + max(0.,
                state.get('saved_epoch', state['started_epoch']) - state.get('session_started_epoch', state['started_epoch']))
        from mirage.contracts.programs import positive
        positive(self.workers, 'workers')
        if type(self.generation_overlap) is not bool:
            raise ValueError('generation_overlap must be boolean')
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='mirage-test')
        # Wrappers may wait for source/replay subprocesses. They must not occupy
        # the execution slots they submit to, which would deadlock a full pool.
        self.representations = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='mirage-representation')
        self.thread = threading.Thread(target=copy_context().run, args=(self._drain,), name='mirage-representations', daemon=False)

    def start(self):
        self.thread.start()

    def feedback_snapshot(self):
        with self.condition:
            return {'version': self.feedback_version, 'completed_cases': deepcopy(self.feedback_rows[-16:]),
                    'scope': 'Completed runtime evidence only; in-flight work is excluded'}

    def observe(self, row):
        with self.condition:
            self.feedback_version += 1
            self.feedback_rows.append({'directory': row['directory'], 'api': row.get('api'),
                'verdict': row.get('verdict'), 'reason': row.get('reason'),
                'representation': row.get('mutation'),
                **({'product_use': row['inspection']['product_use']} if
                   row.get('inspection', {}).get('product_use', {}).get('uses') else {}),
                'runtime_valid': row.get('inspection', {}).get('valid') is True,
                'representation_attained': row.get('representation_candidate_attained')})
            row['feedback_committed_version'] = self.feedback_version

    def register(self, source, row):
        with self.condition:
            identity = self.traversal.register(source, row)
            row['representation_source_id'] = identity
            self.source_registrations.append({'directory': row['directory'], 'source_id': identity,
                'reason': None if identity else 'Source execution/obligations or candidate evidence unavailable'})
            self.persist()
            self.condition.notify_all()

    def wait_capacity(self):
        begin = time.monotonic()
        with self.condition:
            # The controller also calls this before proposal selection, so an
            # alternating round constructs its prompt from fully published feedback.
            def blocked():
                if self.generation_overlap:
                    return self.pending_sources() > 1
                return self.pending_sources() > 0 or any(
                    event.get('finished') is None for event in self.worker_events)
            while blocked() and not self.error and not self.cancel.is_set():
                self.condition.wait(timeout=.2)
            self.queue_seconds += time.monotonic() - begin
            self.persist()
            if self.error:
                raise RuntimeError(self.error)
            if self.cancel.is_set():
                raise InterruptedError('Pipeline interrupted')

    def pending_sources(self):
        return sum(any(c['status'] in ('pending', 'dispatched') for c in s['candidates'])
                   for s in self.traversal.sources)

    def _execute(self, function, directory, mode):
        submitted = time.time()
        def execute():
            start = time.time()
            event = {'directory': str(directory), 'mode': mode, 'started': start,
                     'submitted': submitted, 'queue_seconds': start - submitted, 'finished': None}
            with self.condition:
                self.worker_events.append(event)
                self.persist()
            try:
                from mirage.runtime.process import cancellation_scope
                with cancellation_scope(self.cancel):
                    return function()
            finally:
                with self.condition:
                    event['finished'] = time.time()
                    event['seconds'] = event['finished'] - start
                    self.persist()
                    self.condition.notify_all()
        return self.executor.submit(copy_context().run, execute).result()

    def worker(self, request, directory, timeout):
        return self._execute(lambda: run_worker(request, directory, timeout, cancel=self.cancel),
                             directory, request['mode'])

    def operation(self, function, *args, **kwargs):
        return self._execute(lambda: function(*args, **kwargs), args[0] if args else '', 'replay_or_triage')

    def _drain(self):
        active = {}
        try:
            while True:
                with self.condition:
                    while len(active) < self.workers and not self.cancel.is_set():
                        work = self.traversal.choose(float('inf'), 0)
                        if work is None:
                            break
                        origin, candidate, source, _ = work
                        self.persist()
                        active[self.representations.submit(
                            copy_context().run, self.execute_representation, origin, candidate, source)] = (origin, candidate)
                    if not active:
                        if self.closed or self.cancel.is_set():
                            return
                        self.condition.wait(timeout=.2)
                        continue
                done, _ = wait(active, timeout=.2, return_when=FIRST_COMPLETED)
                for future in done:
                    origin, candidate = active.pop(future)
                    row = future.result()
                    with self.condition:
                        if self.cancel.is_set():
                            # Retain each interrupted attempt; resume requeues all
                            # dispatched candidates after checking parent hashes.
                            continue
                        row['representation_traversal']['status'] = self.traversal.complete(
                            origin['source_id'], candidate['id'], row)
                        self.rows.append(row)
                        self.completed.put(row)
                        self.persist()
                        self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.error = f'{type(error).__name__}: {error}'
                self.cancel.set()
                self.persist()
                self.condition.notify_all()
        finally:
            self.representations.shutdown(wait=True)

    def execute_representation(self, origin, candidate, source):
        from mirage.workflow import mutation_paths_preserved, replay_stateful
        base = self.root / 'representations' / origin['source_id'] / candidate['id']
        base.mkdir(parents=True, exist_ok=True)
        attempt = len(list(base.glob('run-*')))
        directory = base / f'run-{attempt:04d}'
        directory.mkdir()
        config = dict(origin['configuration'], representation_candidate=candidate['plan'],
                      canonical_reference=False)
        start = time.time()
        row = {'attempt': f"representation:{origin['source_id']}:{candidate['id']}:{attempt}",
               'api': config['api'], 'directory': str(directory), 'configuration': config,
               'proposal_origin': 'framework_representation', 'program_generation_dispatched': False,
               'source_reused': True, 'generation_usage': {'input_tokens': 0, 'output_tokens': 0},
               'started_seconds': start - self.started,
               'representation_traversal': {'source_id': origin['source_id'],
                   'candidate_id': candidate['id'], 'plan': candidate['plan']},
               'candidate': {'api': config['api'], 'parent': origin['source_id'],
                             'kind': 'representation', 'goal': candidate['id']},
               'generated': True, 'contract_valid': True, 'verdict': None}
        (directory / 'program.py').write_text(source)
        request = {'mode': 'execute_stateful_pair', 'source': source, 'config': config,
                   'engine': str(self.root / 'engine'), 'initial_reference': origin['initial_reference'],
                   'reference_scope': 'initialization', 'audit_budget_seconds': min(10, self.worker_timeout / 8)}
        save(directory / 'request.json', request)
        process = self.worker(request, directory / 'execution', self.worker_timeout)
        row['process'] = asdict(process)
        summary = directory / 'execution/summary.json'
        if process.outcome == 'completed' and summary.exists():
            row.update(json.loads(summary.read_text()), executable=True, status='executed')
            row['mutation_obligations_valid'] = all(mutation_paths_preserved(origin, row.get(key, {}))
                for key in ('inspection', 'followup_inspection'))
            if not row['mutation_obligations_valid']:
                row['representation_candidate_attained'] = False
        else:
            row.update(status='execution_failed', reason=process.error or process.outcome,
                       representation_candidate_attained=False, mutation_obligations_valid=False)
        save(directory / 'execution-sha256.json', inventory(directory / 'execution'))
        if (directory / 'execution/source.pt').exists():
            save(directory / 'stateful-manifest.json', {'format': 'stateful-1-replay',
                'source': 'program.py', 'configuration': config, 'engine': str(self.root / 'engine'),
                'engine_inventory': inventory(self.root / 'engine'), 'files': inventory(directory),
                'initial_reference': 'execution/source.pt', 'expected_verdict': row.get('verdict')})
            if row.get('verdict') == 'suspected_failure':
                row['automatic_replays'] = []
                for _ in range(self.replay_failures):
                    if self.cancel.is_set():
                        break
                    try:
                        row['automatic_replays'].append(self._execute(
                            lambda: replay_stateful(directory / 'stateful-manifest.json',
                                timeout=self.worker_timeout, cancel=self.cancel), directory, 'replay'))
                    except (OSError, ValueError, RuntimeError) as error:
                        row['automatic_replays'].append({'replay_valid': False, 'reason': str(error), 'provider_calls': 0})
        row['finished_seconds'] = time.time() - self.started
        save(directory / 'outcome.json', row)
        return row

    def finish(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        while self.thread.is_alive():
            self.thread.join(timeout=.2)
            self.consume()
        self.consume()
        if self.error:
            raise RuntimeError(self.error)

    def close(self, interrupted=False):
        if interrupted:
            self.cancel.set()
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join()
        self.representations.shutdown(wait=True)
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.condition:
            self.persist()

    def timing(self):
        models = [(x['request_started'], x['request_finished']) for x in self.model_events if x.get('finished')]
        workers = [(x['started'], x['finished']) for x in self.worker_events if x.get('finished')]
        def union(intervals):
            merged = []
            for a, b in sorted(intervals):
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            return merged
        model_union, worker_union = union(models), union(workers)
        overlap = sum(max(0., min(b, d)-max(a, c)) for a,b in model_union for c,d in worker_union)
        busy = sum(b-a for a,b in worker_union)
        cumulative = sum(x.get('seconds', 0.) for x in self.worker_events)
        active = peak = 0
        for _, delta in sorted([(a, 1) for a,b in workers] + [(b, -1) for a,b in workers]):
            active += delta
            peak = max(peak, active)
        wall = self.prior_session_seconds + time.time() - self.session_started
        return {'version': 'generation-pipeline-timing-2', 'llm_budget_seconds': self.budget,
            'generation_overlap': self.generation_overlap,
            'llm_request_seconds': self.model_seconds,
            'generation_budget_stop': deepcopy(self.generation_budget_stop),
            'framework_worker_seconds': cumulative, 'framework_busy_seconds': busy,
            'framework_worker_limit': self.workers, 'framework_peak_workers': peak,
            'framework_mean_active_workers': cumulative / busy if busy else 0.,
            'framework_queue_seconds': sum(x.get('queue_seconds', 0.) for x in self.worker_events),
            'generation_backpressure_seconds': self.queue_seconds if self.backpressure_measurement_complete else None,
            'recorded_generation_backpressure_seconds': self.queue_seconds,
            'backpressure_measurement_complete': self.backpressure_measurement_complete,
            'overlap_seconds': overlap,
            'wall_seconds': wall, 'elapsed_since_initial_start_seconds': time.time() - self.started,
            'unfinished_model_requests': sum(x.get('finished') is None for x in self.model_events),
            'unfinished_framework_workers': sum(x.get('finished') is None for x in self.worker_events),
            'controller_and_unattributed_seconds': max(0., wall -
                sum(b-a for a,b in model_union) - busy + overlap),
            'provider_process_overhead_seconds': sum(max(0., x['finished']-x['started']-x['seconds'])
                for x in self.model_events if x.get('finished')),
            'scope': 'Request end-to-end time includes service wait; worker wrapper wall includes startup, evidence and any shared dispatch wait. Use the shared scheduling ledger for subprocess slot time. Overlap is not additive wall time.'}

    def summary(self):
        with self.condition:
            return self._summary()

    def _summary(self):
        value = self.traversal.summary()
        value['policy'] = f'Drain all observed eligible candidates independently of the model budget; {self.workers} shared execution slots; no defect blacklist'
        value['generation_overlap'] = self.generation_overlap
        counts = Counter(c['status'] for s in value['sources'] for c in s['candidates'])
        value['status_counts'] = dict(counts)
        value['dispatch_complete'] = not (counts['pending'] or counts['dispatched'])
        value['all_eligible_attained'] = value['dispatch_complete'] and not counts['not_verified']
        return value

    def persist(self):
        if not self.root.exists():
            return
        # Both durable ledgers describe the same traversal snapshot.
        traversal = self.summary()
        save(self.root / 'pipeline-state.json', {'version': 'generation-pipeline-2', 'api': self.api,
            'budget': self.budget, 'worker_timeout': self.worker_timeout, 'replay_failures': self.replay_failures,
            'workers': self.workers, 'generation_overlap': self.generation_overlap, 'queue_seconds': self.queue_seconds,
            'backpressure_measurement_complete': self.backpressure_measurement_complete,
            'started_epoch': self.started, 'session_started_epoch': self.session_started,
            'saved_epoch': time.time(), 'prior_session_seconds': self.prior_session_seconds,
            'model_seconds': self.model_seconds, 'generation_budget_stop': self.generation_budget_stop,
            'model_events': self.model_events, 'worker_events': self.worker_events,
            'feedback_version': self.feedback_version, 'feedback_rows': self.feedback_rows,
            'representation_rows': [{'directory': r['directory']} for r in self.rows],
            'source_registrations': self.source_registrations,
            'traversal': traversal, 'error': self.error})
        save(self.root / 'representation-traversal.json', traversal)


def resume_pipeline(directory):
    """Finish durable generated work without issuing new model requests."""
    from mirage.artifacts.report import summarize_search
    root = Path(directory).resolve()
    state = json.loads((root / 'pipeline-state.json').read_text())
    verify(root / 'engine', json.loads((root / 'engine-sha256.json').read_text()))
    p = ExecutionPipeline(root, state['api'], state['budget'], state['worker_timeout'],
                          replay_failures=state['replay_failures'], resume=True)
    summary = root / 'summary.json'
    report = json.loads(summary.read_text()) if summary.exists() else {
        'strategy': 'stateful', 'records': [], 'provider_calls': len(state['model_events']),
        'stop_reason': 'interrupted', 'elapsed_seconds': 0}
    # The state ledger may be newer than the controller summary.
    known = {r['directory']: r for r in report['records']}
    for row in p.rows:
        known[row['directory']] = row
    for path in root.glob('attempt-*/outcome.json'):
        row = json.loads(path.read_text())
        known[row['directory']] = row
    staged = root / 'pipeline-pending-source.json'
    completed = False
    try:
        if staged.exists():
            pending = json.loads(staged.read_text())
            original = Path(pending['row']['directory'])
            row = known.get(str(original))
            if row is None:
                request = pending['request']
                execution = original / 'execution'
                if not ((execution / 'process.json').exists() and (execution / 'summary.json').exists()):
                    retry = original / f'resume-source-{len(list(original.glob("resume-source-*"))):04d}'
                    retry.mkdir()
                    (retry / 'program.py').write_text(pending['source'])
                    row = dict(pending['row'], directory=str(retry))
                    process = p.worker(request, retry / 'execution', p.worker_timeout)
                    execution = retry / 'execution'
                else:
                    from mirage.runtime.process import ProcessResult
                    process = ProcessResult(**json.loads((execution / 'process.json').read_text()))
                    row = dict(pending['row'])
                row['process'] = asdict(process)
                if process.outcome == 'completed' and (execution / 'summary.json').exists():
                    from mirage.workflow import mutation_paths_preserved
                    row.update(json.loads((execution / 'summary.json').read_text()), executable=True, status='executed')
                    row['mutation_obligations_valid'] = mutation_paths_preserved(pending.get('obligations'), row.get('inspection', {}))
                else:
                    row.update(status='execution_failed', reason=process.error or process.outcome,
                               mutation_obligations_valid=False)
                row['finished_seconds'] = time.time() - p.started
                save(Path(row['directory']) / 'outcome.json', row)
                known[row['directory']] = row
            p.register(pending['source'], row)
            staged.unlink()
        # Recover a completed source written just before its inventory commit.
        registered = {r['directory'] for r in p.source_registrations}
        for row in list(known.values()):
            if row.get('proposal_origin') != 'framework_representation' and row['directory'] not in registered:
                program = Path(row['directory']) / 'program.py'
                p.register(program.read_text() if program.exists() else '', row)
        p.start()
        p.finish()
        for row in p.rows:
            known[row['directory']] = row
        report.update(records=list(known.values()), representation_traversal=p.summary(),
                      pipeline_timing=p.timing(), pipeline_complete=p.summary()['dispatch_complete'],
                      stop_reason='drain_complete' if p.summary()['dispatch_complete'] else 'pipeline_incomplete',
                      generation_resumed=False, elapsed_seconds=p.timing()['wall_seconds'])
        report['search_metrics'] = summarize_search(report)
        save(summary, report)
        completed = True
        return report
    finally:
        p.close(interrupted=not completed)
