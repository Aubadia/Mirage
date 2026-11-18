"""Independent strategy sessions sharing bounded generation and test capacity."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
import threading
import time

from mirage.contracts.scheduling import APIJob, SchedulingOptions
from mirage.runtime.scheduler import SharedScheduler, SessionScope, session_scope


def _write(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def _failed(report):
    if report.get('stop_reason') in (
        'interrupted', 'controller_error', 'pipeline_incomplete', 'metadata_failed',
        'provider_cancellation_unconfirmed', 'interrupted_or_provider_cancellation_unconfirmed',
    ):
        return True
    if any(row.get('status') != 'validated' for row in report.get('canonicals', [])):
        return True
    return any(row.get('status', '').startswith('generation_provider_') or row.get('status') == 'provider_failed'
               for row in report.get('experiments', report.get('records', [])))


def run_multi_api(options, apis, *, stateful_options=None):
    apis = tuple(apis)
    if not apis or len(set(apis)) != len(apis):
        raise ValueError('API sessions require nonempty distinct APIs')
    root = Path(options.output).resolve()
    jobs = tuple(APIJob(f'api-{index:03d}', api,
                       replace(options, output=str(root / f'api-{index:03d}'),
                               seed=options.seed + 1009 * index), dict(stateful_options or {}))
                 for index, api in enumerate(apis))
    limits = SchedulingOptions(options.workers, options.generation_concurrency, options.active_api_limit,
                               options.per_api_worker_limit, options.per_api_generation_limit)
    return run_sessions(jobs, limits, root)


def run_sessions(jobs, limits, output, *, deadline=None):
    """Run APIJob records; each record may have its own device, strategy and budget.

    Queue admission precedes a session's training wall clock. Once admitted,
    shared-resource waiting consumes that clock. Stateful model-only budgets
    begin at request dispatch and retain their independent durable final drain.
    """
    from mirage.artifacts.report import summarize_multi_api
    jobs = tuple(jobs)
    if not jobs or len({job.identifier for job in jobs}) != len(jobs):
        raise ValueError('API jobs require nonempty unique session identifiers')
    outputs = [Path(job.options.output).resolve() for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError('API session output directories must be distinct')
    root = Path(output).resolve()
    if any(path == root or root.is_relative_to(path) for path in outputs):
        raise ValueError('Session outputs cannot contain the campaign output')
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(outputs) for b in outputs[i + 1:]):
        raise ValueError('API session output directories cannot overlap')
    if any(path.exists() for path in outputs):
        raise ValueError('API session outputs already exist; preserve them and use a fresh run')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'multi-api.json').exists():
        raise ValueError('API campaign already attempted; use a fresh output directory')
    scheduler = SharedScheduler(limits, deadline=deadline)
    row_lock = threading.RLock()
    submitted = time.monotonic()
    rows = [dict(session_id=job.identifier, api=job.api, strategy=job.options.strategy,
                 device=job.options.device, seed=job.options.seed, directory=str(outputs[index]),
                 budget_seconds=job.options.time_limit, state='queued', report=None,
                 budget_scope=('model_request_seconds_then_drain' if job.options.strategy == 'stateful'
                               and job.options.generation_pipeline else 'session_wall_seconds'))
            for index, job in enumerate(jobs)]
    started = time.monotonic()
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, lambda signum, _: scheduler.abort(f'Signal {signum}'))

    def persist():
        with row_lock:
            summary = summarize_multi_api(rows, scheduler.snapshot(), asdict(limits), time.monotonic() - started)
        _write(root / 'multi-api.json', summary)
        _write(root / 'summary.json', summary)
        return summary

    def update(index, **changes):
        with row_lock:
            rows[index].update(changes)

    def run_one(index):
        job, row = jobs[index], rows[index]
        if scheduler.cancel.is_set():
            update(index, state='not_started', reason=scheduler.reason)
            return
        scope = SessionScope(scheduler, job.identifier, job.options.device, threading.Event(), deadline)
        update(index, state='running', admission_queue_seconds=time.monotonic() - submitted,
                   started_epoch=time.time())
        begin = time.monotonic()
        try:
            Path(job.options.output).parent.mkdir(parents=True, exist_ok=True)
            with session_scope(scope):
                from mirage.workflow import run
                options = replace(job.options, active_api_limit=1,
                                  workers=min(job.options.workers, limits.per_api_worker_limit),
                                  generation_concurrency=min(job.options.generation_concurrency, limits.per_api_generation_limit),
                                  per_api_worker_limit=None, per_api_generation_limit=None)
                if job.stateful_options:
                    from mirage.workflow import run_stateful
                    report = run_stateful(options, (job.api,), **job.stateful_options)
                else:
                    report = run(options, (job.api,))
                update(index, report=report, state='failed' if _failed(report) else 'completed')
        except Exception as error:
            update(index, state='failed', reason=f'{type(error).__name__}: {error}')
        finally:
            update(index, finished_epoch=time.time(), session_wall_seconds=time.monotonic() - begin)
            try:
                Path(job.options.output).mkdir(parents=True, exist_ok=True)
                with row_lock:
                    session = {k: v for k, v in row.items() if k != 'report'}
                _write(Path(job.options.output) / 'session.json', session)
            except OSError as error:
                update(index, state='failed', reason=f'Cannot save session evidence: {error}')

    try:
        persist()
        with ThreadPoolExecutor(max_workers=min(limits.active_api_limit, len(jobs)),
                                thread_name_prefix='mirage-api') as executor:
            try:
                iterator = iter(range(len(jobs)))
                active = {executor.submit(run_one, index): index
                          for index in (next(iterator, None) for _ in range(min(limits.active_api_limit, len(jobs))))
                          if index is not None}
                while active:
                    if deadline is not None and time.monotonic() >= deadline:
                        scheduler.abort('Campaign allocation deadline')
                    completed, _ = wait(active, timeout=.2, return_when=FIRST_COMPLETED)
                    for future in completed:
                        active.pop(future)
                        future.result()
                        index = next(iterator, None)
                        if index is not None and not scheduler.cancel.is_set():
                            active[executor.submit(run_one, index)] = index
                    if completed:
                        persist()
            except BaseException as error:
                # Cancel before executor shutdown waits for the session controllers.
                scheduler.abort(f'Coordinator interrupted: {type(error).__name__}: {error}')
                raise
        for row in rows:
            if row['state'] == 'queued':
                row.update(state='not_started', reason=scheduler.reason)
    except BaseException as error:
        scheduler.abort(f'Coordinator interrupted: {type(error).__name__}: {error}')
        raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
        final = persist()
    return final
