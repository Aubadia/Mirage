"""Fair resource leases; waiting controllers never occupy execution capacity."""
from collections import Counter, deque
import os
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import threading
import time

from mirage.contracts.scheduling import SchedulingOptions
from mirage.runtime.differential_pool import CAPACITY, DifferentialPool


_scope = ContextVar('mirage_scheduling_scope', default=None)
_generation_held = ContextVar('mirage_generation_lease', default=False)


class SchedulingCancelled(TimeoutError):
    pass


class AnyCancellation:
    def __init__(self, *events):
        self.events = tuple(event for event in events if event is not None)

    def is_set(self):
        return any(event.is_set() for event in self.events)


@dataclass
class SessionScope:
    scheduler: 'SharedScheduler'
    identifier: str
    device: str
    cancel: threading.Event
    deadline: float | None = None


def current_scope():
    return _scope.get()


def cancellation_event():
    scope = current_scope()
    return scope.cancel if scope else threading.Event()


@contextmanager
def session_scope(scope):
    with scope.scheduler.condition:
        scope.scheduler.cancellations[scope.identifier] = scope.cancel
        if scope.scheduler.cancel.is_set():
            scope.cancel.set()
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


@contextmanager
def generation_slot(*, deadline=None, label='generation'):
    scope = current_scope()
    if scope is None or _generation_held.get():
        yield
        return
    if scope.deadline is not None:
        deadline = scope.deadline if deadline is None else min(scope.deadline, deadline)
    with scope.scheduler.lease(scope.identifier, 'generation', deadline=deadline,
                               cancel=scope.cancel, label=label):
        token = _generation_held.set(True)
        try:
            yield
        finally:
            _generation_held.reset(token)


class SharedScheduler:
    """API-fair admission; ordinary CUDA probes run exclusively of differential tasks."""
    def __init__(self, options: SchedulingOptions, *, deadline=None):
        self.options, self.deadline = options, deadline
        self.condition = threading.Condition(threading.RLock())
        self.cancel = threading.Event()
        self.reason = None
        self.uncertain = []
        self.service_stop_file = os.environ.get('MIRAGE_GENERATION_STOP_FILE')
        self.cancellations = {}
        self.events = []
        self.pending = {'execution': {}, 'generation': {}}
        self.rotation = {'execution': deque(), 'generation': deque()}
        self.active = Counter()
        self.by_session = Counter()
        self.cuda_active = 0
        self.lanes = Counter()
        self.differential = DifferentialPool()
        self.started = time.monotonic()

    def abort(self, reason):
        with self.condition:
            self.reason = self.reason or str(reason)
            self.cancel.set()
            for cancellation in self.cancellations.values():
                cancellation.set()
            self.condition.notify_all()

    def _expired(self, event):
        return (self.cancel.is_set() or event['_cancel'] is not None and event['_cancel'].is_set()
                or event['_deadline'] is not None and time.monotonic() >= event['_deadline'])

    def _eligible(self, identifier, kind, event):
        capacity = self.options.workers if kind == 'execution' else self.options.generation_concurrency
        limit = self.options.per_api_worker_limit if kind == 'execution' else self.options.per_api_generation_limit
        lane = event['resource']
        available = lane not in CAPACITY or self.lanes[lane] < CAPACITY[lane]
        if lane == 'cuda':
            available = self.cuda_active == 0
        elif lane == 'device_cuda':
            # Let an older exclusive probe drain the device even under a continuous differential backlog.
            older_exclusive = any(other['resource'] == 'cuda' and other['submitted'] < event['submitted']
                                  for queue in self.pending['execution'].values() for other in queue)
            available = available and self.lanes['cuda'] == 0 and not older_exclusive
        return (self.active[kind] < capacity and self.by_session[identifier, kind] < limit
                and available)

    def _release(self, event):
        event.update(state='finished', finished=time.monotonic())
        self.active[event['kind']] -= 1
        self.by_session[event['session'], event['kind']] -= 1
        self.lanes[event['resource']] -= 1
        self.cuda_active -= int(event['resource'] in ('cuda','device_cuda'))

    def _dispatch(self):
        for event in self.uncertain[:]:
            receipt = event.get('receipt')
            if receipt and Path(receipt).is_file():
                self._release(event)
                event['terminal_receipt_verified'] = True
                self.uncertain.remove(event)
        for kind in ('execution', 'generation'):
            rotation, queues = self.rotation[kind], self.pending[kind]
            while rotation:
                if kind == 'generation' and self.service_stop_file and Path(self.service_stop_file).exists():
                    for queue in queues.values():
                        for event in queue:
                            event.update(state='service_unavailable', finished=time.monotonic())
                    rotation.clear()
                    queues.clear()
                    break
                dispatched = False
                for _ in range(len(rotation)):
                    identifier = rotation.popleft()
                    queue = queues[identifier]
                    for event in list(queue):
                        if self._expired(event):
                            queue.remove(event)
                            event.update(state='cancelled', finished=time.monotonic())
                    chosen = next((event for event in queue if self._eligible(identifier, kind, event)), None)
                    if chosen is not None:
                        queue.remove(chosen)
                        chosen.update(state='running', granted=time.monotonic())
                        self.active[kind] += 1
                        self.by_session[identifier, kind] += 1
                        self.lanes[chosen['resource']] += 1
                        self.cuda_active += int(chosen['resource'] in ('cuda','device_cuda'))
                        chosen['global_active_at_dispatch'] = self.active[kind]
                        chosen['session_active_at_dispatch'] = self.by_session[identifier, kind]
                        dispatched = True
                    if queue:
                        rotation.append(identifier)
                    else:
                        del queues[identifier]
                    if dispatched:
                        break
                if not dispatched:
                    break
        self.condition.notify_all()

    @contextmanager
    def lease(self, identifier, kind, *, resource='cpu', deadline=None, cancel=None, label=''):
        if kind not in self.pending or resource not in ('cpu', 'cuda', *CAPACITY):
            raise ValueError('Unknown scheduling resource')
        if kind == 'generation':
            resource = 'cpu'
        if self.deadline is not None:
            deadline = self.deadline if deadline is None else min(deadline, self.deadline)
        event = dict(session=identifier, kind=kind, resource=resource, label=str(label),
                     submitted=time.monotonic(), state='queued', _deadline=deadline, _cancel=cancel)
        with self.condition:
            if kind == 'generation' and self.service_stop_file and Path(self.service_stop_file).exists():
                from mirage.runtime.pipeline import GenerationBudgetExhausted
                raise GenerationBudgetExhausted('Model service exited; drain submitted tests')
            self.events.append(event)
            queue = self.pending[kind].get(identifier)
            if queue is None:
                queue = self.pending[kind][identifier] = []
                self.rotation[kind].append(identifier)
            queue.append(event)
            self._dispatch()
            while event['state'] == 'queued':
                service_lost = self.service_stop_file and Path(self.service_stop_file).exists()
                cannot_reconcile = (len(self.uncertain) == self.options.generation_concurrency
                                    and all(not e.get('receipt') for e in self.uncertain))
                if kind == 'generation' and (service_lost or cannot_reconcile):
                    from mirage.runtime.pipeline import GenerationBudgetExhausted
                    event['_deadline'] = time.monotonic()
                    self._dispatch()
                    raise GenerationBudgetExhausted('Model service unavailable or no confirmable slots; drain queued execution')
                self.condition.wait(timeout=.05)
                self._dispatch()
            if event['state'] == 'service_unavailable':
                from mirage.runtime.pipeline import GenerationBudgetExhausted
                raise GenerationBudgetExhausted('Model service exited; drain submitted tests')
            if event['state'] == 'cancelled':
                raise SchedulingCancelled(self.reason or 'Task cancelled or deadline expired while queued')
        try:
            yield event
        except BaseException as error:
            if kind == 'generation' and getattr(error, 'cancellation_uncertain', False):
                with self.condition:
                    event.update(state='awaiting_receipt', receipt=getattr(error, 'receipt', None))
                    self.uncertain.append(event)
            raise
        finally:
            with self.condition:
                if event['state'] == 'running':
                    self._release(event)
                self._dispatch()

    def snapshot(self):
        with self.condition:
            events = [{k: v for k, v in event.items() if not k.startswith('_')} for event in self.events]
            for event in events:
                event['queue_seconds'] = max(0., event.get('granted', event.get('finished', time.monotonic())) - event['submitted'])
                event['execution_seconds'] = (event.get('finished', time.monotonic()) - event['granted']
                                               if 'granted' in event else 0.)
            peaks = {}
            for kind in ('execution', 'generation'):
                marks = [(event['granted'], 1) for event in events if event['kind'] == kind and 'granted' in event]
                marks += [(event.get('finished', time.monotonic()), -1) for event in events
                          if event['kind'] == kind and 'granted' in event]
                active = peak = 0
                for _, delta in sorted(marks):
                    active += delta
                    peak = max(peak, active)
                peaks[kind] = peak
            totals = {}
            for event in events:
                key = event['session']
                row = totals.setdefault(key, {})
                kind = event['kind']
                row[kind + '_queue_seconds'] = row.get(kind + '_queue_seconds', 0.) + event['queue_seconds']
                row[kind + '_slot_seconds_sum'] = row.get(kind + '_slot_seconds_sum', 0.) + event['execution_seconds']
            return dict(version='shared-api-scheduling-1', policy='round_robin_with_idle_borrowing',
                        cancelled=self.cancel.is_set(), cancellation_reason=self.reason,
                        active=dict(self.active), cuda_active=self.cuda_active,
                        peaks=peaks, session_resource_totals=totals,
                        events=events, wall_seconds=time.monotonic() - self.started,
                        clock='process monotonic seconds; durations are not GPU utilization')
