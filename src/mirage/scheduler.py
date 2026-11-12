"""Bounded experiment dispatch and serialized sampling/coverage decisions."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict
import threading
import time

from mirage.generator.provider import ProviderFailure
from mirage.sampler.coverage import Coverage
from mirage.sampler.policy import Sampler


class Scheduler:
    def __init__(self, universe, *, seed, epsilon, workers, generation_concurrency, deadline):
        self.universe = universe
        self.coverage = Coverage(universe)
        self.sampler = Sampler(universe, seed=seed, epsilon=epsilon)
        self.workers = workers
        self.deadline = deadline
        self.cancel = threading.Event()
        self.generation_slots = threading.BoundedSemaphore(generation_concurrency)
        self.lock = threading.RLock()
        self.events = []
        self.paused = False
        self.stop_reason = "case_limit"

    def _event(self, kind, **fields):
        self.events.append({"sequence": len(self.events), "event": kind, **fields})

    def draw_api(self, index):
        with self.lock:
            candidate = self.sampler.sample(self.coverage.counts)
            self._event("dispatch", index=index, api=candidate.api,
                        verified_counts=[self.coverage.counts.get(c, 0) for c in self.universe])
            return candidate

    def select(self, api, index):
        # Completion order may change the next draw; the exact snapshot is evidence.
        with self.lock:
            subset = tuple(item for item in self.universe if item.api == api)
            weights = dict(zip(self.universe, self.sampler.probabilities(self.coverage.counts)))
            selected = self.sampler.random.choices(subset, weights=[weights[item] for item in subset], k=1)[0]
            snapshot = [self.coverage.counts.get(item, 0) for item in self.universe]
            self._event("selection", index=index, candidate=asdict(selected), verified_counts=snapshot)
            return selected, snapshot

    def commit(self, result):
        with self.lock:
            for api, execution in result["api_observations"]:
                self.coverage.observe_api(api, execution)
            if result["experiment"] is not None:
                self.coverage.update(result["experiment"], result["executions"], result["inspection"])
            self._event("completion", index=result["index"], status=result["summary"]["status"])

    def stop(self, reason):
        with self.lock:
            if not self.cancel.is_set():
                self.stop_reason = reason
            self.cancel.set()

    @contextmanager
    def generation_slot(self, deadline):
        while True:
            if self.cancel.is_set():
                raise TimeoutError("Batch cancelled while waiting for generation")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Generation queue deadline expired")
            if self.generation_slots.acquire(timeout=min(remaining, 0.1)):
                break
        try:
            if self.cancel.is_set():
                raise TimeoutError("Batch cancelled before generation")
            yield
        except ProviderFailure as error:
            if error.cancellation_uncertain:
                # Releasing a client slot does not prove the server stopped computing.
                self.stop("provider_cancellation_unconfirmed")
            raise
        finally:
            self.generation_slots.release()

    def execute(self, count, work):
        """Yield completed cases; None marks an empty pool for exclusive replay."""
        pending = {}
        index = 0
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="mirage") as pool:
            try:
                while pending or (index < count and not self.cancel.is_set()) or self.paused:
                    if time.monotonic() >= self.deadline:
                        self.stop("time_limit")
                    if self.paused and not pending:
                        yield None
                        self.paused = False
                    while (len(pending) < self.workers and index < count
                           and not self.paused and not self.cancel.is_set()):
                        draw = self.draw_api(index)
                        pending[pool.submit(work, index, draw)] = index
                        index += 1
                    if not pending:
                        break
                    try:
                        done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    except KeyboardInterrupt:
                        self.stop("interrupted")
                        continue
                    for future in sorted(done, key=lambda item: pending[item]):
                        del pending[future]
                        yield future.result()
            finally:
                self.cancel.set()



class LimitedProvider:
    def __init__(self, provider, scheduler):
        self.provider = provider
        self.scheduler = scheduler

    def generate(self, request, *, deadline):
        with self.scheduler.generation_slot(deadline):
            return self.provider.generate(request, deadline=deadline)
