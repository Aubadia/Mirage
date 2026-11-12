from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

from mirage.contracts.models import Experiment
from mirage.generator.provider import ProviderFailure
from mirage.sampler.candidates import candidates
from mirage.scheduler import Scheduler


def make_scheduler(workers=2, generation_concurrency=1):
    return Scheduler(candidates(("torch.matmul",), "eager", "fp32", 2), seed=0, epsilon=0.1,
                     workers=workers, generation_concurrency=generation_concurrency,
                     deadline=time.monotonic() + 10)


class SchedulerTests(unittest.TestCase):
    def test_experiments_overlap_with_bounded_dispatch(self):
        scheduler = make_scheduler()
        barrier = threading.Barrier(2)
        active = 0
        peak = 0
        lock = threading.Lock()

        def work(index, draw):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            if index < 2:
                barrier.wait(timeout=3)
            time.sleep(0.01)
            with lock:
                active -= 1
            return index

        self.assertEqual(sorted(scheduler.execute(5, work)), list(range(5)))
        self.assertEqual(peak, 2)

    def test_pause_drains_pool_before_exclusive_work(self):
        scheduler = make_scheduler()
        active = set()
        lock = threading.Lock()

        def work(index, draw):
            with lock:
                active.add(index)
            time.sleep(0.02)
            with lock:
                active.remove(index)
            return index

        drained = False
        for result in scheduler.execute(4, work):
            if result == 0:
                scheduler.paused = True
            if result is None:
                self.assertEqual(active, set())
                self.assertEqual(len([e for e in scheduler.events if e["event"] == "dispatch"]), 2)
                drained = True
        self.assertTrue(drained)

    def test_generation_limit_is_independent_and_queued_work_can_cancel(self):
        scheduler = make_scheduler()
        active = 0
        peak = 0

        def generate():
            nonlocal active, peak
            with scheduler.generation_slot(scheduler.deadline):
                active += 1
                peak = max(peak, active)
                time.sleep(0.02)
                active -= 1

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: generate(), range(3)))
        self.assertEqual(peak, 1)
        scheduler.stop("interrupted")
        with self.assertRaises(TimeoutError), scheduler.generation_slot(scheduler.deadline):
            self.fail("Cancelled generation was dispatched")

    def test_unconfirmed_remote_cancellation_stops_new_dispatch(self):
        scheduler = make_scheduler()
        with self.assertRaises(ProviderFailure), scheduler.generation_slot(scheduler.deadline):
            raise ProviderFailure("timeout", "server cancellation unknown", cancellation_uncertain=True)
        self.assertEqual(list(scheduler.execute(3, lambda *_: self.fail("Unexpected dispatch"))), [])
        self.assertEqual(scheduler.stop_reason, "provider_cancellation_unconfirmed")

    def test_commit_once_and_planning_does_not_count_as_coverage(self):
        scheduler = make_scheduler()
        selected, snapshot = scheduler.select("torch.matmul", 0)
        self.assertFalse(any(snapshot))
        self.assertEqual(scheduler.coverage.report()["experiments"], 0)
        result = {"index": 0, "api_observations": [], "executions": (), "inspection": None,
                  "experiment": Experiment("one", selected, 0), "summary": {"status": "unresolved"}}
        scheduler.commit(result)
        with self.assertRaises(ValueError):
            scheduler.commit(result)
        report = scheduler.coverage.report()
        self.assertEqual(report["experiments"], 1)
        self.assertEqual(report["combinations"]["covered"], 0)
