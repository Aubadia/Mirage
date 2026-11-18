"""Bounded subprocess execution with file-backed output and process-group cleanup."""

from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


_cancellation = ContextVar("mirage_worker_cancellation", default=None)


@contextmanager
def cancellation_scope(cancel):
    """Allow existing replay callers to share the controller's cancellation token."""
    token = _cancellation.set(cancel)
    try:
        yield
    finally:
        _cancellation.reset(token)


@dataclass(frozen=True)
class ProcessResult:
    outcome: str
    returncode: int | None
    elapsed_seconds: float
    directory: str
    error: str | None = None
    resources: dict = field(default_factory=dict)


def run_worker(request: dict, directory: Path, timeout: float, *, cancel=None) -> ProcessResult:
    cancel = cancel if cancel is not None else _cancellation.get()
    from mirage.runtime.scheduler import AnyCancellation, SchedulingCancelled, current_scope
    scope = current_scope()
    if scope is None:
        return _run_worker(request, directory, timeout, cancel=cancel)
    cancellation = AnyCancellation(cancel, scope.cancel, scope.scheduler.cancel)
    # Provider subprocesses use generation capacity, acquired outside their timer.
    if request.get('mode') == 'provider':
        return _run_worker(request, directory, timeout, cancel=cancellation)
    config = request.get('config', {})
    device = config.get('device', scope.device)
    resource = 'cuda' if str(device).split(':')[0] == 'cuda' else 'cpu'
    begin = time.monotonic()
    try:
        with scope.scheduler.lease(scope.identifier, 'execution', resource=resource,
                                   deadline=scope.deadline, cancel=cancellation, label=directory) as event:
            remaining = timeout if scope.deadline is None else min(timeout, scope.deadline - time.monotonic())
            result = _run_worker(request, directory, remaining, cancel=cancellation)
            queue_seconds = event['granted'] - event['submitted']
    except SchedulingCancelled:
        result = _run_worker(request, directory, 0, cancel=cancellation)
        queue_seconds = time.monotonic() - begin
    result.resources['scheduling'] = dict(session=scope.identifier, resource=resource,
                                         queue_seconds=queue_seconds)
    (directory / 'process.json').write_text(json.dumps(asdict(result), indent=2), encoding='utf-8')
    return result


def _run_worker(request: dict, directory: Path, timeout: float, *, cancel=None) -> ProcessResult:
    cancel = cancel if cancel is not None else _cancellation.get()
    directory.mkdir(parents=True, exist_ok=False)
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, indent=2, allow_nan=False), encoding="utf-8")
    if timeout <= 0 or (cancel is not None and cancel.is_set()):
        result = ProcessResult("timeout", None, 0.0, str(directory), "Budget exhausted or batch cancelled before launch")
        (directory / "process.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result
    environment = dict(os.environ)
    package_root = str(Path(request["engine"]).resolve()) if request.get("engine") else str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join((package_root, environment.get("PYTHONPATH", "")))
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    started = time.monotonic()
    outcome, code, error = "crash", None, None
    resources = {}
    with (directory / "stdout.txt").open("wb") as stdout, (directory / "stderr.txt").open("wb") as stderr:
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "mirage.runtime.worker", str(request_path.resolve())],
                stdout=stdout, stderr=stderr, env=environment, start_new_session=True,
            )
            try:
                expires = started + timeout
                while True:
                    if cancel is not None and cancel.is_set():
                        error = "Batch cancelled; worker process group terminated"
                        raise subprocess.TimeoutExpired(process.args, timeout)
                    remaining = expires - time.monotonic()
                    if remaining <= 0:
                        error = "Worker deadline expired"
                        raise subprocess.TimeoutExpired(process.args, timeout)
                    try:
                        code = process.wait(timeout=min(remaining, 0.1))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except subprocess.TimeoutExpired:
                outcome = "timeout"
                _kill_group(process)
                code = process.wait()
            else:
                status_path = directory / "status.json"
                if status_path.exists():
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    outcome, error = status["outcome"], status.get("error")
                    resources = status.get("resources", {})
                    if code != 0 and outcome == "completed":
                        outcome, error = "crash", "Worker exited nonzero after reporting completion"
                else:
                    error = "Worker exited without a completion record"
        except BaseException:
            if process is not None and process.poll() is None:
                _kill_group(process)
                process.wait()
            raise
        finally:
            # Descendants can outlive a parent that returned or crashed.
            if process is not None:
                _kill_group(process)
    result = ProcessResult(outcome, code, time.monotonic() - started, str(directory), error, resources)
    (directory / "process.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
