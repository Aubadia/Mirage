"""Bounded, replaceable subprocesses for frozen device comparisons."""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
import gc
import json
import os
from pathlib import Path
import random
import select
import subprocess
import sys
import threading
import time
import traceback

from dataclasses import asdict

CAPACITY = {'device_cuda': 4, 'device_cpu': 2, 'comparison': 1}
_local = ContextVar('mirage_local_differential_scope', default=None)


def scope():
    from mirage.runtime.scheduler import current_scope
    return current_scope() or _local.get()


def standalone(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if scope() is not None:
            return function(*args, **kwargs)
        from mirage.contracts.scheduling import SchedulingOptions
        from mirage.runtime.scheduler import SessionScope, SharedScheduler
        scheduler = SharedScheduler(SchedulingOptions(sum(CAPACITY.values())))
        token = _local.set(SessionScope(scheduler, 'standalone', 'cpu', threading.Event()))
        try:
            return function(*args, **kwargs)
        finally:
            _local.reset(token)
            scheduler.differential.close()
    return wrapped


class Worker:
    def __init__(self, engine, identity, lane):
        self.identity = identity
        self.started = time.monotonic()
        self.used = 0
        read_fd, write_fd = os.pipe()
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((engine, os.environ.get('PYTHONPATH', ''))))
        env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(write_fd), lane],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                pass_fds=(write_fd,), start_new_session=True, env=env)
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        self.ack = read_fd

    def close(self):
        from mirage.runtime.process import _kill_group
        _kill_group(self.process)
        self.process.wait()
        self.process.stdin.close()
        os.close(self.ack)


class DifferentialPool:
    """Admission and fairness belong to the owning scheduler; idle identities are interchangeable."""
    def __init__(self):
        self.idle = defaultdict(list)
        self.lock = threading.Lock()

    def close(self):
        with self.lock:
            for workers in self.idle.values():
                for worker in workers:
                    worker.close()
            self.idle.clear()

    def execute(self, request, directory, lane, timeout, cancel):
        from mirage.runtime.process import ProcessResult
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'request.json').write_text(json.dumps(request, allow_nan=False))
        begin = time.monotonic()
        outcome, error, code, resources = 'crash', None, None, {}
        worker = None
        reusable = False
        try:
            if timeout <= 0 or cancel.is_set():
                raise TimeoutError('Case deadline or cancellation before dispatch')
            with self.lock:
                idle = self.idle[lane]
                identity = request['engine_identity']
                after = request.get('fresh_after', 0)
                worker = next((w for w in reversed(idle) if w.identity == identity and w.started >= after), None)
                if worker is not None:
                    idle.remove(worker)
                elif idle:
                    idle.pop().close()
            if worker is None:
                worker = Worker(request['engine'], identity, lane)
            worker.used += 1
            resources.update(worker_pid=worker.process.pid, worker_started=worker.started,
                             worker_case=worker.used, lane=lane)
            worker.process.stdin.write((json.dumps(str(directory.resolve()))+'\n').encode())
            worker.process.stdin.flush()
            while True:
                left = begin + timeout - time.monotonic()
                if left <= 0 or cancel.is_set():
                    raise TimeoutError('Resident case cancelled or timed out')
                if select.select([worker.ack], [], [], min(left, .1))[0]:
                    ack = os.read(worker.ack, 1)
                    if not ack:
                        raise EOFError('Worker exited before reporting completion')
                    status = json.loads((directory/'status.json').read_text())
                    outcome, error = status['outcome'], status.get('error')
                    resources.update(status.get('resources', {}))
                    reusable = ack == b'+' and outcome == 'completed'
                    code = 0 if outcome == 'completed' else 1
                    break
        except TimeoutError as exc:
            outcome, error = 'timeout', str(exc)
        except (EOFError, OSError, ValueError) as exc:
            error = str(exc)
        finally:
            if worker is not None:
                if reusable:
                    with self.lock:
                        self.idle[lane].append(worker)
                else:
                    worker.close()
        result = ProcessResult(outcome, code, time.monotonic()-begin, str(directory), error, resources)
        (directory/'process.json').write_text(json.dumps(asdict(result)))
        return result


@contextmanager
def case_state(torch, cuda):
    torch.compiler.reset()
    from mirage.runtime.environment import configure
    configure()
    grad, dtype = torch.is_grad_enabled(), torch.get_default_dtype()
    default_device = torch.get_default_device()
    cache_enabled = torch.is_autocast_cache_enabled()
    before_cuda = torch.cuda.is_initialized()
    rng = random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state() if cuda else None
    cast = {k:(torch.is_autocast_enabled(k),torch.get_autocast_dtype(k)) for k in ('cpu','cuda')}
    module = torch.nn.modules.module
    backward_hook_mode = module._global_is_full_backward_hook
    hooks = {k:dict(v) for k,v in vars(module).items() if k.startswith('_global_') and isinstance(v,dict)}
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.set_rng_state(rng[2])
        random.setstate(rng[0])
        torch.set_rng_state(rng[1])
        torch.set_grad_enabled(grad)
        torch.set_default_dtype(dtype)
        if torch.get_default_device() != default_device:
            torch.set_default_device(default_device)
        torch.set_autocast_cache_enabled(cache_enabled)
        torch.clear_autocast_cache()
        module._global_is_full_backward_hook = backward_hook_mode
        for k,(enabled,precision) in cast.items():
            torch.set_autocast_enabled(k,enabled)
            torch.set_autocast_dtype(k,precision)
        for k,original in hooks.items():
            current = getattr(module,k)
            current.clear()
            current.update(original)
        gc.collect()
        if not cuda and not before_cuda and torch.cuda.is_initialized():
            # The side record is still valid; retire a CPU worker that acquired device state.
            raise UnexpectedDeviceContext()


class UnexpectedDeviceContext(Exception):
    pass


def serve(fd, lane):
    import torch
    import resource
    from mirage.runtime.worker import dispatch
    torch.set_num_threads(1)
    cuda = lane == 'device_cuda' and torch.cuda.is_available()
    if cuda and torch.cuda.device_count() != 1:
        raise RuntimeError('Each test step must own exactly one visible GPU')
    for line in sys.stdin:
        directory = Path(json.loads(line))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        status = dict(outcome='completed')
        reusable = True
        with ExitStack() as stack:
            for stream, name in ((sys.stdout,'stdout.txt'),(sys.stderr,'stderr.txt')):
                stream.flush()
                prior = os.dup(stream.fileno())
                stack.callback(os.close, prior)
                stack.callback(os.dup2, prior, stream.fileno())
                dest = stack.enter_context((directory/name).open('ab'))
                os.dup2(dest.fileno(),stream.fileno())
                stack.callback(stream.flush)
            try:
                request = json.loads((directory/'request.json').read_text())
                if lane == 'comparison':
                    dispatch(request,directory)
                else:
                    with case_state(torch,cuda):
                        dispatch(request,directory)
                    side = json.loads((directory/'summary.json').read_text())
                    reusable = side.get('status') == 'ready' and not side.get('target_exception')
            except UnexpectedDeviceContext:
                reusable = False
                if not (directory/'summary.json').is_file():
                    status.update(outcome='exception', error='CPU worker exited without a side record')
            except Exception as exc:
                traceback.print_exc()
                status.update(outcome='oom' if isinstance(exc,(MemoryError,torch.OutOfMemoryError)) else 'exception',
                              error=f'{type(exc).__name__}: {exc}')
                reusable = False
            end = resource.getrusage(resource.RUSAGE_SELF)
            status['resources'] = dict(user_cpu_seconds=end.ru_utime-usage.ru_utime,
                                       system_cpu_seconds=end.ru_stime-usage.ru_stime,
                                       worker_peak_rss_bytes=end.ru_maxrss*1024)
            (directory/'status.json').write_text(json.dumps(status))
        os.write(fd,b'+' if reusable else b'-')
        if not reusable:
            return


if __name__ == '__main__':
    serve(int(sys.argv[1]), sys.argv[2])
