"""Bounded subprocess execution with file-backed output and process-group cleanup."""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


@dataclass(frozen=True)
class ProcessResult:
    outcome: str
    returncode: int | None
    elapsed_seconds: float
    directory: str
    error: str | None = None
    resources: dict = field(default_factory=dict)


def run_worker(request: dict, directory: Path, timeout: float) -> ProcessResult:
    directory.mkdir(parents=True, exist_ok=False)
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, indent=2, allow_nan=False), encoding="utf-8")
    if timeout <= 0:
        result = ProcessResult("timeout", None, 0.0, str(directory), "Budget exhausted before worker launch")
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
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                outcome = "timeout"
                os.killpg(process.pid, signal.SIGKILL)
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
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    result = ProcessResult(outcome, code, time.monotonic() - started, str(directory), error, resources)
    (directory / "process.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result
