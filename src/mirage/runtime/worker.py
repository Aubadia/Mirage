"""Private subprocess entry point for initialization, providers and training."""

from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import traceback


def dispatch(request: dict, directory: Path) -> None:
    mode = request["mode"]
    if mode == "provider":
        from mirage.generator.provider import Request, Stage, load_provider

        value = dict(request["request"])
        value["stage"] = Stage(value["stage"])
        response = load_provider(request["provider"], directory=directory).generate(
            Request(**value), deadline=request["deadline"])
        from mirage.generator.provider import Response
        if not isinstance(response, Response):
            raise TypeError("Provider must return Response")
        (directory / "response.json").write_text(json.dumps(asdict(response), allow_nan=False), encoding="utf-8")
        return

    import torch
    if mode == "inspect_pair":
        from mirage.artifacts.bundle import load_tensor_record
        from mirage.rules.conditional import inspect

        records = [load_tensor_record(Path(path)) if path else None for path in request["records"]]
        result = inspect(request["rule"], *records)
        (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if mode in ("api_metadata", "initialize_program", "execute_program"):
        from mirage.runtime.api_metadata import describe as api_description
        from mirage.runtime.program import initialize, run
        from mirage.runtime.probes import Recorder

        if mode == "api_metadata":
            result = api_description(request["api"])
        elif mode == "initialize_program":
            result = initialize(request["source"], request["config"], directory / "initial.pt")
        else:
            result = run(request["source"], request["config"], request["initial"],
                         Recorder(directory / "trace.jsonl"))
            torch.save(result, directory / "result.pt")
            result = {key: result[key] for key in ("api_executed", "backward_executed", "target_state_reached")}
        (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    raise ValueError(f"Unknown worker mode: {mode}")


def main() -> int:
    path = Path(sys.argv[1])
    directory = path.parent
    status = {"outcome": "completed", "error": None}
    try:
        dispatch(json.loads(path.read_text(encoding="utf-8")), directory)
    except Exception as error:
        traceback.print_exc()
        outcome = "exception"
        if isinstance(error, TimeoutError):
            outcome = "timeout"
        elif isinstance(error, MemoryError):
            outcome = "oom"
        elif "torch" in sys.modules and isinstance(error, sys.modules["torch"].OutOfMemoryError):
            outcome = "oom"
        status = {"outcome": outcome, "error": f"{type(error).__name__}: {error}"}
        from mirage.generator.provider import ProviderFailure

        if isinstance(error, ProviderFailure):
            status["provider_failure"] = {
                "category": error.category,
                "cancellation_uncertain": error.cancellation_uncertain,
            }
    usage = resource.getrusage(resource.RUSAGE_SELF)
    status["resources"] = {"user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
                           "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    (directory / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 0 if status["outcome"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
