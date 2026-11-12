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
        response = load_provider(request["provider"]).generate(Request(**value), deadline=request["deadline"])
        from mirage.generator.provider import Response
        if not isinstance(response, Response):
            raise TypeError("Provider must return Response")
        (directory / "response.json").write_text(json.dumps(asdict(response), allow_nan=False), encoding="utf-8")
        return

    import torch
    from mirage.contracts.programs import CaseConfig, Program
    from mirage.runtime.environment import configure, describe

    config = CaseConfig.from_dict(request["config"])
    configure()
    if mode == "initialize":
        from mirage.runtime.state import make_initial

        environment = describe(config.device)
        torch.save(make_initial(config), directory / "initial.pt")
        (directory / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    elif mode == "execute":
        from mirage.runtime.harness import execute
        from mirage.runtime.probes import Recorder

        program = Program.from_dict(request["program"])
        initial = torch.load(request["initial"], map_location="cpu", weights_only=True)
        result = execute(program, config, initial, Recorder(directory / "trace.jsonl"), directory / "partial.pt")
        torch.save(result, directory / "result.pt")
    else:
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
    usage = resource.getrusage(resource.RUSAGE_SELF)
    status["resources"] = {"user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
                           "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    (directory / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 0 if status["outcome"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
