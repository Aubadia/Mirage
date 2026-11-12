"""Batch exploration, immutable evidence inspection and LLM-free replay."""

import argparse
import json
from pathlib import Path
import sys

from mirage.artifacts.store import ArtifactStore
from mirage.contracts.apis import select_apis
from mirage.artifacts.report import summarize


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="mirage", description="PyTorch representation and training-state testing")
    commands = result.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="Validate and display one manifest")
    inspect.add_argument("manifest", type=Path)

    report = commands.add_parser("report", help="Show validated batch coverage and saved outcomes")
    report.add_argument("directory", type=Path)

    run = commands.add_parser("run", help="Generate, execute and inspect a batch of candidates", allow_abbrev=False)
    run.add_argument("--apis", nargs="+")
    run.add_argument("--cases", type=positive_integer, default=100)
    run.add_argument("--time-limit", type=positive_integer, default=14400)
    run.add_argument("--worker-timeout", type=positive_integer, default=600)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--backend", choices=("eager", "compile_eager", "aot_eager", "inductor"), default="eager")
    run.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    run.add_argument("--provider", help="vllm, Python module:factory, or reference integration fixture")
    run.add_argument("--output", type=Path, default=Path("result/experiments"))
    run.add_argument("--workers", type=positive_integer, default=1)
    run.add_argument("--generation-concurrency", type=positive_integer, default=2)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--updates", type=positive_integer, default=3)
    run.add_argument("--size", type=positive_integer, default=4)
    run.add_argument("--epsilon", type=float, default=0.1)
    run.add_argument("--warmup-steps", type=positive_integer, default=1)
    run.add_argument("--replay-failures", type=int, default=1)

    replay = commands.add_parser("replay", help="Execute a saved pair without calling an LLM")
    replay.add_argument("manifest", type=Path)
    replay.add_argument("--worker-timeout", type=positive_integer, default=600)
    replay.add_argument("--allow-environment-change", action="store_true")
    replay.add_argument("--source-program", type=Path)
    replay.add_argument("--followup-program", type=Path)
    replay.add_argument("--size", type=positive_integer)
    replay.add_argument("--updates", type=positive_integer)
    replay.add_argument("--iteration", type=int)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            from mirage.contracts.programs import RunOptions
            from mirage.workflow import run

            apis = select_apis(args.apis)
            if not args.provider:
                raise ValueError("--provider module:factory is required; use --provider reference only for fixture runs")
            values = vars(args).copy()
            for key in ("command", "apis"):
                values.pop(key)
            values["output"] = str(values["output"])
            output = run(RunOptions(**values), apis)
        elif args.command == "replay":
            from mirage.artifacts.replay import replay

            output = replay(args.manifest, timeout=args.worker_timeout,
                            allow_environment_change=args.allow_environment_change,
                            source_program=args.source_program, followup_program=args.followup_program,
                            size=args.size, updates=args.updates, iteration=args.iteration)
        elif args.command == "inspect":
            if args.manifest.suffix != ".json":
                raise ValueError("Expected a .json manifest")
            output = ArtifactStore(args.manifest.parent).read(args.manifest.stem)
            if "files" in output["payload"]:
                from mirage.artifacts.bundle import verify

                verify(args.manifest.parent / output["payload"]["bundle"], output["payload"]["files"])
        else:
            manifests = ArtifactStore(args.directory).list_manifests()
            output = summarize(manifests)
        print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command == "run":
            if output["stop_reason"] in ("interrupted", "controller_error", "provider_cancellation_unconfirmed"):
                return 1
            if any(case["status"].startswith("generation_provider_") for case in output["experiments"]):
                return 1
        return 0
    except (ValueError, OSError, TypeError) as error:
        print(f"mirage: {error}", file=sys.stderr)
        return 2
