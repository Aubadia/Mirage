"""Batch exploration, immutable evidence inspection and LLM-free replay."""

import argparse
import json
from pathlib import Path
import sys

from mirage.artifacts.store import ArtifactStore
from mirage.contracts.apis import select_apis, ACCEPTANCE_APIS
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
    run.add_argument("--strategy", choices=("training","properties","stateful"), default="training")
    run.add_argument("--representation-traversal", action=argparse.BooleanOptionalAction, default=True,
                     help="Traverse applicable target input representations in stateful mode (default: enabled)")
    run.add_argument("--oracle-improvements", action=argparse.BooleanOptionalAction, default=True,
                     help="Use scoped local gradient evidence and audit scheduling in stateful mode (default: enabled)")
    run.add_argument("--generation-pipeline", action=argparse.BooleanOptionalAction, default=True,
                     help="Overlap generation and full traversal; time-limit is model seconds per API in stateful mode")
    run.add_argument("--generation-overlap", action=argparse.BooleanOptionalAction, default=True,
                     help="Allow model/framework overlap; disabling waits for framework feedback before the next proposal")
    run.add_argument("--cases", type=positive_integer, default=100)
    run.add_argument("--time-limit", type=positive_integer, default=14400)
    run.add_argument("--worker-timeout", type=positive_integer, default=600)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--backend", choices=("eager", "compile_eager", "aot_eager", "inductor"), default="eager")
    run.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    run.add_argument("--provider", help="vllm or Python module:factory")
    run.add_argument("--output", type=Path, default=Path("result/experiments"))
    run.add_argument("--workers", type=positive_integer, default=None,
                     help="Execution workers (default: 8 for the stateful pipeline, 1 otherwise)")
    run.add_argument("--generation-concurrency", type=positive_integer, default=2)
    run.add_argument("--canonicals-per-api", type=positive_integer, default=3)
    run.add_argument("--targets-per-canonical", type=positive_integer, default=12)
    run.add_argument("--paths-per-target", type=positive_integer, default=2)
    run.add_argument("--forward-only", action="store_true", help="Do not require a backward path through the target API")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--size", type=positive_integer, default=4)
    run.add_argument("--epsilon", type=float, default=0.1)
    run.add_argument("--replay-failures", type=int, default=1)

    replay = commands.add_parser("replay", help="Execute a saved pair without calling an LLM")
    replay.add_argument("manifest", type=Path)
    replay.add_argument("--worker-timeout", type=positive_integer, default=600)
    replay.add_argument("--allow-environment-change", action="store_true")
    resume = commands.add_parser("resume", help="Drain saved stateful pipeline work without further model calls")
    resume.add_argument("directory", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            from mirage.contracts.programs import RunOptions
            from mirage.workflow import run

            apis = select_apis(args.apis or list(ACCEPTANCE_APIS))
            if not args.provider:
                raise ValueError("--provider vllm or module:factory is required")
            values = vars(args).copy()
            for key in ("command", "apis"):
                values.pop(key)
            if values["strategy"] != "stateful":
                values["generation_pipeline"] = values["representation_traversal"] = values["oracle_improvements"] = False
            values["output"] = str(values["output"])
            values["contract_version"] = 4 if values["strategy"] == "stateful" else 3
            output = run(RunOptions(**values), apis)
        elif args.command == "resume":
            from mirage.runtime.pipeline import resume_pipeline
            output = resume_pipeline(args.directory)
        elif args.command == "replay":
            from mirage.workflow import replay

            output = replay(args.manifest, timeout=args.worker_timeout,
                            allow_environment_change=args.allow_environment_change)
        elif args.command == "inspect":
            if args.manifest.suffix != ".json":
                raise ValueError("Expected a .json manifest")
            output = ArtifactStore(args.manifest.parent).read(args.manifest.stem)
            if "files" in output["payload"]:
                from mirage.artifacts.bundle import verify

                verify(args.manifest.parent / output["payload"]["bundle"], output["payload"]["files"])
        else:
            property_summary=args.directory/"summary.json"
            if property_summary.exists() and "records" in json.loads(property_summary.read_text()):
                from mirage.artifacts.report import summarize_properties
                output=summarize_properties(json.loads(property_summary.read_text()))
            else:
                manifests = ArtifactStore(args.directory).list_manifests()
                output = summarize(manifests)
        print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command == "replay" and output.get("process", {}).get("outcome", "completed") != "completed":
            return 1
        if args.command == "replay" and output.get("replay_valid") is False:
            return 1
        if args.command in ("run", "resume"):
            if output["stop_reason"] in ("interrupted", "controller_error", "pipeline_incomplete", "provider_cancellation_unconfirmed",
                                       "interrupted_or_provider_cancellation_unconfirmed"):
                return 1
            if any(item["status"] != "validated" for item in output.get("canonicals", [])):
                return 1
            if output["stop_reason"] == "metadata_failed":
                return 1
            if any(case["status"].startswith("generation_provider_") or case["status"] == "provider_failed"
                   for case in output.get("experiments", output.get("records", []))):
                return 1
        return 0
    except (ValueError, OSError, TypeError) as error:
        print(f"mirage: {error}", file=sys.stderr)
        return 2
