"""Control-plane CLI. Execution commands remain explicitly unavailable."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from mirage.artifacts.store import ArtifactStore
from mirage.contracts.apis import select_apis


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

    report = commands.add_parser("report", help="Inventory saved artifacts; does not infer coverage")
    report.add_argument("directory", type=Path)

    run = commands.add_parser("run", help="Reserved for the forthcoming execution runtime")
    run.add_argument("--apis", nargs="+")
    run.add_argument("--cases", type=positive_integer, default=100)
    run.add_argument("--time-limit", type=positive_integer, default=14400)
    run.add_argument("--worker-timeout", type=positive_integer, default=600)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--backend", choices=("eager", "compile_eager", "aot_eager", "inductor"), default="eager")
    run.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")

    replay = commands.add_parser("replay", help="Reserved for the forthcoming replay runtime")
    replay.add_argument("manifest", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            select_apis(args.apis)
            if args.seed < 0:
                raise ValueError("seed must be nonnegative")
            raise ValueError("run is unavailable: the PyTorch runtime is not implemented in this batch")
        if args.command == "replay":
            raise ValueError("replay is unavailable: executable replay bundles are not implemented in this batch")
        if args.command == "inspect":
            if args.manifest.suffix != ".json":
                raise ValueError("Expected a .json manifest")
            output = ArtifactStore(args.manifest.parent).read(args.manifest.stem)
        else:
            manifests = ArtifactStore(args.directory).list_manifests()
            output = {
                "artifact_count": len(manifests),
                "kinds": dict(Counter(item["kind"] for item in manifests)),
                "artifact_ids": [item["artifact_id"] for item in manifests],
                "coverage": None,
                "note": "Artifact inventory only; coverage requires validated execution evidence.",
            }
        print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError) as error:
        print(f"mirage: {error}", file=sys.stderr)
        return 2
