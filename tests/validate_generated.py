"""Revalidate preserved LLM programs under the current engine inside Slurm."""

import argparse
import json
from pathlib import Path

from mirage.contracts.apis import ACCEPTANCE_APIS
from mirage.runtime.program import initialize, run
from mirage.runtime.probes import Recorder
from mirage.rules.conditional import inspect
from mirage.workflow import base_config
from mirage.contracts.programs import RunOptions


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", default=["real-*"])
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    selected = {}
    candidates = {path for batch in args.batches for path in args.root.glob(f"{batch}/*/canonical.json")}
    for path in sorted(candidates):
        entry = json.loads(path.read_text())
        if entry.get("canonical_api_verified"):
            for program in sorted(path.parent.glob("canonical-*/program.py")):
                result = program.parent / "validation/summary.json"
                if result.exists() and json.loads(result.read_text()).get("backward_executed"):
                    selected[entry["api"]] = program
    report = {"apis": [], "missing": sorted(set(ACCEPTANCE_APIS) - selected.keys()), "rules": []}
    for index, api in enumerate(ACCEPTANCE_APIS):
        if api not in selected:
            continue
        directory = args.output / f"api-{index:02d}"
        directory.mkdir()
        source = selected[api].read_text()
        (directory / "program.py").write_text(source)
        config = base_config(api, RunOptions(str(directory), "preserved-LLM", device="cuda"), 42)
        entry = {"api": api, "original_program": str(selected[api]), "status": "unresolved"}
        try:
            metadata = initialize(source, config, directory / "initial.pt")
            record = run(source, config, directory / "initial.pt", Recorder(directory / "trace.jsonl"))
            torch.save(record, directory / "result.pt")
            entry.update(status="validated" if record["api_executed"] and record["backward_executed"] else "invalid",
                         graph=record["graph"], metadata=metadata)
            for rule, changes in (("saved_tensor", {"saved_tensors": True}), ("checkpoint", {"checkpoint": True}),
                                  ("accumulation", {"microbatches": 2})):
                try:
                    left = record
                    if rule == "accumulation":
                        left = run(source, dict(config, accumulation_source=True), directory / "initial.pt",
                                   Recorder(directory / f"{rule}-source.jsonl"))
                    right = run(source, dict(config, **changes), directory / "initial.pt",
                                Recorder(directory / f"{rule}-followup.jsonl"))
                    comparison = inspect(rule, left, right)
                    torch.save({"source": left, "followup": right}, directory / f"{rule}.pt")
                except Exception as error:
                    comparison = {"verdict": None, "reason": f"{type(error).__name__}: {error}"}
                report["rules"].append({"api": api, "rule": rule, **comparison})
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        report["apis"].append(entry)
        (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return int(bool(report["missing"]) or any(e["status"] != "validated" for e in report["apis"]))


if __name__ == "__main__":
    raise SystemExit(main())
