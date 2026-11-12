"""Replay saved cases without regeneration, retaining all evidence as a child artifact."""

from dataclasses import asdict, replace
from pathlib import Path
import shutil
import time
import uuid

from mirage.artifacts.bundle import inventory, load_tensor_record, verify
from mirage.artifacts.store import ArtifactStore, json_value
from mirage.contracts.models import Candidate, Experiment
from mirage.contracts.programs import CaseConfig, Program
from mirage.inspector.comparison import equal
from mirage.inspector.oracle import inspect
from mirage.inspector.validation import assess
from mirage.rules.registry import RULES


def replay(manifest_path, *, timeout=600, deadline=None, allow_environment_change=False,
           source_program=None, followup_program=None, size=None, updates=None, iteration=None):
    from mirage.workflow import execute_case, require_torch

    require_torch()
    manifest_path = Path(manifest_path).resolve()
    store = ArtifactStore(manifest_path.parent)
    original = store.read(manifest_path.stem)
    saved = original["payload"]
    if original["kind"] != "experiment" or saved.get("status") != "executed":
        raise ValueError("This artifact does not contain a complete executable pair")
    root = manifest_path.parent / saved["bundle"]
    verify(root, saved["files"])
    candidate = Candidate(**saved["experiment"]["candidate"])
    if candidate.mr and json_value(RULES[candidate.mr].as_dict()) != saved["rule"]:
        raise ValueError("Saved MR version differs from this Mirage installation")
    configs = [CaseConfig.from_dict(value) for value in saved["configs"]]
    programs = [Program.from_dict(value) for value in saved["programs"]]
    original_configs = list(configs)
    for index, path in enumerate((source_program, followup_program)):
        if path is not None:
            programs[index] = replace(programs[index], source=Path(path).read_text(encoding="utf-8"))
    if any(value is not None for value in (size, updates, iteration)):
        for index, config in enumerate(configs):
            target = replace(config.target, iteration=iteration) if iteration is not None else config.target
            configs[index] = replace(config, size=size if size is not None else config.size,
                                     updates=updates if updates is not None else config.updates, target=target)
            programs[index] = replace(programs[index], actions=configs[index].actions)
        candidate = replace(candidate, state=configs[1].target.key, transition=">".join(configs[1].actions))

    child_id = f"replay-{uuid.uuid4().hex[:20]}"
    bundle = manifest_path.parent / child_id
    bundle.mkdir()
    engine = bundle / "engine"
    shutil.copytree(root / saved["engine"], engine)
    initial_path = bundle / "initial.pt"
    shutil.copyfile(root / saved["initial"], initial_path)
    if size is not None:
        import torch

        initial = load_tensor_record(initial_path)
        if size > original_configs[0].size or initial.get("optimizer") is not None:
            raise ValueError("Size minimization requires a smaller initial case with an uninitialized optimizer")
        initial["parameter"] = initial["parameter"][:size, :size].clone()
        initial["bias"] = initial["bias"][:size].clone()
        torch.save(initial, initial_path)
    experiment = Experiment(child_id, candidate, saved["experiment"]["seed"],
                            saved["experiment"]["requires_backward"])
    deadline = deadline if deadline is not None else time.monotonic() + 2 * timeout
    results = []
    for role, program, config in zip(("source", "followup"), programs, configs):
        budget = max(0.0, min(timeout, deadline - time.monotonic()))
        results.append(execute_case(program, config, initial_path, bundle / role, budget, engine))
    processes, records = tuple(zip(*results))
    executions = tuple(assess(experiment, f"{child_id}-{role}", role, config, process, record)
                       for role, config, process, record in zip(("source", "followup"), configs, processes, records))
    environment_changes = []
    for role, directory, record in zip(("source", "followup"), saved["result_directories"], records):
        previous_path = root / directory / "result.pt"
        previous = load_tensor_record(previous_path) if previous_path.exists() else {}
        if record is not None and "environment" in record and "environment" in previous and not equal(previous["environment"], record["environment"]):
            environment_changes.append({"role": role, "before": previous["environment"], "after": record["environment"]})
    inspection, comparison = inspect(experiment, executions, configs, records, saved.get("tolerance_policy"))
    if environment_changes and not allow_environment_change:
        # Preserve the new execution, but do not call it same-environment reproduction.
        from mirage.contracts.models import Inspection
        inspection = Inspection(child_id, tuple(e.execution_id for e in executions), None,
                                reason="Environment changed; explicit cross-version validation was not requested") if candidate.mr else None
        comparison = {"reason": "Environment mismatch", "correctness": "unassessed"}
    payload = {
        "status": "executed", "bundle": child_id, "initial": "initial.pt", "engine": "engine",
        "experiment": asdict(experiment), "configs": [asdict(c) for c in configs],
        "programs": [asdict(p) for p in programs], "executions": [asdict(e) for e in executions],
        "result_directories": ["source", "followup"], "inspection": asdict(inspection) if inspection else None,
        "comparison": comparison, "rule": saved["rule"], "tolerance_policy": saved.get("tolerance_policy"),
        "environment_changes": environment_changes, "allow_environment_change": allow_environment_change,
        "modifications": {"source_program": str(source_program) if source_program else None,
                          "followup_program": str(followup_program) if followup_program else None,
                          "size": size, "updates": updates, "iteration": iteration},
        "provider_calls": 0, "files": inventory(bundle),
    }
    path = store.write(child_id, "experiment", payload, parent_id=original["artifact_id"])
    return {"artifact": str(path), "verdict": inspection.verdict.value if inspection and inspection.verdict else None,
            "outcomes": [p.outcome for p in processes], "environment_changes": environment_changes,
            "execution_seconds": sum(p.elapsed_seconds for p in processes), "provider_calls": 0}
