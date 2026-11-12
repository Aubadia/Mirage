"""Batch orchestration; generation, execution and correctness remain separate."""

from dataclasses import asdict, replace
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time
import signal
import threading
import uuid

from mirage.artifacts.bundle import inventory, load_tensor_record, snapshot_engine
from mirage.artifacts.store import ArtifactStore, json_value
from mirage.contracts.models import Candidate, Experiment
from mirage.contracts.programs import CaseConfig
from mirage.generator.pipeline import generate_stage
from mirage.generator.provider import IsolatedProvider, Stage
from mirage.generator.repair import Diagnostic, DiagnosticKind
from mirage.inspector.oracle import inspect
from mirage.inspector.validation import assess
from mirage.rules.registry import RULES
from mirage.rules.tolerances import policy
from mirage.runtime.process import cancellation_scope, run_worker
from mirage.sampler.candidates import candidates, pair_configs
from mirage.scheduler import LimitedProvider, Scheduler


def require_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        raise ValueError("PyTorch is unavailable in this Python environment; select the prepared PyTorch environment")


def execute_case(program, config, initial, directory, timeout, engine=None, *, cancel=None):
    result = run_worker({"mode": "execute", "config": asdict(config), "program": asdict(program),
                         "initial": str(initial.resolve()), "engine": str(engine) if engine else None}, directory, timeout, cancel=cancel)
    record_path = directory / "result.pt"
    record = load_tensor_record(record_path) if result.outcome == "completed" and record_path.exists() else None
    if record is None and (directory / "trace.jsonl").exists():
        events = []
        for line in (directory / "trace.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break
        record = {"events": events, "snapshots": []}
    return result, record


def _run_experiment(options, root, batch_id, index, api_draw, scheduler):
    store = ArtifactStore(root)
    deadline = scheduler.deadline
    stages = []
    resource_records = []
    api_observations = []
    executions = ()
    inspection = None
    execution_seconds = 0.0
    generation_cost = {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
                       "unknown_usage_attempts": 0, "attempts": 0}

    def remaining():
        if scheduler.cancel.is_set():
            raise TimeoutError("Batch cancelled")
        budget = min(options.worker_timeout, deadline - time.monotonic())
        if budget <= 0:
            raise TimeoutError("Batch wall-clock budget exhausted")
        return budget

    experiment_id = f"{batch_id}-{index:04d}"
    bundle = root / experiment_id
    bundle.mkdir()
    snapshot_engine(bundle / "engine")
    # The first draw chooses only the API. State/MR selection follows canonical validation.
    base = CaseConfig(api_draw.api, (options.seed + index) % (2 ** 63), options.device,
                      options.backend, options.precision, options.size, options.updates,
                      options.warmup_steps)
    experiment = None
    payload = {"batch_id": batch_id, "index": index, "api": base.api,
               "provider": options.provider, "status": "unresolved", "generation": [],
               "bundle": experiment_id, "executions": [], "inspection": None,
               "canonical_config": asdict(base), "initial": "initialize/initial.pt"}
    payload["engine"] = "engine"
    isolated = IsolatedProvider(options.provider, bundle, options.worker_timeout, cancel=scheduler.cancel)
    provider = LimitedProvider(isolated, scheduler)
    validation_number = 0

    def validate_runtime(program, config, stage_directory):
        nonlocal validation_number, execution_seconds
        validation_number += 1
        process, record = execute_case(program, config, bundle / payload["initial"],
                                       stage_directory / f"validation-{validation_number:03d}", remaining(), bundle / "engine", cancel=scheduler.cancel)
        execution_seconds += process.elapsed_seconds
        resource_records.append({"stage": "validation", **asdict(process)})
        intended = Experiment("validation", Candidate(config.api, config.target.key, config.target.checkpoint,
                                                      ">".join(config.actions), None, config.backend, config.precision), config.seed)
        execution = assess(intended, "validation", "followup", config, process, record)
        api_observations.append((config.api, execution))
        if process.outcome != "completed" or record is None:
            return Diagnostic(DiagnosticKind.UNKNOWN_EXCEPTION, process.error or process.outcome), (process, record)
        evidence = execution.evidence
        if evidence.api_executed is not True or evidence.backward_executed is not True:
            return Diagnostic(DiagnosticKind.MISSING_PATH, "Required API/backward evidence is missing"), (process, record)
        if evidence.actions_completed is not True:
            return Diagnostic(DiagnosticKind.UNRESOLVED_PLAN, "Transition actions were not verified in order"), (process, record)
        if evidence.target_state_reached is not True:
            return Diagnostic(DiagnosticKind.STATE_NOT_REACHED, "Runtime did not attain the requested predicate"), (process, record)
        return Diagnostic(DiagnosticKind.ACCEPTED), (process, record)

    def stage(kind, config, previous=None, mr=None):
        program, generation, validation = generate_stage(
            provider, kind, config, bundle / kind.value, deadline, validate_runtime, previous, mr)
        stages.append(generation)
        payload["generation"].append({"stage": kind.value, "status": generation.status})
        if program is None:
            payload["status"] = f"generation_{generation.status}"
            raise GenerationStopped()
        return program, validation

    try:
        initialized = run_worker({"mode": "initialize", "config": asdict(base), "engine": str(bundle / "engine")},
                                 bundle / "initialize", remaining(), cancel=scheduler.cancel)
        execution_seconds += initialized.elapsed_seconds
        resource_records.append({"stage": "initialize", **asdict(initialized)})
        if initialized.outcome != "completed":
            payload.update(status="initialization_failed", error=initialized.error, outcome=initialized.outcome)
            raise GenerationStopped()
        canonical, _ = stage(Stage.CANONICAL, base)
        skeleton, _ = stage(Stage.SKELETON, base, canonical)
        selected, verified_counts = scheduler.select(base.api, index)
        experiment = Experiment(experiment_id, selected, base.seed)
        configs = pair_configs(selected, base)
        payload.update(experiment=asdict(experiment), selection={
            "api_draw": asdict(api_draw), "candidate": asdict(selected),
            "verified_counts": verified_counts,
            "seed": options.seed, "index": index,
        })
        transition, _ = stage(Stage.TRANSITION, configs[1], skeleton, selected.mr)
        followup, followup_result = stage(Stage.FOLLOWUP, configs[1], transition, selected.mr)
        source = replace(canonical, actions=configs[0].actions)
        source_result = execute_case(source, configs[0], bundle / payload["initial"], bundle / "source", remaining(), bundle / "engine", cancel=scheduler.cancel)
        execution_seconds += source_result[0].elapsed_seconds
        resource_records.append({"stage": "source", **asdict(source_result[0])})
        processes, records = tuple(zip(source_result, followup_result))
        executions = tuple(assess(experiment, f"{experiment_id}-{role}", role, config, process, record)
                           for role, config, process, record in zip(("source", "followup"), configs, processes, records))
        inspection, comparison = inspect(experiment, executions, configs, records)
        payload.update(status="executed", programs=[asdict(source), asdict(followup)],
                       configs=[asdict(c) for c in configs], executions=[asdict(e) for e in executions],
                       result_directories=[str(Path(p.directory).relative_to(bundle)) for p in processes],
                       inspection=asdict(inspection) if inspection else None, comparison=comparison,
                       rule=RULES[selected.mr].as_dict() if selected.mr else None,
                       tolerance_policy=policy(selected.api, selected.precision, selected.backend, selected.mr) if selected.mr else None)
    except GenerationStopped:
        pass
    except TimeoutError as error:
        payload.update(status="budget_exhausted", error=str(error))
    except KeyboardInterrupt:
        payload.update(status="interrupted")
        scheduler.stop("interrupted")
    except Exception as error:
        payload.update(status="controller_error", error=f"{type(error).__name__}: {error}")
        scheduler.stop("controller_error")
    finally:
        resource_records.extend({"stage": "provider", **asdict(p)} for p in isolated.processes)
        for generation in stages:
            for attempt in generation.attempts:
                generation_cost["attempts"] += 1
                generation_cost["seconds"] += attempt.elapsed_seconds
                usage = attempt.response.usage if attempt.response else None
                if usage is None or usage.input_tokens is None or usage.output_tokens is None:
                    generation_cost["unknown_usage_attempts"] += 1
                for name in ("input_tokens", "output_tokens"):
                    value = getattr(usage, name, None)
                    if value is not None:
                        generation_cost[name] += value
        payload["files"] = inventory(bundle)
        path = store.write(experiment_id, "experiment", payload)
        summary = {"artifact": str(path), "status": payload["status"],
                   "verdict": payload["inspection"]["verdict"] if payload["inspection"] else None}
    return {"index": index, "summary": summary, "experiment": experiment,
            "executions": executions, "inspection": inspection,
            "api_observations": api_observations, "resources": resource_records,
            "generation": generation_cost, "execution_seconds": execution_seconds}


def run(options, apis):
    require_torch()
    root = Path(options.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(root)
    batch_id = f"batch-{uuid.uuid4().hex[:16]}"
    universe = candidates(apis, options.backend, options.precision, options.updates)
    started = time.monotonic()
    scheduler = Scheduler(universe, seed=options.seed, epsilon=options.epsilon,
                          workers=options.workers, generation_concurrency=options.generation_concurrency,
                          deadline=started + options.time_limit)
    summaries = []
    resources = []
    reproduction_queue = []
    execution_seconds = 0.0
    generation_cost = {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
                       "unknown_usage_attempts": 0, "attempts": 0}

    def report(status):
        coverage = scheduler.coverage.report()
        generation = dict(generation_cost)
        if generation["unknown_usage_attempts"]:
            generation.update(input_tokens=None, output_tokens=None)
        coverage["cost"] = {
            "execution_seconds": execution_seconds, "generation": generation,
            "wall_seconds": time.monotonic() - started,
            "generation_time_includes_validation": True,
            "provider_seconds": sum(p["elapsed_seconds"] for p in resources if p["stage"] == "provider"),
            "resources": resources,
        }
        return {"batch_id": batch_id, "coverage": coverage, "candidates": universe,
                "options": options, "experiments": summaries, "stop_reason": status,
                "attempted_experiments": len(summaries), "schedule": list(scheduler.events),
                "generation_outcomes": dict(Counter(item["status"] for item in summaries))}

    def work(index, api_draw):
        return _run_experiment(options, root, batch_id, index, api_draw, scheduler)

    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: scheduler.stop("interrupted"))
    try:
        for completed in scheduler.execute(options.cases, work):
            if completed is None:
                # No other experiment can use the test GPU during reproduction.
                from mirage.artifacts.replay import replay

                for summary in reproduction_queue:
                    summary["reproductions"] = []
                    for _ in range(options.replay_failures):
                        if scheduler.cancel.is_set() or time.monotonic() >= scheduler.deadline:
                            break
                        with cancellation_scope(scheduler.cancel):
                            reproduction = replay(Path(summary["artifact"]), timeout=options.worker_timeout,
                                                  deadline=scheduler.deadline)
                        execution_seconds += reproduction["execution_seconds"]
                        summary["reproductions"].append(reproduction)
                reproduction_queue.clear()
                continue
            scheduler.commit(completed)
            summaries.append(completed["summary"])
            resources.extend(completed["resources"])
            execution_seconds += completed["execution_seconds"]
            for key, value in completed["generation"].items():
                generation_cost[key] += value
            if completed["summary"]["verdict"] == "suspected_failure" and options.replay_failures:
                reproduction_queue.append(completed["summary"])
                scheduler.paused = True
            store.write(f"{batch_id}-progress-{len(summaries):04d}", "report", report("in_progress"))
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if time.monotonic() >= scheduler.deadline and scheduler.stop_reason == "case_limit":
        scheduler.stop_reason = "time_limit"
    result = report(scheduler.stop_reason)
    report_path = store.write(batch_id, "report", result)
    return {"report": str(report_path), **json_value(result)}


class GenerationStopped(Exception):
    """Expected generation or initialization outcome already preserved in artifacts."""
