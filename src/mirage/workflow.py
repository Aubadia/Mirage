"""Canonical reuse, bounded LLM plans, and evidence-backed exploration batches."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
import uuid

from mirage.artifacts.bundle import inventory, load_tensor_record, snapshot_engine, verify
from mirage.artifacts.store import ArtifactStore, json_value
from mirage.contracts.exploration import targets, source_config, validate_history, history_request
from mirage.generator.program import analyze, request, ContractError
from mirage.generator.provider import IsolatedProvider
from mirage.generator.repair import generate, Diagnostic, DiagnosticKind
from mirage.rules.conditional import inspect, RULE_VERSION
from mirage.runtime.process import run_worker
from mirage.sampler.policy import Sampler


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def base_config(api, options, seed):
    return {"api": api, "seed": seed, "device": options.device, "backend": options.backend,
            "precision": options.precision, "representation": "contiguous", "input_name": "",
            "gradient_state": "none", "optimizer_steps": 0, "graph_mode": "fresh",
            "checkpoint": False, "saved_tensors": False, "microbatches": 1,
            "accumulation_source": False, "requires_backward": not options.forward_only}


def execute(source, config, initial, directory, budget, engine=None, cancel=None):
    process = run_worker({"mode": "execute_program", "source": source, "config": config,
                          "initial": str(Path(initial).resolve()),
                          "engine": str(engine) if engine else None}, directory, budget, cancel=cancel)
    path = directory / "result.pt"
    record = load_tensor_record(path) if process.outcome == "completed" and path.exists() else None
    return process, record


def run(options, apis):
    if options.strategy == "properties":
        return run_properties(options, apis)
    root = Path(options.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    batch_id = f"batch-{uuid.uuid4().hex[:16]}"
    started = time.monotonic()
    deadline = started + options.time_limit
    cancel = threading.Event()
    gate = threading.BoundedSemaphore(options.generation_concurrency)
    resources = []
    generations = []
    cases = []
    canonical_results = []
    lock = threading.Lock()
    api_locks = {api: threading.Lock() for api in apis}
    metadata_cache = {}
    target_attempts = Counter()
    stop_reason = "completed"
    reserved_cases = 0
    old_signals = {}
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            old_signals[number] = signal.signal(number, lambda *_: cancel.set())

    def budget():
        return max(0, min(options.worker_timeout, deadline - time.monotonic()))

    class GatedProvider:
        def __init__(self, provider):
            self.provider = provider

        def generate(self, prompt, *, deadline):
            while not cancel.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Generation budget exhausted")
                if gate.acquire(timeout=min(0.2, remaining)):
                    break
            else:
                raise TimeoutError("Batch cancelled")
            try:
                response = self.provider.generate(prompt, deadline=deadline)
                return response
            except Exception as error:
                if getattr(error, "cancellation_uncertain", False):
                    cancel.set()
                    session = os.environ.get("MIRAGE_SESSION_DIR")
                    if session:
                        (Path(session) / "cancellation-unconfirmed").write_text(str(error))
                raise
            finally:
                gate.release()

    def canonical(api, index):
        nonlocal reserved_cases
        identifier = f"{batch_id}-{apis.index(api):03d}-{index:02d}"
        directory = root / identifier
        directory.mkdir()
        snapshot_engine(directory / "engine")
        config = base_config(api, options, options.seed + 1009 * apis.index(api) + index)
        processes = []
        isolated = IsolatedProvider(options.provider, directory, options.worker_timeout, cancel=cancel)
        provider = GatedProvider(isolated)
        entry = {"id": identifier, "api": api, "status": "unresolved", "cases": []}
        try:
            with api_locks[api]:
                if api not in metadata_cache:
                    metadata_process = run_worker({"mode": "api_metadata", "api": api}, directory / "api", budget(), cancel=cancel)
                    processes.append(metadata_process)
                    if metadata_process.outcome != "completed":
                        entry.update(status="api_unavailable", error=metadata_process.error)
                        return entry
                    metadata_cache[api] = json.loads((directory / "api/summary.json").read_text())
                metadata = metadata_cache[api]
            write_json(directory / "api-context.json", metadata)
            selected = {}
            attempt_number = 0

            def validate(content):
                nonlocal attempt_number
                attempt_number += 1
                attempt_dir = directory / f"canonical-{attempt_number:02d}"
                attempt_dir.mkdir()
                try:
                    value = json.loads(content)
                    if not isinstance(value, dict) or set(value) != {"source"}:
                        raise ContractError("Return JSON with exactly source")
                    graph = analyze(value["source"])
                except SyntaxError as error:
                    return Diagnostic(DiagnosticKind.SYNTAX,
                                      f"Python syntax error on line {error.lineno}: {error.msg}; "
                                      f"line text: {error.text!r}. Remove unmatched braces from the Python source; "
                                      "JSON closing braces must not be included inside the source string.")
                except (ValueError, TypeError) as error:
                    return Diagnostic(DiagnosticKind.CONTRACT, str(error))
                source = value["source"]
                (attempt_dir / "program.py").write_text(source, encoding="utf-8")
                init = run_worker({"mode": "initialize_program", "source": source, "config": config},
                                  attempt_dir / "initialize", budget(), cancel=cancel)
                processes.append(init)
                if init.outcome != "completed":
                    kind = DiagnosticKind.CONTRACT if (init.error or "").startswith("ContractError:") else DiagnosticKind.UNKNOWN_EXCEPTION
                    return Diagnostic(kind, init.error or init.outcome)
                initial = attempt_dir / "initialize/initial.pt"
                execution, record = execute(source, config, initial, attempt_dir / "validation",
                                            budget(), directory / "engine", cancel)
                processes.append(execution)
                if record is None:
                    kind = DiagnosticKind.CONTRACT if (execution.error or "").startswith("ContractError:") else DiagnosticKind.UNKNOWN_EXCEPTION
                    return Diagnostic(kind, execution.error or execution.outcome)
                if not record["api_executed"] or config["requires_backward"] and not record["backward_executed"]:
                    return Diagnostic(DiagnosticKind.UNKNOWN_EXCEPTION, "Target API/backward execution not verified")
                selected.update(source=source, initial=initial, graph=graph,
                                metadata=json.loads((attempt_dir / "initialize/summary.json").read_text()),
                                record=record)
                return Diagnostic(DiagnosticKind.ACCEPTED)

            saved_attempts = []

            def save_attempt(attempt):
                saved_attempts.append(attempt)
                write_json(directory / f"attempt-{len(saved_attempts):02d}.json", asdict(attempt))

            generation = generate(provider, request(metadata, config["seed"], options.device, options.size,
                                                    config["requires_backward"]),
                                  validate, deadline=deadline, record=save_attempt)
            write_json(directory / "generation.json", asdict(generation))
            with lock:
                generations.append(generation)
            if generation.status != "accepted":
                entry["status"] = f"generation_{generation.status}"
                return entry
            entry.update(status="validated", graph=selected["graph"],
                         canonical_api_verified=True,
                         canonical_backward_verified=selected["record"]["backward_executed"])
            consumed = set().union(*(set(call.get("input_names", []))
                                     for call in selected["record"]["target_calls"]))
            operand_metadata = {name: value for name, value in selected["metadata"]["metadata"].items()
                                if name.removeprefix(".0.") in consumed}
            choices = targets(config, operand_metadata, None)
            if not selected["graph"]["sample_loss"] or not selected["metadata"]["batch_axes"]:
                choices = [(rule, target) for rule, target in choices if rule != "accumulation"]
            entry["declared_targets"] = len(choices)
            remaining_targets = list(range(len(choices)))
            for target_index in range(min(options.targets_per_canonical, len(choices))):
                keys = {i: json.dumps({k: v for k, v in choices[i][1].items() if k != "seed"}, sort_keys=True)
                        for i in remaining_targets}
                with lock:
                    sampler = Sampler(tuple(remaining_targets), seed=config["seed"] + target_index, epsilon=options.epsilon)
                    chosen = sampler.sample({i: target_attempts[key] for i, key in keys.items()})
                    target_attempts[keys[chosen]] += 1
                remaining_targets.remove(chosen)
                rule, followup = choices[chosen]
                if (rule in ("layout", "storage", "checkpoint", "saved_tensor")
                        and not selected["record"]["pure"]
                        or rule in ("layout", "storage", "saved_tensor") and selected["record"]["stochastic"]):
                    # Keep the legal state trajectory when this oracle is unavailable.
                    rule = None
                for path_index in range(options.paths_per_target):
                    if cancel.is_set() or budget() <= 0:
                        return entry
                    with lock:
                        if reserved_cases >= options.cases:
                            return entry
                        reserved_cases += 1
                    case_id = f"{identifier}-t{target_index:02d}-p{path_index}"
                    bundle = root / case_id
                    bundle.mkdir()
                    plan = {}

                    def validate_plan(content):
                        try:
                            value = json.loads(content)
                            if not isinstance(value, dict) or set(value) != {"actions"}:
                                raise ContractError("Return exactly actions")
                            plan["history"] = list(validate_history(value["actions"], followup))
                        except (ValueError, TypeError) as error:
                            return Diagnostic(DiagnosticKind.CONTRACT, str(error))
                        return Diagnostic(DiagnosticKind.ACCEPTED)

                    planned = generate(provider, history_request(followup, selected["source"], path_index),
                                       validate_plan, deadline=deadline)
                    write_json(bundle / "generation.json", asdict(planned))
                    with lock:
                        generations.append(planned)
                    if planned.status != "accepted":
                        item = {"id": case_id, "api": api, "status": f"plan_{planned.status}"}
                        entry["cases"].append(item)
                        with lock:
                            cases.append(item)
                        continue
                    followup = dict(followup, **plan)
                    source = source_config(rule, followup)
                    shutil.copyfile(selected["initial"], bundle / "initial.pt")
                    (bundle / "program.py").write_text(selected["source"], encoding="utf-8")
                    shutil.copytree(directory / "engine", bundle / "engine")
                    write_json(bundle / "api.json", metadata)
                    configs = [source, followup] if rule else [followup]
                    records = []
                    results = []
                    for role, case_config in zip(("source", "followup") if rule else ("followup",), configs):
                        result, record = execute(selected["source"], case_config, bundle / "initial.pt",
                                                 bundle / role, budget(), bundle / "engine", cancel)
                        processes.append(result)
                        results.append(result)
                        records.append(record)
                    inspection = inspect(rule, *(records if rule else [None, records[0]]))
                    verified = all(r and r["api_executed"] and (not config["requires_backward"] or r["backward_executed"])
                                   and r["target_state_reached"] for r in records)
                    payload = {"generation_contract_version": 3, "bundle": case_id,
                               "status": "executed", "api": api, "canonical": identifier,
                               "rule": rule, "rule_version": RULE_VERSION, "configs": configs,
                               "results": [asdict(p) for p in results], "inspection": inspection,
                               "verified": verified, "files": inventory(bundle)}
                    manifest = ArtifactStore(root).write(case_id, "experiment", payload)
                    item = {"id": case_id, "api": api, "status": "executed", "rule": rule,
                            "verified": verified, "verdict": inspection.get("verdict"),
                            "outcomes": [result.outcome for result in results],
                            "state": {k: v for k, v in followup.items() if k not in ("seed", "history")},
                            "path": plan["history"], "manifest": str(manifest)}
                    item["transition"] = records[-1].get("transition", []) if records[-1] else []
                    if inspection.get("verdict") == "suspected_failure":
                        item["reproductions"] = []
                        for _ in range(options.replay_failures):
                            if budget() <= 0 or cancel.is_set():
                                break
                            reproduction = replay(manifest, timeout=budget(), deadline=deadline, cancel=cancel)
                            with lock:
                                resources.extend(reproduction["resources"])
                            item["reproductions"].append(reproduction)
                    entry["cases"].append(item)
                    with lock:
                        cases.append(item)
            return entry
        except Exception as error:
            entry.update(status="unresolved", error=f"{type(error).__name__}: {error}")
            return entry
        finally:
            write_json(directory / "canonical.json", entry)
            with lock:
                resources.extend(asdict(p) for p in processes + isolated.processes)

    try:
        tasks = [(api, index) for index in range(options.canonicals_per_api) for api in apis]
        with ThreadPoolExecutor(max_workers=options.workers) as executor:
            # Limit submitted tasks too; deadlines must not leave a large queued batch.
            futures = {}
            iterator = iter(tasks)
            for _ in range(options.workers):
                item = next(iterator, None)
                if item:
                    futures[executor.submit(canonical, *item)] = item
            while futures:
                done = next(as_completed(futures))
                futures.pop(done)
                canonical_results.append(done.result())
                item = next(iterator, None)
                if item and not cancel.is_set() and budget() > 0:
                    futures[executor.submit(canonical, *item)] = item
    finally:
        for number, handler in old_signals.items():
            signal.signal(number, handler)
    if cancel.is_set():
        stop_reason = "interrupted_or_provider_cancellation_unconfirmed"
    elif time.monotonic() >= deadline:
        stop_reason = "time_limit"
    attempts = [attempt for generation in generations for attempt in generation.attempts]
    verified_cases = [case for case in cases if case.get("verified")]
    report = {
        "batch_id": batch_id, "generation_contract_version": 3, "stop_reason": stop_reason,
        "wall_seconds": time.monotonic() - started, "options": asdict(options),
        "canonicals": canonical_results, "experiments": cases,
        "case_statuses": dict(Counter(case["status"] for case in cases)),
        "verdicts": dict(Counter(case.get("verdict") or "unassessed" for case in cases if case["status"] == "executed")),
        "runtime_outcomes": dict(Counter(outcome for case in cases for outcome in case.get("outcomes", []))),
        "generation": {"calls": len(attempts), "repairs": sum(max(0, len(g.attempts) - 1) for g in generations),
                       "input_tokens": sum(a.response.usage.input_tokens or 0 for a in attempts if a.response),
                       "output_tokens": sum(a.response.usage.output_tokens or 0 for a in attempts if a.response),
                       "unknown_usage": sum(a.response is None or a.response.usage.input_tokens is None
                                            or a.response.usage.output_tokens is None for a in attempts)},
        "coverage": {"api": {"covered": len({c["api"] for c in canonical_results if c.get("canonical_api_verified")}),
                             "total": len(apis)},
                     "verified_cases": len(verified_cases),
                     "verified_coverage_only": sum(case["rule"] is None for case in verified_cases),
                     "unique_verified_states": len({json.dumps(c["state"], sort_keys=True) for c in verified_cases}),
                     "unique_verified_paths": len({tuple(c["path"]) for c in verified_cases}),
                     "unique_verified_transitions": len({tuple(c["transition"]) for c in verified_cases}),
                     "declared_program_targets": sum(c.get("declared_targets", 0) for c in canonical_results),
                     "api_mr": {"applicable": len({(c["api"], c["rule"]) for c in cases
                                                     if c.get("verdict") in ("pass", "suspected_failure")}),
                                "declared": len(apis) * 6},
                     "candidate_basis": "bounded declared targets; feasibility is established only by execution"},
        "resources": resources,
    }
    ArtifactStore(root).write(f"{batch_id}-report", "report", report)
    return report


def replay(manifest_path, *, timeout=600, allow_environment_change=False, deadline=None, cancel=None):
    path = Path(manifest_path).resolve()
    raw=json.loads(path.read_text())
    if raw.get('mode')=='execute_property':
        return replay_property(path,timeout=timeout,allow_environment_change=allow_environment_change,cancel=cancel)
    original = ArtifactStore(path.parent).read(path.stem)
    payload = original["payload"]
    if payload.get("generation_contract_version") != 3:
        raise ValueError("Legacy reproduction is no longer supported")
    source = path.parent / payload["bundle"]
    verify(source, payload["files"])
    identifier = f"replay-{uuid.uuid4().hex[:16]}"
    directory = path.parent / identifier
    directory.mkdir()
    shutil.copytree(source / "engine", directory / "engine")
    shutil.copyfile(source / "initial.pt", directory / "initial.pt")
    code = (source / "program.py").read_text()
    results, records = [], []
    deadline = deadline if deadline is not None else time.monotonic() + 3 * timeout
    for index, config in enumerate(payload["configs"]):
        process, record = execute(code, config, directory / "initial.pt", directory / f"execution-{index}",
                                  max(0, min(timeout, deadline - time.monotonic())), directory / "engine", cancel)
        results.append(asdict(process))
        records.append(record)
    record_paths = [str(directory / f"execution-{index}" / "result.pt") if record else None
                    for index, record in enumerate(records)]
    inspection_process = run_worker({"mode": "inspect_pair", "rule": payload["rule"],
                                     "records": record_paths if len(records) == 2 else [None, record_paths[0]],
                                     "engine": str(directory / "engine")}, directory / "inspection",
                                    max(0, min(timeout, deadline - time.monotonic())), cancel=cancel)
    if inspection_process.outcome == "completed":
        inspection = json.loads((directory / "inspection/summary.json").read_text())
    else:
        inspection = {"verdict": None, "reason": inspection_process.error or inspection_process.outcome}
    environment_changes = []
    for index, (record, role) in enumerate(zip(records, ("source", "followup") if payload["rule"] else ("followup",))):
        previous_path = source / role / "result.pt"
        if record and previous_path.exists():
            previous = load_tensor_record(previous_path)
            if previous["environment"] != record["environment"]:
                environment_changes.append(index)
    if environment_changes and not allow_environment_change:
        inspection = {"verdict": None, "reason": "Environment changed; cross-environment comparison was not requested"}
    replay_resources = [*results, asdict(inspection_process)]
    result = {"generation_contract_version": 3, "provider_calls": 0, "results": results,
              "resources": replay_resources,
              "inspection": inspection, "environment_changes": environment_changes,
              "bundle": identifier, "files": inventory(directory)}
    manifest = ArtifactStore(path.parent).write(identifier, "execution", result, parent_id=original["artifact_id"])
    return {"artifact": str(manifest), "verdict": inspection.get("verdict"), "provider_calls": 0,
            "environment_changes": environment_changes, "resources": replay_resources}


def run_properties(options, apis):
    """Blind property generation through the same isolated worker as regression fixtures."""
    from mirage.generator.program import analyze_property, property_request
    from mirage.rules.properties import families
    from mirage.contracts.exploration import property_configuration
    from mirage.generator.provider import ProviderFailure
    root=Path(options.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    snapshot_engine(root/'engine')
    write_json(root/'engine-sha256.json',inventory(root/'engine'))
    started=time.monotonic(); deadline=started+options.time_limit
    results=[]; provider=IsolatedProvider(options.provider,root,options.worker_timeout)
    stop='time_limit'
    # API metadata is measured separately by the outer experiment protocol when supplied.
    metadata={}
    for api in apis:
        directory=root/f'api-{len(metadata)}'
        process=run_worker({'mode':'api_metadata','api':api,'engine':str(root/'engine')},directory,
                           max(.01,min(options.worker_timeout,deadline-time.monotonic())))
        if process.outcome!='completed':
            write_json(root/'summary.json',{'stop_reason':'metadata_failed','records':results,'error':process.error})
            return {'stop_reason':'metadata_failed','records':results}
        metadata[api]=json.loads((directory/'summary.json').read_text())
    metadata_seconds=time.monotonic()-started
    started=time.monotonic(); deadline=started+options.time_limit
    pending=None; repair_count=0
    for index in range(options.cases):
        remaining=deadline-time.monotonic()
        # Leave a short window for persisting evidence; do not dispatch requests at expiry.
        if remaining<45: break
        api=apis[index%len(apis)]; choices=families(api)
        if not choices:
            results.append(dict(api=api,status='unsupported',attempt=index)); break
        config=property_configuration(api,options.device,index,choices[index%len(choices)])
        request=property_request(metadata[api],config)
        if pending is not None:
            request=pending; config=dict(request.context['configuration']); pending=None
        attempt=root/f'attempt-{index:04d}'; attempt.mkdir()
        write_json(attempt/'request.json',asdict(request))
        row=dict(api=api,attempt=index,configuration=config,status='generation_failed',
                 generated=False,contract_valid=False,executable=False,api_executed=False,
                 verdict=None,directory=str(attempt),started_seconds=time.monotonic()-started)
        try:
            response=provider.generate(request,deadline=deadline)
            write_json(attempt/'response.json',asdict(response))
            value=json.loads(response.content)
            source=value['source']; row['generated']=True
            (attempt/'program.py').write_text(source)
            row['fingerprint']=analyze_property(source)['fingerprint']; row['contract_valid']=True
            remaining=deadline-time.monotonic()
            if remaining<=0:
                row['status']='budget_before_execution'
            else:
                process=run_worker({'mode':'execute_property','source':source,'config':config,
                                    'engine':str(root/'engine')},attempt/'execution',min(options.worker_timeout,remaining))
                row['process']=asdict(process)
                write_json(attempt/'execution.sha256.json',inventory(attempt/'execution'))
                path=attempt/'execution/summary.json'
                if process.outcome=='completed' and path.exists():
                    outcome=json.loads(path.read_text()); row.update(status='executed',executable=True,
                        api_executed=outcome.get('api_executed',False),verdict=outcome.get('verdict'),
                        reason=outcome.get('reason'),predicates=outcome.get('predicates',{}))
                else: row.update(status='execution_failed',error=process.error)
        except ProviderFailure as error:
            row.update(status='provider_failed',error=str(error))
            if error.cancellation_uncertain:
                session=os.environ.get('MIRAGE_SESSION_DIR')
                if session: (Path(session)/'cancellation-unconfirmed').write_text(str(error))
                stop='provider_cancellation_unconfirmed'
        except Exception as error:
            row.update(status='invalid_program' if row['generated'] else 'generation_failed',
                       error=f'{type(error).__name__}: {error}')
        if row['status']=='invalid_program' and repair_count<2:
            from dataclasses import replace
            pending=replace(request,previous_content=response.content,diagnostic=row['error'])
            repair_count+=1
        else:
            repair_count=0
        row['repair_index']=repair_count
        row['finished_seconds']=time.monotonic()-started
        write_json(attempt/'outcome.json',row); results.append(row)
        write_json(root/'summary.json',dict(stop_reason=stop,records=results,elapsed_seconds=time.monotonic()-started,
                                          provider_calls=provider.calls))
        if stop=='provider_cancellation_unconfirmed': break
    if len(results)>=options.cases and stop=='time_limit': stop='case_limit'
    output=dict(stop_reason=stop,records=results,elapsed_seconds=time.monotonic()-started,provider_calls=provider.calls,metadata_seconds=metadata_seconds,dispatch_reserve_seconds=45)
    write_json(root/'summary.json',output)
    return output


def replay_property(path, *, timeout, allow_environment_change=False, cancel=None):
    """Replay saved property inputs and a content-verified engine, without a provider."""
    request=json.loads(path.read_text())
    evidence=path.parent
    checksums=evidence.parent/(evidence.name+'.sha256.json')
    verify(evidence,json.loads(checksums.read_text()))
    engine=Path(request['engine'])
    verify(engine,json.loads((engine.parent/'engine-sha256.json').read_text()))
    request['initial']=str(evidence/'initial.pt')
    directory=evidence.parent/('replay-'+uuid.uuid4().hex[:16])
    process=run_worker(request,directory,timeout,cancel=cancel)
    original=load_tensor_record(evidence/'result.pt')
    result=load_tensor_record(directory/'result.pt') if process.outcome=='completed' else None
    changes=result is not None and result.get('environment')!=original.get('environment')
    verdict=result.get('verdict') if result else None
    if changes and not allow_environment_change: verdict=None
    report=dict(verdict=verdict,provider_calls=0,environment_changed=changes,process=asdict(process),
                reason='Environment changed' if changes else process.error)
    write_json(directory/'replay-summary.json',report)
    return report
