"""Canonical reuse, bounded LLM plans, and evidence-backed exploration batches."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
import tempfile
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
from mirage.contracts.candidates import CANDIDATE_VERSION, CapabilitySnapshot
from mirage.sampler.candidates import registry, legal_paths, unmet_requirements


_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def _report_json_value(value):
    # Most reports already contain builtin JSON values. Walk each call afresh,
    # copying only containers whose special values actually need conversion.
    kind = type(value)
    if kind in _JSON_SCALAR_TYPES:
        return value
    if kind is dict:
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        converted = None
        for key, item in value.items():
            child = item if type(item) in _JSON_SCALAR_TYPES else _report_json_value(item)
            if child is not item:
                if converted is None:
                    converted = dict(value)
                converted[key] = child
        return value if converted is None else converted
    if kind is list:
        converted = None
        for index, item in enumerate(value):
            child = item if type(item) in _JSON_SCALAR_TYPES else _report_json_value(item)
            if child is not item:
                if converted is None:
                    converted = list(value)
                converted[index] = child
        return value if converted is None else converted
    if kind in (tuple, list):
        return [item if type(item) in _JSON_SCALAR_TYPES else _report_json_value(item)
                for item in value]
    return json_value(value)


def write_json(path, value):
    payload = json.dumps(_report_json_value(value), indent=2, allow_nan=False)
    pending = None
    try:
        # Publish only a complete write; interrupted/full-storage writes keep the old report.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.pending', delete=False) as stream:
            pending = Path(stream.name)
            stream.write(payload)
        pending.replace(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def state_repair_progress(inspection):
    """Compare attained obligations, not incidental numerical changes between edits."""
    checks = inspection.get('predicates', [])
    if not checks or any(type(p.get('attained')) is not bool for p in checks):
        return None
    identities, attained = [], []
    for check in checks:
        identity = json.dumps({k: v for k, v in check.items()
                               if k not in ('observed', 'evidence_status', 'reason', 'attained')},
                              sort_keys=True)
        identities.append(identity)
        if check['attained']:
            attained.append(identity)
    return {'predicates': sorted(identities), 'attained': sorted(attained)}


def state_repair_stalled(previous, current):
    return (previous is not None and current is not None
            and previous['predicates'] == current['predicates']
            and not set(current['attained']) > set(previous['attained']))


def same_repair_payload(previous, current):
    if previous is None:
        return False
    if previous == current:
        return True
    try:
        return json.loads(previous) == json.loads(current)
    except (ValueError, TypeError):
        return False


def stateful_runtime_diagnostic(process):
    """Legacy exception strings do not establish that generated code is illegal."""
    status_path = Path(process.directory) / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    diagnostic = status.get('diagnostic', {})
    confirmed = (process.outcome == 'exception'
                 and diagnostic.get('version') == 'runtime-diagnostic-1'
                 and diagnostic.get('kind') == 'contract_violation'
                 and diagnostic.get('exception_type') == 'ContractError'
                 and diagnostic.get('origin') not in (None, 'generated_or_external')
                 and diagnostic.get('repair_allowed') is True)
    return {'kind': 'contract_violation' if confirmed else 'unknown_exception',
            'repair_allowed': confirmed, 'evidence': diagnostic,
            'reason': process.error or process.outcome, 'correctness': 'unassessed'}


def rerun_stateful_exception(process, destination, timeout):
    """Repeat the saved invocation, without claiming identical initial state or a bug."""
    request_path = Path(process.directory) / 'request.json'
    if timeout <= 0 or not request_path.exists():
        return {'status': 'not_run', 'reason': 'budget or saved request unavailable', 'provider_calls': 0}
    repeated = run_worker(json.loads(request_path.read_text()), destination, timeout)
    result = {'status': 'executed', 'process': asdict(repeated), 'provider_calls': 0,
              'same_exception': repeated.outcome == process.outcome == 'exception'
                                and repeated.error == process.error,
              'initial_state_equivalence': 'unassessed', 'correctness': 'unassessed',
              'scope': 'same source/configuration rerun; exception agreement is not defect confirmation',
              'diagnostic': stateful_runtime_diagnostic(repeated)}
    write_json(destination / 'triage.json', result)
    return result


def mutation_backward_requirements(evidence):
    """Retain callable obligations separately from internal target-node evidence."""
    callable_calls = {w['call'] for w in evidence.get('witnesses', ())
                      if w['checkpoint'] == 'final'
                      and w['relations'].get('callable_backward_node', {}).get('status') == 'verified'}
    return {'required_backward_calls': evidence.get('backward_calls', []),
            'required_callable_backward_calls': sorted(callable_calls)}


def mutation_paths_preserved(mutation, inspection):
    witnesses = inspection.get('semantic_coverage', {}).get('witnesses', ())
    for requirement, relation in (('required_backward_calls', 'target_backward_node'),
                                  ('required_callable_backward_calls', 'callable_backward_node')):
        observed = {w['call'] for w in witnesses if w['checkpoint'] == 'final'
                    and w['relations'].get(relation, {}).get('status') == 'verified'}
        if not set((mutation or {}).get(requirement, ())) <= observed:
            return False
    return True


def optimizer_configuration_reusable(parent, config):
    """Reuse a verified source only when its new obligations are observer actions."""
    from mirage.contracts.capabilities import OPTIMIZER_COVERAGE_VERSION, StatePredicate
    from mirage.generator.state_plan import optimizer_action_for_predicates
    from mirage.inspector.stateful import optimizer_witness_fact
    if (parent.get('optimizer_coverage_version') != OPTIMIZER_COVERAGE_VERSION
            or not parent.get('optimizer_coverage')
            or parent.get('evidence', {}).get('optimizer_target_update_count') != 1):
        return False
    ignored = {'attempt', 'proposal_index', 'predicates', 'optimizer_gradient_action', 'canonical_reference', 'oracle_seen_units'}
    if ({k: v for k, v in parent['configuration'].items() if k not in ignored}
            != {k: v for k, v in config.items() if k not in ignored}):
        return False
    try:
        predicates = [StatePredicate(**p) for p in json.loads(json.dumps(config.get('predicates', [])))]
        action = optimizer_action_for_predicates(config.get('predicates', []))
    except (TypeError, ValueError):
        return False
    gradients = [p for p in predicates if p.dimension == 'gradient_representation']
    companions = [p for p in predicates if p.dimension == 'optimizer' and p.field == 'completed'
                  and p.expected is True]
    if (not gradients or len(gradients) + len(companions) != len(predicates)
            or action is None or action != config.get('optimizer_gradient_action')):
        return False
    if action == parent['configuration'].get('optimizer_gradient_action'):
        return False
    witnesses = parent.get('evidence', {}).get('optimizer_witnesses', [])
    requirements, unmet = [], False
    for gradient in gradients:
        required = StatePredicate('optimizer', 'completed', True,
            gradient.checkpoint.rsplit('/', 1)[0] + '/post_update',
            call_index=gradient.call_index, operand_path=gradient.operand_path,
            relation='target_gradient_update', parameter_key=gradient.parameter_key)
        before = optimizer_witness_fact(witnesses, gradient)
        complete = optimizer_witness_fact(witnesses, required)
        if (required not in companions or not before.established or not complete.established
                or complete.value is not True):
            return False
        requirements.append(required)
        unmet |= before.value != gradient.expected
    return unmet and all(p in requirements for p in companions)


def library_configuration_pool(parent, catalog, cache):
    """Cache accepted and rejected metadata proposals, never runtime attainment."""
    import hashlib
    from mirage.generator.state_plan import library_configuration_plans
    evidence = parent.get('evidence', {})
    key = hashlib.sha256(json.dumps({'catalog': catalog,
        'configuration': parent['configuration'], 'source_sha256': parent.get('source_sha256'),
        'optimizer_coverage_version': parent.get('optimizer_coverage_version'),
        'optimizer_coverage': parent.get('optimizer_coverage'),
        'optimizer_witnesses': evidence.get('optimizer_witnesses'),
        'optimizer_target_update_count': evidence.get('optimizer_target_update_count')},
        sort_keys=True).encode()).hexdigest()
    previous = cache.get(parent['id'])
    if previous and previous['cache_key'] == key:
        return previous
    started = time.monotonic()
    plans, rejected = library_configuration_plans(catalog)
    accepted = []
    for plan in plans:
        config = dict(parent['configuration'], predicates=plan['predicates'],
                      optimizer_gradient_action=plan['optimizer_gradient_action'])
        if optimizer_configuration_reusable(parent, config):
            accepted.append(plan)
        else:
            rejected.append({'goal_ids': plan['goal_ids'], 'status': 'rejected_library_proposal',
                             'reason': 'Parent configuration or verified update chain does not permit reuse',
                             'reachability': 'unassessed'})
    result = {'cache_key': key, 'parent_id': parent['id'], 'source_sha256': parent.get('source_sha256'),
              'accepted': accepted, 'rejected': rejected, 'execution_rejections': [],
              'metadata_planning_seconds': time.monotonic() - started,
              'scope': 'Bounded static proposals; child execution must revalidate every obligation'}
    cache[parent['id']] = result
    return result


def available_library_plans(pool, selected_goal_ids):
    """LLM examined/unresolved IDs are not executions of a known transition."""
    return [plan for plan in pool['accepted'] if not set(plan['goal_ids']) & selected_goal_ids]


def configuration_dispatch_budget(parent, rows, fallback):
    """Estimate a known action from its verified parent's completed execution."""
    import math
    measurements = []
    for row in rows:
        admission = row.get('corpus_admission', {})
        process = row.get('process', {})
        seconds = process.get('elapsed_seconds')
        if (admission.get('admitted') is True and admission.get('seed_id') == parent['id']
                and row.get('configuration') == parent['configuration']
                and row.get('inspection', {}).get('valid') is True
                and row.get('mutation_obligations_valid') is True
                and process.get('outcome') == 'completed' and type(process.get('returncode')) is int
                and process['returncode'] == 0 and type(seconds) in (int, float)
                and math.isfinite(seconds) and seconds > 0):
            measurements.append(seconds)
    seconds = max(measurements) if measurements else None
    return {'reserve_seconds': max(10., 1.5 * seconds + 5.) if seconds is not None else fallback,
            'cost_basis': 'verified_parent_worker' if seconds is not None else 'whole_round_fallback',
            'parent_id': parent['id'], 'source_sha256': parent.get('source_sha256'),
            'measured_worker_seconds': seconds,
            'scope': 'Dispatch estimate only; execution remains bounded by the remaining deadline'}


def affordable_candidates(candidates, remaining, fallback):
    """Apply after bootstrap/fresh quotas; unknown work retains the old reserve."""
    return [candidate for candidate in candidates if remaining >
            candidate.get('dispatch_budget', {}).get('reserve_seconds', fallback)]


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


def check_api(api, *, output=None):
    """Run a name-only check using prepared environment settings and isolated probes."""
    from mirage.contracts.apis import validate_api_name
    from mirage.contracts.programs import RunOptions
    from mirage.artifacts.report import summarize_api_check

    started = time.monotonic()
    root = Path(output) if output is not None else Path(os.environ.get(
        "MIRAGE_CHECK_OUTPUT_ROOT", "result/check")) / ("run-" + uuid.uuid4().hex)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    from fcntl import flock, LOCK_EX
    run_lock = (root/'resume.lock').open('a')
    flock(run_lock, LOCK_EX)
    report = {"version": "api-check-1", "api": api, "directory": str(root),
              "status": "running", "stage": "configuration", "terminal": False,
              "correctness": "unassessed", "processes": [], "metrics": None,
              "implementation_stage": "spec-fragments-1", "differential": {"status": "not_evaluated"}}
    write_json(root / "summary.json", report)

    def finish(status, reason, exit_status):
        report.update(status=status, reason=reason, terminal=True,
                      exit_status=exit_status, elapsed_seconds=time.monotonic() - started)
        write_json(root / "summary.json", report)
        return report

    try:
        validate_api_name(api)
        limit = float(os.environ.get("MIRAGE_CHECK_TIME_LIMIT", "300"))
        worker_timeout = float(os.environ.get("MIRAGE_CHECK_WORKER_TIMEOUT", "60"))
        device = os.environ.get("MIRAGE_CHECK_DEVICE", "auto")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("MIRAGE_CHECK_DEVICE must be auto, cpu or cuda")
        provider = os.environ.get("MIRAGE_PROVIDER", "vllm")
        # Configuration is frozen before any worker starts. Auto begins on CPU;
        # claim-specific device exploration is implemented by later milestones.
        options = RunOptions(output=str(root / "execution"), provider=provider,
            cases=int(os.environ.get("MIRAGE_CHECK_CASES", "4")), time_limit=limit,
            worker_timeout=worker_timeout, seed=int(os.environ.get("MIRAGE_CHECK_SEED", "0")),
            strategy="stateful", contract_version=4, forward_only=True,
            device="cpu" if device == "auto" else device,
            representation_traversal=True, oracle_improvements=True)
        toggle = os.environ.get("MIRAGE_CHECK_DIFFERENTIAL", "1")
        if toggle not in ("0", "1"):
            raise ValueError("MIRAGE_CHECK_DIFFERENTIAL must be 0 or 1")
        report["configuration"] = dict(asdict(options), differential=toggle == "1")
        report["configuration"]["budget_scope"] = "whole-check wall time, including preflight and metadata"
        deadline = started + limit
        if time.monotonic() >= deadline:
            return finish("skipped", "budget_exhausted", 0)
        report["stage"] = "specification"
        write_json(root / "summary.json", report)
        process = run_worker({"mode": "api_spec", "api": api}, root / "specification",
                             min(worker_timeout, deadline - time.monotonic()))
        report["processes"].append(asdict(process))
        if process.outcome != "completed":
            if process.outcome == "timeout" and time.monotonic() >= deadline:
                return finish("skipped", "budget_exhausted", 0)
            return finish("failed", "specification_worker_" + process.outcome, 1)
        spec = json.loads((root / "specification/summary.json").read_text())
        write_json(root / "api-spec.json", spec)
        report["specification"] = "api-spec.json"
        if spec["status"] != "resolved":
            return finish("skipped", "api_unavailable", 0)
        if device == "auto" and api.startswith("torch.cuda."):
            from dataclasses import replace
            options = replace(options, device="cuda")
            report["configuration"]["device"] = "cuda"
        if options.device == "cuda" and not spec["environment"]["cuda_available"]:
            return finish("skipped", "requested_device_unavailable", 0)
        if time.monotonic() >= deadline:
            return finish("skipped", "budget_exhausted", 0)
        report["stage"] = "provider_preflight"
        write_json(root / "summary.json", report)
        process = run_worker({"mode": "provider_preflight", "provider": provider},
            root / "provider-preflight", min(worker_timeout, deadline - time.monotonic()))
        report["processes"].append(asdict(process))
        if process.outcome != "completed":
            if process.outcome == "timeout" and time.monotonic() >= deadline:
                return finish("skipped", "budget_exhausted", 0)
            return finish("failed", "provider_preflight_" + process.outcome, 1)
        if time.monotonic() >= deadline:
            return finish("skipped", "budget_exhausted", 0)
        report["stage"] = "execution"
        write_json(root / "summary.json", report)
        # Reserve a bounded slice for real-device comparison. Both experimental arms
        # use the same generation budget to preserve paired-seed attribution.
        generation_deadline = deadline - min(45., limit / 5) if spec["environment"]["cuda_available"] else deadline
        result = run_stateful(options, (api,), path_policy="mixed", predicates=[],
            representation_traversal=True, oracle_improvements=True, wall_deadline=generation_deadline, api_spec=spec)
        report["execution_summary"] = "execution/summary.json"
        report["metrics"] = summarize_api_check(result)
        stop = result.get("stop_reason")
        report["native_stop_reason"] = stop
        write_json(root/"execution-complete.json", {"stop_reason": stop, "records": len(result.get("records", []))})
        report["stage"] = "differential"
        write_json(root/"summary.json", report)
        from mirage.artifacts.replay import assess_devices
        report["differential"] = assess_devices(root, result, spec, enabled=toggle == "1",
                                                timeout=worker_timeout, deadline=deadline)
        failures = [row for row in result.get("records", [])
                    if row.get("status") == "provider_failed"]
        worker_failures = [row for row in result.get("records", [])
                           if row.get("process", {}).get("outcome") in ("crash", "timeout")]
        if failures or worker_failures or stop in ("metadata_failed", "controller_error", "provider_cancellation_unconfirmed"):
            return finish("failed", "execution_incomplete", 1)
        if report["metrics"]["runtime_valid"] == 0:
            reason = "budget_exhausted" if stop == "time_limit" else "no_verified_execution"
            return finish("skipped", reason, 0)
        if report["differential"]["incomplete"]:
            return finish("partial", "differential_incomplete", 1)
        return finish("completed", "bounded_execution_finished", 0)
    except KeyboardInterrupt:
        if report['stage'] == 'differential':
            report['differential']['incomplete'] = True
        return finish("interrupted", "interrupted", 130)
    except Exception as error:
        import traceback
        (root / "error.txt").write_text(traceback.format_exc())
        report["error"] = f"{type(error).__name__}: {error}"
        reason = "configuration_error" if report["stage"] == "configuration" else "controller_error"
        return finish("failed", reason, 2 if reason == "configuration_error" else 1)
    finally:
        run_lock.close()


def run(options, apis):
    from mirage.runtime.scheduler import current_scope
    if options.active_api_limit > 1 and current_scope() is None:
        from mirage.runtime.multi_api import run_multi_api
        return run_multi_api(options, apis)
    return _run(options, apis)


def _run(options, apis):
    if options.strategy == "stateful":
        return run_stateful(options, apis, representation_traversal=options.representation_traversal,
                            oracle_improvements=options.oracle_improvements)
    if options.strategy == "properties":
        return run_properties(options, apis)
    root = Path(options.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    batch_id = f"batch-{uuid.uuid4().hex[:16]}"
    started = time.monotonic()
    deadline = started + options.time_limit
    from mirage.runtime.scheduler import cancellation_event, current_scope
    scope = current_scope()
    if scope is not None:
        scope.deadline = deadline if scope.deadline is None else min(scope.deadline, deadline)
    cancel = cancellation_event()
    gate = threading.BoundedSemaphore(options.generation_concurrency)
    resources = []
    generations = []
    cases = []
    canonical_results = []
    lock = threading.Lock()
    api_locks = {api: threading.Lock() for api in apis}
    metadata_cache = {}
    target_attempts = Counter()
    unmet_by_api = {}
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

            with lock:
                requirements = tuple(unmet_by_api.get(api, ()))
            generation = generate(provider, request(metadata, config["seed"], options.device, options.size,
                                                    config["requires_backward"], requirements),
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
            snapshot = CapabilitySnapshot.read(selected["record"]["capabilities"])
            candidates = registry(config, snapshot)
            entry["candidate_contract_version"] = CANDIDATE_VERSION
            entry["capabilities"] = snapshot.payload()
            entry["candidates"] = [candidate.payload() for candidate in candidates]
            requirements = unmet_requirements(candidates)
            entry["unmet_capability_requirements"] = requirements
            with lock:
                unmet_by_api[api] = requirements
            admitted = [candidate for candidate in candidates if candidate.status == "admitted"]
            choices = [(candidate.rule, dict(candidate.configuration, candidate_id=candidate.identifier,
                        candidate_prerequisites=list(candidate.prerequisites))) for candidate in admitted]
            entry["declared_targets"] = len(choices)
            remaining_targets = list(range(len(choices)))
            for target_index in range(min(options.targets_per_canonical, len(choices))):
                keys = {i: json.dumps({k: v for k, v in choices[i][1].items()
                                      if k not in ("seed", "candidate_id", "candidate_prerequisites")}, sort_keys=True)
                        for i in remaining_targets}
                with lock:
                    sampler = Sampler(tuple(remaining_targets), seed=config["seed"] + target_index, epsilon=options.epsilon)
                    chosen = sampler.sample({i: target_attempts[key] for i, key in keys.items()})
                    target_attempts[keys[chosen]] += 1
                remaining_targets.remove(chosen)
                rule, followup = choices[chosen]
                candidate_id = followup["candidate_id"]
                offered = legal_paths(followup, backward=snapshot.facts["backward"].established,
                                      optimizer=snapshot.facts["optimizer"].established)
                if not offered:
                    entry.setdefault("unresolved_paths", []).append(candidate_id)
                    continue
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
                            if tuple(plan["history"]) not in offered:
                                raise ContractError("Select exactly one offered complete legal path")
                        except (ValueError, TypeError) as error:
                            return Diagnostic(DiagnosticKind.CONTRACT, str(error))
                        return Diagnostic(DiagnosticKind.ACCEPTED)

                    planned = generate(provider, history_request(followup, selected["source"], path_index, offered),
                                       validate_plan, deadline=deadline)
                    write_json(bundle / "generation.json", asdict(planned))
                    with lock:
                        generations.append(planned)
                    if planned.status != "accepted":
                        item = {"id": case_id, "api": api, "candidate_id": candidate_id,
                                "status": f"plan_{planned.status}"}
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
                               "candidate_contract_version": CANDIDATE_VERSION,
                               "candidate": admitted[chosen].payload(),
                               "status": "executed", "api": api, "canonical": identifier,
                               "rule": rule, "rule_version": RULE_VERSION, "configs": configs,
                               "results": [asdict(p) for p in results], "inspection": inspection,
                               "verified": verified, "files": inventory(bundle)}
                    manifest = ArtifactStore(root).write(case_id, "experiment", payload)
                    item = {"id": case_id, "api": api, "status": "executed", "rule": rule,
                            "candidate_id": candidate_id,
                            "verified": verified, "verdict": inspection.get("verdict"),
                            "outcomes": [result.outcome for result in results],
                            "state": {k: v for k, v in followup.items()
                                      if k not in ("seed", "history", "candidate_id", "candidate_prerequisites")},
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
                    futures[executor.submit(copy_context().run, canonical, *item)] = item
            while futures:
                done = next(as_completed(futures))
                futures.pop(done)
                canonical_results.append(done.result())
                item = next(iterator, None)
                if item and not cancel.is_set() and budget() > 0:
                    futures[executor.submit(copy_context().run, canonical, *item)] = item
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
        "candidate_contract_version": CANDIDATE_VERSION,
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
                     "candidate_basis": "bounded evidence-admitted candidates; attainment requires execution"},
        "resources": resources,
    }
    from mirage.artifacts.report import summarize_training_candidates
    report["candidate_metrics"] = summarize_training_candidates(canonical_results, cases)
    ArtifactStore(root).write(f"{batch_id}-report", "report", report)
    return report


def replay(manifest_path, *, timeout=600, allow_environment_change=False, deadline=None, cancel=None):
    candidate = json.loads(Path(manifest_path).read_text())
    if candidate.get('format') == 'device-differential-1':
        from mirage.artifacts.replay import replay_devices
        result = replay_devices(manifest_path, timeout=timeout, deadline=deadline or time.monotonic()+timeout)
        return dict(result, replay_valid=result['status'] != 'incomplete')
    if candidate.get('format') == 'stateful-1-replay':
        return replay_stateful(Path(manifest_path), timeout=timeout,
                               allow_environment_change=allow_environment_change, cancel=cancel)

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
        write_json(root/'summary.json',dict(strategy='properties',strategy_status='deprecated_reference_only',stop_reason=stop,records=results,elapsed_seconds=time.monotonic()-started,
                                          provider_calls=provider.calls))
        if stop=='provider_cancellation_unconfirmed': break
    if len(results)>=options.cases and stop=='time_limit': stop='case_limit'
    output=dict(strategy='properties',strategy_status='deprecated_reference_only',stop_reason=stop,records=results,elapsed_seconds=time.monotonic()-started,provider_calls=provider.calls,metadata_seconds=metadata_seconds,dispatch_reserve_seconds=45)
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



def run_stateful(options, apis, **kwargs):
    """Run independent per-API model budgets, or the explicit legacy wall budget."""
    from mirage.runtime.scheduler import current_scope
    if options.active_api_limit > 1 and current_scope() is None:
        from mirage.runtime.multi_api import run_multi_api
        return run_multi_api(options, apis, stateful_options=kwargs)
    if not options.generation_pipeline:
        return _run_stateful(options, apis, **kwargs)
    from dataclasses import replace
    from mirage.runtime.pipeline import ExecutionPipeline
    from mirage.artifacts.report import summarize_search
    if not kwargs.get('representation_traversal', False):
        raise ValueError('Generation pipeline requires representation traversal')
    root = Path(options.output).resolve()
    if len(apis) > 1:
        root.mkdir(parents=True, exist_ok=False)
    reports = []
    for api_index, api in enumerate(apis):
        directory = root if len(apis) == 1 else root / f'api-{api_index:03d}'
        pipeline = ExecutionPipeline(directory, api, options.time_limit, options.worker_timeout,
                                     replay_failures=options.replay_failures, workers=options.workers,
                                     generation_overlap=options.generation_overlap)
        completed = False
        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.getsignal(signal.SIGTERM)
            def terminate(signum, frame):
                raise KeyboardInterrupt('Pipeline termination requested; durable work can be resumed')
            signal.signal(signal.SIGTERM, terminate)
        try:
            report = _run_stateful(replace(options, output=str(directory)), (api,), _pipeline=pipeline, **kwargs)
            reports.append(report)
            completed = True
            if report.get('stop_reason') == 'provider_cancellation_unconfirmed':
                break
        finally:
            try:
                pipeline.close(interrupted=not completed)
            finally:
                if old_handler is not None:
                    signal.signal(signal.SIGTERM, old_handler)
    if len(apis) == 1:
        return reports[0]
    result = {'strategy': 'stateful', 'per_api': reports,
              'records': [r for report in reports for r in report['records']],
              'stop_reason': 'llm_time_limit' if len(reports) == len(apis) and all(r.get('pipeline_complete') for r in reports) else 'pipeline_incomplete',
              'provider_calls': sum(r['provider_calls'] for r in reports),
              'elapsed_seconds': sum(r['elapsed_seconds'] for r in reports),
              'pipeline_complete': len(reports) == len(apis) and all(r.get('pipeline_complete') for r in reports),
              'expected_apis': list(apis), 'completed_api_runs': len(reports)}
    result['search_metrics'] = summarize_search(result)
    write_json(root / 'summary.json', result)
    return result


def _run_stateful(options, apis, *, edges=True, predicates=None, path_policy="backward", coverage_refinement=True,
                 search_policy="adaptive", generation_mode="local", coverage_basis="semantic",
                 representation_traversal=False, oracle_improvements=False, _pipeline=None, wall_deadline=None, api_spec=None):
    """Corpus-guided local generation, runtime mutation and conditional MR assessment.

    API/issue benchmark data never enters this path. Every candidate and repair
    retains its request, response and execution evidence under one bounded run.
    """
    from dataclasses import replace
    from mirage.runtime.pipeline import GenerationBudgetExhausted
    from mirage.generator.program import (stateful_request, analyze_stateful, unmet_predicate_feedback,
                                          source_operand_bindings)
    from mirage.contracts.exploration import stateful_input_profile, stateful_state_predicates, stateful_diversity_predicates
    from mirage.generator.provider import ProviderFailure
    from mirage.inspector.stateful import failure_record, input_category_feedback
    from mirage.generator.mutation import plan_mutation, apply_mutation, plan_regions, apply_regions
    from mirage.generator.state_plan import (goal_catalog, goal_family, planning_request,
                                            validate_plans, optimizer_action_for_predicates)
    from mirage.artifacts.corpus import Corpus
    from mirage.sampler.policy import FeedbackSampler
    from mirage.artifacts.report import summarize_search
    from mirage.backends.pytorch import PyTorchBackend
    explain_certificate = PyTorchBackend().explain_certificate
    from mirage.sampler.representations import RepresentationTraversal, program_goal_catalog
    traversal = (_pipeline.traversal if _pipeline else RepresentationTraversal()) if representation_traversal else None
    audit_seen_units = set()
    def planning_catalog(entry, explicit):
        catalog = goal_catalog(entry, explicit)
        return program_goal_catalog(catalog) if traversal else catalog
    if traversal and any(p.get('dimension') in ('representation', 'tensor_relation') for p in (predicates or [])):
        raise ValueError('Explicit input representation predicates belong to framework traversal, not LLM planning')
    if search_policy not in ('adaptive', 'round_robin') or generation_mode not in ('local', 'full') or coverage_basis not in ('semantic', 'transition'):
        raise ValueError('Unsupported stateful search treatment')
    if path_policy not in ('backward','mixed'):
        raise ValueError('Unsupported stateful path policy')
    root = Path(options.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshot_engine(root / 'engine')
    write_json(root / 'engine-sha256.json', inventory(root / 'engine'))
    metadata, rows = {}, []
    from mirage.runtime.scheduler import current_scope
    provider = IsolatedProvider(options.provider, root, options.worker_timeout,
                                cancel=_pipeline.cancel if _pipeline else (current_scope().cancel if current_scope() else None))
    metadata_started = time.monotonic()
    for index, api in enumerate(apis):
        directory = root / f'api-{index}'
        if api_spec is not None:
            from mirage.generator.state_plan import check_context, fragment_claims
            if api_spec['api'] != api or api_spec['status'] != 'resolved':
                raise ValueError('Acquired Spec does not identify the requested API')
            metadata[api] = check_context(api_spec)
            directory.mkdir()
            write_json(directory/'summary.json', metadata[api])
            continue
        metadata_timeout = options.worker_timeout
        if wall_deadline is not None:
            metadata_timeout = min(metadata_timeout, wall_deadline - time.monotonic())
        process = (_pipeline.worker if _pipeline else run_worker)({'mode':'api_metadata', 'api':api, 'contract':'stateful-1',
                              'engine':str(root/'engine')}, directory, metadata_timeout)
        if process.outcome != 'completed':
            result = {'strategy':'stateful', 'records':[], 'stop_reason':'metadata_failed', 'error':process.error}
            write_json(root/'summary.json', result); return result
        metadata[api] = json.loads((directory/'summary.json').read_text())
    metadata_seconds = time.monotonic()-metadata_started
    started = time.monotonic()
    # In pipeline mode only model calls spend the per-API budget. Each test
    # retains its own timeout; allocation interruption is handled by the owner.
    deadline = started + (1e9 if _pipeline else options.time_limit)
    if wall_deadline is not None:
        deadline = min(deadline, wall_deadline)
    scope = current_scope()
    if scope is not None and not _pipeline:
        scope.deadline = deadline if scope.deadline is None else min(scope.deadline, deadline)
    pending = None; repair_count = 0; stop = 'case_limit'
    corpus = Corpus(root, coverage_basis=coverage_basis)
    feedback = FeedbackSampler(seed=options.seed, epsilon=options.epsilon, coverage_basis=coverage_basis)
    source_cache = {}
    pending_selection = None
    plan_cache = {}
    library_cache = {}
    attempted_goals = {}
    selected_goals = {}
    planning_counts = Counter()
    planning_failures = Counter()
    corpus.save()
    signatures, proposal_counts = {}, Counter()
    dispatch_reserve = 10.
    dispatch_stop = None
    relation_ledger = {}
    claim_index = 0
    input_index = 0
    exception_reruns = Counter()
    def consume_representations():
        if not _pipeline:
            return
        import queue
        while True:
            try:
                child = _pipeline.completed.get_nowait()
            except queue.Empty:
                break
            feedback_row = dict(child, verdict=child.get('mr_verdict', child.get('verdict')))
            if child.get('representation_candidate_attained') is True:
                feedback_row['inspection'] = child.get('followup_inspection', {})
            child['search_feedback'] = feedback.observe(child['candidate'], feedback_row)
            _pipeline.observe(child)
            audit_seen_units.update(child.get('audit_candidate_units', []))
            write_json(Path(child['directory']) / 'outcome.json', child)
            rows.append(child)
    if _pipeline:
        from mirage.runtime.pipeline import BudgetedProvider
        _pipeline.consume = consume_representations
        provider = BudgetedProvider(provider, _pipeline)
        _pipeline.start()
        execute_worker = _pipeline.worker
    else:
        execute_worker = run_worker
    for index in range(options.cases):
        if scope is not None and scope.cancel.is_set():
            stop = 'interrupted'
            break
        if _pipeline:
            _pipeline.wait_capacity()
            consume_representations()
            if _pipeline.model_seconds >= options.time_limit - .01:
                stop = 'llm_time_limit'
                break
        if rows:
            dispatch_reserve = max(10., 1.5*max(r['finished_seconds']-r['started_seconds'] for r in rows)+5.)
        early_reserve = dispatch_reserve if pending or search_policy != 'adaptive' else 10.
        if deadline-time.monotonic() <= early_reserve:
            dispatch_stop = {'reason': 'pending_or_minimum_reserve', 'reserve_seconds': early_reserve,
                             'remaining_seconds': deadline-time.monotonic()}
            stop = 'time_limit'; break
        relation_id = next((key for key, item in relation_ledger.items()
                            if item['status'] == 'proposed'), None) if index % 4 == 1 and not pending else None
        representation_work = (traversal.choose(deadline-time.monotonic(), dispatch_reserve)
                               if traversal and not _pipeline and not pending and not relation_id and (input_index % 2 == 1
                                   or deadline-time.monotonic() <= dispatch_reserve) else None)
        if representation_work:
            origin, candidate, parent_source, reserve = representation_work
            api = origin['configuration']['api']
            config = dict(origin['configuration'], attempt=index, proposal_index=input_index,
                          canonical_reference=False, representation_candidate=candidate['plan'])
            request = stateful_request(metadata[api], config, edges=edges, predicates=config['predicates'],
                                       canonical=parent_source)
            request.context.update(generation_reason='framework_representation_traversal',
                configuration_source_reuse=True, proposal_origin='framework_representation',
                representation_source_id=origin['source_id'], representation_candidate_id=candidate['id'],
                traversal_initial_reference=origin['initial_reference'],
                configuration_reuse_obligations={k: origin[k] for k in
                    ('required_backward_calls', 'required_callable_backward_calls')})
            choice = {'api': api, 'parent': origin['source_id'], 'kind': 'representation',
                      'goal': candidate['id']}
            selection = {'policy': 'representation-traversal-1', 'candidate_count': 1,
                         'selected_probability': 1., 'dispatch_reserve_seconds': reserve}
            input_index += 1
        elif pending:
            request = pending
            choice, selection = pending_selection
            config = dict(request.context['configuration'])
            api = config['api']
            pending = None
        elif api_spec is not None:
            api = api_spec['api']
            protocol = fragment_claims(api_spec)
            claim = protocol[claim_index % len(protocol)]
            parent_source = None
            requested = []
            if relation_id:
                candidate = relation_ledger[relation_id]
                candidate['status'] = 'dispatched'
                candidate['history'].append('dispatched')
                write_json(root/'relation-ledger.json', relation_ledger)
                parent_source = (Path(candidate['source'])/'program.py').read_text()
                requested = [candidate['predicate']]
                claim = 'operand_relation'
            else:
                claim_index += 1
            config = dict(api=api, device=options.device, seed=options.seed, attempt=index,
                          proposal_index=input_index, predicates=requested, check_claim=claim,
                          requires_backward=claim in ('backward', 'accumulation') or any(
                              p.get('relation') or p.get('peer_relation') for p in requested),
                          input_categories=['ordinary'], representation_traversal=True)
            request = stateful_request(metadata[api], config, predicates=requested, canonical=parent_source)
            if relation_id:
                request.context['relation_id'] = relation_id
            choice = dict(api=api, parent=None, kind='fresh', goal=claim)
            selection = dict(policy='spec_and_observed_relations', candidate_count=len(protocol))
            input_index += 1
        else:
            mutation = None
            if search_policy == 'adaptive':
                candidates = [{'api':a, 'parent':None, 'kind':'fresh'} for a in apis]
                for entry in corpus.entries:
                    if not entry['active']:
                        continue
                    if entry['id'] not in source_cache:
                        source_cache[entry['id']] = corpus.source(entry)
                    catalog = planning_catalog(entry, predicates) if predicates != [] else []
                    attempted = attempted_goals.get(entry['id'], set())
                    if planning_failures[entry['id']] < 2 and (predicates == [] or any(g['id'] not in attempted for g in catalog)):
                        candidates.append({'api':entry['configuration']['api'], 'parent':entry['id'], 'kind':'planned'})
                    if predicates != []:
                        pool = library_configuration_pool(entry, catalog, library_cache)
                        ready = available_library_plans(pool, selected_goals.get(entry['id'], set()))
                        if ready:
                            # One branch per parent prevents goal-count multiplicity bias.
                            candidates.append({'api':entry['configuration']['api'], 'parent':entry['id'],
                                               'kind':'configuration', 'proposal_origin':'library_known_action',
                                               'dispatch_budget': configuration_dispatch_budget(entry, rows, dispatch_reserve)})
                if library_cache:
                    write_json(root/'library-proposals.json', list(library_cache.values()))
                known_count = len(candidates)
                unseen = next((a for a in apis if not proposal_counts[a]), None)
                if unseen:
                    candidates = [c for c in candidates if c['api'] == unseen and c['kind'] == 'fresh']
                    selection_reason = 'api_bootstrap'
                elif input_index % 4 == 0:
                    candidates = [c for c in candidates if c['kind'] == 'fresh']
                    selection_reason = 'fresh_generation_quota'
                else:
                    selection_reason = 'adaptive'
                remaining = deadline-time.monotonic()
                quota_count = len(candidates)
                candidates = affordable_candidates(candidates, remaining, dispatch_reserve)
                if not candidates:
                    dispatch_stop = {'reason': 'no_affordable_quota_candidate', 'quota': selection_reason,
                                     'remaining_seconds': remaining, 'quota_candidate_count': quota_count,
                                     'whole_round_reserve_seconds': dispatch_reserve}
                    stop = 'time_limit'; break
                choice, selection = feedback.choose(candidates, reason=selection_reason)
                selection['known_candidate_count'] = known_count
                selection['budget_filtered_candidates'] = quota_count-len(candidates)
                selection['dispatch_budget'] = dict(choice.get('dispatch_budget', {
                    'reserve_seconds': dispatch_reserve, 'cost_basis': 'whole_round_fallback'}),
                    remaining_seconds=remaining)
                api = choice['api']
                parent = corpus.select(api, seed_id=choice['parent']) if choice['parent'] else None
            else:
                api = apis[index % len(apis)]
                parent = corpus.select(api) if proposal_counts[api] % 4 else None
                if parent and planning_failures[parent['id']] >= 2:
                    parent = None
                if parent and predicates != [] and not any(
                        g['id'] not in attempted_goals.get(parent['id'], set()) for g in planning_catalog(parent, predicates)):
                    parent = None
                choice = {'api':api, 'parent':parent['id'] if parent else None, 'kind':'fresh'}
                selection = {'policy':'round_robin', 'candidate_count':1, 'selected_probability':1.}
            proposal = proposal_counts[api]
            proposal_counts[api] += 1
            config = {'api':api, 'device':options.device, 'seed':options.seed, 'attempt':index, 'proposal_index':input_index,
                      'requires_backward':not options.forward_only and (path_policy != 'mixed' or index % 3 != 0)
                          and not metadata[api].get('callable_capabilities',{}).get('context_manager_factory',False),
                      'input_categories':list(stateful_input_profile(input_index).requested),
                      'predicates':[],
                      'canonical_reference':bool(index % 2)}
            if parent:
                # Mixed policy chooses canonical obligations, never silently adds a
                # backward requirement to an operand-only child edit.
                config['requires_backward'] = bool(parent['configuration'].get('requires_backward', True)
                                                   or parent['evidence'].get('backward_executed'))
            if parent and choice['kind'] == 'configuration':
                config = dict(parent['configuration'], attempt=index, proposal_index=input_index, predicates=[])
            if not config['requires_backward'] and predicates is None:
                config['predicates'] = [p for p in config['predicates'] if p['dimension'] == 'context']
            # A verified canonical seeds explicit planning; no synthetic count rotation.
            mutation, parent_source = None, None
            reason = 'periodic_fresh_generation' if proposal % 4 == 0 else 'no_verified_seed'
            if parent:
                parent_source = corpus.source(parent)
                reason = ('library_configuration_selection' if choice['kind'] == 'configuration'
                          else 'explicit_state_planning')
                if choice['kind'] != 'configuration':
                    choice = dict(choice, kind='planned')
                if predicates == []:
                    mutation = plan_mutation(parent_source, parent['id'], proposal - 1) if generation_mode == 'local' else None
                    if mutation:
                        mutation.update(mutation_backward_requirements(parent['evidence']))
                    reason = 'local_mutation' if mutation else 'full_regeneration'
                    choice = dict(choice, kind=mutation['kind'] if mutation else 'full')
            if traversal:
                config['representation_traversal'] = True
            request = stateful_request(metadata[api], config, edges=edges, predicates=config['predicates'],
                                       canonical=parent_source, mutation=mutation)
            request.context['generation_parent_id'] = parent['id'] if parent else None
            if parent_source:
                request.context['parent_certificate_diagnostic'] = explain_certificate(parent_source)
                request.context['parent_optimizer_certificate_diagnostic'] = explain_certificate(
                    parent_source, local_vjp=True, optimizer_update=True)
            request.context['generation_reason'] = reason
            request.context['prior_transition_signatures'] = signatures.get(api, [])[-8:]
            if traversal:
                request.context['representation_feedback'] = [
                    {'source_id': entry['source_id'], 'candidates': [
                        {'action': c['plan']['action'], 'status': c['status'],
                         'reason': c['eligibility'].get('reason')}
                        for c in entry['candidates'][:16]]}
                    for entry in traversal.sources if entry['configuration']['api'] == api][-4:]
            input_index += 1
        config = json.loads(json.dumps(config))
        if oracle_improvements:
            config.update(oracle_improvements=True, oracle_seen_units=sorted(audit_seen_units))
        request.context['configuration'] = config
        mutation = request.context.get('mutation_contract')
        parent_id = request.context.get('generation_parent_id') or (mutation['parent_id'] if mutation else None)
        directory = root/f'attempt-{index:04d}';directory.mkdir()
        write_json(directory/'request.json', asdict(request))
        row = {'attempt':index, 'api':api, 'directory':str(directory), 'generated':False,
               'contract_valid':False, 'executable':False, 'api_executed':False,
               'repair_index':repair_count, 'refinement_kind':request.context.get('refinement_kind'), 'started_seconds':time.monotonic()-started,
               'configuration':config, 'verdict':None, 'generation_stage':request.stage.value,
               'generation_reason':request.context.get('generation_reason'), 'parent_seed_id':parent_id,
               'generation_mode':'local' if mutation else 'full', 'selection':selection, 'candidate':choice,
               'proposal_origin':request.context.get('proposal_origin', 'fresh_or_legacy')}
        if request.context.get('transition_plan'):
            row['state_plan'] = request.context['transition_plan']
        phase = 'generation'; response = None; repairable = False; coverage_feedback = None; edit = None
        try:
            if parent_id and not request.previous_content and request.context.get('generation_reason') in (
                    'explicit_state_planning', 'library_configuration_selection'):
                phase = 'state_plan'
                parent_entry = next(e for e in corpus.entries if e['id'] == parent_id)
                library_plan = request.context.get('generation_reason') == 'library_configuration_selection'
                consumed = (selected_goals if library_plan else attempted_goals).get(parent_id, set())
                catalog = [g for g in planning_catalog(parent_entry, predicates) if g['id'] not in consumed]
                if not catalog:
                    raise ContractError('No unattained supported goal: reselect a seed or canonical')
                row['proposal_origin'] = 'library_known_action' if library_plan else 'llm'
                if library_plan:
                    pool = library_configuration_pool(parent_entry, planning_catalog(parent_entry, predicates), library_cache)
                    plans = available_library_plans(pool, selected_goals.get(parent_id, set()))
                    row.update(planning_dispatched=False, planning_outcome='library_planned',
                               library_candidate_count=len(plans), library_cache_key=pool['cache_key'])
                    write_json(directory/'plans.json', dict(pool, available=plans))
                else:
                    if parent_id not in plan_cache or not plan_cache[parent_id]:
                        families = tuple(sorted({goal_family(g) for g in catalog}))
                        family_counts = {f: planning_counts[(api, f)] for f in families}
                        portfolio = Sampler(families, seed=options.seed + input_index, epsilon=options.epsilon)
                        planning_family = portfolio.sample(family_counts)
                        probabilities = portfolio.probabilities(family_counts)
                        planning_counts[(api, planning_family)] += 1
                        row['planning_family_selection'] = {
                            'family': planning_family, 'counts_before': family_counts,
                            'probabilities': dict(zip(families, probabilities)),
                            'scope': 'known available goal families; proposal allocation, not reachability'}
                        plan_request = planning_request(metadata[api], request.context['canonical_source'], parent_entry,
                                                        catalog, config, planning_family=planning_family)
                        if traversal:
                            plan_request.context['representation_feedback'] = request.context.get('representation_feedback', [])
                        plan_request.context['parent_certificate_diagnostic'] = explain_certificate(request.context['canonical_source'])
                        plan_request.context['parent_optimizer_certificate_diagnostic'] = explain_certificate(
                            request.context['canonical_source'], local_vjp=True, optimizer_update=True)
                        write_json(directory/'plan-request.json', asdict(plan_request))
                        row['planning_dispatched'] = True
                        row['planning_usage'] = None
                        plan_response = provider.generate(plan_request, deadline=deadline)
                        write_json(directory/'plan-response.json', asdict(plan_response))
                        row['planning_usage'] = asdict(plan_response.usage)
                        plans, rejected = validate_plans(plan_response.content, catalog, planning_family=planning_family)
                        write_json(directory/'plans.json', {'accepted':plans, 'rejected':rejected})
                        unresolved = [p for p in rejected if p.get('status') == 'unresolved']
                        row['planning_unresolved'] = unresolved
                        row['planning_outcome'] = ('planned' if plans else
                                                   'unresolved' if unresolved or not rejected else 'rejected')
                        attempted_goals.setdefault(parent_id, set()).update(
                            goal for p in unresolved for goal in p.get('examined_goal_ids', []))
                        plan_cache[parent_id] = plans
                    plans = plan_cache[parent_id]
                if not plans:
                    if library_plan:
                        raise ContractError('Known configuration candidate became unavailable; runtime reachability unassessed')
                    # Exhausting a proposal budget does not prove the whole catalog unreachable.
                    planning_failures[parent_id] += 1
                    row['planning_budget_failures'] = planning_failures[parent_id]
                    raise ContractError('No compatible state plan: replan without requesting a patch')
                if not library_plan:
                    planning_failures[parent_id] = 0
                plan_kind = 'configuration' if library_plan else 'planned'
                candidates = [{'api':api, 'parent':parent_id, 'kind':plan_kind, 'goal':p['family'],
                               'feedback_family':p['family'], 'plan_index':i} for i,p in enumerate(plans)]
                # Equivalent goal families are distinct only when their exact predicates differ.
                for candidate in candidates:
                    candidate['goal'] += ':' + ','.join(sorted(plans[candidate['plan_index']]['goal_ids']))
                selected, plan_selection = feedback.choose(candidates, reason='library_configuration' if library_plan else 'state_goal')
                plan = plans[selected['plan_index']] if library_plan else plans.pop(selected['plan_index'])
                selected_goals.setdefault(parent_id, set()).update(plan['goal_ids'])
                attempted_goals.setdefault(parent_id, set()).update(plan['goal_ids'])
                plan_cache[parent_id] = [p for p in plan_cache.get(parent_id, [])
                                        if not set(p['goal_ids']) & attempted_goals[parent_id]]
                config['predicates'] = plan['predicates']
                # The source must attain its own gradient-layout goal. This action
                # is executed by the observer at the native optimizer boundary.
                action = optimizer_action_for_predicates(config['predicates'])
                if action is not None:
                    config['optimizer_gradient_action'] = action
                else:
                    config.pop('optimizer_gradient_action', None)
                if not library_plan:
                    config['input_categories'] = parent_entry['configuration'].get('input_categories', ['ordinary'])
                regions = ['full'] if generation_mode == 'full' else plan['regions']
                reuse_configuration = optimizer_configuration_reusable(parent_entry, config)
                if library_plan and not reuse_configuration:
                    raise ContractError('Selected known action failed exact source/configuration reuse checks')
                mutation = (None if reuse_configuration else
                            plan_regions(request.context['canonical_source'], parent_id, index, regions))
                if mutation:
                    mutation.update(mutation_backward_requirements(parent_entry['evidence']))
                    mutation['desired_coverage'] = {'predicates':config['predicates']}
                choice = dict(selected, kind=plan_kind, proposal_origin=row['proposal_origin'])
                representation_feedback = request.context.get('representation_feedback', [])
                request = stateful_request(metadata[api], config, edges=edges, predicates=config['predicates'],
                    canonical=request.context['canonical_source'], mutation=mutation)
                if traversal:
                    request.context['representation_feedback'] = representation_feedback
                request.context.update(transition_plan=plan, generation_parent_id=parent_id,
                                       generation_reason='planned_local_edit' if mutation else 'planned_full_generation',
                                       proposal_origin=row['proposal_origin'])
                if reuse_configuration:
                    request.context.update(configuration_source_reuse=True,
                        configuration_reuse_obligations=mutation_backward_requirements(parent_entry['evidence']),
                        generation_reason='planned_execution_configuration')
                request.context['parent_certificate_diagnostic'] = explain_certificate(request.context['canonical_source'])
                request.context['parent_optimizer_certificate_diagnostic'] = explain_certificate(
                    request.context['canonical_source'], local_vjp=True, optimizer_update=True)
                row.update(configuration=config, state_plan=plan, plan_selection=plan_selection,
                           generation_reason=request.context['generation_reason'],
                           generation_mode='configuration' if reuse_configuration else 'local' if mutation else 'full', candidate=choice)
                write_json(directory/'request.json', asdict(request))
            phase = 'generation'
            reuse_configuration = bool(request.context.get('configuration_source_reuse') and not request.previous_content)
            row['program_generation_dispatched'] = not reuse_configuration
            if reuse_configuration:
                from mirage.generator.provider import Response, Usage
                response = Response(json.dumps({'source': request.context['canonical_source']}), Usage(0, 0),
                    metadata={'origin': 'verified_source_reuse', 'provider_call': False})
                row['source_reused'] = True
                representation_plan = config.get('representation_candidate')
                row['realized_transition'] = {'kind': 'execution_configuration',
                    'source_changed': False, 'action': representation_plan or config.get('optimizer_gradient_action'),
                    'scope': 'realized harness action; free-text planning actions are proposals'}
                edit = {'kind': 'execution_configuration', 'parent_id': parent_id,
                        'representation_candidate' if representation_plan else 'optimizer_gradient_action':
                            representation_plan or config.get('optimizer_gradient_action')}
                if representation_plan:
                    row['generation_mode'] = 'representation'
                    row['representation_traversal'] = {
                        'source_id': request.context['representation_source_id'],
                        'candidate_id': request.context['representation_candidate_id'],
                        'plan': representation_plan, 'status': 'dispatched'}
                write_json(directory/'edit.json', edit)
            else:
                response = provider.generate(request, deadline=deadline)
            write_json(directory/'response.json', asdict(response))
            row['generation_usage'] = asdict(response.usage)
            if same_repair_payload(request.previous_content, response.content):
                row['duplicate_repair_kind'] = 'response'
                raise ContractError('Repair repeated a previously rejected response; prior evidence is retained')
            phase = 'parse'
            value = json.loads(response.content)
            multi_region = mutation and mutation.get('version') == 'local-regions-2'
            expected = {'patches'} if multi_region else {'source'}
            if not isinstance(value, dict) or set(value) != expected:
                raise ContractError('Response must have exactly ' + ','.join(expected))
            row['generated'] = True
            if multi_region:
                phase = 'static_contract'
                source, edit = apply_regions(request.context['canonical_source'], value['patches'], mutation)
                write_json(directory/'edit.json', edit)
                row.update(source_normalization=None, raw_syntax_valid=True)
            else:
                source = value['source']
                if not isinstance(source, str):
                    raise ContractError('source must be a Python string')
            if mutation and not multi_region:
                (directory/'replacement.py').write_text(source)
                phase = 'static_contract'
                source, edit = apply_mutation(request.context['canonical_source'], source, mutation)
                write_json(directory/'edit.json', edit)
                row['source_normalization'] = None
                row['raw_syntax_valid'] = True
            elif not mutation:
                (directory/'raw-program.py').write_text(source)
                from mirage.generator.repair import normalize_source_delimiter
                source, normalization = normalize_source_delimiter(source)
                row['source_normalization'] = normalization
                row['raw_syntax_valid'] = normalization is None
            (directory/'program.py').write_text(source)
            phase = 'static_contract'; graph = analyze_stateful(source)
            row.update(contract_valid=True, fingerprint=graph['fingerprint'])
            if graph['fingerprint'] == request.context.get('repair_rejected_fingerprint'):
                row['duplicate_repair_kind'] = 'ast'
                raise ContractError('Repair did not change the previously rejected program AST; prior evidence is retained')
            row['program_certificate_diagnostic'] = explain_certificate(source)
            row['optimizer_certificate_diagnostic'] = explain_certificate(source, local_vjp=True, optimizer_update=True)
            row['source_operand_bindings'] = source_operand_bindings(source, api, config['predicates'])
            phase = 'initialization'
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError('Budget exhausted before execution')
            worker_request = {'mode':'execute_stateful_pair', 'source':source, 'config':config,
                                  'engine':str(root/'engine'),
                                  **({'initial_reference': request.context['traversal_initial_reference'],
                                      'reference_scope': 'initialization'}
                                     if request.context.get('traversal_initial_reference') else {}),
                                  'audit_budget_seconds':min(10, max(0.1, min(options.worker_timeout, remaining) / 8))}
            if _pipeline:
                from mirage.runtime.pipeline import save
                save(root/'pipeline-pending-source.json', {'source': source, 'row': row,
                    'request': worker_request,
                    'obligations': mutation or request.context.get('configuration_reuse_obligations')})
            process = execute_worker(worker_request, directory/'execution',
                                     min(options.worker_timeout, remaining))
            row['process'] = asdict(process)
            write_json(directory/'execution-sha256.json', inventory(directory/'execution'))
            outcome_path = directory/'execution/summary.json'
            if process.outcome == 'completed' and outcome_path.exists():
                outcome = json.loads(outcome_path.read_text()); row.update(outcome, status='executed', executable=True)
                backward_calls = {w['call'] for w in outcome.get('inspection', {}).get('semantic_coverage', {}).get('witnesses', [])
                                  if w['checkpoint'] == 'final' and w['relations']['target_backward_node']['status'] == 'verified'}
                row['mutation_obligations_valid'] = mutation_paths_preserved(
                    mutation or request.context.get('configuration_reuse_obligations'), outcome.get('inspection', {}))
                if row.get('proposal_origin') == 'framework_representation':
                    row['mutation_obligations_valid'] &= mutation_paths_preserved(
                        request.context.get('configuration_reuse_obligations'), outcome.get('followup_inspection', {}))
                    if row['mutation_obligations_valid'] is not True:
                        row['representation_candidate_attained'] = False
                if not outcome.get('inspection', {}).get('valid'):
                    check = outcome.get('inspection', {})
                    blocking = ('target_return' if not check.get('api_executed') else
                                'backward' if not check.get('required_backward_executed') else 'state')
                    diagnostic = unmet_predicate_feedback(check, bindings=row['source_operand_bindings'])
                    if any(p.get('dimension') in ('gradient_representation', 'optimizer')
                           for p in config['predicates']):
                        diagnostic['optimizer_source_effects'] = row['optimizer_certificate_diagnostic']
                        diagnostic['optimizer_limitations'] = check.get('optimizer_coverage', {}).get('limitations', [])[:8]
                    row['repair_evidence'] = diagnostic
                    row.update(failure_record(blocking, 'Required runtime evidence missing: '+json.dumps(diagnostic)))
                    repairable = True
                else:
                    signature = outcome.get('transition_signature', [])
                    if signature not in signatures.setdefault(api, []): signatures[api].append(signature)
                    if not row['mutation_obligations_valid']:
                        row.update(failure_record('backward', 'Mutation lost a protected target or callable backward path'))
                        repairable = True
                    else:
                        coverage_feedback = input_category_feedback(outcome) if coverage_refinement else None
                        if coverage_feedback: row['coverage_refinement_requested'] = coverage_feedback
            else:
                # Only a typed harness rejection permits semantic repair. Unknown
                # exceptions retain their original source and are independently rerun.
                diagnosis = stateful_runtime_diagnostic(process)
                row.update(status='execution_failed', reason=process.error,
                           first_blocking_stage='runtime_contract' if diagnosis['repair_allowed'] else 'unresolved_runtime',
                           secondary_diagnostics=[], runtime_diagnostic=diagnosis)
                repairable = diagnosis['repair_allowed']
                if not repairable:
                    row['repair_stopped'] = 'unresolved_runtime; retained for independent triage'
                    row['quarantined'] = True
                    if process.outcome == 'exception' and exception_reruns[api] < options.replay_failures:
                        remaining = deadline-time.monotonic()-5
                        if remaining > 5:
                            exception_reruns[api] += 1
                            row['exception_rerun'] = (_pipeline.operation if _pipeline else lambda fn, *args: fn(*args))(rerun_stateful_exception, process, directory/'exception-rerun',
                                min(options.worker_timeout, max(5, 2*process.elapsed_seconds+2), remaining))
        except GenerationBudgetExhausted as error:
            # No request was sent; retain prior planning evidence and drain existing work.
            stop = 'llm_time_limit'
            if phase == 'state_plan':
                row['planning_dispatched'] = False
            row.update(status='budget_before_generation', reason=str(error),
                       program_generation_dispatched=False, budget_rejected_request_dispatched=False,
                       generation_budget_stop=_pipeline.generation_budget_stop,
                       finished_seconds=time.monotonic()-started)
            write_json(directory/'outcome.json', row)
            rows.append(row)
            break
        except (ContractError, SyntaxError, json.JSONDecodeError) as error:
            status = 'invalid_program'
            if phase == 'state_plan':
                row.setdefault('planning_outcome', 'rejected')
                status = 'planning_unresolved' if row['planning_outcome'] == 'unresolved' else 'planning_rejected'
            row.update(status=status, **failure_record(phase, str(error)))
            repairable = phase != 'state_plan'
        except ProviderFailure as error:
            row.update(status='provider_failed', reason=str(error), cancellation_uncertain=error.cancellation_uncertain)
            if error.cancellation_uncertain:
                stop = 'provider_cancellation_unconfirmed'
                session = os.environ.get('MIRAGE_SESSION_DIR')
                if session: (Path(session)/'cancellation-unconfirmed').write_text(str(error))
        except Exception as error:
            row.update(status='unresolved', reason=f'{type(error).__name__}: {error}')
        if (phase == 'state_plan' and 'planning_usage' in row
                and row.get('status') in ('planning_rejected', 'planning_unresolved', 'provider_failed', 'unresolved')
                and 'planning_budget_failures' not in row):
            planning_failures[parent_id] += 1
            row['planning_budget_failures'] = planning_failures[parent_id]
        if row.get('proposal_origin') == 'library_known_action':
            # A rejected known configuration retains its evidence; changing Python
            # would turn this into a different proposal and incur unreported model cost.
            repairable = False
            attained = (row.get('inspection', {}).get('valid') is True
                        and row.get('mutation_obligations_valid') is True)
            row['library_configuration_outcome'] = 'verified' if attained else 'not_verified'
            if not attained:
                row['repair_stopped'] = 'library_configuration_evidence_retained'
                pool = library_cache[parent_id]
                pool['execution_rejections'].append({'goal_ids': row.get('state_plan', {}).get('goal_ids', []),
                    'directory': str(directory), 'reason': row.get('reason', 'Runtime obligations not established'),
                    'reachability': 'unassessed'})
                write_json(root/'library-proposals.json', list(library_cache.values()))
        if row.get('proposal_origin') == 'framework_representation':
            repairable = False
            coverage_feedback = None
            if row.get('representation_candidate_attained') is not True:
                row['repair_stopped'] = 'representation_evidence_retained_without_source_edit'
        noop = 'no semantic AST edit' in row.get('reason', '')
        if row.get('duplicate_repair_kind'):
            repairable = False
            row.update(status='repair_unchanged', repair_stopped='unchanged_before_execution',
                       repair_evidence_reference=request.context.get('repair_rejected_directory'))
        progress = state_repair_progress(row.get('inspection', {})) if row.get('first_blocking_stage') == 'state' else None
        if repairable and state_repair_stalled(request.context.get('repair_state_progress'), progress):
            repairable = False
            row['repair_stopped'] = 'no_monotonic_state_progress'
        if repairable and row.get('verdict') != 'suspected_failure' and response is not None and repair_count < (1 if noop else 2):
            repair_count += 1
            diagnostic=row.get('reason','contract violation')
            if noop:
                diagnostic += '. The patch is AST-identical. Implement the missing transition_plan delta; formatting changes do not attain the goal.'
            if mutation and 'protected target' in diagnostic:
                diagnostic += (' For operand_setup, emit ONLY operand assignments inserted before the existing target. '
                               'Do not copy probe.call, backward or return: those statements are already preserved. '
                               'The desired state predicate must follow from the inserted assignments.')
            context=dict(request.context, refinement_kind='contract_or_state', repair_state_progress=progress,
                         repair_rejected_fingerprint=row.get('fingerprint'), repair_rejected_directory=str(directory))
            pending = replace(request, context=context,previous_content=response.content, diagnostic=diagnostic)
            pending_selection = (choice, {'policy':'contract_repair', 'candidate_count':1, 'selected_probability':1.})
        else:
            repair_count = 0
        row['finished_seconds'] = time.monotonic()-started
        if oracle_improvements:
            audit_seen_units.update(row.get('audit_candidate_units', []))
        if row.get('representation_traversal'):
            item = row['representation_traversal']
            item['status'] = traversal.complete(item['source_id'], item['candidate_id'], row)
        if api_spec is not None:
            from mirage.generator.state_plan import offer_relations
            key = request.context.get('relation_id')
            if key:
                item = relation_ledger[key]
                item['attempts'].append(index)
                item.update(status='reached' if row.get('inspection', {}).get('valid') else 'not_reached',
                            checkpoint=row.get('inspection', {}).get('predicates', []),
                            reason=row.get('reason'), directory=str(directory))
                item['history'].append(item['status'])
            if row.get('proposal_origin') != 'framework_representation':
                offer_relations(row, relation_ledger)
            write_json(root/'relation-ledger.json', relation_ledger)
        if traversal and not _pipeline and row.get('executable'):
            row['representation_source_id'] = traversal.register(source, row)
        if traversal and not _pipeline:
            write_json(root/'representation-traversal.json', traversal.summary())
        # MR novelty must retain its own applicability when numerical evidence raises suspicion.
        feedback_row = dict(row, verdict=row.get('mr_verdict', row.get('verdict')))
        if row.get('proposal_origin') == 'framework_representation' and row.get('representation_candidate_attained') is True:
            feedback_row['inspection'] = row.get('followup_inspection', {})
        row['search_feedback'] = feedback.observe(choice, feedback_row)
        if _pipeline:
            _pipeline.observe(row)
        write_json(directory/'outcome.json', row);rows.append(row)
        if (directory/'execution/source.pt').exists():
            write_json(directory/'stateful-manifest.json', {
                'format':'stateful-1-replay', 'source':'program.py', 'configuration':config,
                'engine':str(root/'engine'), 'engine_inventory':inventory(root/'engine'),
                'files':inventory(directory), 'initial_reference':'execution/source.pt',
                'expected_verdict':row.get('verdict')})
            row['corpus_admission'] = ({'admitted': False, 'reason': 'representation_child_retained_in_traversal'}
                if row.get('proposal_origin') == 'framework_representation' else
                corpus.admit(source, row, directory/'stateful-manifest.json', parent_id=parent_id, edit=edit))
            if row.get('verdict') == 'suspected_failure':
                row['automatic_replays'] = []
                replay_started = time.monotonic()
                for _ in range(options.replay_failures):
                    remaining = deadline-time.monotonic()-5
                    if remaining <= 5:
                        row['automatic_replay_stop'] = 'search_budget_exhausted'
                        break
                    try:
                        repeated = (_pipeline.operation if _pipeline else lambda fn, *args, **kw: fn(*args, **kw))(
                            replay_stateful, directory/'stateful-manifest.json', timeout=min(options.worker_timeout, remaining))
                    except (OSError, ValueError, RuntimeError) as error:
                        repeated = {'replay_valid': False, 'provider_calls': 0,
                                    'reason': f'{type(error).__name__}: {error}'}
                    row['automatic_replays'].append(repeated)
                # Detection time and immutable outcome precede verification. The
                # overall search deadline includes this separately reported cost.
                row['automatic_replay_seconds'] = time.monotonic()-replay_started
                write_json(directory/'automatic-replays.json', row['automatic_replays'])
            if (api_spec is None and predicates == [] and generation_mode == 'local' and coverage_feedback and row['corpus_admission'].get('admitted') and not repairable
                    and row.get('verdict') != 'suspected_failure' and request.context.get('refinement_kind') != 'input_coverage'):
                refinement = plan_mutation(source, row['corpus_admission']['seed_id'], input_index, kind='initialization')
                if refinement is not None:
                    refinement.update(mutation_backward_requirements({
                        'backward_calls': sorted(backward_calls),
                        'witnesses': outcome.get('inspection', {}).get('semantic_coverage', {}).get('witnesses', [])}))
                    refinement['desired_coverage'] = {'input_categories':config['input_categories'], 'predicates':config['predicates']}
                    pending = stateful_request(metadata[api], config, edges=edges, canonical=source,
                                               predicates=config['predicates'], mutation=refinement)
                    pending.context.update(refinement_kind='input_coverage', generation_reason='local_input_refinement',
                                           coverage_feedback=coverage_feedback, generation_parent_id=refinement['parent_id'])
                    pending_selection = ({'api':api, 'parent':refinement['parent_id'], 'kind':'initialization'},
                                         {'policy':'input_refinement', 'candidate_count':1, 'selected_probability':1.})
        if _pipeline:
            if row.get('executable'):
                _pipeline.register(source, row)
            else:
                _pipeline.register('', row)
            (root/'pipeline-pending-source.json').unlink(missing_ok=True)
            consume_representations()
        result = {'strategy':'stateful', 'records':rows, 'stop_reason':stop, 'provider_calls':provider.calls,
                  'elapsed_seconds':time.monotonic()-started, 'metadata_seconds':metadata_seconds,
                  'dispatch_reserve_seconds':dispatch_reserve,
                  'dispatch_admission': {'policy':'known-configuration-worker-cost-1', 'stop':dispatch_stop},
                  'prompt_treatment':'base_plus_edges' if edges else 'base',
                  'corpus':corpus.summary(), 'search_policy':search_policy,
                  'generation_mode':generation_mode, 'coverage_basis':coverage_basis,
                  'library_planning': {'parent_pools': len(library_cache),
                      'accepted_proposals': sum(len(p['accepted']) for p in library_cache.values()),
                      'rejected_proposals': sum(len(p['rejected']) for p in library_cache.values()),
                      'metadata_seconds': sum(p['metadata_planning_seconds'] for p in library_cache.values()),
                      'scope': 'Cached bounded proposals, not proven reachable states'}}
        result['representation_traversal'] = (_pipeline.summary() if _pipeline else traversal.summary()) if traversal else {'enabled': False}
        result['oracle_improvements'] = oracle_improvements
        result['search_metrics'] = summarize_search(result)
        write_json(root/'summary.json', result)
        if stop == 'provider_cancellation_unconfirmed': break
    if _pipeline:
        _pipeline.finish()

    result = {'strategy':'stateful', 'records':rows, 'stop_reason':stop, 'provider_calls':provider.calls,
              'elapsed_seconds':time.monotonic()-started, 'metadata_seconds':metadata_seconds,
              'dispatch_reserve_seconds':dispatch_reserve,
                  'dispatch_admission': {'policy':'known-configuration-worker-cost-1', 'stop':dispatch_stop},
                  'prompt_treatment':'base_plus_edges' if edges else 'base',
                  'corpus':corpus.summary(), 'search_policy':search_policy,
                  'generation_mode':generation_mode, 'coverage_basis':coverage_basis,
                  'library_planning': {'parent_pools': len(library_cache),
                      'accepted_proposals': sum(len(p['accepted']) for p in library_cache.values()),
                      'rejected_proposals': sum(len(p['rejected']) for p in library_cache.values()),
                      'metadata_seconds': sum(p['metadata_planning_seconds'] for p in library_cache.values()),
                      'scope': 'Cached bounded proposals, not proven reachable states'}}
    result['representation_traversal'] = (_pipeline.summary() if _pipeline else traversal.summary()) if traversal else {'enabled': False}
    result['oracle_improvements'] = oracle_improvements
    if _pipeline:
        result['dispatch_reserve_seconds'] = None
        result['dispatch_admission'] = {'policy': 'model_budget_only_then_full_drain'}
        result['pipeline_timing'] = _pipeline.timing()
        result['pipeline_complete'] = _pipeline.summary()['dispatch_complete']
        result['generation_stop_reason'] = stop
        if not result['pipeline_complete']:
            result['stop_reason'] = 'pipeline_incomplete'
        result['budget_scope'] = 'Cumulative model request seconds per API; framework drain is independent'
    if api_spec is not None:
        result['model_request_timings'] = provider.request_timings
        result['relation_goal_counts'] = dict(Counter(item['status'] for item in relation_ledger.values()))
        result['relation_goal_candidates'] = len(relation_ledger)
    result['search_metrics'] = summarize_search(result)
    write_json(root/'summary.json', result)
    return result



def replay_stateful(path, *, timeout=600, allow_environment_change=False, cancel=None):
    """Replay an immutable source/input-checked execution pair without a provider."""
    manifest=json.loads(path.read_text());directory=path.parent
    verify(directory,manifest['files'])
    engine=Path(manifest['engine']);verify(engine,manifest['engine_inventory'])
    original_request = directory/'execution/request.json'
    audit_budget = (json.loads(original_request.read_text()).get('audit_budget_seconds', 10)
                    if original_request.exists() else 10)
    destination=directory/f'replay-{uuid.uuid4().hex[:12]}'
    process=run_worker({'mode':'execute_stateful_pair','source':(directory/manifest['source']).read_text(),
                        'config':manifest['configuration'],'engine':str(engine),
                        'initial_reference':str(directory/manifest['initial_reference']),
                        'audit_budget_seconds':audit_budget,
                        'allow_environment_change':allow_environment_change},destination,timeout,cancel=cancel)
    summary=json.loads((destination/'summary.json').read_text()) if (destination/'summary.json').exists() else {}
    result={'process':asdict(process),'summary':summary,'provider_calls':0,
            'matches_original_verdict':process.outcome=='completed' and summary.get('verdict')==manifest['expected_verdict'],
            'directory':str(destination)}
    result['replay_valid']=bool(result['matches_original_verdict'] and summary.get('replay_source_evidence_equal')
                                and summary.get('replay_followup_evidence_equal'))
    write_json(destination/'replay.json',result)
    return result
