"""Private subprocess entry point for initialization, providers and training."""

from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import traceback


def stateful_replay_evidence(record):
    """Required semantic evidence; process-local identities and timings are excluded."""
    keys=('observations','final_training_state','checkpoints','transition_signature',
          'inspection','target_gradients','context_events','input_coverage')
    result={key:record.get(key) for key in keys}
    # Compare captured tensors; audit timings are not replay evidence.
    if 'gradient_evidence' in record:
        result['gradient_evidence'] = record['gradient_evidence']
    result['target_calls']=[{key:call.get(key) for key in
        ('index','inputs','output','metadata','write_effects','writable_paths','kind','evidence','schema_read_only','entry_context')}
        for call in record.get('target_calls',[])]
    return result


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
    if mode == "execute_stateful_pair":
        from mirage.runtime.stateful import execute
        from mirage.runtime.probes import Recorder
        from mirage.backends.pytorch import PyTorchBackend
        from mirage.rules.catalog import representation_pair
        initial_reference = torch.load(request["initial_reference"], map_location="cpu", weights_only=True) if request.get("initial_reference") else None
        left = execute(request["source"], dict(request["config"], compiler_reference=True), Recorder(directory / "source-trace.jsonl"), initial_reference)
        if initial_reference is not None and not request.get("allow_environment_change"):
            if left["environment"] != initial_reference.get("environment"):
                raise ValueError("Replay environment differs from the recorded execution")
        torch.save(left, directory / "source.pt")
        eligible = [p for p in left["mutation_candidates"] if p["eligibility"]["status"] == "verified"]
        summary = {"api_executed": left["api_executed"], "backward_executed": left["backward_executed"],
                   "target_state_reached": left["target_state_reached"], "input_coverage": left["input_coverage"],
                   "transition_signature": left["transition_signature"], "inspection": left["inspection"],
                   "eligible_mutations": len(eligible), "verdict": left["verdict"],
                   "correctness": "unassessed", "reason": "no eligible mutation"}
        from mirage.runtime.operands import MutationInapplicable
        summary['source_completed'] = True
        summary['followup_completed'] = False
        right = None
        try:
            if left.get("context_contract"):
                from mirage.rules.catalog import context_restoration
                summary.update(context_restoration(left, PyTorchBackend()))
            elif left.get("capture_factories"):
                from mirage.rules.catalog import captured_execution_pair
                right = execute(request["source"], dict(request["config"], capture_reference=True), Recorder(directory / "followup-trace.jsonl"), left)
                torch.save(right, directory / "followup.pt")
                summary.update(captured_execution_pair(left, right, PyTorchBackend()))
            elif left.get("compiled_graphs"):
                from mirage.rules.catalog import execution_mode_pair
                right = execute(request["source"], request["config"], Recorder(directory / "followup-trace.jsonl"), left)
                torch.save(right, directory / "followup.pt")
                summary.update(execution_mode_pair(left, right, PyTorchBackend()))
            elif request["config"].get("canonical_reference") and left["inspection"]["valid"]:
                config = dict(request["config"], canonicalize_inputs=True)
                right = execute(request["source"], config, Recorder(directory / "followup-trace.jsonl"), left)
                torch.save(right, directory / "followup.pt")
                summary.update(representation_pair(left, right, PyTorchBackend()), mutation={"action":"canonicalize_actual_operands"})
            elif eligible and left["inspection"]["valid"]:
                plan = eligible[request["config"].get("attempt", 0) % len(eligible)]
                config = dict(request["config"], mutation={"path":plan["path"], "action":plan["action"], "call_index":plan["call_index"]})
                right = execute(request["source"], config, Recorder(directory / "followup-trace.jsonl"), left)
                torch.save(right, directory / "followup.pt")
                summary.update(representation_pair(left, right, PyTorchBackend()), mutation=config["mutation"])
            summary['followup_completed'] = (directory / "followup.pt").exists()
            if right is not None: summary['followup_input_coverage'] = right['input_coverage']
        except MutationInapplicable as error:
            summary.update(verdict='inapplicable', correctness='unassessed', reason=str(error),
                           first_blocking_stage='mutation_application')
        from mirage.inspector.oracles import audit_pair
        selected = (request['config'].get('numerical_audit') is True
                    or summary.get('verdict') == 'suspected_failure'
                    or request['config'].get('attempt', 0) % 4 == 0)
        if request['config'].get('numerical_audit') is False:
            selected = False
        audit_pair(left, right, summary, budget_seconds=request.get('audit_budget_seconds', 10), enabled=selected)
        torch.save(left, directory / 'source.pt')
        if right is not None:
            torch.save(right, directory / 'followup.pt')
        if initial_reference is not None:
            comparator=PyTorchBackend()
            summary['replay_source_evidence_equal']=comparator.compare(
                stateful_replay_evidence(initial_reference),stateful_replay_evidence(left),'exact')['equal'] is True
            original_followup=Path(request['initial_reference']).with_name('followup.pt')
            if original_followup.exists() and right is not None:
                original=torch.load(original_followup,map_location='cpu',weights_only=True)
                summary['replay_followup_evidence_equal']=comparator.compare(
                    stateful_replay_evidence(original),stateful_replay_evidence(right),'exact')['equal'] is True
            else:
                summary['replay_followup_evidence_equal']=not original_followup.exists() and right is None
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
        return
    if mode == "execute_stateful":
        from mirage.runtime.stateful import execute
        from mirage.runtime.probes import Recorder
        result = execute(request["source"], request["config"], Recorder(directory / "trace.jsonl"))
        torch.save(result, directory / "result.pt")
        keys = ("version", "completed", "api_executed", "backward_executed", "determinism",
                "correctness", "verdict", "reason")
        (directory / "summary.json").write_text(json.dumps({k: result[k] for k in keys}, indent=2))
        return
    if mode == "execute_property":
        from mirage.runtime.program import run_property
        from mirage.runtime.probes import Recorder
        result=run_property(request["source"],request["config"],Recorder(directory/"trace.jsonl"),directory,request.get("initial"))
        torch.save(result,directory/"result.pt")
        summary={k:v for k,v in result.items() if k not in ("observables","reference","control")}
        (directory/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
        return
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
            if request.get("contract") == "stateful-1":
                import inspect
                from mirage.runtime.api_metadata import resolve
                result["documentation"] = inspect.getdoc(resolve(request["api"])) or ""
                result["documentation_truncated"] = False
                target = resolve(request["api"])
                result["callable_capabilities"] = {
                    "context_manager_factory": inspect.isclass(target) and hasattr(target,"__enter__") and hasattr(target,"__exit__"),
                    "tensor_return": "unknown", "backward": "runtime_evidence_required"}
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
