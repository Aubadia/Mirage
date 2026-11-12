"""Derive execution evidence from traces rather than requested configuration."""

from mirage.contracts.models import Evidence, Execution, Outcome
from mirage.runtime.representations import representation_matches


def assess(experiment, execution_id, role, config, process, record) -> Execution:
    target = config.target
    if record is None:
        evidence = Evidence(target.checkpoint)
        return Execution(execution_id, experiment.experiment_id, role, Outcome(process.outcome),
                         evidence, process.elapsed_seconds)

    events = record.get("events", [])
    at_target = [event for event in events if event.get("phase") == "training"
                 and event.get("iteration") == target.iteration]
    api_executed = any(event.get("event") == "api_exit" and event.get("api") == config.api
                       and event.get("recompute") is False for event in at_target)
    if config.backend != "eager":
        name = config.api.split(".")[-1]
        graphs = [event for event in events if event.get("event") == "compiled_graph"]
        has_target = any(any(name in node for node in event.get("nodes", [])) for event in graphs)
        api_executed = api_executed and has_target and any(
            event.get("event") == "compiled_graph_exit" for event in at_target)

    gradients = {event.get("name") for event in at_target if event.get("event") == "gradient"}
    backward = gradients >= {"input", "output"}
    relevant_actions = [event["name"] for event in events if event.get("event") == "action"
                        and (event.get("phase") == "setup" or
                             (event.get("phase") == "training" and event.get("iteration") == target.iteration))]
    cursor = iter(relevant_actions)
    actions = all(any(observed == expected for observed in cursor) for expected in config.actions)
    snapshot = next((item for item in record.get("snapshots", [])
                     if item.get("checkpoint") == target.checkpoint and item.get("iteration") == target.iteration), None)
    reached = False
    if snapshot is not None:
        reached = (
            representation_matches(snapshot["metadata"], target.representation)
            and snapshot["checkpointing"] == target.checkpointing
            and snapshot["warmup"] == target.warmup and snapshot["backend"] == config.backend
        )
        if target.checkpointing:
            reached = reached and any(event.get("event") == "checkpoint_enter" for event in at_target)
    if target.warmup != "none":
        reached = reached and any(event.get("phase") == "warmup" and event.get("event") == "api_exit"
                                  for event in events)
    evidence = Evidence(target.checkpoint, api_executed, backward, actions, bool(reached),
                        (f"{process.directory}/trace.jsonl",))
    return Execution(execution_id, experiment.experiment_id, role, Outcome(process.outcome),
                     evidence, process.elapsed_seconds)
