"""Dynamic MR conditions, evaluated only on valid controlled executions."""

from dataclasses import asdict

from mirage.contracts.apis import PROFILES
from mirage.inspector.comparison import equal


def conditions(rule_id, source_config, followup_config, source, followup) -> tuple[bool, bool, bool, list[str]]:
    reasons = []
    profile = PROFILES[source_config.api]
    preconditions = source.get("grammar_verified") is True and followup.get("grammar_verified") is True
    preconditions = preconditions and equal(source["initial"], source["restored"])
    preconditions = preconditions and equal(followup["initial"], followup["restored"])
    preconditions = preconditions and equal(source["environment"], followup["environment"])
    if rule_id in ("layout", "storage"):
        preconditions = preconditions and not profile.mutation and not profile.stochastic
    elif rule_id == "checkpoint":
        preconditions = preconditions and not profile.mutation
        preconditions = preconditions and not source_config.target.checkpointing and followup_config.target.checkpointing
    elif rule_id == "mutation":
        preconditions = preconditions and profile.mutation and source_config.backend == "eager"
        preconditions = preconditions and followup_config.backend != "eager"
    else:
        raise ValueError("Unknown MR")

    a, b = asdict(source_config), asdict(followup_config)
    for item in (a, b):
        if rule_id in ("layout", "storage"):
            item["target"].pop("representation")
            item["target"].pop("warmup")
        elif rule_id == "checkpoint":
            item["target"].pop("checkpointing")
        elif rule_id == "mutation":
            item.pop("backend")
            item["target"].pop("warmup")
    input_relation = a == b and equal(source["initial"], followup["initial"])
    input_relation = input_relation and source["program_ast"] == followup["program_ast"]
    input_relation = input_relation and equal(source["bias"], followup["bias"])
    left, right = source["observations"], followup["observations"]
    input_relation = input_relation and bool(left and right)
    if left and right:
        input_relation = input_relation and equal(left[0]["inputs"], right[0]["inputs"])
        input_relation = input_relation and equal(left[0]["rng_before"], right[0]["rng_before"])
        for name in ("x", "weight"):
            first, second = left[0]["input_metadata"][name], right[0]["input_metadata"][name]
            input_relation = input_relation and all(first[key] == second[key] for key in ("shape", "dtype", "device"))
        if rule_id in ("mutation", "checkpoint"):
            input_relation = input_relation and equal(left[0]["input_metadata"], right[0]["input_metadata"])
        if rule_id == "mutation":
            preconditions = preconditions and not left[0]["input_metadata"]["x"]["is_leaf"]
            preconditions = preconditions and not right[0]["input_metadata"]["x"]["is_leaf"]

    # Compare post-update states only when successful update counts align.
    keys = lambda records: [(item["iteration"], item["attempted_update"], item["successful_updates"], item["skipped"])
                            for item in records]
    aligned = bool(left) and keys(left) == keys(right) and len(left) == source_config.updates
    if not preconditions:
        reasons.append("MR preconditions failed (purity, restoration, environment or mode)")
    if not input_relation:
        reasons.append("Initial values, RNG, metadata or permitted configuration differences do not match")
    if not aligned:
        reasons.append("Semantic checkpoints or successful optimizer updates cannot align")
    return bool(preconditions), bool(input_relation), bool(aligned), reasons
