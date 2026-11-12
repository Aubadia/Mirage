"""Enumerate the finite candidate universe before sampling or counting coverage."""

from dataclasses import replace

from mirage.contracts.models import Candidate
from mirage.contracts.programs import CaseConfig
from mirage.contracts.states import CHECKPOINTS, StateTarget, transition_actions
from mirage.rules.registry import applicable_rules


def candidates(apis: tuple[str, ...], backend: str, precision: str, updates: int) -> tuple[Candidate, ...]:
    if updates <= 0:
        raise ValueError("updates must be positive")
    result = []
    for api in apis:
        rules = applicable_rules(api, backend)
        transformations = [(None, "strided", False), (None, "offset", False)]
        if "layout" in rules:
            transformations += [("layout", "strided", False), ("layout", "offset", False)]
        if "storage" in rules:
            transformations += [("storage", "clone", False), ("storage", "shared", False)]
        if "checkpoint" in rules:
            transformations += [("checkpoint", "contiguous", True)]
        if "mutation" in rules:
            transformations += [("mutation", rep, False) for rep in ("contiguous", "strided", "offset", "shared")]
        histories = ("none",) if backend == "eager" else ("none", "same", "alternate")
        for mr, representation, checkpointing in transformations:
            for iteration in sorted({0, updates - 1}):
                for checkpoint in CHECKPOINTS:
                    for warmup in histories:
                        target = StateTarget(representation, checkpointing, iteration, checkpoint, warmup)
                        result.append(Candidate(api, target.key, checkpoint,
                                                ">".join(transition_actions(target)), mr, backend, precision))
    return tuple(result)


def pair_configs(candidate: Candidate, base: CaseConfig) -> tuple[CaseConfig, CaseConfig]:
    target = StateTarget.from_key(candidate.state, candidate.checkpoint)
    followup = replace(base, api=candidate.api, backend=candidate.backend,
                       precision=candidate.precision, target=target)
    source_target = target
    source_backend = candidate.backend
    if candidate.mr in ("layout", "storage"):
        source_target = replace(target, representation="contiguous", warmup="none")
    elif candidate.mr == "checkpoint":
        source_target = replace(target, checkpointing=False)
    elif candidate.mr == "mutation":
        source_backend = "eager"
        source_target = replace(target, warmup="none")
    return replace(followup, backend=source_backend, target=source_target), followup
