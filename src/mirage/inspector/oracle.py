"""Conditional MR oracle; no numerical comparison before applicability is proven."""

from mirage.contracts.models import Inspection, Outcome, Verdict
from mirage.inspector.comparison import differences
from mirage.rules.registry import RULES
from mirage.rules.relations import conditions
from mirage.rules.tolerances import policy


def inspect(experiment, executions, configs, records, saved_policy=None):
    mr = experiment.candidate.mr
    if mr is None:
        return None, {"correctness": "unassessed", "reason": "coverage-only case"}
    ids = tuple(item.execution_id for item in executions)
    if len(executions) != 2 or any(item.outcome != Outcome.COMPLETED for item in executions) or any(r is None for r in records):
        return Inspection(experiment.experiment_id, ids, None, reason="Incomplete execution or missing evidence"), {}

    valid = all(item.evidence.api_executed is True and item.evidence.actions_completed is True
                and (not experiment.requires_backward or item.evidence.backward_executed is True)
                and item.evidence.target_state_reached is True for item in executions)
    if not valid:
        return Inspection(experiment.experiment_id, ids, Verdict.INVALID, False,
                          reason="Required API, backward, transition or target state was not verified"), {}
    pre, relation, aligned, reasons = conditions(mr, *configs, *records)
    if not all((pre, relation, aligned)):
        return Inspection(experiment.experiment_id, ids, Verdict.INAPPLICABLE, True,
                          pre, relation, aligned, reason="; ".join(reasons)), {}

    expected_policy = policy(experiment.candidate.api, experiment.candidate.precision,
                             experiment.candidate.backend, mr)
    if saved_policy is not None and saved_policy != expected_policy:
        raise ValueError("Saved tolerance policy is unsupported or was modified")
    tolerances = saved_policy or expected_policy
    mismatches = []
    for left, right in zip(records[0]["observations"], records[1]["observations"]):
        for observable in RULES[mr].observables:
            mismatches.extend(differences(left[observable], right[observable],
                                          tolerances["observables"][observable],
                                          f"iteration[{left['iteration']}].{observable}"))
        if mr == "checkpoint":
            mismatches.extend(differences(left["rng_after"], right["rng_after"], path=f"iteration[{left['iteration']}].rng"))
    verdict = Verdict.SUSPECTED_FAILURE if mismatches else Verdict.PASS
    inspection = Inspection(experiment.experiment_id, ids, verdict, True, True, True, True,
                            not mismatches, "Output relation violated" if mismatches else "Output relation holds")
    return inspection, {"tolerance_policy": tolerances, "differences": mismatches,
                        "confirmed_defect": False}
