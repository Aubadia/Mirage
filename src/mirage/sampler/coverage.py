"""Coverage over an explicit candidate universe, never a theoretical space."""

from collections import Counter

from mirage.contracts.models import Candidate, Execution, Experiment, Inspection, Outcome


class Coverage:
    def __init__(self, candidates: tuple[Candidate, ...]) -> None:
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("Candidate universe must be nonempty and unique")

        self.candidates = tuple(candidates)
        self.counts: Counter[Candidate] = Counter()
        self._observed: set[Candidate] = set()
        self._api_observed: set[str] = set()
        self._transitions_observed: set[str] = set()
        self._applied: set[Candidate] = set()
        self._experiments: set[str] = set()
        self._verdicts: Counter[str] = Counter()
        self._applicability: Counter[str] = Counter()
        self.execution_seconds = 0.0

    def observe_api(self, api: str, execution: Execution) -> None:
        """Retain canonical/incremental API evidence even if later generation fails."""
        if api not in {candidate.api for candidate in self.candidates}:
            raise ValueError("API is outside the declared universe")
        if execution.evidence.trace_refs and execution.evidence.api_executed is True:
            self._api_observed.add(api)

    def update(
        self,
        experiment: Experiment,
        executions: tuple[Execution, ...],
        inspection: Inspection | None = None,
    ) -> None:
        """Accept one final observation per experiment, not polling snapshots.

        Only the follow-up is required to reach the sampled target predicate.
        Source and follow-up may deliberately have different representations.
        """
        candidate = experiment.candidate
        if candidate not in self.candidates:
            raise ValueError("Candidate is outside the declared universe")
        if experiment.experiment_id in self._experiments:
            raise ValueError("Experiment was already recorded")
        if any(item.experiment_id != experiment.experiment_id for item in executions):
            raise ValueError("Execution belongs to another experiment")

        ids = {item.execution_id for item in executions}
        roles = {item.role for item in executions}
        if len(ids) != len(executions) or len(roles) != len(executions):
            raise ValueError("Duplicate execution ID or role")

        if inspection is not None:
            if inspection.experiment_id != experiment.experiment_id:
                raise ValueError("Inspection belongs to another experiment")
            if set(inspection.execution_ids) != ids:
                raise ValueError("Inspection must reference exactly the supplied executions")
            if candidate.mr is None:
                raise ValueError("Coverage-only experiments cannot have an MR inspection")
            if inspection.applicable:
                if roles != {"source", "followup"}:
                    raise ValueError("Applicable MR requires both pair members")
                if any(item.outcome != Outcome.COMPLETED for item in executions):
                    raise ValueError("Applicable MR requires completed executions")
                for item in executions:
                    evidence = item.evidence
                    if (not evidence.trace_refs or evidence.api_executed is not True
                            or evidence.actions_completed is not True
                            or (experiment.requires_backward and evidence.backward_executed is not True)):
                        raise ValueError("Applicable MR requires path evidence for both executions")

        reached = any(
            item.role == "followup"
            and item.outcome == Outcome.COMPLETED
            and item.evidence.verifies(experiment)
            for item in executions
        )
        if inspection is not None and inspection.applicable and not reached:
            raise ValueError("Applicable MR cannot contradict the required target state")

        # Commit only after validation, so a rejected update cannot partially count.
        self._experiments.add(experiment.experiment_id)
        self.execution_seconds += sum(item.elapsed_seconds for item in executions)
        for item in executions:
            if item.evidence.trace_refs and item.evidence.api_executed is True:
                self._api_observed.add(candidate.api)
            if (item.role == "followup" and item.evidence.trace_refs
                    and item.evidence.actions_completed is True):
                self._transitions_observed.add(candidate.transition)
        if reached:
            self._observed.add(candidate)
            if candidate.mr is None:
                self.counts[candidate] += 1

        if inspection is not None and inspection.applicable:
            self._applied.add(candidate)
            self.counts[candidate] += 1

        verdict = inspection.verdict.value if inspection and inspection.verdict else "unassessed"
        self._verdicts[verdict] += 1
        if candidate.mr is not None:
            conditions = () if inspection is None else (
                inspection.preconditions, inspection.input_relation, inspection.checkpoints_aligned,
            )
            status = "unknown"
            if inspection is not None and inspection.applicable:
                status = "applicable"
            elif inspection is not None and inspection.executions_valid is True and False in conditions:
                status = "inapplicable"
            self._applicability[status] += 1

    def report(self) -> dict[str, object]:
        def metric(observed: set, candidates: set) -> dict[str, object]:
            return {
                "covered": len(observed),
                "denominator": len(candidates),
                "fraction": len(observed) / len(candidates) if candidates else None,
            }

        universe = set(self.candidates)
        mr_universe = {c for c in universe if c.mr is not None}
        return {
            "denominator_basis": "declared_candidates_not_proven_feasible",
            "experiments": len(self._experiments),
            "api": metric(self._api_observed, {c.api for c in universe}),
            "state": metric({c.state_key for c in self._observed}, {c.state_key for c in universe}),
            "transition": metric(self._transitions_observed, {c.transition for c in universe}),
            "api_mr": metric({(c.api, c.mr) for c in self._applied}, {(c.api, c.mr) for c in mr_universe}),
            "combinations": metric(set(self.counts), universe),
            "applicability": dict(self._applicability),
            "applicability_denominator": sum(self._applicability.values()),
            "applicability_fractions": {
                key: count / sum(self._applicability.values()) for key, count in self._applicability.items()
            },
            "verdicts": dict(self._verdicts),
            "cost": {"execution_seconds": self.execution_seconds, "generation_tokens": None},
        }
