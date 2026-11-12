"""Serializable experiment intent and runtime evidence, without torch imports.

These records enforce structural consistency. They do not establish that a
probe is truthful or replace the runtime inspector.
"""

from dataclasses import dataclass
from enum import StrEnum
import math

from mirage.contracts.apis import API_IDS


class Outcome(StrEnum):
    COMPLETED = "completed"
    EXCEPTION = "exception"
    CRASH = "crash"
    TIMEOUT = "timeout"
    OOM = "oom"


class Verdict(StrEnum):
    INVALID = "invalid"
    INAPPLICABLE = "inapplicable"
    PASS = "pass"
    SUSPECTED_FAILURE = "suspected_failure"


def require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def require_check(value: bool | None, name: str) -> None:
    if value is not None and type(value) is not bool:
        raise ValueError(f"{name} must be true, false, or unknown")


@dataclass(frozen=True)
class Candidate:
    api: str
    state: str
    checkpoint: str
    transition: str
    mr: str | None
    backend: str = "eager"
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.api not in API_IDS:
            raise ValueError(f"Unknown API: {self.api}")

        for name in ("state", "checkpoint", "transition"):
            require_text(getattr(self, name), name)

        if self.mr is not None:
            require_text(self.mr, "mr")

        if self.backend not in ("eager", "compile_eager", "aot_eager", "inductor"):
            raise ValueError("Unknown backend")

        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unknown precision")

    @property
    def state_key(self) -> tuple[str, str, str, str]:
        # A state observed at a different checkpoint is a different target.
        return self.state, self.checkpoint, self.backend, self.precision


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    candidate: Candidate
    seed: int
    requires_backward: bool = True

    def __post_init__(self) -> None:
        require_text(self.experiment_id, "experiment_id")
        if not isinstance(self.candidate, Candidate):
            raise ValueError("candidate must be a Candidate")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.requires_backward) is not bool:
            raise ValueError("requires_backward must be boolean")


@dataclass(frozen=True)
class Evidence:
    checkpoint: str
    api_executed: bool | None = None
    backward_executed: bool | None = None
    actions_completed: bool | None = None
    target_state_reached: bool | None = None
    trace_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.checkpoint, "checkpoint")
        for name in ("api_executed", "backward_executed", "actions_completed", "target_state_reached"):
            require_check(getattr(self, name), name)
        if not isinstance(self.trace_refs, tuple):
            raise ValueError("trace_refs must be a tuple")
        for ref in self.trace_refs:
            require_text(ref, "trace reference")

    def verifies(self, experiment: Experiment) -> bool:
        """Require recorded runtime evidence at the requested checkpoint."""
        return (
            bool(self.trace_refs)
            and self.checkpoint == experiment.candidate.checkpoint
            and self.api_executed is True
            and (not experiment.requires_backward or self.backward_executed is True)
            and self.actions_completed is True
            and self.target_state_reached is True
        )


@dataclass(frozen=True)
class Execution:
    execution_id: str
    experiment_id: str
    role: str
    outcome: Outcome
    evidence: Evidence
    elapsed_seconds: float

    def __post_init__(self) -> None:
        require_text(self.execution_id, "execution_id")
        require_text(self.experiment_id, "experiment_id")
        if self.role not in ("source", "followup"):
            raise ValueError("role must be source or followup")
        if not isinstance(self.outcome, Outcome) or not isinstance(self.evidence, Evidence):
            raise ValueError("Invalid outcome or evidence")
        if isinstance(self.elapsed_seconds, bool) or not isinstance(self.elapsed_seconds, (int, float)):
            raise ValueError("elapsed_seconds must be numeric")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")


@dataclass(frozen=True)
class Inspection:
    experiment_id: str
    execution_ids: tuple[str, ...]
    verdict: Verdict | None
    executions_valid: bool | None = None
    preconditions: bool | None = None
    input_relation: bool | None = None
    checkpoints_aligned: bool | None = None
    output_relation: bool | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        require_text(self.experiment_id, "experiment_id")
        if not isinstance(self.execution_ids, tuple) or not self.execution_ids:
            raise ValueError("Inspection requires execution references")
        for execution_id in self.execution_ids:
            require_text(execution_id, "execution_id")
        if len(set(self.execution_ids)) != len(self.execution_ids):
            raise ValueError("Duplicate execution reference")
        if self.verdict is not None and not isinstance(self.verdict, Verdict):
            raise ValueError("Invalid verdict")
        for name in ("executions_valid", "preconditions", "input_relation", "checkpoints_aligned", "output_relation"):
            require_check(getattr(self, name), name)

        if self.verdict in (Verdict.PASS, Verdict.SUSPECTED_FAILURE):
            if not self.applicable:
                raise ValueError("An oracle verdict requires verified applicability")
            expected = self.verdict == Verdict.PASS
            if self.output_relation is not expected:
                raise ValueError("Output relation contradicts verdict")
        elif self.output_relation is not None:
            raise ValueError("Do not evaluate output before applicability is established")

        if self.verdict == Verdict.INVALID and self.executions_valid is not False:
            raise ValueError("Invalid requires established invalidity, not missing evidence")
        if self.verdict == Verdict.INAPPLICABLE:
            conditions = (self.preconditions, self.input_relation, self.checkpoints_aligned)
            if self.executions_valid is not True or False not in conditions:
                raise ValueError("Inapplicable requires a failed MR condition on valid executions")

    @property
    def applicable(self) -> bool:
        return all(value is True for value in (
            self.executions_valid, self.preconditions,
            self.input_relation, self.checkpoints_aligned,
        ))
