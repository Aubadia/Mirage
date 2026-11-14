"""Backend-neutral stateful evidence; unknown is never an affirmative certificate."""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

CONTRACT_VERSION = "stateful-1"
SEMANTIC_COVERAGE_VERSION = "state-semantic-1"
GRADIENT_EVIDENCE_VERSION = "gradient-evidence-1"
GRADIENT_EVIDENCE_MAX_EVENTS = 16
GRADIENT_EVIDENCE_MAX_ELEMENTS = 4096
STATE_DIMENSIONS = ("gradient", "context", "graph_lifecycle", "execution_mode", "representation", "data")
INPUT_CATEGORIES = ("ordinary", "zero", "small_finite", "large_finite", "repeated",
                    "nonuniform", "mixed_sign", "singleton", "broadcast", "supported_layout")


class EvidenceStatus(StrEnum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class Verdict(StrEnum):
    INVALID = "invalid"
    INAPPLICABLE = "inapplicable"
    PASS = "pass"
    SUSPECTED_FAILURE = "suspected_failure"
    UNASSESSED = "unassessed"


@dataclass(frozen=True)
class Evidence:
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    reason: str = "not observed"
    value: Any = None
    checkpoint: str | None = None
    provenance: str = "inspector"

    def __post_init__(self):
        if not isinstance(self.status, EvidenceStatus):
            raise TypeError("Use an explicit EvidenceStatus")
        if self.status == EvidenceStatus.VERIFIED and not self.checkpoint:
            raise ValueError("Verified evidence requires an observation checkpoint")
        if self.provenance not in ("inspector", "adapter", "oracle", "llm", "mutation"):
            raise ValueError("Unknown evidence provenance")

    @property
    def established(self):
        return self.status == EvidenceStatus.VERIFIED and self.provenance != "llm"


@dataclass(frozen=True)
class StatePredicate:
    dimension: str
    field: str
    expected: Any
    checkpoint: str
    call_index: int | None = None
    operand_path: tuple | None = None
    relation: str | None = None

    def __post_init__(self):
        if self.dimension not in STATE_DIMENSIONS or not self.field or not self.checkpoint:
            raise ValueError("State predicate requires a supported dimension, field and checkpoint")

        if self.dimension in ("representation", "data") and (type(self.call_index) is not int or self.call_index < 0 or not self.operand_path):
            raise ValueError("Operand predicates require a call index and operand path")
        if self.dimension not in ("representation", "data") and any(value is not None for value in (self.call_index, self.operand_path, self.relation)):
            raise ValueError("Scoped predicates require representation or data dimensions")
        if self.relation not in (None, "target_backward_node", "operand_gradient_contribution", "output_gradient"):
            raise ValueError("Unsupported path relation")

    def holds(self, observed: Evidence):
        return (observed.established and observed.checkpoint == self.checkpoint
                and type(observed.value) is type(self.expected) and observed.value == self.expected)


@dataclass(frozen=True)
class OperandDescriptor:
    path: tuple
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    object_group: int
    storage_group: int | None = None
    strides: tuple[int, ...] | None = None
    offset: int | None = None
    requires_grad: bool = False
    leaf: bool | None = None
    role: str = "unknown"
    properties: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MutationObligations:
    logical_values: bool = True
    preserve_aliases: bool = True
    preserve_gradient: bool = True
    preserve_identity: bool = False
    writable: bool | None = None


@dataclass(frozen=True)
class MutationCandidate:
    path: tuple
    action: str
    eligibility: Evidence


@dataclass(frozen=True)
class InputCoverage:
    requested: tuple[str, ...] = ()
    observed: dict[str, Evidence] = field(default_factory=dict)
    relation_eligible: dict[str, Evidence] = field(default_factory=dict)

    def __post_init__(self):
        if any(category not in INPUT_CATEGORIES for category in self.requested):
            raise ValueError("Unknown generic input category")


@dataclass(frozen=True)
class RelationSpec:
    name: str
    version: int
    preconditions: tuple[str, ...]
    input_relations: tuple[str, ...]
    observables: tuple[str, ...]
    allowed_differences: tuple[str, ...]
    checkpoint: str
    numerical_policy: str

    def __post_init__(self):
        if not (self.name and self.version > 0 and self.preconditions and self.input_relations
                and self.observables and self.checkpoint and self.numerical_policy):
            raise ValueError("Relations require explicit applicability, observations and numerical policy")

    def missing(self, evidence):
        return tuple(key for key in self.preconditions + self.input_relations
                     if key not in evidence or not evidence[key].established
                     or evidence[key].value is not True
                     or evidence[key].checkpoint != self.checkpoint)


# These are candidate predicates, not claims that each API can reach them.
INITIAL_PREDICATES = (
    StatePredicate("gradient", "backward_count", 2, "final"),
    StatePredicate("context", "grad_enabled", True, "final"),
    StatePredicate("graph_lifecycle", "target_backward_count", 2, "final"),
    StatePredicate("execution_mode", "compiled_execution", True, "final"),
)


@dataclass(frozen=True)
class PathEvidence:
    """A scoped relation; unknown observations never become positive coverage."""
    status: EvidenceStatus
    reason: str
    events: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.status, EvidenceStatus):
            raise TypeError("Use an explicit EvidenceStatus")
        if self.status == EvidenceStatus.VERIFIED and not self.events:
            raise ValueError("Verified paths require runtime event references")

    def record(self):
        return {'status': self.status.value, 'reason': self.reason, 'events': list(self.events)}
