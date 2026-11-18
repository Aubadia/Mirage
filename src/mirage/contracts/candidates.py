"""Portable training candidate contract. Admission is not runtime attainment."""
from dataclasses import asdict, dataclass, field
from typing import Any

from mirage.contracts.capabilities import Evidence, EvidenceStatus

CANDIDATE_VERSION = "training-candidates-1"


def path_key(path):
    import json
    return path[0] if len(path) == 1 and type(path[0]) is str and "." not in path[0] else json.dumps(list(path), separators=(",", ":"))


def tensor_paths(value, is_tensor, path=()):
    if is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            if type(key) not in (str, int):
                raise ValueError("Training input paths require string or integer keys")
            yield from tensor_paths(child, is_tensor, (*path, key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensor_paths(child, is_tensor, (*path, index))


def path_value(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def replace_path(tree, path, value):
    """Copy containers, retaining every unaffected tensor identity."""
    if not path:
        return value
    key, *rest = path
    if isinstance(tree, dict):
        result = dict(tree)
        result[key] = replace_path(tree[key], rest, value)
    else:
        result = list(tree)
        result[key] = replace_path(tree[key], rest, value)
        if isinstance(tree, tuple):
            result = tuple(result)
    return result


@dataclass(frozen=True)
class CapabilitySnapshot:
    backend: str
    checkpoint: str
    provenance: dict
    subjects: tuple[dict, ...]
    facts: dict[str, Evidence]
    version: str = CANDIDATE_VERSION

    def payload(self):
        import json
        # Saved runtime records use weights_only loading: keep evidence as
        # primitive data, without pickled enum/dataclass implementation objects.
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def read(cls, value):
        if value.get("version") != CANDIDATE_VERSION:
            raise ValueError("Unknown training capability version")
        facts = {name: Evidence(**dict(fact, status=EvidenceStatus(fact["status"])))
                 for name, fact in value["facts"].items()}
        return cls(value["backend"], value["checkpoint"], value["provenance"],
                   tuple(value["subjects"]), facts)


@dataclass(frozen=True)
class Candidate:
    identifier: str
    kind: str
    subjects: tuple
    checkpoint: str
    goals: dict
    prerequisites: tuple[str, ...]
    permitted_differences: tuple[str, ...]
    effects: tuple[str, ...]
    required_evidence: tuple[str, ...]
    status: str
    reasons: tuple[str, ...]
    rule: str | None
    configuration: dict
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ("admitted", "rejected", "unresolved"):
            raise ValueError("Unknown candidate admission status")

    def payload(self):
        return asdict(self)
