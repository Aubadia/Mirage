"""Reviewed MR families and their evidence requirements, versioned for replay."""

from dataclasses import asdict, dataclass

from mirage.contracts.apis import PROFILES


@dataclass(frozen=True)
class Rule:
    rule_id: str
    allowed_differences: tuple[str, ...]
    preconditions: tuple[str, ...]
    observables: tuple[str, ...]
    version: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


TRAINING_OBSERVABLES = ("loss", "output", "gradient", "parameter", "optimizer", "scaler")
RULES = {
    "layout": Rule("layout", ("representation", "warmup"),
                   ("pure deterministic forward", "equal logical inputs", "equal initial training state"),
                   TRAINING_OBSERVABLES),
    "storage": Rule("storage", ("representation", "warmup"),
                    ("no mutation", "equal logical inputs", "identity-independent computation"),
                    TRAINING_OBSERVABLES),
    "checkpoint": Rule("checkpoint", ("checkpointing",),
                       ("no forward side effects", "preserved RNG", "equal backend and representation"),
                       TRAINING_OBSERVABLES),
    "mutation": Rule("mutation", ("backend", "warmup"),
                     ("nonleaf mutation operand", "equal alias topology", "equal initial training state"),
                     TRAINING_OBSERVABLES + ("mutation",)),
}


def applicable_rules(api: str, backend: str) -> tuple[str, ...]:
    profile = PROFILES[api]
    if profile.mutation:
        return ("mutation",) if backend != "eager" else ()
    if profile.stochastic:
        # Layout changes can change RNG-to-element mapping, even under one seed.
        return ("checkpoint",)
    return ("layout", "storage", "checkpoint")
