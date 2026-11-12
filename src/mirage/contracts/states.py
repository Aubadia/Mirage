"""Versioned, finite state vocabulary shared by planning and inspection."""

from dataclasses import dataclass


REPRESENTATIONS = ("contiguous", "strided", "offset", "clone", "shared")
CHECKPOINTS = ("before_api", "after_backward", "after_update")


@dataclass(frozen=True)
class StateTarget:
    representation: str = "contiguous"
    checkpointing: bool = False
    iteration: int = 0
    checkpoint: str = "before_api"
    warmup: str = "none"

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError("Unknown representation")
        if type(self.checkpointing) is not bool:
            raise ValueError("checkpointing must be boolean")
        if type(self.iteration) is not int or self.iteration < 0:
            raise ValueError("iteration must be nonnegative")
        if self.checkpoint not in CHECKPOINTS:
            raise ValueError("Unknown semantic checkpoint")
        if self.warmup not in ("none", "same", "alternate"):
            raise ValueError("Unknown warm-up history")

    @property
    def key(self) -> str:
        return ":".join((self.representation, str(int(self.checkpointing)),
                         str(self.iteration), self.warmup))

    @classmethod
    def from_key(cls, key: str, checkpoint: str) -> "StateTarget":
        representation, checkpointing, iteration, warmup = key.split(":")
        if checkpointing not in ("0", "1"):
            raise ValueError("Invalid checkpoint flag in state key")
        return cls(representation, checkpointing == "1", int(iteration), checkpoint, warmup)


def transition_actions(target: StateTarget) -> tuple[str, ...]:
    """Record ordered semantic actions; execution must independently verify them."""
    return (
        "restore_initial_state", f"warmup:{target.warmup}", "restore_training_state",
        f"checkpoint:{int(target.checkpointing)}", f"construct:{target.representation}",
        "forward", "backward", "adam_update",
    )
