"""Portable execution specifications; importing these records needs no PyTorch."""

from dataclasses import dataclass, field
import math

from mirage.contracts.apis import API_IDS
from mirage.contracts.states import StateTarget, transition_actions


def positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Program:
    source: str
    actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Program source must be nonempty")
        if not isinstance(self.actions, tuple) or any(not isinstance(a, str) for a in self.actions):
            raise ValueError("Program actions must be a tuple of strings")

    @classmethod
    def from_dict(cls, value: dict) -> "Program":
        if set(value) != {"source", "actions"}:
            raise ValueError("Program requires source and actions")
        if not isinstance(value["actions"], list):
            raise ValueError("actions must be a JSON array")
        return cls(value["source"], tuple(value["actions"]))


@dataclass(frozen=True)
class CaseConfig:
    api: str
    seed: int = 0
    device: str = "cpu"
    backend: str = "eager"
    precision: str = "fp32"
    size: int = 4
    updates: int = 3
    warmup_steps: int = 1
    target: StateTarget = field(default_factory=StateTarget)

    def __post_init__(self) -> None:
        if self.api not in API_IDS:
            raise ValueError("Unknown API")
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 63:
            raise ValueError("seed must fit a nonnegative signed 64-bit integer")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda (one visible GPU)")
        if self.backend not in ("eager", "compile_eager", "aot_eager", "inductor"):
            raise ValueError("Unknown backend")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unknown precision")
        for name in ("size", "updates", "warmup_steps"):
            positive(getattr(self, name), name)
        if self.size < 2:
            raise ValueError("size must be at least two to distinguish strides")
        if not isinstance(self.target, StateTarget) or self.target.iteration >= self.updates:
            raise ValueError("Target iteration must occur within training")

    @classmethod
    def from_dict(cls, value: dict) -> "CaseConfig":
        value = dict(value)
        value["target"] = StateTarget(**value["target"])
        return cls(**value)

    @property
    def actions(self) -> tuple[str, ...]:
        return transition_actions(self.target)


@dataclass(frozen=True)
class RunOptions:
    output: str
    provider: str
    cases: int = 100
    time_limit: float = 14400
    worker_timeout: float = 600
    seed: int = 0
    device: str = "cpu"
    backend: str = "eager"
    precision: str = "fp32"
    updates: int = 3
    size: int = 4
    epsilon: float = 0.1
    warmup_steps: int = 1
    replay_failures: int = 1
    workers: int = 1
    generation_concurrency: int = 2

    def __post_init__(self) -> None:
        positive(self.cases, "cases")
        positive(self.workers, "workers")
        positive(self.generation_concurrency, "generation_concurrency")
        for name in ("time_limit", "worker_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.output or not self.provider:
            raise ValueError("output and provider are required")
        if isinstance(self.epsilon, bool) or not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if type(self.replay_failures) is not int or self.replay_failures < 0:
            raise ValueError("replay_failures must be nonnegative")
        CaseConfig(API_IDS[0], self.seed, self.device, self.backend, self.precision,
                   self.size, self.updates, self.warmup_steps)
