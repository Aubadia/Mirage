"""Validated exploration resource budgets and execution options."""

from dataclasses import dataclass
import math


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


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
    size: int = 4
    epsilon: float = 0.1
    replay_failures: int = 1
    workers: int = 1
    generation_concurrency: int = 2
    contract_version: int = 3
    canonicals_per_api: int = 3
    targets_per_canonical: int = 12
    paths_per_target: int = 2
    forward_only: bool = False

    def __post_init__(self) -> None:
        if self.contract_version != 3:
            raise ValueError("Unknown generation contract")
        for name in ("canonicals_per_api", "targets_per_canonical", "paths_per_target"):
            positive(getattr(self, name), name)
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
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Expected cpu or cuda")
        if self.backend not in ("eager", "compile_eager", "aot_eager", "inductor"):
            raise ValueError("Unknown backend")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unknown precision")
        positive(self.size, "size")
