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
    strategy: str = "training"
    generation_pipeline: bool = False
    representation_traversal: bool = False
    oracle_improvements: bool = False

    def __post_init__(self) -> None:
        for name in ("generation_pipeline", "representation_traversal", "oracle_improvements"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.strategy not in ("training","properties","stateful"):
            raise ValueError("Unknown execution strategy")
        if self.contract_version != (4 if self.strategy == "stateful" else 3):
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


def validate_checkpoint_data(value, is_tensor, path="root"):
    """Portable checkpoint payloads exclude executable functions and live objects.

    Live modules/optimizers belong to ctx; their explicit state_dict payloads can
    use this contract. This does not claim arbitrary context serialization.
    """
    if value is None or type(value) in (bool, int, float, str) or is_tensor(value):
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            validate_checkpoint_data(item, is_tensor, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ValueError(f"Unsupported checkpoint key at {path}")
            validate_checkpoint_data(item, is_tensor, f"{path}[{key!r}]")
        return
    raise ValueError(f"Unsupported checkpoint data at {path}: {type(value).__name__}")


def bind_call(target, args, kwargs):
    """Inspect only trustworthy signatures; absence is not evidence of legality."""
    import inspect
    from mirage.contracts.capabilities import Evidence, EvidenceStatus
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return Evidence(EvidenceStatus.UNKNOWN, "callable signature unavailable")
    try:
        signature.bind(*args, **kwargs)
    except TypeError as error:
        return Evidence(EvidenceStatus.REJECTED, str(error))
    return Evidence(EvidenceStatus.VERIFIED, "signature binding", True, "call_binding")


def initialization_call_kind(name):
    """Classify diagnostics without treating a seed as an initialization certificate."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf in ("manual_seed", "manual_seed_all", "seed"):
        return "seed_setting"
    if leaf in ("rand", "randn", "randint", "rand_like", "randn_like", "randperm",
                "normal", "normal_", "uniform_", "bernoulli", "bernoulli_", "dropout"):
        return "sampling"
    return "unresolved"
