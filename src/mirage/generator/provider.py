"""Model-independent Python interface for local generation providers."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class Stage(StrEnum):
    STATE_GOAL = "state_goal"
    CANONICAL = "canonical"
    TRANSITION = "transition"


@dataclass(frozen=True)
class Usage:
    # None means unavailable; zero means measured and zero.
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for count in (self.input_tokens, self.output_tokens):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError("Token counts must be nonnegative integers or unknown")


@dataclass(frozen=True)
class Request:
    stage: Stage
    prompt: str
    context: dict[str, object] = field(default_factory=dict)
    previous_content: str | None = None
    diagnostic: str | None = None


@dataclass(frozen=True)
class Response:
    content: str
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class ProviderFailure(RuntimeError):
    """A service failure, distinct from an invalid generated program."""

    def __init__(self, category: str, message: str, *, cancellation_uncertain: bool = False):
        super().__init__(message)
        self.category = category
        self.cancellation_uncertain = cancellation_uncertain


class Provider(Protocol):
    def generate(self, request: Request, *, deadline: float | None) -> Response:
        """Generate within an absolute time.monotonic() deadline.

        Implementations must cancel their work and raise TimeoutError when the
        deadline expires. A generic synchronous Python adapter cannot forcibly
        stop an arbitrary implementation; the caller also checks on return.
        """
        ...


def load_provider(specification: str, *, directory=None) -> Provider:
    import importlib

    if specification == "vllm":
        from mirage.generator.http_provider import HTTPProvider

        return HTTPProvider.from_environment(directory=directory)
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Provider must be 'module:factory' or 'vllm'")
    factory = getattr(importlib.import_module(module_name), attribute)
    provider = factory()
    if not callable(getattr(provider, "generate", None)):
        raise TypeError("Provider factory must return an object with generate()")
    return provider


class IsolatedProvider:
    """Enforce deadlines externally even when a provider ignores cancellation."""

    def __init__(self, specification, directory, timeout, *, cancel=None):
        self.specification = specification
        self.directory = directory
        self.timeout = timeout
        self.calls = 0
        self.processes = []
        self.request_timings = []
        self.cancel = cancel

    def generate(self, request: Request, *, deadline: float | None) -> Response:
        from mirage.runtime.scheduler import current_scope, generation_slot
        scope = current_scope()
        try:
            with generation_slot(deadline=deadline, label=self.directory / f'provider-{self.calls + 1:03d}'):
                return self._timed_generate(request, deadline=deadline)
        except ProviderFailure as error:
            if scope and error.cancellation_uncertain:
                scope.scheduler.abort(f'Shared provider cancellation unconfirmed: {error}')
            raise

    def _timed_generate(self, request: Request, *, deadline: float | None) -> Response:
        import json
        import time
        started, before = time.time(), self.calls
        try:
            return self._generate(request, deadline=deadline)
        finally:
            finished = time.time()
            path = self.directory / f"provider-{self.calls:03d}" / "provider-timing.json"
            timing = (json.loads(path.read_text()) if self.calls > before and path.exists() else
                      {'started_epoch': started, 'finished_epoch': finished,
                       'elapsed_seconds': finished - started, 'basis': 'provider_process_upper_bound'})
            self.request_timings.append(timing)

    def _generate(self, request: Request, *, deadline: float | None) -> Response:
        from dataclasses import asdict
        import json
        import time
        from mirage.runtime.process import run_worker

        remaining = self.timeout if deadline is None else min(self.timeout, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Provider budget exhausted")
        self.calls += 1
        directory = self.directory / f"provider-{self.calls:03d}"
        process = run_worker({"mode": "provider", "provider": self.specification,
                              "request": asdict(request), "deadline": time.monotonic() + remaining},
                             directory, remaining, cancel=self.cancel)
        self.processes.append(process)
        if process.outcome == "timeout":
            if self.specification == "vllm":
                raise ProviderFailure("timeout", process.error or "Provider deadline expired",
                                      cancellation_uncertain=True)
            raise TimeoutError("Provider process exceeded its deadline")
        if process.outcome != "completed":
            status_path = directory / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            if "provider_failure" in status:
                failure = status["provider_failure"]
                raise ProviderFailure(failure["category"], process.error or process.outcome,
                                      cancellation_uncertain=failure["cancellation_uncertain"])
            if self.specification == "vllm":
                raise ProviderFailure("process", process.error or process.outcome,
                                      cancellation_uncertain=True)
            raise RuntimeError(process.error or "Provider process failed")
        value = json.loads((directory / "response.json").read_text(encoding="utf-8"))
        value["usage"] = Usage(**value["usage"])
        return Response(**value)
