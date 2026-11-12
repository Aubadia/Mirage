"""Model-independent Python interface for local generation providers."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class Stage(StrEnum):
    CANONICAL = "canonical"
    SKELETON = "skeleton"
    TRANSITION = "transition"
    FOLLOWUP = "followup"


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


class Provider(Protocol):
    def generate(self, request: Request, *, deadline: float | None) -> Response:
        """Generate within an absolute time.monotonic() deadline.

        Implementations must cancel their work and raise TimeoutError when the
        deadline expires. A generic synchronous Python adapter cannot forcibly
        stop an arbitrary implementation; the caller also checks on return.
        """
        ...
