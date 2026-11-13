"""Bounded generation repair. Framework-failure candidates never enter repair."""

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
import math
import time
from typing import Callable

from mirage.generator.provider import Provider, ProviderFailure, Request, Response


class DiagnosticKind(StrEnum):
    ACCEPTED = "accepted"
    SYNTAX = "syntax"
    CONFIRMED_ILLEGAL = "confirmed_illegal"
    MISSING_PATH = "missing_path"
    UNKNOWN_EXCEPTION = "unknown_exception"
    SUSPECTED_FAILURE = "suspected_failure"
    UNSUPPORTED = "unsupported"
    UNRESOLVED_PLAN = "unresolved_plan"
    STATE_NOT_REACHED = "state_not_reached"
    CONTRACT = "contract_violation"


@dataclass(frozen=True)
class Diagnostic:
    kind: DiagnosticKind
    message: str = ""


@dataclass(frozen=True)
class Attempt:
    request: Request
    response: Response | None
    diagnostic: Diagnostic | None
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    status: str
    attempts: tuple[Attempt, ...]


REPAIRABLE = frozenset((
    DiagnosticKind.CONTRACT,
    DiagnosticKind.SYNTAX,
    DiagnosticKind.CONFIRMED_ILLEGAL,
    DiagnosticKind.MISSING_PATH,
))


def generate(
    provider: Provider,
    request: Request,
    validate: Callable[[str], Diagnostic],
    *,
    max_repairs: int = 2,
    deadline: float | None = None,
    record: Callable[[Attempt], None] | None = None,
) -> GenerationResult:
    """Preserve every attempt, including unknown provider/validator errors.

    The validator must establish illegality before requesting semantic repair.
    A runtime exception by itself is not evidence of an illegal program.
    """
    if type(max_repairs) is not int or not 0 <= max_repairs <= 2:
        raise ValueError("max_repairs must be between zero and two")
    if deadline is not None and (isinstance(deadline, bool) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic timestamp")

    attempts: list[Attempt] = []
    current = deepcopy(request)

    def save(attempt: Attempt) -> None:
        attempts.append(attempt)
        if record is not None:
            record(attempt)

    for index in range(max_repairs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            return GenerationResult("timeout", tuple(attempts))

        started = time.monotonic()
        response = None
        try:
            response = provider.generate(deepcopy(current), deadline=deadline)
            if not isinstance(response, Response) or not isinstance(response.content, str):
                raise TypeError("Provider must return a Response with text content")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Generation deadline expired")
            diagnostic = validate(response.content)
            if not isinstance(diagnostic, Diagnostic) or not isinstance(diagnostic.kind, DiagnosticKind):
                raise TypeError("Validator must return a typed Diagnostic")
        except Exception as error:
            # Preserve unknown failures; automatically repairing them could hide a bug.
            save(Attempt(current, response, None, time.monotonic() - started,
                         f"{type(error).__name__}: {error}"))
            status = "timeout" if isinstance(error, TimeoutError) else "unresolved"
            if isinstance(error, ProviderFailure):
                status = f"provider_{error.category}"
            return GenerationResult(status, tuple(attempts))

        save(Attempt(current, response, diagnostic, time.monotonic() - started))
        # Validation belongs to the same budget, and its evidence is still kept.
        if deadline is not None and time.monotonic() >= deadline:
            return GenerationResult("timeout", tuple(attempts))
        if diagnostic.kind == DiagnosticKind.ACCEPTED:
            return GenerationResult("accepted", tuple(attempts))
        if diagnostic.kind not in REPAIRABLE:
            return GenerationResult(diagnostic.kind.value, tuple(attempts))
        if index == max_repairs:
            return GenerationResult("repair_exhausted", tuple(attempts))

        current = replace(current, previous_content=response.content,
                          diagnostic=diagnostic.message)

    raise AssertionError("Unreachable generation state")


def normalize_source_delimiter(source):
    """Remove only one diagnosed, standalone trailing JSON delimiter.

    No executable statement is added or rewritten. Both raw and normalized source
    are retained by the caller; all other syntax errors remain generation failures.
    """
    import ast
    if not isinstance(source,str): raise TypeError('source must be a Python source string')
    try:
        ast.parse(source)
        return source,None
    except SyntaxError as error:
        lines=source.rstrip().splitlines()
        if (error.msg != "unmatched '}'" or not lines or lines[-1].strip()!='}'
                or error.lineno!=len(lines)):
            raise
        candidate='\n'.join(lines[:-1])+'\n'
        ast.parse(candidate)
        return candidate,{'kind':'trailing_json_delimiter','removed_line':len(lines),
                          'training_statements_added':0,'training_statements_rewritten':0}
