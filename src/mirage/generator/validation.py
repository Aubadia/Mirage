"""Validate a deliberately small, inspectable program grammar.

The MVP varies state transitions and memory representations. Forward programs
must preserve the profiled computation; arbitrary Python cannot establish MR
purity from syntax alone and is not admitted to this controlled harness.
"""

import ast
import json

from mirage.contracts.apis import PROFILES
from mirage.contracts.programs import Program
from mirage.generator.repair import Diagnostic, DiagnosticKind


class UnsupportedProgram(ValueError):
    """A program is outside the supported grammar, not necessarily illegal PyTorch."""


def normalized(source: str) -> str:
    tree = ast.parse(source)
    # Comments and a function's explanatory docstring do not change the grammar.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def validate_source(source: str, api: str) -> None:
    actual = ast.parse(source)
    expected = ast.parse(PROFILES[api].source)
    # Let canonical generation explore documented operator options while the
    # surrounding forward/loss grammar remains inspectable and side-effect free.
    allowed = {
        "torch.nn.functional.dropout": {"p": lambda v: type(v) in (int, float) and 0 < v < 1},
        "torch.nn.functional.softmax": {"dim": lambda v: type(v) is int and v in (-2, -1, 0, 1)},
        "torch.nn.functional.gelu": {"approximate": lambda v: v in ("none", "tanh")},
        "torch.nn.functional.layer_norm": {"eps": lambda v: type(v) in (int, float) and 1e-8 <= v <= 1e-3},
    }.get(api, {})
    forward = next((node for node in actual.body if isinstance(node, ast.FunctionDef) and node.name == "forward"), None)
    returned = [node for node in forward.body if isinstance(node, ast.Return)] if forward else []
    if len(returned) == 1 and isinstance(returned[0].value, ast.Call):
        call = returned[0].value
        expected_call = expected.body[0].body[0].value
        for keyword in call.keywords:
            if keyword.arg not in allowed:
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                raise UnsupportedProgram("Operator options must be literal values") from None
            if not allowed[keyword.arg](value):
                raise UnsupportedProgram(f"Unsupported operator option: {keyword.arg}")
            existing = next((k for k in expected_call.keywords if k.arg == keyword.arg), None)
            if existing is None:
                expected_call.keywords.append(keyword)
            else:
                existing.value = keyword.value
    if normalized(source) != normalized(ast.unparse(expected)):
        raise UnsupportedProgram("Source does not match the profiled pure forward/loss grammar")


def decode(content: str, api: str, actions: tuple[str, ...]) -> Program:
    value = json.loads(content)
    program = Program.from_dict(value)
    validate_source(program.source, api)
    if program.actions != actions:
        raise UnsupportedProgram("Action sequence differs from the requested legal transition plan")
    return program


def validate(content: str, api: str, actions: tuple[str, ...]) -> Diagnostic:
    try:
        decode(content, api, actions)
    except (SyntaxError, json.JSONDecodeError) as error:
        return Diagnostic(DiagnosticKind.SYNTAX, str(error))
    except (TypeError, KeyError, ValueError) as error:
        # An unsupported candidate is not evidence that the framework rejected legal code.
        return Diagnostic(DiagnosticKind.UNSUPPORTED, str(error))
    return Diagnostic(DiagnosticKind.ACCEPTED)
