"""Incremental generation, with runtime feedback after every accepted filling."""

from dataclasses import asdict
import json

from mirage.artifacts.store import json_value
from mirage.generator.prompts import request
from mirage.generator.provider import Stage
from mirage.generator.repair import Diagnostic, DiagnosticKind, generate
from mirage.generator.validation import decode, normalized, validate


def generate_stage(provider, stage, config, directory, deadline, runtime_validate, previous=None, mr=None):
    directory.mkdir(parents=True, exist_ok=False)
    materialized = None
    validation_result = None

    def validator(content):
        nonlocal materialized, validation_result
        diagnostic = validate(content, config.api, config.actions)
        if diagnostic.kind != DiagnosticKind.ACCEPTED:
            return diagnostic
        materialized = decode(content, config.api, config.actions)
        if (previous is not None and stage != Stage.FOLLOWUP
                and normalized(materialized.source) != normalized(previous.source)):
            return Diagnostic(DiagnosticKind.UNSUPPORTED, "Stage changed the canonical computation")
        diagnostic, validation_result = runtime_validate(materialized, config, directory)
        return diagnostic

    def record(attempt):
        index = len(list(directory.glob("attempt-*.json"))) + 1
        (directory / f"attempt-{index:02d}.json").write_text(
            json.dumps(json_value(asdict(attempt)), indent=2, allow_nan=False), encoding="utf-8")

    result = generate(provider, request(stage, config, previous, mr), validator,
                      deadline=deadline, record=record)
    (directory / "generation.json").write_text(
        json.dumps(json_value(asdict(result)), indent=2, allow_nan=False), encoding="utf-8")
    if result.status != "accepted":
        return None, result, validation_result
    (directory / "program.py").write_text(materialized.source, encoding="utf-8")
    return materialized, result, validation_result
