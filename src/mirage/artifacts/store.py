"""Versioned, immutable JSON manifests with atomic publication.

Tensor payloads and executable replay bundles belong to the runtime batch.
This store deliberately only accepts portable JSON data.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

SCHEMA_VERSION = 1


def json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def validate_manifest(manifest: Any) -> dict[str, Any]:
    required = {"schema_version", "artifact_id", "kind", "created_at", "parent_id", "payload"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Invalid manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema version")
    validate_id(manifest["artifact_id"])
    if manifest["parent_id"] is not None:
        validate_id(manifest["parent_id"])
        if manifest["parent_id"] == manifest["artifact_id"]:
            raise ValueError("Artifact cannot be its own parent")
    if manifest["kind"] not in ("experiment", "generation", "execution", "inspection", "report"):
        raise ValueError("Unknown artifact kind")
    if not isinstance(manifest["payload"], dict):
        raise ValueError("payload must be an object")
    if not isinstance(manifest["created_at"], str):
        raise ValueError("created_at must be an ISO timestamp")
    try:
        stamp = datetime.fromisoformat(manifest["created_at"])
    except ValueError as error:
        raise ValueError("Invalid created_at timestamp") from error
    if stamp.tzinfo is None:
        raise ValueError("created_at requires timezone information")
    # Also reject non-finite values when validation is called before writing.
    json.dumps(manifest, allow_nan=False)
    return manifest


def validate_id(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Artifact ID must be a simple filename identifier")


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def read(self, artifact_id: str) -> dict[str, Any]:
        validate_id(artifact_id)
        path = self.root / f"{artifact_id}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant)
        validate_manifest(manifest)
        if manifest["artifact_id"] != artifact_id:
            raise ValueError("Manifest ID does not match filename")
        return manifest

    def write(self, artifact_id: str, kind: str, payload: dict[str, Any], *, parent_id: str | None = None) -> Path:
        manifest = validate_manifest({
            "schema_version": SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "kind": kind,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent_id": parent_id,
            "payload": json_value(payload),
        })
        if parent_id is not None:
            self.read(parent_id)

        encoded = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"{artifact_id}.json"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root,
                                             prefix=".pending-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())

            # Hard-link publication is atomic and fails if the ID already exists.
            # Unlike replace(), it cannot silently overwrite original evidence.
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    def list_manifests(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            raise ValueError(f"Artifact directory does not exist: {self.root}")
        return [self.read(path.stem) for path in sorted(self.root.glob("*.json"))]
