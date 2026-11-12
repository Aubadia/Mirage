"""Content-verified replay payloads alongside immutable JSON manifests."""

import hashlib
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def snapshot_engine(destination: Path) -> None:
    """Save the small Python execution engine so future replay uses the same code."""
    source = Path(__file__).resolve().parents[1]
    for path in source.rglob("*.py"):
        target = destination / "mirage" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def verify(root: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid bundle member: {relative}")
        if digest(path) != expected:
            raise ValueError(f"Bundle checksum mismatch: {relative}")


def load_tensor_record(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)
