"""Acceptance examples and open public API path validation."""

API_IDS = (
    "torch.matmul",
    "torch.nn.functional.linear",
    "torch.nn.functional.layer_norm",
    "torch.nn.functional.gelu",
    "torch.nn.functional.softmax",
    "torch.nn.functional.dropout",
    "Tensor.add_",
    "Tensor.mul_",
)

ACCEPTANCE_APIS = API_IDS + (
    "torch.addmm", "torch.bmm", "torch.nn.functional.conv2d", "torch.sum",
    "torch.mean", "torch.gather", "torch.max", "torch.nn.functional.log_softmax",
)


def validate_api_name(api: str) -> str:
    """Validate a public path without importing the tested framework on login nodes."""
    parts = api.split(".") if isinstance(api, str) else []
    if len(parts) < 2 or parts[0] not in ("torch", "Tensor"):
        raise ValueError("Expected a torch.* or Tensor.* API path")
    if any(not part.isidentifier() or part.startswith("_") for part in parts):
        raise ValueError("Expected a public API path")
    return api


def select_apis(values: list[str] | None = None) -> tuple[str, ...]:
    """Validate a CLI subset, preserving order and removing duplicates."""
    if values is None:
        return ACCEPTANCE_APIS

    if not values:
        raise ValueError("At least one API is required")

    for value in values:
        validate_api_name(value)

    return tuple(dict.fromkeys(values))
