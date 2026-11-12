"""The approved API pool; framework-specific legality profiles come later."""

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


def select_apis(values: list[str] | None = None) -> tuple[str, ...]:
    """Validate a CLI subset, preserving order and removing duplicates."""
    if values is None:
        return API_IDS

    if not values:
        raise ValueError("At least one API is required")

    unknown = set(values).difference(API_IDS)
    if unknown:
        raise ValueError(f"Unknown API(s): {', '.join(sorted(unknown))}")

    return tuple(dict.fromkeys(values))
