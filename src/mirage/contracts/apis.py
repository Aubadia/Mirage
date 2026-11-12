"""The approved API pool and the controlled training-program grammar."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class APIProfile:
    api: str
    expression: str
    mutation: bool = False
    stochastic: bool = False
    uses_weight: bool = False
    input_constraint: str = "square rank-two FP32 trainable input; size >= 2"
    backward_required: bool = True

    @property
    def source(self) -> str:
        return (
            "def forward(x, weight, bias):\n"
            f"    return {self.expression}\n\n\n"
            "def training_loss(output):\n"
            "    return output.float().square().mean()\n"
        )


PROFILES = {
    p.api: p for p in (
        APIProfile(API_IDS[0], "torch.matmul(x, weight)", uses_weight=True),
        APIProfile(API_IDS[1], "torch.nn.functional.linear(x, weight, bias)", uses_weight=True),
        APIProfile(API_IDS[2], "torch.nn.functional.layer_norm(x, (x.shape[-1],))"),
        APIProfile(API_IDS[3], "torch.nn.functional.gelu(x)"),
        APIProfile(API_IDS[4], "torch.nn.functional.softmax(x, dim=-1)"),
        APIProfile(API_IDS[5], "torch.nn.functional.dropout(x, p=0.25, training=True)", stochastic=True),
        APIProfile(API_IDS[6], "x.add_(bias)", mutation=True),
        APIProfile(API_IDS[7], "x.mul_(bias)", mutation=True),
    )
}


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
