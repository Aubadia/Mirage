"""Explicit tolerance policy for the bounded MVP workload (version 1).

These are screening thresholds, not proof of a framework defect. Failure
validation must revisit conditioning and numerical error for the saved case.
"""

from dataclasses import asdict, dataclass

from mirage.contracts.apis import API_IDS


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


def policy(api: str, precision: str, backend: str, rule_id: str) -> dict:
    if api not in API_IDS or rule_id not in ("layout", "storage", "checkpoint", "mutation"):
        raise ValueError("Unknown tolerance context")
    if backend not in ("eager", "compile_eager", "aot_eager", "inductor"):
        raise ValueError("Unknown tolerance backend")
    base = {"fp32": (1e-6, 1e-5), "fp16": (2e-4, 5e-3), "bf16": (2e-3, 4e-2)}[precision]
    reduction = api in API_IDS[:3] or api == API_IDS[4]
    factor = 4 if reduction else 1
    if backend == "inductor":
        factor *= 2
    forward = Tolerance(base[0] * factor, base[1] * factor)
    gradient = Tolerance(forward.atol * 2, forward.rtol * 2)
    # Adam updates operate on FP32 parameters even under autocast.
    parameter = Tolerance(1e-6 * factor, 1e-4 * factor)
    return {
        "version": 1, "api": api, "precision": precision, "backend": backend,
        "mr": rule_id, "nonfinite": "reject", "discrete": "exact",
        "observables": {name: asdict(tolerance) for name, tolerance in (
            ("output", forward), ("loss", forward), ("gradient", gradient),
            ("parameter", parameter), ("optimizer", gradient),
            ("scaler", Tolerance(0, 0)), ("mutation", forward),
        )},
    }
