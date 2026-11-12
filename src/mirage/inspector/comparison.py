"""Recursive numerical comparison with explicit finite-value and dtype checks."""

import math


def differences(left, right, tolerance: dict | None = None, path: str = "value") -> list[dict]:
    import torch

    tolerance = tolerance or {"atol": 0.0, "rtol": 0.0}
    if path.endswith(".step") or ".optimizer.param_groups" in path:
        tolerance = {"atol": 0.0, "rtol": 0.0}
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return [{"path": path, "reason": "tensor/type mismatch"}]
        if left.shape != right.shape or left.dtype != right.dtype:
            return [{"path": path, "reason": "shape/dtype mismatch"}]
        if left.is_floating_point() or left.is_complex():
            if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                return [{"path": path, "reason": "nonfinite tensor"}]
            dtype = torch.complex128 if left.is_complex() else torch.float64
            a, b = left.to(dtype), right.to(dtype)
            error = (a - b).abs()
            allowed = tolerance["atol"] + tolerance["rtol"] * b.abs()
            if bool((error <= allowed).all()):
                return []
            return [{"path": path, "reason": "numerical mismatch",
                     "max_absolute_error": float(error.max()), "tolerance": tolerance}]
        return [] if torch.equal(left, right) else [{"path": path, "reason": "discrete tensor mismatch"}]
    if type(left) is not type(right):
        return [{"path": path, "reason": "type mismatch"}]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [{"path": path, "reason": "mapping keys mismatch"}]
        return [item for key in left for item in differences(left[key], right[key], tolerance, f"{path}.{key}")]
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return [{"path": path, "reason": "sequence length mismatch"}]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, tolerance, f"{path}[{index}]")]
    if isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            return [{"path": path, "reason": "nonfinite scalar"}]
        if abs(left - right) <= tolerance["atol"] + tolerance["rtol"] * abs(right):
            return []
    elif left == right:
        return []
    return [{"path": path, "reason": "scalar mismatch", "left": left, "right": right}]


def equal(left, right) -> bool:
    return not differences(left, right)
