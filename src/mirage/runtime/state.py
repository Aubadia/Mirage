"""Capture actual initial values and RNG state, not only seeds."""

from copy import deepcopy
import random


def cpu_copy(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return deepcopy(value)


def rng_state(device: str) -> dict:
    import torch

    return {"python": random.getstate(), "cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state().cpu() if device == "cuda" else None}


def restore_rng(state: dict, device: str) -> None:
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"])
    if device == "cuda":
        if state["cuda"] is None:
            raise ValueError("Saved initial state has no CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"])
