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


def make_initial(config) -> dict:
    import torch

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed(config.seed)
    parameter = torch.randn(config.size, config.size) * 0.25 + 0.5
    bias = torch.randn(config.size) * 0.1 + 0.5
    return {"version": 1, "parameter": parameter, "bias": bias,
            "rng": rng_state(config.device), "optimizer": None, "scaler": None,
            "gradient": None}


def training_state(parameter, optimizer, scaler, device: str) -> dict:
    return {"parameter": cpu_copy(parameter), "optimizer": cpu_copy(optimizer.state_dict()),
            "scaler": cpu_copy(scaler.state_dict()), "rng": rng_state(device),
            "gradient": cpu_copy(parameter.grad)}


def restore_training(state, parameter, optimizer, scaler, device: str) -> None:
    import torch

    with torch.no_grad():
        parameter.copy_(state["parameter"].to(device))
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    scaler.load_state_dict(deepcopy(state["scaler"]))
    gradient = state["gradient"]
    parameter.grad = None if gradient is None else gradient.to(device).clone()
    restore_rng(state["rng"], device)
