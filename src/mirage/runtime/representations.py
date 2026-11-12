"""Differentiable representation changes with explicit within-run storage relations."""


def construct(parameter, representation: str):
    import torch

    # The mutable API operand is never a leaf or a view of the trainable leaf.
    anchor = parameter * 1.0
    if representation in ("contiguous", "shared"):
        value = anchor
    elif representation == "strided":
        value = anchor.t().contiguous().t()
    elif representation == "offset":
        prefix = torch.zeros((1, parameter.shape[1]), device=parameter.device, dtype=parameter.dtype)
        value = torch.cat((prefix, anchor), dim=0)[1:]
    elif representation == "clone":
        value = anchor.clone()
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    weight = value.detach() if representation == "shared" else value.detach().clone()
    return value, weight, anchor


def metadata(tensors: dict) -> dict:
    groups = {}
    result = {}
    for name, tensor in tensors.items():
        identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
        group = groups.setdefault(identity, len(groups))
        result[name] = {
            "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "offset": tensor.storage_offset(), "contiguous": tensor.is_contiguous(),
            "storage_group": group, "dtype": str(tensor.dtype), "device": str(tensor.device),
            "requires_grad": tensor.requires_grad, "is_leaf": tensor.is_leaf,
            "has_base": tensor._base is not None, "version": tensor._version,
            "grad_fn": type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None,
        }
    return result


def representation_matches(observed: dict, representation: str) -> bool:
    x, weight, anchor = (observed[name] for name in ("x", "weight", "anchor"))
    if representation == "strided":
        return not x["contiguous"] and x["offset"] == 0
    if representation == "offset":
        return x["contiguous"] and x["offset"] > 0
    if representation == "clone":
        return x["contiguous"] and x["storage_group"] != anchor["storage_group"]
    if representation == "shared":
        return x["storage_group"] == weight["storage_group"]
    return x["contiguous"] and x["offset"] == 0 and x["storage_group"] == anchor["storage_group"]
