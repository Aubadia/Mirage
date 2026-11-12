"""Record within-execution tensor metadata and storage relationships."""


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
