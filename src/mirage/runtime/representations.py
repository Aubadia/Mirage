"""Record within-execution tensor metadata and storage relationships."""


def metadata(tensors: dict) -> dict:
    import torch

    groups = {}
    result = {}
    for name, tensor in tensors.items():
        if tensor.layout != torch.strided or tensor.is_mkldnn:
            result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "device": str(tensor.device), "layout": str(tensor.layout),
                            "requires_grad": tensor.requires_grad, "storage_observable": False}
            if tensor.layout == torch.sparse_csr:
                result[name]["values"] = metadata({"values": tensor.values()})["values"]
            continue
        identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
        group = groups.setdefault(identity, len(groups))
        result[name] = {
            "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "offset": tensor.storage_offset(), "contiguous": tensor.is_contiguous(),
            "storage_group": group, "dtype": str(tensor.dtype), "device": str(tensor.device),
            "requires_grad": tensor.requires_grad, "is_leaf": tensor.is_leaf,
            "has_base": tensor._base is not None, "version": tensor._version,
            "grad_fn": type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None,
            "conj": tensor.is_conj(), "neg": tensor.is_neg(), "layout": str(tensor.layout),
        }
    return result


def equivalent_view(tensor, representation):
    """Return a logical-value-preserving representation, or reject applicability."""
    import torch
    if tensor.layout != torch.strided or tensor.is_mkldnn:
        raise ValueError('Representation action requires a dense strided tensor')
    if representation == 'original': return tensor
    if representation == 'strided':
        storage = torch.zeros((*tensor.shape,2),dtype=tensor.dtype,device=tensor.device)
        result=storage[...,0]
        result.copy_(tensor.detach())
    elif representation == 'offset':
        storage=torch.zeros(tensor.numel()+1,dtype=tensor.dtype,device=tensor.device)
        result=storage[1:].view(tensor.shape)
        result.copy_(tensor.detach())
    elif representation == 'conjugate':
        if not tensor.is_complex(): raise ValueError('Conjugate representation requires complex values')
        result=tensor.detach().conj_physical().conj()
    elif representation == 'negative':
        if tensor.dtype not in (torch.float32,torch.float64): raise ValueError('Negative representation requires real FP32/FP64')
        result=torch.complex(torch.zeros_like(tensor),-tensor.detach()).conj().imag
    elif representation == 'singleton_stride':
        axes=[i for i,n in enumerate(tensor.shape) if n==1]
        if not axes: raise ValueError('No singleton dimension')
        strides=list(tensor.stride()); strides[axes[0]]=1
        result=tensor.detach().as_strided(tensor.shape,strides,tensor.storage_offset())
    elif representation == 'expanded':
        axes=[i for i,n in enumerate(tensor.shape) if n>1]
        if not axes: raise ValueError('No expandable dimension')
        axis=axes[-1]
        result=tensor.detach().narrow(axis,0,1).expand_as(tensor)
        if not torch.equal(result,tensor): raise ValueError('Expansion would change logical values')
    else: raise ValueError('Unknown representation action')
    result.requires_grad_(tensor.requires_grad)
    if not torch.equal(result,tensor): raise ValueError('Representation changed logical values')
    return result


def connected_representation(tensor, action):
    """Stateful transformations retain the source differentiation coordinates.

    Unlike the legacy fresh-leaf property path, copies here participate in the
    original graph. Expanded independent coordinates are rejected by eligibility.
    """
    import torch
    if tensor.is_mkldnn and action == "logical_materialize":
        return tensor.to_dense()
    if tensor.layout == torch.sparse_csr and action.startswith("values_"):
        values = connected_representation(tensor.values(), action.removeprefix("values_"))
        return torch.sparse_csr_tensor(tensor.crow_indices(), tensor.col_indices(), values,
                                       size=tensor.shape, dtype=tensor.dtype, device=tensor.device)
    if tensor.layout != torch.strided:
        raise ValueError("Unsupported representation layout")
    if action == "clone": return tensor.clone()
    if action == "canonical": return tensor.resolve_conj().resolve_neg().clone(memory_format=torch.contiguous_format)
    if action == "strided":
        result = torch.zeros((*tensor.shape, 2), dtype=tensor.dtype, device=tensor.device)[..., 0]
        return result.copy_(tensor)
    if action == "offset":
        result = torch.zeros(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)[1:].view(tensor.shape)
        return result.copy_(tensor)
    if action == "conjugate": return tensor.conj_physical().conj()
    if action == "negative": return torch.complex(torch.zeros_like(tensor), -tensor).conj().imag
    if action == "singleton_stride":
        axis = next(i for i, n in enumerate(tensor.shape) if n == 1)
        strides = list(tensor.stride())
        strides[axis] = max(1, strides[axis] + 1)
        return tensor.as_strided(tensor.shape, strides, tensor.storage_offset())
    if action == "expanded":
        for axis, size in enumerate(tensor.shape):
            if size > 1:
                candidate = tensor.narrow(axis, 0, 1).expand_as(tensor)
                if torch.equal(candidate, tensor): return candidate
        raise ValueError("No value-preserving expansion")
    raise ValueError(f"Unsupported action: {action}")
