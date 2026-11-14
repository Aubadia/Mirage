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


class ReconstructionError(ValueError):
    """Recorded arguments cannot be assigned an unambiguous bounded coordinate chart."""


class StorageCoordinateChart:
    """Rebuild dense arguments from independent occupied storage coordinates.

    This is a linear representation map, not an operator derivative. Cotangents
    are pulled back by summing their signed contributions to shared coordinates.
    Unobserved storage holes are never perturbed or treated as observed values.
    """
    def __init__(self, call, *, max_coordinates=512, max_elements=4096):
        import itertools
        import math
        import torch
        from mirage.runtime.operands import discover
        from mirage.backends.pytorch import PyTorchBackend

        self.template = call.get('inputs')
        descriptors = call.get('metadata')
        if not isinstance(self.template, dict) or not isinstance(descriptors, list) or not descriptors:
            raise ReconstructionError('missing argument tree or tensor metadata')
        observed = {tuple(d.path): t for d, t in discover(self.template, PyTorchBackend())}
        self.items, self.groups, self.objects = [], {}, {}
        total = 0
        for d in descriptors:
            path = tuple(d['path'])
            if path not in observed or any(item['path'] == path for item in self.items):
                raise ReconstructionError('missing or duplicate tensor path')
            tensor = observed[path]
            shape, strides = list(d['shape']), d.get('strides')
            props = d.get('properties', {})
            if d.get('layout') != 'torch.strided' or not isinstance(strides, (list, tuple)):
                raise ReconstructionError('only dense strided coordinate charts are supported')
            if tensor.is_complex() or props.get('conjugate'):
                raise ReconstructionError('complex/conjugate coordinates are unsupported')
            if list(tensor.shape) != shape or str(tensor.dtype) != d['dtype']:
                raise ReconstructionError('saved tensor and metadata disagree')
            if len(shape) != len(strides) or any(type(s) is not int or s < 0 for s in strides):
                raise ReconstructionError('invalid or negative strides')
            offset = d.get('offset')
            if type(offset) is not int or offset < 0 or not tensor.numel():
                raise ReconstructionError('invalid offset or empty tensor')
            total += tensor.numel()
            if total > max_elements:
                raise ReconstructionError('argument element budget exceeded')
            requires_grad = d.get('requires_grad') is True
            if requires_grad and tensor.dtype not in (torch.float32, torch.float64):
                raise ReconstructionError('differentiable coordinates require float32/float64')
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ReconstructionError('nonfinite argument values')
            negative = props.get('negative') is True
            if negative and not tensor.is_floating_point():
                raise ReconstructionError('negative view requires real floating values')
            group_id = d.get('storage_group')
            object_id = d.get('object_group')
            if group_id is None or object_id is None:
                raise ReconstructionError('missing storage/object identity')
            positions = [offset + sum(i * s for i, s in zip(index, strides))
                         for index in itertools.product(*(range(n) for n in shape))]
            extent = max(positions) + 1
            if extent > max_elements:
                raise ReconstructionError('storage span budget exceeded')
            group = self.groups.setdefault(group_id, {'dtype': tensor.dtype, 'device': d['device'],
                'requires_grad': requires_grad, 'values': {}, 'extent': 0})
            if (group['dtype'] != tensor.dtype or group['device'] != d['device'] or
                    group['requires_grad'] != requires_grad):
                raise ReconstructionError('mixed dtype/device/differentiability in shared storage')
            logical = tensor.detach().resolve_neg().reshape(-1).tolist()
            for position, value in zip(positions, logical):
                physical = -value if negative else value
                if position in group['values'] and group['values'][position] != physical:
                    raise ReconstructionError('aliased recorded values disagree')
                group['values'][position] = physical
            group['extent'] = max(group['extent'], extent)
            item = {'path': path, 'descriptor': d, 'group': group_id, 'object': object_id,
                    'positions': positions, 'negative': negative, 'requires_grad': requires_grad}
            if object_id in self.objects:
                prior = self.objects[object_id]
                fields = ('group', 'positions', 'negative', 'requires_grad')
                if (any(prior[k] != item[k] for k in fields) or
                        tuple(prior['descriptor']['shape']) != tuple(shape) or
                        tuple(prior['descriptor']['strides']) != tuple(strides)):
                    raise ReconstructionError('repeated object metadata disagree')
            else:
                self.objects[object_id] = item
            self.items.append(item)
        if set(observed) != {item['path'] for item in self.items}:
            raise ReconstructionError('tensor metadata is incomplete')
        self.coordinates = []
        for group_id, group in self.groups.items():
            group['indices'] = sorted(group['values'])
            group['coordinate_indices'] = []
            if group['requires_grad']:
                for position in group['indices']:
                    group['coordinate_indices'].append(len(self.coordinates))
                    self.coordinates.append((group_id, position, float(group['values'][position])))
        if not self.coordinates or len(self.coordinates) > max_coordinates:
            raise ReconstructionError('empty or excessive independent coordinate count')
        self._coordinate_index = {(g, p): i for i, (g, p, _) in enumerate(self.coordinates)}
        self._torch = torch

    @property
    def values(self):
        return [value for _, _, value in self.coordinates]

    @property
    def epsilon(self):
        return max(self._torch.finfo(g['dtype']).eps for g in self.groups.values() if g['requires_grad'])

    def build(self, values=None):
        """Fresh tensors per call prevent finite-difference evaluations sharing state."""
        torch = self._torch
        values = self.values if values is None else values
        if len(values) != len(self.coordinates):
            raise ReconstructionError('coordinate size mismatch')
        storages, bases = {}, []
        for group_id, group in self.groups.items():
            indices = group['indices']
            data = ([values[i] for i in group['coordinate_indices']] if group['requires_grad']
                    else [group['values'][p] for p in indices])
            base = torch.tensor(data, dtype=group['dtype'], device=group['device'],
                                requires_grad=group['requires_grad'])
            if group['requires_grad']:
                bases.append((group_id, base))
            index = torch.tensor(indices, dtype=torch.int64, device=group['device'])
            storages[group_id] = torch.zeros(group['extent'], dtype=group['dtype'], device=group['device']).index_copy(0, index, base)
        objects, tensors = {}, {}
        for item in self.items:
            if item['object'] not in objects:
                d = item['descriptor']
                tensor = storages[item['group']].as_strided(d['shape'], d['strides'], d['offset'])
                if item['negative']:
                    tensor = torch._neg_view(tensor)
                objects[item['object']] = tensor
            tensors[item['path']] = objects[item['object']]
        def replace(value, path=()):
            if path in tensors:
                return tensors[path]
            if isinstance(value, dict):
                if set(value) == {'framework_literal'}:
                    literal = value['framework_literal']
                    if literal.startswith('torch.') and hasattr(torch, literal[6:]):
                        return getattr(torch, literal[6:])
                    return torch.device(literal)
                return {k: replace(v, path + (k,)) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [replace(v, path + (i,)) for i, v in enumerate(value)]
            if value is None or type(value) in (bool, int, float, str):
                return value
            raise ReconstructionError('unsupported argument literal')
        return replace(self.template), bases

    def pullback(self, inputs):
        """Independent signed index sums of observed operand cotangents."""
        import math
        torch = self._torch
        by_path = {tuple(item['path']): item for item in inputs}
        if len(by_path) != len(inputs):
            raise ReconstructionError('duplicate observed operand cotangent')
        expected = {item['path'] for item in self.items if item['requires_grad']}
        if set(by_path) != expected:
            raise ReconstructionError('incomplete differentiable operand cotangents')
        sums = [[] for _ in self.coordinates]
        for item in self.items:
            if not item['requires_grad']:
                continue
            gradient = by_path[item['path']].get('gradient')
            if not isinstance(gradient, torch.Tensor) or tuple(gradient.shape) != tuple(item['descriptor']['shape']):
                raise ReconstructionError('missing or wrong-shaped operand cotangent')
            if str(gradient.dtype) != item['descriptor']['dtype'] or not bool(torch.isfinite(gradient).all()):
                raise ReconstructionError('wrong-dtype or nonfinite operand cotangent')
            values = gradient.detach().resolve_neg().reshape(-1).tolist()
            for position, value in zip(item['positions'], values):
                index = self._coordinate_index[(item['group'], position)]
                sums[index].append(-float(value) if item['negative'] else float(value))
        return [math.fsum(values) for values in sums]

    def verify(self, tree):
        """Check logical values, layout and alias partitions, ignoring chart leaf status."""
        from mirage.runtime.operands import discover
        from mirage.backends.pytorch import PyTorchBackend
        torch = self._torch
        actual = {tuple(d.path): (d, t) for d, t in discover(tree, PyTorchBackend())}
        original = {tuple(d.path): t for d, t in discover(self.template, PyTorchBackend())}
        for item in self.items:
            d, tensor = actual[item['path']]
            saved = item['descriptor']
            for field in ('shape', 'strides'):
                if list(getattr(d, field)) != list(saved[field]):
                    raise ReconstructionError('rebuilt tensor ' + field + ' differs')
            for field in ('offset', 'dtype', 'device', 'requires_grad', 'layout'):
                if getattr(d, field) != saved[field]:
                    raise ReconstructionError('rebuilt tensor ' + field + ' differs')
            if tensor.is_neg() != item['negative'] or tensor.is_conj() != saved.get('properties', {}).get('conjugate', False):
                raise ReconstructionError('rebuilt view bits differ')
            if not torch.equal(tensor.detach().cpu(), original[item['path']]):
                raise ReconstructionError('rebuilt logical values differ')
        for left in self.items:
            for right in self.items:
                a, b = actual[left['path']][0], actual[right['path']][0]
                if ((a.storage_group == b.storage_group) != (left['group'] == right['group']) or
                        (a.object_group == b.object_group) != (left['object'] == right['object'])):
                    raise ReconstructionError('rebuilt alias or object identity differs')
        return True
