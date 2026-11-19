"""PyTorch-specific facts behind the backend-neutral descriptor boundary."""
from mirage.contracts.capabilities import Evidence, EvidenceStatus, OperandDescriptor


_REPRESENTATION_SEMANTIC_CLASSES = {
    'pointwise': {'sin', 'cos', 'exp', 'log', 'sqrt', 'tanh', 'sigmoid', 'abs', 'neg',
                  'add', 'sub', 'mul', 'div', 'square'},
    'reduction': {'sum', 'mean', 'std'},
    'multiplication': {'matmul', 'mm', 'bmm'},
    'linear_algebra': {'slogdet', 'lu_solve'},
    'indexed_values': {'gather', 'scatter_add'},
    'normalization': {'batch_norm', 'layer_norm', 'group_norm'},
    'write_values': {'copy_', 'add_', 'sub_', 'mul_', 'div_', 'fill_', 'zero_'},
}


class PyTorchBackend:
    name = "pytorch"

    def training_capabilities(self, context, *, checkpoint, provenance, observed=None):
        """Concrete runner facts, scoped to this context and checkpoint.

        Tensor shape/schema facts authorize mechanics only. They never certify
        purity, mathematical equivalence, or sample independence.
        """
        import torch
        from mirage.contracts.candidates import CapabilitySnapshot, tensor_paths, path_key

        def fact(value, reason):
            status = (EvidenceStatus.UNKNOWN if value is None else
                      EvidenceStatus.VERIFIED if value else EvidenceStatus.REJECTED)
            return Evidence(status, reason, value, checkpoint, "adapter")

        observed = observed or {}
        inputs = list(tensor_paths(context["inputs"], self.is_tensor))
        parameters = context["parameters"]
        optimizer = context["optimizer"]
        bound = [p for g in optimizer.param_groups for p in g["params"]] if optimizer is not None else []
        binding = bool(bound) and all(any(p is q for q in parameters.values()) for p in bound)
        linked = observed.get("target_parameter_names")
        connected = None if linked is None else bool(set(linked) &
            {name for name, p in parameters.items() if any(p is q for q in bound)})
        usable = False if optimizer is None or not binding else connected
        axes = context["batch_axes"]
        lengths = {context["inputs"][name].shape[0] for name in axes}
        slicing = (bool(axes) and len(lengths) == 1 and min(lengths) >= 2
                   and observed.get("sample_loss") is True and provenance.get("precision") == "fp32")
        subjects = []
        object_groups, storage_groups = {}, {}
        for path, tensor in inputs:
            storage = self.storage_identity(tensor)
            object_group = object_groups.setdefault(id(tensor), len(object_groups))
            storage_group = storage_groups.setdefault(storage, len(storage_groups))
            aliases = [other_path for other_path, other in inputs
                       if other_path != path and self.storage_identity(other) == storage]
            parameter_alias = any(self.storage_identity(p) == storage for p in parameters.values())
            independent = not aliases and not parameter_alias
            partners = [other_path for other_path, other in inputs if other_path != path
                        and other.shape == tensor.shape and other.dtype == tensor.dtype
                        and other.device == tensor.device and torch.equal(other, tensor)
                        and not other.requires_grad and not tensor.requires_grad]
            name = path_key(path)
            consumed = observed.get("input_names")
            local = {
                "target_operand": fact(None if consumed is None else name in consumed,
                                       "observed target operand dependency"),
                "alias_preserved": fact(independent,
                    "single-coordinate adapter requires independent input storage; alias topology retained as an unmet goal"),
                "clone": fact(tensor.layout == torch.strided, "dense clone mechanics"),
                "offset": fact(tensor.layout == torch.strided and tensor.numel() > 0, "dense nonempty offset mechanics"),
                "strided": fact(tensor.layout == torch.strided and tensor.ndim > 0 and tensor.numel() > 1,
                                "noncontiguity requires multiple elements"),
                "shared": fact(bool(partners) and not parameter_alias,
                               "shared storage requires a distinct equal-valued nontrainable partner"),
            }
            subjects.append(dict(path=list(path), shape=list(tensor.shape), dtype=str(tensor.dtype),
                device=str(tensor.device), layout=str(tensor.layout), requires_grad=tensor.requires_grad,
                is_leaf=tensor.is_leaf, object_group=object_group, storage_group=storage_group,
                aliases=[list(p) for p in aliases], parameter_alias=parameter_alias,
                shared_partners=[list(p) for p in partners], facts={k: vars(v) for k, v in local.items()}))
        backward = observed.get("backward_executed")
        gradients = bool(parameters) or any(t.requires_grad for _, t in inputs)
        facts = {
            "optimizer": fact(usable, "optimizer presence, registered bindings and target-to-parameter backward path"),
            "optimizer_present": fact(optimizer is not None, "runtime optimizer object"),
            "optimizer_bound": fact(binding, "runtime optimizer parameter identities"),
            "backward": fact(backward, "canonical target backward observed"),
            "gradients": fact(gradients, "registered gradient coordinates"),
            "accumulated_gradients": fact(observed.get("all_gradient_coordinates"),
                                          "canonical backward populates every registered gradient coordinate"),
            "microbatch": fact(slicing, "sample_loss, common nonempty batch lengths and FP32 slicing mechanics"),
            "mutation": fact(None if "mutated_operations" not in observed else bool(observed["mutated_operations"]),
                             "observed writes authorize compiled mutation exploration, not equivalence"),
        }
        return CapabilitySnapshot(self.name, checkpoint, provenance, tuple(subjects), facts)

    def is_tensor(self, value):
        import torch
        return isinstance(value, torch.Tensor)

    def storage_identity(self, value):
        import torch
        if value.layout == torch.sparse_csr:
            value = value.values()
        if value.layout != torch.strided or value.is_mkldnn:
            return None
        storage = value.untyped_storage()
        # Storage object identity distinguishes empty allocations with data_ptr == 0.
        return (str(value.device), storage._cdata)

    def describe_tensor(self, value, path, object_group, storage_group):
        import torch
        dense = value.layout == torch.strided and not value.is_mkldnn
        return OperandDescriptor(
            path=path, shape=tuple(value.shape), dtype=str(value.dtype), device=str(value.device),
            layout=str(value.layout), object_group=object_group, storage_group=storage_group,
            strides=tuple(value.stride()) if dense else None,
            offset=value.storage_offset() if dense else None,
            requires_grad=value.requires_grad, leaf=value.is_leaf,
            properties={"complex": value.is_complex(), "floating": value.is_floating_point(),
                        "conjugate": value.is_conj(), "negative": value.is_neg(),
                        "mkldnn": value.is_mkldnn,
                        "contiguous": value.is_contiguous() if dense else None})

    def gradient_edges(self, operands):
        """Capture edges before the call, including before any in-place mutation."""
        import torch
        return [(tuple(d.path), torch.autograd.graph.get_gradient_edge(t))
                for d, t in operands if t.requires_grad]

    def backward_binding(self, tensor, input_edges, *, boundary=False):
        """Certify only a fresh direct node with known operand edges.

        Composite/fused internals and repeated differentiation coordinates do not
        acquire per-operand attribution from a tensor hook or a shared leaf grad.
        Factory-result boundaries explicitly certify the callable, not its internals.
        """
        node = tensor.grad_fn
        if node is None or any(node is edge.node for _, edge in input_edges):
            return None, [], "no fresh attributable output node"
        links = []
        for slot, (parent, output_nr) in enumerate(node.next_functions):
            if parent is None:
                continue
            paths = [path for path, edge in input_edges
                     if parent is edge.node and output_nr == edge.output_nr]
            if not paths and not boundary:
                return None, [], "composite or unobserved input edge"
            if len(paths) == 1 and not boundary:
                links.append((slot, paths[0]))
        if not input_edges and not boundary:
            return None, [], "no differentiable operand edge"
        return node, links, "factory-result boundary" if boundary else "direct operand edges"

    def local_vjp_coordinate_edges(self, node, input_edges, descriptors):
        """Bind exact-object repeats only for their summed storage-coordinate VJP.

        Slot-to-path order is not observable for repeated identical operands. It
        cannot support per-position semantic or optimizer provenance; any bijection
        has the same pullback only when every occurrence is the exact same object.
        """
        by_path = {tuple(d['path']): d for d in descriptors}
        if len(by_path) != len(descriptors) or len({p for p, _ in input_edges}) != len(input_edges):
            return None
        groups = {}
        for path, edge in input_edges:
            groups.setdefault((id(edge.node), edge.output_nr), []).append(path)
        slots = {}
        for slot, (parent, output_nr) in enumerate(node.next_functions):
            if parent is None:
                continue
            key = (id(parent), output_nr)
            if key not in groups:
                return None
            slots.setdefault(key, []).append(slot)
        links = []
        for key, paths in groups.items():
            positions = slots.get(key, [])
            if len(paths) != len(positions):
                return None
            if len(paths) > 1:
                metadata = [by_path.get(path) for path in paths]
                if any(d is None for d in metadata):
                    return None
                first = metadata[0]
                fields = ('object_group', 'storage_group', 'shape', 'strides', 'offset',
                          'dtype', 'device', 'layout', 'requires_grad', 'properties')
                if (first.get('object_group') is None or first.get('storage_group') is None
                        or any(any(d.get(field) != first.get(field) for field in fields) for d in metadata[1:])):
                    return None
            links.extend(zip(positions, paths))
        return tuple(sorted(links)) if links else None

    def capability(self, name):
        if name in ("tensor_descriptors", "dense_storage_identity"):
            return Evidence(EvidenceStatus.VERIFIED, "implemented adapter capability", True,
                            "capability_inventory", "adapter")
        return Evidence(EvidenceStatus.UNSUPPORTED, "not implemented by this adapter")

    def mutation_actions(self, descriptor):
        if descriptor.properties.get("mkldnn"):
            return ("logical_materialize",)
        if descriptor.layout == "torch.sparse_csr":
            return ("values_strided", "values_offset", "values_canonical")
        if descriptor.layout != "torch.strided":
            return ()
        return ("clone", "strided", "offset", "conjugate", "negative", "singleton_stride", "expanded", "canonical")

    def mutation_eligibility(self, tensor, action, obligations):
        import torch
        reason = None
        if isinstance(tensor, torch.nn.Parameter):
            reason = "registered parameter identity cannot be replaced"
        elif (obligations.writable and tensor.layout == torch.strided and not tensor.is_mkldnn
              and torch._debug_has_internal_overlap(tensor) != 0):
            reason = "overlapping or unverified writable storage"
        elif obligations.writable and tensor.requires_grad:
            reason = "writable differentiable operands require an effect certificate"
        elif obligations.writable and action in ("expanded", "logical_materialize"):
            reason = "write propagation is not certified for this action"
        elif tensor.layout == torch.sparse_csr and tensor.requires_grad:
            reason = "CSR gradient-preserving reconstruction is not certified"
        elif action == "conjugate" and not tensor.is_complex():
            reason = "conjugate requires complex dtype"
        elif action == "negative" and tensor.dtype not in (torch.float32, torch.float64):
            reason = "negative-bit action requires float32 or float64"
        elif action == "singleton_stride":
            axes = [i for i, n in enumerate(tensor.shape) if n == 1]
            if not axes:
                reason = "missing singleton dimension"
        elif action == "expanded":
            axes = [i for i, n in enumerate(tensor.shape) if n > 1]
            if tensor.requires_grad:
                reason = "expansion would alter independent differentiation coordinates"
            elif not any(torch.equal(tensor.narrow(i, 0, 1).expand_as(tensor), tensor) for i in axes):
                reason = "no value-preserving expansion axis"
        if reason:
            return Evidence(EvidenceStatus.REJECTED, reason)
        return Evidence(EvidenceStatus.VERIFIED, "runtime mutation preconditions", True,
                        "mutation_eligibility", "adapter")

    def mutate(self, tensor, action, obligations):
        from mirage.runtime.representations import connected_representation
        evidence = self.mutation_eligibility(tensor, action, obligations)
        if not evidence.established:
            raise ValueError(evidence.reason)
        return connected_representation(tensor, action)

    def verify_mutation(self, source, result, action, obligations):
        import torch
        def logical(t):
            return t.to_dense() if t.layout != torch.strided or t.is_mkldnn else t
        left, right = logical(source), logical(result)
        good = (source.shape == result.shape and source.dtype == result.dtype
                and source.device == result.device and torch.equal(left, right))
        if obligations.preserve_gradient:
            good &= source.requires_grad == result.requires_grad
            if source.requires_grad:
                good &= result.grad_fn is not None
        if source.layout == torch.sparse_csr:
            good &= torch.equal(source.crow_indices(), result.crow_indices())
            good &= torch.equal(source.col_indices(), result.col_indices())
        if action == "canonical": good &= result.is_contiguous() and not result.is_conj() and not result.is_neg()
        if action == "conjugate": good &= result.is_conj()
        if action == "negative": good &= result.is_neg()
        if action == "offset": good &= result.storage_offset() > 0
        if action == "strided" and result.numel() > 1: good &= not result.is_contiguous()
        if action == "expanded": good &= any(n > 1 and s == 0 for n, s in zip(result.shape, result.stride()))
        if action == "singleton_stride": good &= result.stride() != source.stride()
        return Evidence(EvidenceStatus.VERIFIED if good else EvidenceStatus.REJECTED,
                        "logical values and representation obligations" if good else "mutation not attained",
                        bool(good), "mutation_applied", "mutation")

    def input_properties(self, tensor):
        """Observe values at the target boundary using dtype-relative magnitude bins."""
        import math
        import torch
        logical = tensor.to_dense() if tensor.layout != torch.strided or tensor.is_mkldnn else tensor
        value = logical.detach()
        finite = bool(torch.isfinite(value).all())
        flat = value.reshape(-1)
        facts = {"finite": finite, "empty": value.numel() == 0,
                 "singleton": 1 in value.shape, "layout": str(tensor.layout),
                 "supported_layout": tensor.layout != torch.strided or tensor.is_mkldnn or not tensor.is_contiguous(),
                 "zero": bool((value == 0).any()) if value.numel() else False,
                 "small_finite": False, "large_finite": False,
                 "repeated": len(set(flat[:4096].cpu().tolist())) < min(value.numel(),4096) if finite and value.numel()>1 else False,
                 "nonuniform": bool((flat != flat[0]).any()) if value.numel() > 1 else False}
        facts['mixed_sign'] = (bool((value < 0).any() and (value > 0).any())
                               if not value.is_complex() else False)
        if value.is_floating_point() or value.is_complex():
            limits = torch.finfo(value.real.dtype)
            small, large = math.sqrt(limits.tiny), math.sqrt(limits.max)
            magnitudes = value.abs()
            facts.update(small_threshold=small, large_threshold=large,
                         small_finite=bool(((magnitudes > 0) & (magnitudes <= small) & torch.isfinite(magnitudes)).any()),
                         large_finite=bool(((magnitudes >= large) & torch.isfinite(magnitudes)).any()))
        facts['ordinary'] = finite and value.numel()>0 and bool((value != 0).any()) and not facts['small_finite'] and not facts['large_finite']
        return facts

    def compare(self, left, right, policy):
        """Scale-aware screening; no absolute floor masks tiny nonzero outputs."""
        import math
        import torch
        mismatches, uncertainties = [], []
        def walk(a, b, path):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.shape != b.shape or a.dtype != b.dtype:
                    mismatches.append({'path': path, 'reason': 'shape/dtype mismatch'}); return
                if a.layout != torch.strided or a.is_mkldnn: a = a.to_dense()
                if b.layout != torch.strided or b.is_mkldnn: b = b.to_dense()
                if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                    uncertainties.append(path + ': nonfinite observation'); return
                if policy == 'exact' or not (a.is_floating_point() or a.is_complex()):
                    if not torch.equal(a, b): mismatches.append({'path': path, 'reason': 'exact mismatch'})
                    return
                if policy != 'dtype_scaled_v1':
                    uncertainties.append(path + ': unsupported numerical policy'); return
                if not a.numel(): return
                epsilon = torch.finfo(a.real.dtype).eps
                aa, bb = a.to(torch.complex128 if a.is_complex() else torch.float64), b.to(torch.complex128 if b.is_complex() else torch.float64)
                # Normalize before subtraction to avoid overflow at finite dtype extremes.
                scale = max(float(aa.abs().max()), float(bb.abs().max()))
                error = 0.0 if scale == 0 else float((aa / scale - bb / scale).abs().max())
                limit = min(0.01, 64 * epsilon)
                if error > limit: mismatches.append({'path': path, 'relative_max_error': error, 'limit': limit, 'scale': scale})
            elif type(a) is dict and type(b) is dict and a.keys() == b.keys():
                for key in a: walk(a[key], b[key], path + '.' + str(key))
            elif isinstance(a, (list, tuple)) and type(a) is type(b) and len(a) == len(b):
                for index, (x, y) in enumerate(zip(a, b)): walk(x, y, f'{path}[{index}]')
            elif type(a) in (type(None), bool, int, float, str) and type(a) is type(b):
                if isinstance(a, float) and not (math.isfinite(a) and math.isfinite(b)):
                    uncertainties.append(path + ': nonfinite scalar')
                elif a != b: mismatches.append({'path': path, 'reason': 'scalar mismatch'})
            else:
                mismatches.append({'path': path, 'reason': 'structure mismatch'})
        walk(left, right, 'observations')
        return {'equal': None if uncertainties else not mismatches, 'differences': mismatches,
                'reason': '; '.join(uncertainties) if uncertainties else 'comparison completed',
                'policy': policy, 'screening_only': True}

    def reduction_numerical_domain(self, operation, call):
        """Bound rounding amplification for the existing 64-epsilon screen.

        Admission depends on operands, never agreement of the compared outputs.
        Wider or cancellation-dominated reductions require another numerical policy.
        """
        import math
        import torch
        tree = call.get('inputs') or {}
        args, kwargs = tree.get('args', ()), tree.get('kwargs', {})
        value = args[0] if args else kwargs.get('input')
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.is_mkldnn or value.dtype not in (torch.float32, torch.float64)
                or not value.numel() or not bool(torch.isfinite(value).all())
                or kwargs.get('dtype', value.dtype) != value.dtype):
            return False
        dim = kwargs.get('dim', args[1] if len(args) > 1 else None)
        correction = kwargs.get('correction', 1)
        if operation == 'std':
            unbiased = kwargs.get('unbiased', args[2] if len(args) > 2 else None)
            if type(dim) is bool:
                unbiased, dim = dim, None
            if unbiased is not None:
                if type(unbiased) is not bool or 'correction' in kwargs: return False
                correction = int(unbiased)
            if type(correction) not in (int, float) or not math.isfinite(correction): return False
        if dim is None or isinstance(dim, (list, tuple)) and not dim:
            dims = tuple(range(value.ndim))
        else:
            dims = (dim,) if type(dim) is int else dim
            if not isinstance(dims, (list, tuple)) or any(type(d) is not int for d in dims): return False
            if value.ndim == 0:
                if tuple(dims) not in ((0,), (-1,)): return False
                dims = ()
            else:
                if any(d < -value.ndim or d >= value.ndim for d in dims): return False
                dims = tuple(d % value.ndim for d in dims)
                if len(set(dims)) != len(dims): return False
        width = math.prod(value.shape[d] for d in dims)
        if width > (8 if operation == 'std' else 16): return False
        normalized = value.to(torch.float64)
        scale = normalized.abs().amax(dim=dims, keepdim=True)
        normalized = normalized / torch.where(scale > 0, scale, torch.ones_like(scale))
        limits = torch.finfo(value.dtype)
        if operation == 'std':
            if not 0 <= correction < width: return False
            centered = normalized - normalized.mean(dim=dims, keepdim=True)
            spread = (centered.square().sum(dim=dims, keepdim=True) / (width-correction)).sqrt()
            # Near-constant inputs amplify subtraction and derivative roundoff.
            expected_scale = spread * scale
            return bool((spread >= .25).all() and (expected_scale >= limits.tiny).all()
                        and (expected_scale <= limits.max).all())
        total = normalized.sum(dim=dims, keepdim=True).abs()
        mass = normalized.abs().sum(dim=dims, keepdim=True)
        expected_scale = total * (scale / width if operation == 'mean' else scale)
        normal_or_zero = (expected_scale >= limits.tiny) | (mass == 0)
        return bool((total >= .5 * mass).all() and normal_or_zero.all()
                    and (expected_scale <= limits.max).all())

    def multiplication_numerical_domain(self, call):
        """Rectangular products need a contraction bound, not invertibility."""
        import math
        import torch
        tree = call.get('inputs') or {}
        args, kwargs = tree.get('args', ()), tree.get('kwargs', {})
        left = args[0] if args else kwargs.get('input')
        right = args[1] if len(args) > 1 else kwargs.get('other', kwargs.get('mat2'))
        operands = (left, right)
        if any(not isinstance(t, torch.Tensor) or t.layout != torch.strided or t.is_mkldnn
               or t.dtype not in (torch.float32, torch.float64) or not t.numel()
               or t.ndim == 0 or not bool(torch.isfinite(t).all()) for t in operands):
            return False
        width = left.shape[-1]
        if width != right.shape[-2 if right.ndim > 1 else 0] or width > 16: return False
        a, b = [t.to(torch.float64) for t in operands]
        scales = (float(a.abs().max()), float(b.abs().max()))
        if 0 in scales: return True
        a = a / scales[0]
        b = b / scales[1]
        try:
            magnitude = (a @ b).abs().max()
            mass = (a.abs() @ b.abs()).max()
        except RuntimeError:
            return False
        if not bool(torch.isfinite(magnitude) and torch.isfinite(mass) and magnitude >= .5 * mass): return False
        if magnitude == 0: return False
        log_scale = math.log(float(magnitude)) + math.log(scales[0]) + math.log(scales[1])
        limits = torch.finfo(left.dtype)
        return math.log(limits.tiny) <= log_scale <= math.log(limits.max)

    def representation_evidence(self, left, right):
        # One semantic-class registry serves all value-representation comparisons.
        # This is capability admission, never an API-specific expected-value formula.
        classes = _REPRESENTATION_SEMANTIC_CLASSES
        api = left.get('config', {}).get('api', '')
        operation = api.rsplit('.', 1)[-1]
        value_semantics = any(operation in names for names in classes.values())
        results = {}
        def fact(key, established, reason):
            results[key] = Evidence(EvidenceStatus.VERIFIED if established else EvidenceStatus.UNKNOWN,
                                    reason, True if established else None, 'comparison', 'adapter')
        fact('valid_executions', all(r.get('inspection', {}).get('valid') for r in (left, right)), 'inspector validity')
        fact('logical_value_semantics', value_semantics, 'reviewed semantic class')
        numerical = operation in classes['pointwise'] | classes['indexed_values'] | classes['normalization'] | classes['write_values']
        numerical_reason = 'pointwise representation screening'
        if operation in classes['reduction']:
            numerical = all(self.reduction_numerical_domain(operation, call)
                            for record in (left, right) for call in record.get('target_calls', ()))
            numerical_reason = 'bounded reduction width and operand cancellation/spread'
        elif operation in classes['multiplication']:
            numerical = all(self.multiplication_numerical_domain(call)
                            for record in (left, right) for call in record.get('target_calls', ()))
            numerical_reason = 'bounded contraction width and product cancellation; no square-matrix requirement'
        elif operation in classes['linear_algebra']:
            import torch
            matrices = []
            try:
                for record in (left, right):
                    for call in record.get('target_calls', ()):
                        tree = call['inputs']
                        args, kwargs = tree.get('args', ()), tree.get('kwargs', {})
                        matrix = args[0] if args else (kwargs.get('LU') if operation == 'lu_solve' else kwargs.get('A', kwargs.get('input')))
                        if operation == 'lu_solve':
                            pivots = args[1] if len(args) > 1 else kwargs.get('pivots')
                            permutation, lower, upper = torch.lu_unpack(matrix, pivots)
                            matrix = permutation @ lower @ upper
                        matrices.append(matrix)
                numerical = bool(matrices) and all(isinstance(t, torch.Tensor) and t.ndim >= 2
                    and t.shape[-1] == t.shape[-2] and t.numel() > 0
                    and (t.is_floating_point() or t.is_complex()) and bool(torch.isfinite(t).all())
                    and bool((torch.linalg.cond(t.to(torch.complex128 if t.is_complex() else torch.float64)) <= 100).all())
                    for t in matrices)
            except (RuntimeError, ValueError, TypeError):
                numerical = False
            numerical_reason = 'bounded coefficient-matrix conditioning; RHS shape and conditioning are not invertibility requirements'
        fact('numerical_domain', numerical, numerical_reason)

        fact('read_only_target', value_semantics and operation not in classes['write_values'] and all(not c.get('write_effects') for r in (left,right) for c in r.get('target_calls',())), 'reviewed value/effect class')
        fact('same_program', left.get('graph', {}).get('fingerprint') == right.get('graph', {}).get('fingerprint'), 'source fingerprint')
        fact('same_initial_state', self.compare(left.get('initial'), right.get('initial'), 'exact')['equal'] is True and left.get('initial_metadata') == right.get('initial_metadata'), 'exact initial logical state and alias topology')
        calls = [r.get('target_calls', []) for r in (left, right)]
        aligned = bool(calls[0]) and len(calls[0]) == len(calls[1])
        fact('aligned_checkpoints', aligned and [c['name'] for c in left.get('checkpoints', [])] == [c['name'] for c in right.get('checkpoints', [])], 'ordered semantic checkpoints')
        fact('same_target_inputs', aligned and all(self.compare(a['inputs'], b['inputs'], 'exact')['equal'] is True for a,b in zip(*calls)), 'exact target logical inputs')
        fact('preserved_coordinates', bool(right.get('mutations')) and all(m['status'] == 'verified' for m in right['mutations']), 'verified graph-preserving transformation')
        finite = all(self.compare(r.get('observations'), r.get('observations'), 'exact')['equal'] is True for r in (left,right))
        fact('finite_observations', finite, 'finite target observations')
        # A seed or unchanged RNG alone cannot certify arbitrary user code. Only
        # the restricted straight-line numerical subset is currently certified.
        pure = all(r.get('program_certificate') == 'explicit_straight_line_values_v1' for r in (left,right))
        fact('deterministic', pure, 'reviewed straight-line initialization and effects certificate')
        return results

    def program_certificate(self, source):
        """Compatibility tag for the reviewed source subset, never runtime proof."""
        return self.explain_certificate(source)['certificate']

    def numerical_source_certificate(self, source):
        """Source effects eligible for local VJP replay, not a representation MR."""
        return self.explain_certificate(source, local_vjp=True)['certificate']

    def optimizer_source_certificate(self, source):
        """Restricted source effects for the separate optimizer-gradient MR."""
        return self.explain_certificate(source, local_vjp=True, optimizer_update=True)['certificate']

    def scoped_vjp_source_certificate(self, source):
        """Reviewed surrounding effects; runtime local evidence is also mandatory."""
        return self.explain_certificate(source, local_vjp=True, scoped_vjp=True)['certificate']

    def explain_certificate(self, source, *, local_vjp=False, optimizer_update=False, scoped_vjp=False):
        """Bounded binding/effect diagnostics; unsupported programs stay executable.

        Values are tracked through straight-line assignments and the build context.
        A method spelling is insufficient: its receiver must have a reviewed kind.
        Target semantics, numerical domains and runtime MR conditions remain separate.
        """
        import ast
        from mirage.generator.program import analyze_stateful, ContractError
        report = {'version': 'source-certificate-analysis-1', 'status': 'unknown',
                  'certificate': None, 'blockers': [],
                  'scope': 'reviewed straight-line source bindings/effects around the target; target value/effect semantics and runtime MR conditions remain separate'}
        from mirage.runtime.input_constraints import OPERAND_CONSTRUCTOR
        phase = 'module'
        optimizer_effects = optimizer_update or scoped_vjp

        class Uncertified(Exception):
            pass

        def reject(node, code, reason):
            report['blockers'].append({'code': code, 'phase': phase,
                'line': getattr(node, 'lineno', 0), 'expression': ast.unparse(node)[:160], 'reason': reason})
            raise Uncertified

        # These describe reviewed value/effect and return kinds, not correctness
        # formulas. Target calls outside this table can execute, but their return
        # values cannot acquire arbitrary Tensor methods from a spelling alone.
        value_ops = {'sin', 'cos', 'exp', 'log', 'sqrt', 'tanh', 'sigmoid', 'abs', 'neg',
                     'add', 'sub', 'mul', 'div', 'square', 'sum', 'mean', 'std', 'matmul',
                     'mm', 'bmm', 'gather', 'scatter_add', 'lu_solve'}
        views = {'reshape', 'view', 'expand', 'conj', 'conj_physical', 'detach', 'clone',
                 'contiguous', 'transpose', 'permute', 'unsqueeze', 'squeeze', 'to',
                 'float', 'double', 'resolve_conj', 'resolve_neg', 'to_mkldnn'}
        factories = {'torch.tensor', 'torch.zeros', 'torch.ones', 'torch.arange', 'torch.linspace',
                     'torch.full', 'torch.complex', 'torch.sparse_csr_tensor', 'torch.eye'}
        pure_functions = {'torch.' + name for name in value_ops | views - {'to', 'float', 'double', 'to_mkldnn'}}
        pure_functions |= {'torch.zeros_like', 'torch.ones_like', 'torch.cat', 'torch.stack'}
        tuple_methods = {'slogdet': ('sign', 'logabsdet'), 'lu_factor': ('LU', 'pivots')}
        tuple_functions = {prefix + name: fields for name, fields in tuple_methods.items()
                           for prefix in ('torch.', 'torch.linalg.', 'torch.Tensor.')}
        target_tensor_ops = set().union(*_REPRESENTATION_SEMANTIC_CLASSES.values()) - set(tuple_methods)
        target_tensor_functions = {prefix + name for name in target_tensor_ops for prefix in ('torch.', 'torch.Tensor.')}
        target_tensor_functions |= {'torch.nn.functional.' + name for name in _REPRESENTATION_SEMANTIC_CLASSES['normalization']}
        target_tensor_functions.add('torch.linalg.lu_solve')
        pure_functions |= {'torch.linalg.lu_solve'}
        optimizers = {'torch.optim.SGD', 'torch.optim.Adam', 'torch.optim.AdamW'}
        # Native dtype aliases retain the same binding kind as their canonical
        # names; operator/dtype support is still checked by runtime MR guards.
        dtype_names = {'torch.' + name for name in ('float16', 'float32', 'float64', 'bfloat16',
                       'complex32', 'complex64', 'complex128', 'int8', 'int16', 'int32', 'int64', 'uint8', 'bool',
                       'half', 'float', 'double', 'short', 'int', 'long', 'chalf', 'cfloat', 'cdouble')}
        tensor = ('tensor', None)
        scalar = ('scalar', None)
        protected = set()

        def safe(value, *, build=False, active=None):
            kind, payload = value
            if kind in ('tensor', 'scalar', 'dtype', 'device', 'shape', 'target_result'):
                return True
            if kind in ('sequence', 'result_tuple', 'dict'):
                active = set() if active is None else active
                if id(value) in active:
                    return False
                active.add(id(value))
                children = payload.values() if isinstance(payload, dict) else payload
                result = all(safe(item, build=build, active=active) for item in children)
                active.remove(id(value))
                return result
            return build and kind == 'optimizer'

        def contains(value, sought, visited=None):
            if value is sought:
                return True
            visited = set() if visited is None else visited
            if id(value) in visited or value[0] not in ('dict', 'sequence', 'result_tuple'):
                return False
            visited.add(id(value))
            children = value[1].values() if isinstance(value[1], dict) else value[1]
            return any(contains(item, sought, visited) for item in children)

        def primitive_helper(node):
            def literal(item):
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in ('min', 'max') and not item.keywords:
                    return (min if item.func.id == 'min' else max)(*[literal(arg) for arg in item.args])
                value = ast.literal_eval(item)
                def primitive(v):
                    return type(v) in (int, float, bool) or isinstance(v, (list, tuple)) and all(primitive(x) for x in v)
                if not primitive(value):
                    raise ValueError('nonprimitive helper operand')
                return value
            try:
                if sum(1 for _ in ast.walk(node)) > 64:
                    return False
                literal(node)
                return True
            except (ValueError, TypeError, SyntaxError):
                return False

        def native_tensor_result(binding, positional, keywords):
            # Native schemas establish return kinds, not derivative correctness.
            # Restrict this extension to the scoped route; unknown Python wrappers
            # and heterogeneous overload returns cannot acquire Tensor methods.
            if not scoped_vjp:
                return False
            import inspect
            import torch
            from mirage.runtime.api_metadata import resolve
            name = ('torch.Tensor.' + binding[1] if binding[0] == 'tensor_method'
                    else binding[1])
            try:
                target = resolve(name)
                if not (inspect.isbuiltin(target) or inspect.ismethoddescriptor(target)):
                    return False
                owner = getattr(target, '__objclass__', None)
                module = getattr(target, '__module__', None) or getattr(owner, '__module__', '')
                if module != 'torch' and not module.startswith('torch.'):
                    return False
                operator = 'aten::' + target.__name__
                schemas = torch._C._jit_get_schemas_for_operator(operator)
                supplied = ([tensor] if binding[0] == 'tensor_method' else []) + positional
                def accepts_tensor(kind):
                    tag = kind.kind()
                    if tag == 'OptionalType':
                        return accepts_tensor(kind.getElementType())
                    # Unknown schema types remain possible, so their return kinds
                    # must still pass. Never discard an overload by guesswork.
                    return tag not in {'NumberType', 'FloatType', 'IntType', 'ComplexType',
                                       'BoolType', 'StringType', 'NoneType', 'ListType',
                                       'TupleType', 'DictType', 'DeviceObjType', 'SymIntType'}
                def compatible(schema):
                    args = [a for a in schema.arguments if not a.kwarg_only]
                    for index, value in enumerate(supplied):
                        if value[0] == 'tensor' and (index >= len(args) or not accepts_tensor(args[index].type)):
                            return False
                    named = {a.name: a for a in schema.arguments}
                    for key, (value, _) in keywords.items():
                        if value[0] == 'tensor' and key in named and not accepts_tensor(named[key].type):
                            return False
                    return True
                schemas = [schema for schema in schemas if compatible(schema)]
                established = bool(schemas) and all(len(schema.returns) == 1
                    and schema.returns[0].type.kind() == 'TensorType' for schema in schemas)
            except (AttributeError, ValueError, RuntimeError, TypeError):
                return False
            if established:
                fact = {'target': name, 'operator': operator, 'return_kind': 'Tensor',
                        'source': 'installed native schemas compatible with proven Tensor arguments',
                        'overloads': len(schemas)}
                facts = report.setdefault('native_return_bindings', [])
                if fact not in facts:
                    facts.append(fact)
            return established

        def expr(node, env, *, target=False):
            if isinstance(node, ast.Constant) and type(node.value) in (type(None), bool, int, float, complex, str):
                return scalar
            if isinstance(node, ast.Name):
                if node.id in env:
                    return env[node.id]
                reject(node, 'unbound_name', 'No reviewed binding for this name')
            if isinstance(node, (ast.List, ast.Tuple)):
                return ('sequence', [expr(item, env) for item in node.elts])
            if isinstance(node, ast.Dict):
                if any(not isinstance(key, ast.Constant) or type(key.value) not in (str, int) for key in node.keys):
                    reject(node, 'dynamic_container', 'Only explicit primitive dictionary keys are reviewed')
                return ('dict', {key.value: expr(value, env) for key, value in zip(node.keys, node.values)})
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    reject(node, 'private_attribute', 'Private attributes and overrides are not reviewed')
                base = expr(node.value, env)
                kind, payload = base
                if kind == 'symbol':
                    qualified = payload + '.' + node.attr
                    if qualified in dtype_names:
                        return ('dtype', qualified)
                    if qualified in ('math.pi', 'math.e'):
                        return scalar
                    return ('symbol', qualified)
                if kind == 'tensor':
                    if node.attr in ('grad', 'real', 'imag', 'T', 'mT'):
                        return tensor
                    if node.attr in ('dtype', 'device', 'shape'):
                        return (node.attr, None)
                    if node.attr in value_ops | views | set(tuple_methods) | {'backward', 'requires_grad_'} or target:
                        return ('tensor_method', node.attr)
                    reject(node, 'metadata_or_unknown_attribute', 'Tensor metadata is not a logical-value observable under this certificate')
                if kind == 'optimizer' and optimizer_effects and phase == 'run' and node.attr in ('step', 'zero_grad'):
                    return ('optimizer_method', node.attr)
                if scoped_vjp and kind == 'dict' and node.attr == 'values':
                    return ('dict_values', list(payload.values()))
                if kind == 'finfo' and node.attr in ('tiny', 'max', 'min', 'eps', 'smallest_normal', 'bits'):
                    return scalar
                if kind == 'result_tuple' and node.attr in payload:
                    return payload[node.attr]
                reject(node, 'unproved_receiver', 'Attribute receiver has no reviewed Tensor, namespace or result binding')
            if isinstance(node, ast.Subscript):
                kind, payload = expr(node.value, env)
                if kind in ('dict', 'sequence', 'result_tuple'):
                    if not isinstance(node.slice, ast.Constant) or type(node.slice.value) not in (int, str):
                        reject(node, 'dynamic_container', 'Container binding requires a literal key/index')
                    try:
                        return (list(payload.values())[node.slice.value] if kind == 'result_tuple' else payload[node.slice.value])
                    except (KeyError, IndexError, TypeError):
                        reject(node, 'missing_container_binding', 'Container key/index is not established by build')
                if kind in ('tensor', 'shape'):
                    def index(item):
                        if isinstance(item, ast.Slice):
                            for part in (item.lower, item.upper, item.step):
                                if part is not None and expr(part, env)[0] != 'scalar':
                                    reject(part, 'index_binding', 'Slice bounds must be primitive')
                        elif isinstance(item, ast.Tuple):
                            for part in item.elts: index(part)
                        elif expr(item, env)[0] not in ('scalar', 'tensor'):
                            reject(item, 'index_binding', 'Index is not a reviewed primitive/Tensor')
                    index(node.slice)
                    return tensor if kind == 'tensor' else scalar
                reject(node, 'unproved_receiver', 'Subscript receiver binding is unknown')
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)):
                values = [expr(node.left, env), expr(node.right, env)]
                if all(value[0] in ('tensor', 'scalar') for value in values):
                    return tensor if any(value[0] == 'tensor' for value in values) else scalar
                reject(node, 'overloaded_operator', 'Operators require known primitive or Tensor operands')
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                value = expr(node.operand, env)
                if value[0] in ('tensor', 'scalar'):
                    return value
                reject(node, 'overloaded_operator', 'Unary operator receiver is not proven')
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == OPERAND_CONSTRUCTOR:
                    if (phase != 'build' or node.func.id in env or len(node.args) != 1
                            or not isinstance(node.args[0], ast.Constant) or type(node.args[0].value) != str
                            or any(k.arg not in ('requires_grad', 'sample') or not isinstance(k.value, ast.Constant)
                                   or (type(k.value.value) != bool if k.arg == 'requires_grad'
                                       else k.value.value is not None and type(k.value.value) != int)
                                   for k in node.keywords)):
                        reject(node, 'operand_constructor_binding', 'Expected injected named operand with literal options')
                    return tensor
                if scoped_vjp and isinstance(node.func, ast.Name) and node.func.id == 'list' and node.func.id not in env:
                    if len(node.args) == 1 and not node.keywords:
                        value = expr(node.args[0], env)
                        if value[0] == 'sequence':
                            return value
                    reject(node, 'container_construction', 'list requires an established finite sequence')
                if isinstance(node.func, ast.Name) and node.func.id in ('min', 'max') and node.func.id not in env:
                    if primitive_helper(node):
                        return scalar
                    reject(node, 'overloaded_helper', 'min/max are admitted only on bounded literal primitive operands')
                function = expr(node.func, env)
                is_target = function == ('symbol', 'probe.call')
                positional = []
                keywords = {}
                for arg in node.args:
                    if isinstance(arg, ast.Starred):
                        value = expr(arg.value, env)
                        if value[0] not in ('sequence', 'result_tuple'):
                            reject(arg, 'dynamic_arguments', 'Positional expansion requires a known sequence binding')
                        positional.extend(value[1].values() if value[0] == 'result_tuple' else value[1])
                    else:
                        positional.append(expr(arg, env, target=is_target and not positional))
                for keyword in node.keywords:
                    value = expr(keyword.value, env)
                    if keyword.arg is None:
                        if value[0] != 'dict' or any(type(key) is not str for key in value[1]):
                            reject(keyword.value, 'dynamic_arguments', 'Keyword expansion requires a known string-key dictionary binding')
                        entries = [(key, child, None) for key, child in value[1].items()]
                    else:
                        entries = [(keyword.arg, value, keyword.value)]
                    for key, child, origin in entries:
                        if key in keywords:
                            reject(node, 'duplicate_keyword', 'Expanded and explicit keyword bindings must not overlap')
                        keywords[key] = (child, origin)
                keyword_values = [value for value, _ in keywords.values()]
                if is_target:
                    if not positional:
                        reject(node, 'target_binding', 'Target callable is missing')
                    target_value = positional[0]
                    if target_value[0] != 'tensor_method' and not (target_value[0] == 'symbol' and target_value[1].startswith('torch.')):
                        reject(node, 'target_binding', 'Target must retain its imported framework or native Tensor binding')
                    arguments = positional[1:] + keyword_values
                    if not all(safe(value) for value in arguments):
                        reject(node, 'target_arguments', 'Unknown target argument binding')
                    qualified = target_value[1]
                    fields = tuple_methods.get(qualified) if target_value[0] == 'tensor_method' else tuple_functions.get(qualified)
                    if fields:
                        return ('result_tuple', {key: tensor for key in fields})
                    method_tensor = target_value[0] == 'tensor_method' and qualified in target_tensor_ops | views
                    return tensor if (method_tensor or qualified in pure_functions | factories | target_tensor_functions
                                      or native_tensor_result(target_value, positional[1:], keywords)) else ('target_result', None)
                arguments = positional + keyword_values
                if not all(safe(value) for value in arguments):
                    reject(node, 'argument_binding', 'Callbacks, opaque objects and framework bindings are not value arguments')
                if any(key in ('out', 'inplace') and not (key == 'out' and isinstance(origin, ast.Constant) and origin.value is None)
                       for key, (_, origin) in keywords.items()):
                    reject(node, 'write_argument', 'Write-capable helper arguments require a separate effect certificate')
                kind, qualified = function
                unbound_tensor_method = kind == 'symbol' and qualified.startswith('torch.Tensor.')
                if unbound_tensor_method:
                    if not positional or positional[0][0] != 'tensor':
                        reject(node, 'unproved_receiver', 'Unbound Tensor methods require an explicit reviewed Tensor receiver')
                    kind, qualified = 'tensor_method', qualified.rsplit('.', 1)[-1]
                if kind == 'symbol':
                    if qualified == 'torch.nn.Parameter' and optimizer_effects and phase == 'build':
                        # Native Parameter construction gets no general constructor
                        # exemption: data must retain its reviewed Tensor binding.
                        data = positional[0] if len(positional) == 1 and 'data' not in keywords else (
                            keywords['data'][0] if not positional and 'data' in keywords else None)
                        flag = keywords.get('requires_grad')
                        if (data is not None and data[0] == 'tensor' and len(positional) <= 1
                                and set(keywords) <= {'data', 'requires_grad'}
                                and not any(isinstance(arg, ast.Starred) for arg in node.args)
                                and not any(keyword.arg is None for keyword in node.keywords)
                                and (flag is None or isinstance(flag[1], ast.Constant) and flag[1].value is True)):
                            return tensor
                        reject(node, 'parameter_constructor', 'Parameter requires reviewed Tensor data and literal requires_grad=True')
                    if qualified in ('probe.mark',):
                        return scalar
                    if qualified in factories | pure_functions:
                        return tensor
                    if qualified in tuple_functions:
                        return ('result_tuple', {key: tensor for key in tuple_functions[qualified]})
                    if qualified == 'torch.finfo':
                        return ('finfo', None)
                    if qualified in optimizers and phase == 'build':
                        return ('optimizer', None)
                    if qualified == 'torch.manual_seed' and phase == 'build' and all(value[0] == 'scalar' for value in arguments):
                        return ('seed_result', None)
                if kind == 'dict_values' and scoped_vjp and not node.args and not node.keywords:
                    return ('sequence', qualified)
                if kind == 'optimizer_method':
                    if qualified == 'step' and not node.args and not node.keywords:
                        return scalar
                    if qualified == 'zero_grad' and not node.args and (not node.keywords or
                            len(node.keywords) == 1 and node.keywords[0].arg == 'set_to_none'
                            and isinstance(node.keywords[0].value, ast.Constant) and node.keywords[0].value.value is True):
                        return scalar
                    reject(node, 'optimizer_effect', 'Reviewed optimizer actions exclude closures, dynamic flags and step arguments')
                if kind == 'tensor_method':
                    if qualified in tuple_methods:
                        return ('result_tuple', {key: tensor for key in tuple_methods[qualified]})
                    if qualified in value_ops | views:
                        return tensor
                    if qualified == 'requires_grad_' and phase == 'build':
                        return tensor
                    if qualified == 'requires_grad_' and phase == 'run' and local_vjp:
                        # Only explicit native gradient enabling is reviewed here.
                        # Keep literal provenance; expanded/dynamic flags can hide effects.
                        arguments = node.args[1:] if unbound_tensor_method else node.args
                        flag = (arguments[0] if len(arguments) == 1 and not node.keywords
                                else node.keywords[0].value if not arguments and len(node.keywords) == 1
                                and node.keywords[0].arg == 'requires_grad' else None)
                        if (not any(isinstance(arg, ast.Starred) for arg in node.args)
                                and isinstance(flag, ast.Constant) and flag.value is True):
                            return tensor
                        reject(node, 'gradient_setup', 'Local VJP gradient setup requires an explicit literal True')
                    if qualified == 'backward' and phase == 'run':
                        return scalar
                reject(node, 'unsupported_call', 'Callable or phase lacks a reviewed value/effect binding')
            reject(node, 'unsupported_expression', 'Expression is outside the reviewed straight-line value subset')

        def assign(target, value, env, origin=None):
            if (scoped_vjp and phase == 'run' and isinstance(target, ast.Attribute)
                    and target.attr == 'grad' and isinstance(origin, ast.Constant) and origin.value is None
                    and expr(target.value, env)[0] == 'tensor'):
                return
            if isinstance(target, ast.Name):
                if target.id in protected:
                    reject(target, 'binding_reassignment', 'Framework, builtin or interface bindings cannot be replaced')
                env[target.id] = value
                return
            if isinstance(target, (ast.Tuple, ast.List)) and value[0] in ('sequence', 'result_tuple'):
                values = list(value[1].values()) if value[0] == 'result_tuple' else value[1]
                if len(target.elts) != len(values):
                    reject(target, 'unpacking_binding', 'Unpacking arity is not established')
                for item, child in zip(target.elts, values): assign(item, child, env)
                return
            if isinstance(target, ast.Subscript) and phase == 'build' and isinstance(target.value, ast.Name):
                base = env.get(target.value.id)
                if base and base[0] == 'dict' and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                    if contains(value, base):
                        reject(target, 'container_cycle', 'Direct or indirect context container cycles are not reviewed')
                    base[1][target.slice.value] = value
                    return
            reject(target, 'write_effect', 'Only local bindings and owned build dictionary entries may be assigned')

        def body(function, env):
            result = None
            for statement in function.body:
                if result is not None:
                    reject(statement, 'unreachable_statement', 'Statements after return are not reviewed')
                if isinstance(statement, ast.Assign):
                    value = expr(statement.value, env)
                    for target in statement.targets: assign(target, value, env, statement.value)
                elif isinstance(statement, ast.Expr):
                    expr(statement.value, env)
                elif isinstance(statement, ast.Return):
                    result = expr(statement.value, env)
                    if not safe(result, build=phase == 'build'):
                        reject(statement, 'opaque_return', 'Returned state/observations must have reviewed value bindings')
                elif not isinstance(statement, ast.Pass):
                    reject(statement, 'statement_effect', 'Control flow, deletion, augmented assignment and definitions are not certified')
            if result is None:
                reject(function, 'missing_return', 'Explicit return required for binding analysis')
            return result

        try:
            graph = analyze_stateful(source)
            tree = ast.parse(source)
            definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            if set(definitions) != {'build', 'run'}:
                reject(tree, 'helper_definition', 'Only build/run functions are reviewed; helper bodies cannot be hidden')
            if any(not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) for node in tree.body):
                reject(next(node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))),
                       'module_definition', 'Classes and opaque method receivers require a separate certificate')
            aliases = graph['aliases']
            imported = {key: (('dtype', value) if value in dtype_names else ('symbol', value)) for key, value in aliases.items()}
            if scoped_vjp:
                protected.add('list')
            protected.add(OPERAND_CONSTRUCTOR)
            protected.update(aliases.keys() | {'min', 'max', 'build', 'run', 'ctx', 'probe', 'device', 'seed'})
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        local = alias.asname or alias.name.split('.')[0]
                        if local in (OPERAND_CONSTRUCTOR, 'build', 'run', 'ctx', 'probe', 'device', 'seed', 'min', 'max'):
                            reject(node, 'binding_reassignment', 'Import shadows an interface or reviewed builtin')
            phase = 'build'
            context = body(definitions['build'], dict(imported, device=('device', None), seed=scalar))
            if context[0] != 'dict':
                reject(definitions['build'], 'context_binding', 'build must return a known context dictionary')
            phase = 'run'
            body(definitions['run'], dict(imported, ctx=context, probe=('symbol', 'probe')))
        except Uncertified:
            return report
        except (ContractError, SyntaxError, ValueError, TypeError, RecursionError) as error:
            report['blockers'].append({'code': 'source_contract', 'phase': phase,
                'line': getattr(error, 'lineno', 0) or 0, 'expression': '', 'reason': str(error)[:240]})
            return report
        report.update(status='verified', certificate=('explicit_scoped_local_vjp_v1' if scoped_vjp else
                      'explicit_optimizer_gradient_update_v1' if optimizer_update else
                      'explicit_straight_line_local_vjp_v1'
                      if local_vjp else 'explicit_straight_line_values_v1'))
        return report

    def operand_obligations(self, target, tree):
        """Obtain write roles from installed native schemas, never mutation recipes."""
        import inspect
        import torch
        from mirage.contracts.capabilities import MutationObligations
        from mirage.runtime.operands import discover
        paths = [d.path for d, _ in discover(tree, self)]
        unknown = {path: MutationObligations(writable=None) for path in paths}
        if not (inspect.isbuiltin(target) or inspect.ismethoddescriptor(target)):
            try:
                bound = inspect.signature(target).bind(*tree['args'], **tree['kwargs']); bound.apply_defaults()
            except (TypeError, ValueError): return unknown
            if (getattr(target,'__module__','') == 'torch.nn.functional'
                    and getattr(target,'__name__','') in ('batch_norm','layer_norm','group_norm')
                    and bound.arguments.get('training',False) is False):
                return {path:MutationObligations(writable=False) for path in paths}
            return unknown
        name = getattr(target, '__name__', '')
        try: schemas = torch._C._jit_get_schemas_for_operator('aten::' + name)
        except RuntimeError: return unknown
        role_sets = []
        for schema in schemas:
            arguments = list(schema.arguments)
            positional = [a for a in arguments if not a.kwarg_only]
            if len(tree['args']) > len(positional): continue
            bound = {a.name for a in positional[:len(tree['args'])]} | set(tree['kwargs'])
            if set(tree['kwargs']) - {a.name for a in arguments}: continue
            if any(a.name not in bound and not a.has_default_value() for a in arguments): continue
            roles = {}
            for path in paths:
                argument = (positional[path[1]] if path[0] == 'args' else
                            next((a for a in arguments if a.name == path[1]), None))
                if argument is None: break
                roles[path] = bool(argument.alias_info and argument.alias_info.is_write)
            if len(roles) == len(paths): role_sets.append(roles)
        if not role_sets: return unknown
        return {path: MutationObligations(writable=role_sets[0][path]
                    if all(roles[path] == role_sets[0][path] for roles in role_sets) else None)
                for path in paths}


    def bind_native(self, target, args, kwargs):
        import inspect
        import torch
        if not (inspect.isbuiltin(target) or inspect.ismethoddescriptor(target)):
            return Evidence(EvidenceStatus.UNKNOWN, 'native callable schema unavailable')
        try: schemas=torch._C._jit_get_schemas_for_operator('aten::'+getattr(target,'__name__',''))
        except RuntimeError: schemas=[]
        if not schemas: return Evidence(EvidenceStatus.UNKNOWN,'native callable schema unavailable')
        if set(kwargs)-{a.name for schema in schemas for a in schema.arguments}:
            return Evidence(EvidenceStatus.UNKNOWN,'Python/native keyword aliases unresolved')
        for schema in schemas:
            positional=[a for a in schema.arguments if not a.kwarg_only]
            if len(args)>len(positional): continue
            bound={a.name for a in positional[:len(args)]}|set(kwargs)
            if all(a.name in bound or a.has_default_value() for a in schema.arguments):
                return Evidence(EvidenceStatus.VERIFIED,'native arity binding only',True,'call_binding','adapter')
        return Evidence(EvidenceStatus.REJECTED,'No installed native schema binds these arguments; check receiver and required operands')


def context_state():
    import torch
    return {'grad_enabled':torch.is_grad_enabled(),
            'cpu_enabled':torch.is_autocast_enabled('cpu'),
            'cpu_dtype':str(torch.get_autocast_dtype('cpu')),
            'cuda_enabled':torch.is_autocast_enabled('cuda'),
            'cuda_dtype':str(torch.get_autocast_dtype('cuda')),
            'cache':torch.is_autocast_cache_enabled()}


class ContextObserver:
    """Observe original context methods; never generate recursion or enter scopes."""
    def __init__(self, target, records, on_action=None):
        import torch
        self.target, self.records, self.stack = target, records, []
        self.instances = []
        self.wrappers = {}
        self.decorator_methods = set()
        call_code = getattr(getattr(target, '__call__', None), '__code__', None)
        if isinstance(target, type) and call_code:
            self.decorator_methods.add(call_code)
        self.on_action = on_action
        self.supported = target in (torch.autocast, torch.no_grad, torch.enable_grad)
        methods = [getattr(getattr(target, name, None), '__code__', None) for name in ('__enter__', '__exit__')]
        self.codes = dict(zip(methods, ('enter', 'exit'))) if isinstance(target, type) and all(methods) else {}

    def __enter__(self):
        import sys
        import torch
        self.previous = sys.getprofile()
        self.restore = {'grad':torch.is_grad_enabled(), 'cache':torch.is_autocast_cache_enabled(),
                        'cpu':(torch.is_autocast_enabled('cpu'), torch.get_autocast_dtype('cpu')),
                        'cuda':(torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))}
        if self.codes: sys.setprofile(self.profile)
        return self

    def register_product(self, value):
        # A function factory's concrete context type becomes known only after return.
        import sys
        methods = [getattr(getattr(type(value), name, None), '__code__', None)
                   for name in ('__enter__', '__exit__')]
        if all(methods):
            if not any(item is value for item in self.instances):
                self.instances.append(value)
            if not self.codes:
                sys.setprofile(self.profile)
            self.codes.update(zip(methods, ('enter','exit')))
            call_code = getattr(getattr(type(value), '__call__', None), '__code__', None)
            if call_code:
                self.decorator_methods.add(call_code)

    def profile(self, frame, event, arg):
        action = self.codes.get(frame.f_code)
        instance = frame.f_locals.get('self')
        registering = event == 'return' and frame.f_code in self.decorator_methods
        if (action or registering) and ((isinstance(self.target,type) and isinstance(instance,self.target))
                       or any(item is instance for item in self.instances)):
            if registering:
                from types import FunctionType
                if isinstance(arg, FunctionType) and arg.__closure__:
                    self.wrappers.setdefault(arg.__code__, []).append((instance, arg))
            if action == 'enter' and event == 'call':
                if not any(item is instance for item in self.instances): self.instances.append(instance)
                group = next(i for i,item in enumerate(self.instances) if item is instance)
                caller=frame.f_back
                # The exact wrapper returned by a target product owns this entry.
                # Code identity alone would conflate independent decorated functions.
                owner = None
                if caller:
                    for candidate, fn in self.wrappers.get(caller.f_code, ()):
                        if all(caller.f_locals.get(name) is cell.cell_contents
                               for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)):
                            owner = candidate
                            break
                decorated = owner is not None
                reentrant=any(item[0] is instance for item in self.stack)
                owner_group = next((i for i, item in enumerate(self.instances) if item is owner), None) if decorated else None
                self.stack.append((instance, context_state(), group, len(self.stack), owner_group, reentrant))
                if self.on_action: self.on_action('context_enter')
            elif action == 'exit' and event == 'return':
                if self.stack and self.stack[-1][0] is instance:
                    _, before, group, depth, owner_group, reentrant = self.stack.pop()
                    self.records.append({'before':before, 'after':context_state(), 'balanced':True,
                                         'instance_group':group, 'entry_depth':depth,
                                         'decorated_entry':owner_group is not None and (not reentrant or group == owner_group),
                                         'wrapper_owner_group':owner_group, 'reentrant_same_instance':reentrant})
                    if self.on_action: self.on_action('context_exit')
                else:
                    self.records.append({'balanced':False})
        if self.previous: self.previous(frame, event, arg)

    def __exit__(self, kind, value, traceback):
        import sys
        import torch
        if self.codes: sys.setprofile(self.previous)
        if self.stack: self.records.append({'balanced':False, 'reason':'unclosed observed context'})
        # Cleanup happens after observations, including failures; it is not a
        # transition credited to the generated program or an oracle control.
        torch.set_grad_enabled(self.restore['grad'])
        torch.set_autocast_cache_enabled(self.restore['cache'])
        for device in ('cpu', 'cuda'):
            enabled, dtype = self.restore[device]
            torch.set_autocast_enabled(device, enabled)
            torch.set_autocast_dtype(device, dtype)


def probe_context_decorator(product):
    """Exercise an independent copy after observation. No probe event grants coverage."""
    import copy
    if not callable(product) or not hasattr(type(product), '__enter__') or not hasattr(type(product), '__exit__'):
        return 'unsupported'
    try:
        with ContextObserver(type(product), []) as observer:
            context = copy.deepcopy(product)
            if context is product:
                return 'unknown'
            observer.register_product(context)
            entries = []
            token = object()

            def body():
                entries.append(tuple(frame[0] for frame in observer.stack))
                return token

            wrapped = context(body)
            if not callable(wrapped):
                return 'unsupported'
            with context:
                returned = wrapped()
            if returned is token and len(entries) == 1 and len(entries[0]) == 2 and entries[0][0] is context:
                return 'same_instance' if entries[0][1] is context else 'different_instance'
    except Exception:
        return 'unknown'
    return 'unknown'


class CompilerObserver:
    """Observe compiler-created graph execution without inventing a compile call."""
    def __init__(self, probe, namespace, config):
        self.probe, self.namespace, self.config = probe, namespace, config
        self.graphs = []
        self.factory_records = []

    def __enter__(self):
        import torch
        from functools import wraps
        self.original = torch.compile
        self.factory_target = self.config['api'] if self.probe._target is self.original else 'torch.compile'
        if self.probe._target is self.original:
            self.probe.factory_requires_invocation = True
        self.aliases = {k:v for k,v in self.namespace.items() if v is self.original}
        @wraps(self.original)
        def wrapped(model=None, **kwargs):
            if model is None: return lambda fn: wrapped(fn, **kwargs)
            return self.compile(model, kwargs)
        self.wrapped = wrapped
        torch.compile = wrapped
        for key in self.aliases: self.namespace[key] = wrapped
        return self

    def __exit__(self, kind, value, traceback):
        import torch
        torch.compile = self.original
        self.namespace.update(self.aliases)

    def compile(self, model, kwargs):
        import hashlib
        import torch
        from contextlib import nullcontext
        from torch._dynamo.backends.registry import lookup_backend
        from torch.utils._pytree import tree_flatten
        from torch.overrides import _get_current_function_mode, _pop_mode_temporarily
        from mirage.runtime.stateful import snapshot
        from mirage.runtime.program import leaves
        from mirage.runtime.operands import discover
        from dataclasses import asdict
        self.probe._validate_callable_model(model)
        factory_id = len(self.factory_records)
        callable_id = factory_id
        invocation = {'next': 0, 'active': None}
        requested = kwargs.pop('backend', 'inductor')
        selected = 'eager' if self.config.get('compiler_reference') else requested
        actual_backend = lookup_backend(selected) if isinstance(selected, str) else selected
        probe = self.probe

        def backend(graph, examples, **backend_options):
            fingerprint = hashlib.sha256(graph.code.encode()).hexdigest()
            placeholders = [n for n in graph.graph.nodes if n.op == 'placeholder']
            matches = [n for n in graph.graph.nodes if n.op.startswith('call_') and
                       (n.target is probe._target or n.op == 'call_method' and n.target == getattr(probe._target, '__name__', None))]
            macro = probe._target is self.original
            output_node = next(n for n in graph.graph.nodes if n.op == 'output')
            output_nodes, _ = tree_flatten(output_node.args[0])
            def connected(node, destination):
                pending, seen = [node], set()
                while pending:
                    current = pending.pop()
                    if current in seen: continue
                    seen.add(current)
                    if current is destination: return True
                    value = current.meta.get('example_value', current.meta.get('val'))
                    if not any(t.requires_grad for _,t in leaves(value)): continue
                    if getattr(current.target, '__name__', current.target) in ('detach','detach_'): continue
                    pending.extend(user for user in current.users if user.op != 'output')
                return False
            differentiable = [macro or any(connected(n, o) for n in matches) for o in output_nodes]
            compiled = actual_backend(graph, examples, **backend_options)

            def invoke(*values):
                mapping = dict(zip(placeholders, values))
                def resolve_node(value):
                    if isinstance(value, torch.fx.Node):
                        if value not in mapping: raise ValueError('intermediate target operands not observed')
                        return mapping[value]
                    if isinstance(value, tuple): return tuple(resolve_node(v) for v in value)
                    if isinstance(value, list): return [resolve_node(v) for v in value]
                    if isinstance(value, dict): return {k:resolve_node(v) for k,v in value.items()}
                    return value
                # Boundary metadata is valid only when graph inputs directly feed
                # the sole operation, without intervening mutations/materialization.
                direct = (len(matches) == 1 and
                          sum(n.op.startswith('call_') for n in graph.graph.nodes) == 1 and
                          not getattr(probe._target, '__name__', '').endswith('_'))
                tree, descriptors, properties, edges = None, [], [], []
                if direct:
                    try:
                        tree = {'args':resolve_node(matches[0].args), 'kwargs':resolve_node(matches[0].kwargs)}
                        operands = discover(tree, probe._backend)
                        descriptors = [asdict(d) for d,_ in operands]
                        properties = [{'path':list(d.path), 'facts':probe._backend.input_properties(t)} for d,t in operands]
                        edges = probe._backend.gradient_edges(operands)
                    except ValueError:
                        tree, descriptors, properties, edges = None, [], [], []
                if macro:
                    # Graph-input edges protect identity outputs from being credited
                    # as a backward through the returned compiled callable.
                    edges = probe._backend.gradient_edges(discover({'args': values}, probe._backend))
                entry_context = dict(context_state(), inference_enabled=torch.is_inference_mode_enabled())
                target_snapshot = snapshot(tree)
                input_snapshot = snapshot(values)
                outputs = compiled(*values)
                self.graphs.append({'fingerprint':fingerprint, 'backend':str(selected),
                                    'target_matches':len(matches), 'macro_target':macro, 'inputs':input_snapshot,
                                    'factory_id':factory_id, 'callable_id':callable_id,
                                    'invocation_index':invocation['active'],
                                    'target_execution_certified':selected in ('eager','inductor','aot_eager') and bool(matches or macro)})
                probe.operations.append({'name':'compiled_execute'})
                probe._recorder.emit('compiled_execute', fingerprint=fingerprint, target_matches=len(matches))
                if (not matches and not macro) or selected not in ('eager','inductor','aot_eager'): return outputs
                index = len(probe.calls)
                probe.calls.append({'index':index, 'inputs':target_snapshot, 'metadata':descriptors,
                                    'input_properties':properties, 'output':snapshot(outputs),
                                    'evidence':'executed FX graph', 'target_operands_observed':tree is not None,
                                    'execution_mode':'compiled', 'backend':str(selected), 'entry_context':entry_context,
                                    **invocation.get('product_observation', {}),
                                    **({'kind':'compiled_callable_execution', 'factory_target':self.factory_target,
                                        'factory_id':factory_id, 'callable_id':callable_id,
                                        'invocation_index':invocation['active'],
                                        'callable_identity_verified':invocation['active'] is not None} if macro else {})})
                probe._recorder.emit('api_exit', call=index, evidence='executed FX graph')
                actual_outputs, _ = tree_flatten(outputs)
                if len(actual_outputs) == len(differentiable):
                    observed = [(str(position), tensor) for position, (tensor, reaches)
                                in enumerate(zip(actual_outputs, differentiable))
                                if reaches and isinstance(tensor, torch.Tensor)]
                    direct = (direct and tree is not None and
                              all(o is matches[0] for o, reaches in zip(output_nodes, differentiable) if reaches))
                    probe.observe_outputs(index, observed, edges, node_admissible=macro or direct, boundary=macro)
                return outputs
            return invoke
        compiled_model = self.original(model, backend=backend, **kwargs)
        def invoke_model(*args, **kw):
            # Only suspend our observation mode, never a user-selected mode.
            mode = (_pop_mode_temporarily() if _get_current_function_mode() is getattr(probe, '_action_observer', None)
                    and _get_current_function_mode() is not None else nullcontext())
            previous_resources = invocation.get('product_observation', {})
            if probe._target is self.original and self.config.get('state_goal'):
                tree = {'args': args, 'kwargs': kw}
                resources = probe._target_resources(discover(tree, probe._backend))
                invocation['product_observation'] = dict(product_resources=resources,
                    product_inputs=snapshot(tree) if resources else None)
            previous = invocation['active']
            invocation['active'] = invocation['next']
            invocation['next'] += 1
            try:
                with mode: return compiled_model(*args, **kw)
            finally:
                invocation['active'] = previous
                invocation['product_observation'] = previous_resources
        self.factory_records.append({'factory_id':factory_id, 'factory_target':self.factory_target,
                                     'callable_ids':[callable_id]})
        probe._register_callable(invoke_model, factory_target=self.factory_target,
                                 factory_id=factory_id, callable_id=callable_id)
        return invoke_model


class CaptureObserver:
    """Observe public graph hooks and calls to factory-returned modules/functions."""
    def __init__(self, probe, namespace, config):
        self.probe, self.namespace, self.config = probe, namespace, config
        self.captured, self.replays, self.factory_records = {}, [], []
        self.active = False
        self.substitutions = 0

    def __enter__(self):
        import torch
        from functools import wraps
        self.original = torch.cuda.make_graphed_callables
        self.factory_target = self.config['api'] if self.probe._target is self.original else 'torch.cuda.make_graphed_callables'
        if self.probe._target is self.original:
            self.probe.factory_requires_invocation = True
        self.aliases = {k:v for k,v in self.namespace.items() if v is self.original}
        def captured(graph):
            if self.active: self.captured.setdefault(graph, len(self.captured))
        def replayed(graph):
            if graph in self.captured:
                self.replays.append(self.captured[graph])
                self.probe._recorder.emit('cuda_graph_replay', graph=self.captured[graph])
        self.handles = [torch.cuda.graphs.register_graph_capture_end_hook(captured),
                        torch.cuda.graphs.register_graph_replay_end_hook(replayed)]
        @wraps(self.original)
        def wrapped(callables, sample_args, **kwargs):
            return self.factory(callables, sample_args, kwargs)
        torch.cuda.make_graphed_callables = wrapped
        for key in self.aliases: self.namespace[key] = wrapped
        return self

    def __exit__(self, kind, value, traceback):
        import torch
        for handle in self.handles: handle.remove()
        torch.cuda.make_graphed_callables = self.original
        self.namespace.update(self.aliases)

    def factory(self, callables, sample_args, kwargs):
        import torch
        from contextlib import nullcontext
        from torch.overrides import _get_current_function_mode, _pop_mode_temporarily
        from mirage.runtime.stateful import snapshot
        from mirage.runtime.program import leaves
        from mirage.runtime.operands import discover
        from dataclasses import asdict
        modules = callables if isinstance(callables, tuple) else (callables,)
        probe = self.probe
        probe._validate_capture_callables(callables, sample_args)
        factory_id = len(self.factory_records)
        def model_state():
            observing=probe._observing;probe._observing=True
            try: return [snapshot(m.state_dict()) if isinstance(m,torch.nn.Module) else None for m in modules]
            finally: probe._observing=observing
        model_before = model_state()
        previous = probe._observing; probe._observing = True
        self.active = True
        try:
            mode = (_pop_mode_temporarily() if _get_current_function_mode() is getattr(probe,'_action_observer',None)
                    and _get_current_function_mode() is not None else nullcontext())
            with mode:
                if self.config.get('capture_reference'):
                    self.substitutions += 1
                    result = callables
                else:
                    result = self.original(callables, sample_args, **kwargs)
        finally:
            self.active = False; probe._observing = previous
        model_after = model_state()
        self.factory_records.append({'factory_id':factory_id, 'factory_target':self.factory_target,
                                     'callable_ids':list(range(len(modules))),
                                     'before':model_before,'after':model_after,
                                     'buffer_free':all(isinstance(m,torch.nn.Module) and not list(m.buffers()) for m in modules),
                                     'reference':bool(self.config.get('capture_reference'))})
        owner = self
        class ObservedCallable:
            def __init__(self, original, callable_id):
                self.original, self.callable_id, self.invocations = original, callable_id, 0
            def __getattr__(self, name): return getattr(self.original, name)
            def __call__(self, *args, **kw):
                if probe._target is not owner.original:
                    return self.original(*args, **kw)
                invocation_index = self.invocations
                self.invocations += 1
                tree = {'args':args,'kwargs':kw}
                observing = probe._observing; probe._observing = True
                try:
                    operands = discover(tree, probe._backend)
                    descriptors = [asdict(d) for d, _ in operands]
                    properties = [{'path':list(d.path),'facts':probe._backend.input_properties(t)} for d,t in operands]
                    entry_context = dict(context_state(), inference_enabled=torch.is_inference_mode_enabled())
                    before = snapshot(tree)
                    input_edges = probe._backend.gradient_edges(operands)
                    reference = bool(owner.config.get('capture_reference'))
                    if isinstance(self.original, torch.nn.Module):
                        input_edges += probe._backend.gradient_edges(discover(
                            {'parameters':dict(self.original.named_parameters())}, probe._backend))
                finally:
                    probe._observing = observing
                start = len(owner.replays)
                output = self.original(*args, **kw)
                completed_replays = len(owner.replays)-start
                index = len(probe.calls)
                probe._observing = True
                try:
                    probe.calls.append({'index':index,'kind':'factory_result_execution',
                                        'inputs':before,'metadata':descriptors, 'input_properties':properties,
                                        'registered_resources':probe._target_resources(operands) if owner.config.get('state_goal') else [],
                                        'output':snapshot(output),'completed_replays':completed_replays,
                                        'reference':reference, 'entry_context':entry_context,
                                        'factory_target':owner.factory_target,
                                        'factory_id':factory_id, 'callable_id':self.callable_id,
                                        'invocation_index':invocation_index, 'callable_identity_verified':True,
                                        'execution_mode':'reference' if reference else 'captured'})
                    probe.operations.append({'name':'factory_result_execution'})
                    probe._recorder.emit('factory_result_execution',call=index,completed_replays=completed_replays)
                    probe.observe_outputs(index, leaves(output), input_edges, boundary=True,
                                          node_admissible=reference or completed_replays > 0)
                finally: probe._observing = observing
                return output
        wrapped = tuple(ObservedCallable(x, i) for i, x in enumerate(result)) if isinstance(result,tuple) else (ObservedCallable(result, 0),)
        for item in wrapped:
            probe._register_callable(item, factory_target=self.factory_target,
                                     factory_id=factory_id, callable_id=item.callable_id)
        return wrapped if isinstance(result,tuple) else wrapped[0]


def capture_source_certificate(source):
    """Initial capture reference domain: buffer-free modules with pure forward expressions."""
    import ast
    tree = ast.parse(source)
    forwards = [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == 'forward']
    if not forwards: return False
    operations = {'sum','mean','mul','add','sub','square','sin','cos','matmul','relu'}
    for function in forwards:
        for node in ast.walk(function):
            if isinstance(node,(ast.If,ast.For,ast.While,ast.Try,ast.With,ast.Global,ast.Nonlocal)):
                return False
            if isinstance(node,(ast.Attribute,ast.Subscript)) and isinstance(node.ctx,ast.Store): return False
            if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Attribute) or node.func.attr not in operations): return False
    return True


class OptimizerObserver:
    """Observe completed source-owned updates and clearing; never schedule them."""
    def __init__(self,optimizer,on_action, *, before_step=None, after_step=None):
        self.optimizer,self.on_action=optimizer,on_action
        self.before_step,self.after_step=before_step,after_step

    def __enter__(self):
        import sys
        import inspect
        self.previous=sys.getprofile();self.handle=None;self.pre_handle=None;self.zero_code=None
        if self.optimizer is not None:
            import torch
            if not isinstance(self.optimizer,torch.optim.Optimizer):
                # Custom stateful objects may execute but cannot supply native
                # optimizer completion or gradient-consumption evidence.
                return self
            if self.before_step is not None:
                self.pre_handle=self.optimizer.register_step_pre_hook(self.before_step)
            def completed(*args):
                if self.after_step is not None:
                    self.after_step(*args)
                self.on_action('optimizer_step')
            self.handle=self.optimizer.register_step_post_hook(completed)
            self.zero_code=getattr(inspect.unwrap(self.optimizer.zero_grad),'__code__',None)
            sys.setprofile(self.profile)
        return self

    def profile(self,frame,event,arg):
        if event=='return' and frame.f_code is self.zero_code and frame.f_locals.get('self') is self.optimizer:
            self.on_action('zero_grad')
        if self.previous: self.previous(frame,event,arg)

    def __exit__(self,kind,value,traceback):
        import sys
        if self.optimizer is not None: sys.setprofile(self.previous)
        if self.handle is not None: self.handle.remove()
        if self.pre_handle is not None: self.pre_handle.remove()


def execution_source_certificate(source):
    """Reviewed numerical-source subset; implementation identity is not an observable.

    This is deliberately conservative: arbitrary callbacks, metadata inspection,
    data-dependent Python branches and unreviewed stateful objects remain unknown.
    """
    import ast
    from mirage.generator.program import analyze_stateful
    aliases=analyze_stateful(source)['aliases'];tree=ast.parse(source)
    def qualified(node):
        if isinstance(node,ast.Name): return aliases.get(node.id,node.id)
        if isinstance(node,ast.Attribute): return qualified(node.value)+'.'+node.attr
        return ''
    helpers={n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    compiled=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
            call=node.value
            factory=(qualified(call.func)=='torch.compile' or qualified(call.func)=='probe.call'
                     and call.args and qualified(call.args[0])=='torch.compile')
            if factory: compiled.update(t.id for t in node.targets if isinstance(t,ast.Name))
    operations={'sin','cos','exp','log','sqrt','tanh','sigmoid','abs','neg','add','sub','mul','div',
                'square','sum','mean','std','matmul','mm','bmm','slogdet','lu_solve','gather','scatter_add',
                'clone','detach','view','reshape','transpose','permute','contiguous','conj','conj_physical',
                'resolve_conj','resolve_neg','real','imag','add_','sub_','mul_','copy_','zero_','fill_',
                'backward','requires_grad_','to','float','double','unsqueeze','squeeze','expand'}
    constructors={'tensor','zeros','ones','arange','linspace','full','eye','complex','finfo','manual_seed'}
    library={"torch."+name for name in operations|constructors}|{"torch.Tensor."+name for name in operations}
    library|={'torch.compile','torch.nn.Parameter','torch.optim.SGD','torch.optim.Adam','torch.optim.AdamW',
              'torch.linalg.slogdet','torch.linalg.lu_solve','torch.linalg.lu_factor','torch.autograd.grad'}
    attributes=operations|{'grad','shape','dtype','device','get','logabsdet','sign','T','mT','tiny','max','min','eps','smallest_normal'}
    namespaces={'torch','torch.Tensor','torch.nn','torch.optim','torch.linalg','torch.autograd'}
    literals={'torch.'+n for n in ('float16','float32','float64','bfloat16','complex64','complex128','int32','int64','bool')}
    for function in [n for n in tree.body if isinstance(n,ast.FunctionDef)]:
        local_dicts={target.id for item in function.body if isinstance(item,ast.Assign) and isinstance(item.value,ast.Dict)
                     for target in item.targets if isinstance(target,ast.Name)}
        for node in ast.walk(function):
            if isinstance(node,(ast.ClassDef,ast.If,ast.IfExp,ast.While,ast.Try,ast.With,ast.AsyncWith,
                                ast.ListComp,ast.DictComp,ast.SetComp,ast.GeneratorExp,ast.Compare,ast.BoolOp)):
                return False
            if isinstance(node,ast.FunctionDef) and node.decorator_list: return False
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store) and node.id in helpers|set(aliases): return False
            if isinstance(node,ast.For):
                if not isinstance(node.iter,ast.Call) or qualified(node.iter.func)!='range': return False
                try:
                    arguments=[ast.literal_eval(x) for x in node.iter.args]
                    if any(type(x) is not int for x in arguments) or len(range(*arguments))>4: return False
                except (ValueError,TypeError): return False
            if isinstance(node,(ast.Attribute,ast.Subscript)) and isinstance(node.ctx,ast.Store):
                if not (function.name=='build' and isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name)
                        and node.value.id in local_dicts): return False
            if isinstance(node,ast.Attribute):
                name=qualified(node)
                if name.startswith('torch.'):
                    if name not in library|namespaces|literals: return False
                elif node.attr not in attributes and name not in ('probe.call','probe.mark','math.sqrt'): return False
            if isinstance(node,ast.Call):
                name=qualified(node.func)
                if name in library|{'probe.call','probe.mark','range','math.sqrt'}: continue
                if isinstance(node.func,ast.Name) and node.func.id in helpers|compiled: continue
                if isinstance(node.func,ast.Attribute) and not name.startswith('torch.') and node.func.attr in operations|{'get'}: continue
                return False
    return True
