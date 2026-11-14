"""PyTorch-specific facts behind the backend-neutral descriptor boundary."""
from mirage.contracts.capabilities import Evidence, EvidenceStatus, OperandDescriptor


class PyTorchBackend:
    name = "pytorch"

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

    def representation_evidence(self, left, right):
        from mirage.runtime.program import leaves
        # One semantic-class registry serves all value-representation comparisons.
        # This is capability admission, never an API-specific expected-value formula.
        classes = {
            'pointwise': {'sin', 'cos', 'exp', 'log', 'sqrt', 'tanh', 'sigmoid', 'abs', 'neg',
                          'add', 'sub', 'mul', 'div', 'square'},
            'reduction': {'sum', 'mean', 'std'},
            'linear_algebra': {'matmul', 'mm', 'bmm', 'slogdet', 'lu_solve'},
            'indexed_values': {'gather', 'scatter_add'},
            'normalization': {'batch_norm', 'layer_norm', 'group_norm'},
            'write_values': {'copy_', 'add_', 'sub_', 'mul_', 'div_', 'fill_', 'zero_'},
        }
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
        if operation in classes['linear_algebra']:
            import torch
            matrices = [t for call in left.get('target_calls', ())
                        for _, t in leaves(call['inputs'])
                        if t.ndim >= 2 and t.shape[-1] == t.shape[-2] and (t.is_floating_point() or t.is_complex())]
            try:
                numerical = bool(matrices) and all(bool(torch.isfinite(t).all())
                    and bool((torch.linalg.cond(t.to(torch.complex128 if t.is_complex() else torch.float64)) <= 100).all())
                    for t in matrices)
            except (RuntimeError, ValueError):
                numerical = False
            numerical_reason = 'bounded observed matrix conditioning; no universal matrix certificate'
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
        """Conservative straight-line certificate; unsupported source remains executable.

        This admits a semantic subset, not arbitrary Python purity. In particular,
        callbacks, modules, data-dependent control flow and metadata inspection
        cannot acquire a certificate merely from successful execution.
        """
        import ast
        from mirage.generator.program import analyze_stateful
        graph = analyze_stateful(source)
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if {n.name for n in functions} != {'build', 'run'}:
            return None
        aliases = graph['aliases']
        def name(node):
            if isinstance(node, ast.Name): return aliases.get(node.id, node.id)
            if isinstance(node, ast.Attribute): return name(node.value) + '.' + node.attr
            return ''
        def primitive_helper(node):
            def literal(item):
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id in ('min', 'max') and not item.keywords:
                    values = [literal(argument) for argument in item.args]
                    return (min if item.func.id == 'min' else max)(*values)
                value = ast.literal_eval(item)
                def primitive(v):
                    return type(v) in (int, float, bool) or isinstance(v, (list, tuple)) and all(primitive(x) for x in v)
                if not primitive(value): raise ValueError('nonprimitive helper operand')
                return value
            try:
                if sum(1 for _ in ast.walk(node)) > 64: return False
                literal(node)
                return True
            except (ValueError, TypeError, SyntaxError):
                return False
        constructors = {'torch.tensor', 'torch.zeros', 'torch.ones', 'torch.arange',
                        'torch.linspace', 'torch.full', 'torch.finfo', 'torch.complex',
                        'torch.sparse_csr_tensor', 'torch.linalg.lu_factor', 'torch.manual_seed',
                        'torch.optim.SGD', 'torch.optim.Adam', 'torch.optim.AdamW'}
        value_ops = {'sin', 'cos', 'exp', 'log', 'sqrt', 'tanh', 'sigmoid', 'abs', 'neg',
                     'add', 'sub', 'mul', 'div', 'square', 'sum', 'mean', 'std', 'matmul',
                     'mm', 'bmm', 'slogdet', 'lu_solve', 'gather', 'scatter_add'}
        forbidden_attributes = {'data', 'grad_fn', 'is_leaf', 'is_conj', 'is_neg',
                                'stride', 'storage_offset', 'is_contiguous', 'data_ptr'}
        for function in functions:
            local_dicts = {target.id for statement in function.body if isinstance(statement,ast.Assign)
                           and isinstance(statement.value,ast.Dict) for target in statement.targets if isinstance(target,ast.Name)}
            for node in ast.walk(function):
                if isinstance(node, (ast.For, ast.While, ast.If, ast.IfExp, ast.Try, ast.With,
                                     ast.Lambda, ast.ClassDef, ast.ListComp, ast.DictComp,
                                     ast.SetComp, ast.GeneratorExp, ast.Await, ast.Yield)):
                    return None
                if isinstance(node, ast.FunctionDef) and node is not function: return None
                if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, ast.Store):
                    local_build_write = (function.name == 'build' and isinstance(node,ast.Subscript)
                        and isinstance(node.value,ast.Name) and node.value.id in local_dicts
                        and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str))
                    if not local_build_write: return None
                if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in forbidden_attributes): return None
                if isinstance(node, ast.Call):
                    qualified = name(node.func)
                    leaf = node.func.attr if isinstance(node.func, ast.Attribute) else ''
                    if qualified in ('min', 'max') and primitive_helper(node): continue
                    if function.name == 'build':
                        if qualified not in constructors and leaf not in ('requires_grad_', 'reshape', 'view', 'expand', 'conj', 'conj_physical', 'detach', 'to_mkldnn', 'clone', 'contiguous', 'zeros_like', 'ones_like'): return None
                    elif qualified not in ('probe.call', 'probe.mark') and leaf not in value_ops | {'backward', 'clone'}:
                        return None
        return 'explicit_straight_line_values_v1'

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
        self.on_action = on_action
        self.supported = target in (torch.autocast, torch.no_grad, torch.enable_grad)
        self.codes = ({target.__enter__.__code__: 'enter', target.__exit__.__code__: 'exit'}
                      if self.supported else {})

    def __enter__(self):
        import sys
        import torch
        self.previous = sys.getprofile()
        self.restore = {'grad':torch.is_grad_enabled(), 'cache':torch.is_autocast_cache_enabled(),
                        'cpu':(torch.is_autocast_enabled('cpu'), torch.get_autocast_dtype('cpu')),
                        'cuda':(torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))}
        if self.supported: sys.setprofile(self.profile)
        return self

    def profile(self, frame, event, arg):
        action = self.codes.get(frame.f_code)
        instance = frame.f_locals.get('self')
        if action and isinstance(instance, self.target):
            if action == 'enter' and event == 'call':
                if not any(item is instance for item in self.instances): self.instances.append(instance)
                group = next(i for i,item in enumerate(self.instances) if item is instance)
                caller=frame.f_back
                decorated=bool(caller and (caller.f_globals.get('__name__'),caller.f_code.co_name) in {
                    ('torch.amp.autocast_mode','decorate_autocast'),
                    ('torch.utils._contextlib','decorate_context')})
                reentrant=any(item[0] is instance for item in self.stack)
                self.stack.append((instance, context_state(), group, len(self.stack), decorated, reentrant))
                if self.on_action: self.on_action('context_enter')
            elif action == 'exit' and event == 'return':
                if self.stack and self.stack[-1][0] is instance:
                    _, before, group, depth, decorated, reentrant = self.stack.pop()
                    self.records.append({'before':before, 'after':context_state(), 'balanced':True,
                                         'instance_group':group, 'entry_depth':depth,
                                         'decorated_entry':decorated,'reentrant_same_instance':reentrant})
                    if self.on_action: self.on_action('context_exit')
                else:
                    self.records.append({'balanced':False})
        if self.previous: self.previous(frame, event, arg)

    def __exit__(self, kind, value, traceback):
        import sys
        import torch
        if self.supported: sys.setprofile(self.previous)
        if self.stack: self.records.append({'balanced':False, 'reason':'unclosed observed context'})
        # Cleanup happens after observations, including failures; it is not a
        # transition credited to the generated program or an oracle control.
        torch.set_grad_enabled(self.restore['grad'])
        torch.set_autocast_cache_enabled(self.restore['cache'])
        for device in ('cpu', 'cuda'):
            enabled, dtype = self.restore[device]
            torch.set_autocast_enabled(device, enabled)
            torch.set_autocast_dtype(device, dtype)


class CompilerObserver:
    """Observe compiler-created graph execution without inventing a compile call."""
    def __init__(self, probe, namespace, config):
        self.probe, self.namespace, self.config = probe, namespace, config
        self.graphs = []

    def __enter__(self):
        import torch
        from functools import wraps
        self.original = torch.compile
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
                entry_context = dict(context_state(), inference_enabled=torch.is_inference_mode_enabled())
                target_snapshot = snapshot(tree)
                input_snapshot = snapshot(values)
                outputs = compiled(*values)
                self.graphs.append({'fingerprint':fingerprint, 'backend':str(selected),
                                    'target_matches':len(matches), 'macro_target':macro, 'inputs':input_snapshot,
                                    'target_execution_certified':selected in ('eager','inductor','aot_eager') and bool(matches or macro)})
                probe.operations.append({'name':'compiled_execute'})
                probe._recorder.emit('compiled_execute', fingerprint=fingerprint, target_matches=len(matches))
                if (not matches and not macro) or selected not in ('eager','inductor','aot_eager'): return outputs
                index = len(probe.calls)
                probe.calls.append({'index':index, 'inputs':target_snapshot, 'metadata':descriptors,
                                    'input_properties':properties, 'output':snapshot(outputs),
                                    'evidence':'executed FX graph', 'target_operands_observed':tree is not None,
                                    'execution_mode':'compiled', 'backend':str(selected), 'entry_context':entry_context})
                probe._recorder.emit('api_exit', call=index, evidence='executed FX graph')
                actual_outputs, _ = tree_flatten(outputs)
                if len(actual_outputs) == len(differentiable):
                    observed = [(str(position), tensor) for position, (tensor, reaches)
                                in enumerate(zip(actual_outputs, differentiable))
                                if reaches and isinstance(tensor, torch.Tensor)]
                    direct = (direct and tree is not None and
                              all(o is matches[0] for o, reaches in zip(output_nodes, differentiable) if reaches))
                    probe.observe_outputs(index, observed, edges, node_admissible=direct)
                return outputs
            return invoke
        compiled_model = self.original(model, backend=backend, **kwargs)
        def invoke_model(*args, **kw):
            # Only suspend our observation mode, never a user-selected mode.
            mode = (_pop_mode_temporarily() if _get_current_function_mode() is getattr(probe, '_action_observer', None)
                    and _get_current_function_mode() is not None else nullcontext())
            with mode: return compiled_model(*args, **kw)
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
        self.factory_records.append({'before':model_before,'after':model_after,
                                     'buffer_free':all(isinstance(m,torch.nn.Module) and not list(m.buffers()) for m in modules),
                                     'reference':bool(self.config.get('capture_reference'))})
        owner = self
        class ObservedCallable:
            def __init__(self, original): self.original = original
            def __getattr__(self, name): return getattr(self.original, name)
            def __call__(self, *args, **kw):
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
                    if reference and isinstance(self.original, torch.nn.Module):
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
                                        'output':snapshot(output),'completed_replays':completed_replays,
                                        'reference':reference, 'entry_context':entry_context,
                                        'execution_mode':'reference' if reference else 'captured'})
                    probe.operations.append({'name':'factory_result_execution'})
                    probe._recorder.emit('factory_result_execution',call=index,completed_replays=completed_replays)
                    probe.observe_outputs(index, leaves(output), input_edges, boundary=not reference,
                                          node_admissible=reference or completed_replays > 0)
                finally: probe._observing = observing
                return output
        return tuple(ObservedCallable(x) for x in result) if isinstance(result,tuple) else ObservedCallable(result)


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
    def __init__(self,optimizer,on_action):
        self.optimizer,self.on_action=optimizer,on_action

    def __enter__(self):
        import sys
        import inspect
        self.previous=sys.getprofile();self.handle=None;self.zero_code=None
        if self.optimizer is not None:
            import torch
            if not isinstance(self.optimizer,torch.optim.Optimizer):
                raise ValueError('Registered optimizer must implement the PyTorch optimizer contract')
            self.handle=self.optimizer.register_step_post_hook(lambda *args:self.on_action('optimizer_step'))
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
