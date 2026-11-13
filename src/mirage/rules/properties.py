"""Reviewed property families over runtime observations, independent of benchmarks.

The registry describes API semantics, not known failures. Reference formulas and
materialization are valid only after the corresponding predicates are established.
"""
from contextlib import nullcontext
import copy
import itertools
import math

from mirage.inspector.comparison import differences


VERSION = 1
PURE = {'torch.sin', 'torch.cos', 'torch.neg', 'torch.clone', 'torch.matmul',
        'torch.mm', 'torch.mul', 'torch.sum', 'torch.linalg.solve'}
MATERIAL = PURE | {'torch.Tensor.scatter_add', 'Tensor.scatter_add', 'torch.scatter_add',
                   'torch.linalg.lu_solve', 'torch.nn.functional.batch_norm', 'torch.add',
                   'torch.Tensor.copy_', 'Tensor.copy_'}
GRADIENT = {'torch.linalg.slogdet', 'torch.linalg.det', 'torch.std', 'torch.var'}


def families(api):
    if api in MATERIAL:
        return ('materialization',)
    if api in GRADIENT:
        return ('gradient_reference',)
    if api in ('torch.autocast', 'torch.no_grad', 'torch.enable_grad', 'torch.inference_mode'):
        return ('context_restore',)
    if api == 'torch.cuda.make_graphed_callables':
        return ('graph_accumulation',)
    if api == 'torch.compile':
        return ('compiled_alias',)
    return ()


def description(family):
    return {
        'materialization': 'build returns args (tuple), kwargs (dict). They are the exact positional/keyword operands for the target API, including self for Tensor methods. The runner calls the API and compares legal logical materialization. Sparse CSR and MKLDNN values are supported. Avoid aliased writable operands and uninitialized data.',
        'gradient_reference': 'build returns args and kwargs for the target. The first argument must be a real floating tensor requiring gradients. For std/var use a scalar reduction with dim=None; for determinant APIs use a square unbatched nonsingular matrix. The runner differentiates the scalar result (logabsdet for slogdet). Backend is controlled by configuration.',
        'context_restore': 'build returns args and kwargs for constructing the target context manager, plus depth (integer 0..4). The runner checks nested decorator invocation against fresh contexts and observes entry/exit state. No threads are used.',
        'graph_accumulation': 'build returns module (a deterministic torch.nn.Module on device), batches (2..4 tuples of fixed-shape input tensors). Use a small differentiable module without random layers, mutable buffers or control flow. A local class inside build is allowed. The runner sums real tensor outputs, accumulates gradients, and compares captured versus eager execution. Do not capture graphs yourself.',
        'compiled_alias': 'build returns inputs (dict), output_path (list of tuple/list indices or dictionary keys), delta (finite scalar). Also define forward(ctx) returning tensors or a nested tensor structure. Choose a legal program whose output may contain views. The runner compiles forward, checks initial outputs and tests an add_ on the selected returned tensor against eager execution. Do not call compile, assert, or inspect storage/strides inside forward.',
    }[family]


def tensor_leaves(value, path=()):
    import torch
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, path+(key,))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, path+(index,))


def materialize(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.layout == torch.sparse_csr:
            return torch.sparse_csr_tensor(value.crow_indices().clone(), value.col_indices().clone(),
                                           materialize(value.values()), size=value.shape, check_invariants=True)
        if value.is_mkldnn:
            return value.to_dense().clone().to_mkldnn()
        if value.layout != torch.strided:
            raise ValueError('Unsupported tensor layout')
        return value.detach().resolve_conj().resolve_neg().contiguous().clone().requires_grad_(value.requires_grad)
    if isinstance(value, dict):
        return {k: materialize(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(materialize(v) for v in value)
    if isinstance(value, list):
        return [materialize(v) for v in value]
    return copy.deepcopy(value)


def observation(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_mkldnn:
            value = value.to_dense()
        return value.detach().resolve_conj().resolve_neg().cpu().clone()
    if isinstance(value, dict):
        return {k: observation(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(observation(v) for v in value)
    return value


def finite(value):
    import torch
    if isinstance(value,torch.Tensor):
        return bool(torch.isfinite(value.to_dense() if value.layout!=torch.strided else value).all())
    if isinstance(value,dict): return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)): return all(finite(v) for v in value)
    if isinstance(value,float): return math.isfinite(value)
    if isinstance(value,complex): return math.isfinite(value.real) and math.isfinite(value.imag)
    return True


def validate(actual, reference, control, predicates, *, atol=1e-5, rtol=1e-4):
    """Controls and predicates gate every correctness verdict, never vice versa."""
    result = dict(rule_version=VERSION, predicates=predicates, confirmed_defect=False,
                  tolerance=dict(atol=atol, rtol=rtol), observables=observation(actual))
    if not predicates or not all(v is True for v in predicates.values()):
        return dict(result, verdict='inapplicable', reason='Property predicates not established')
    if not finite((actual, reference, control)):
        return dict(result, verdict=None, correctness='unassessed', reason='Nonfinite observations')
    policy = result['tolerance']
    result['control_differences'] = differences(observation(control), observation(reference), policy)
    if result['control_differences']:
        return dict(result, verdict=None, correctness='unassessed', reason='Independent reference/control disagreement')
    result['differences'] = differences(observation(actual), observation(reference), policy)
    return dict(result, verdict='suspected_failure' if result['differences'] else 'pass',
                reference=observation(reference), control=observation(control))


def metadata_tree(value):
    from mirage.runtime.representations import metadata
    return metadata({str(path): t for path, t in tensor_leaves(value)})


def snapshot_signature(value):
    """Serialization increments tensor versions; these properties start fresh graphs.

    Retain every other recorded representation field. Replay is rejected when
    serialization loses required representation or autograd metadata.
    """
    if isinstance(value,dict):
        return {k:snapshot_signature(v) for k,v in value.items() if k!='version'}
    if isinstance(value,list): return [snapshot_signature(v) for v in value]
    return value


def _no_writable_alias(args, kwargs):
    import torch
    dest = kwargs.get('out')
    if dest is None:
        return True
    def storage(t):
        if t.layout == torch.sparse_csr:
            t = t.values()
        return (str(t.device), t.untyped_storage().data_ptr()) if t.layout == torch.strided and not t.is_mkldnn else None
    ds = storage(dest)
    return ds is not None and all(storage(t) != ds for _, t in tensor_leaves(args))


def materialization(target, api, ctx, config, call):
    import torch
    args, kwargs = ctx.get('args', ()), ctx.get('kwargs', {})
    controls = materialize((args, kwargs))
    reference_args = materialize(args)
    predicates = dict(certified_api=api in MATERIAL, independent_output=_no_writable_alias(args, kwargs))
    name = api.split('.')[-1]
    if name == 'batch_norm':
        predicates['inference_statistics'] = not kwargs.get('training', args[5] if len(args)>5 else False)
    if not all(predicates.values()):
        return dict(verdict='inapplicable', predicates=predicates, reason='Unsupported value relation')
    actual = call(target, *args, **kwargs)
    control = target(*controls[0], **controls[1])
    reference = control
    # Independent reference paths are selected by mathematical API semantics.
    if name == 'scatter_add':
        dest, dim, index, src = reference_args
        dim %= dest.ndim
        if index.numel() > 8192:
            return dict(verdict='inapplicable', reason='Scalar reference budget exceeded')
        reference = dest.clone()
        for point in itertools.product(*(range(n) for n in index.shape)):
            destination = list(point)
            destination[dim] = int(index[point])
            reference[tuple(destination)] += src[point]
    elif name == 'lu_solve':
        lu, piv, rhs = reference_args[:3]
        p, lower, upper = torch.lu_unpack(lu, piv)
        matrix = p @ lower @ upper
        if kwargs.get('adjoint', False):
            matrix = matrix.mH
        reference = torch.linalg.solve(matrix, rhs, left=kwargs.get('left', True))
        # The alternative solve must itself satisfy the defining equation.
        residual = matrix @ reference if kwargs.get('left', True) else reference @ matrix
        predicates['reference_residual'] = bool(torch.allclose(residual, rhs.expand_as(residual), atol=1e-8, rtol=1e-7))
    elif name == 'batch_norm':
        x = reference_args[0].to_dense() if reference_args[0].is_mkldnn else reference_args[0]
        def arg(n, key, default=None):
            return reference_args[n] if len(reference_args)>n else kwargs.get(key, default)
        mean, var, weight, bias = arg(1,'running_mean'), arg(2,'running_var'), arg(3,'weight'), arg(4,'bias')
        if mean is None or var is None:
            return dict(verdict='inapplicable', reason='Running statistics required')
        shape = (1, -1) + (1,)*(x.ndim-2)
        eps = arg(7,'eps',1e-5)
        reference = (x-mean.reshape(shape))/torch.sqrt(var.reshape(shape)+eps)
        if weight is not None: reference = reference*weight.reshape(shape)
        if bias is not None: reference = reference+bias.reshape(shape)
    elif name == 'add' and any(t.layout == torch.sparse_csr for _,t in tensor_leaves(args)):
        reference = reference_args[0].to_dense()+kwargs.get('alpha',1)*reference_args[1].to_dense()
    elif name == 'copy_':
        dst, src = reference_args[:2]
        reference = src.to(device=dst.device, dtype=dst.dtype).expand_as(dst).clone()
        if dst.device.type=='cuda' or src.device.type=='cuda': torch.cuda.synchronize()
    result = validate(actual, reference, control, predicates)
    result['actual_metadata'] = metadata_tree((args, kwargs))
    return result


def gradient_reference(target, api, ctx, config, call):
    import torch
    args, kwargs = ctx.get('args', ()), ctx.get('kwargs', {})
    x = args[0]
    if x.is_complex() or x.ndim > 2 or not x.requires_grad or x.numel() > 4096:
        return dict(verdict='inapplicable', reason='Real small differentiable operands required')
    name = api.split('.')[-1]
    def scalar(output):
        return output.logabsdet if name=='slogdet' else output
    def evaluation(fn, values, record=False):
        out = scalar(call(fn, *values, **kwargs) if record else fn(*values, **kwargs))
        if out.ndim: raise ValueError('Scalar output required by this gradient property')
        grad, = torch.autograd.grad(out, values[0])
        return {'output':out, 'gradient':grad}
    fn = target
    compiled_graphs=[]
    if config.get('backend','eager')!='eager':
        from torch._dynamo.backends.registry import lookup_backend
        def backend(gm, examples):
            lowered=lookup_backend(config['backend'])(gm,examples)
            def invoke(*values):
                compiled_graphs.append([str(n.target) for n in gm.graph.nodes if n.op.startswith('call_')])
                return lowered(*values)
            return invoke
        fn = torch.compile(target, backend=backend, fullgraph=True)
    actual = evaluation(fn, args, True)
    control = evaluation(target, materialize(args))
    high = x.detach().resolve_neg().resolve_conj().double()
    predicates = dict(api_certified=api in GRADIENT, execution_mode_reached=config.get('backend','eager')=='eager' or bool(compiled_graphs))
    if name in ('std','var'):
        if kwargs.get('dim') is not None or len(args)>1:
            return dict(verdict='inapplicable', reason='Scalar all-element reduction required')
        correction = kwargs.get('correction', 1 if kwargs.get('unbiased',True) else 0)
        denom = high.numel()-correction
        predicates['positive_degrees_of_freedom'] = denom > 0
        mean = high.mean()
        variance = ((high-mean)**2).sum()/max(denom,1)
        predicates['nonzero_variance'] = bool(variance>0)
        if not all(predicates.values()): return dict(verdict='inapplicable',predicates=predicates)
        out = variance.sqrt() if name=='std' else variance
        grad = (high-mean)/(denom*out) if name=='std' else 2*(high-mean)/denom
    else:
        if x.ndim!=2 or x.shape[0]!=x.shape[1]:
            return dict(verdict='inapplicable', reason='Unbatched square matrix required')
        det = torch.linalg.det(high)
        predicates['well_conditioned'] = bool(torch.linalg.cond(high)<1e6 and det.abs()>1e-12)
        if not all(predicates.values()): return dict(verdict='inapplicable',predicates=predicates)
        inverse = torch.linalg.solve(high, torch.eye(high.shape[0],dtype=high.dtype,device=high.device))
        out = det.abs().log() if name=='slogdet' else det
        grad = inverse.mT if name=='slogdet' else det*inverse.mT
    reference = dict(output=out.to(x.dtype), gradient=grad.to(x.dtype))
    # Relative output tolerance preserves scale; the gradient supplies independent evidence.
    result = validate(actual, reference, control, predicates, atol=0., rtol=1e-4)
    result['actual_metadata'] = metadata_tree(args)
    result['backward_executed'] = True
    result['compiled_graphs'] = compiled_graphs
    return result


def context_restore(target, api, ctx, config, call):
    import torch
    args, kwargs = ctx.get('args',()), ctx.get('kwargs',{})
    depth = ctx.get('depth',1)
    if type(depth) is not int or not 0<=depth<=4: raise ValueError('depth must be 0..4')
    device = kwargs.get('device_type',args[0] if args else 'cpu') if api=='torch.autocast' else 'cpu'
    def state():
        return dict(grad=torch.is_grad_enabled(), inference=torch.is_inference_mode_enabled(),
                    amp=torch.is_autocast_enabled(device), dtype=str(torch.get_autocast_dtype(device)),
                    cache=torch.is_autocast_cache_enabled())
    before = state()
    def fresh(n):
        with target(*args,**kwargs):
            if n: fresh(n-1)
    fresh(depth)
    control = state()
    context = call(target,*args,**kwargs)
    @context
    def nested(n):
        if n: nested(n-1)
    nested(depth)
    result=validate(state(),before,control,dict(bounded_depth=True),atol=0.,rtol=0.)
    result['depth']=depth
    return result


def graph_accumulation(target, api, ctx, config, call):
    import torch
    import torch.fx
    module, batches = ctx['module'], ctx['batches']
    if not 2<=len(batches)<=4: raise ValueError('Expected 2..4 microbatches')
    if list(module.buffers()): return dict(verdict='inapplicable',reason='Mutable module buffers not certified')
    graph = torch.fx.symbolic_trace(module)
    allowed = {'add','mul','matmul','sub','truediv','linear','relu','tanh','sigmoid','sum','mean','getattr'}
    for n in graph.graph.nodes:
        if n.op in ('call_function','call_method') and getattr(n.target,'__name__',str(n.target)) not in allowed:
            return dict(verdict='inapplicable',reason='Uncertified graph operation')
        if n.op=='call_module' and not isinstance(graph.get_submodule(n.target),(torch.nn.Linear,torch.nn.ReLU,torch.nn.Tanh,torch.nn.Sigmoid)):
            return dict(verdict='inapplicable',reason='Uncertified graph module')
    prototype = [(t.shape,t.dtype,t.device) for _,t in tensor_leaves(batches[0])]
    predicates = dict(fixed_signature=all([(t.shape,t.dtype,t.device) for _,t in tensor_leaves(b)]==prototype for b in batches),
                      cuda=all(t.device.type=='cuda' for _,t in tensor_leaves(batches)))
    if not all(predicates.values()): return dict(verdict='inapplicable',predicates=predicates)
    reference = copy.deepcopy(module)
    def accumulate(fn, owner):
        for p in owner.parameters(): p.grad=None
        for batch in batches:
            out = fn(*batch)
            loss = sum(t.sum() for _,t in tensor_leaves(out))
            loss.backward()
        return {n:p.grad for n,p in owner.named_parameters()}
    control = observation(accumulate(reference,reference))
    captured = call(target,module,batches[0])
    actual = accumulate(captured,module)
    torch.cuda.synchronize()
    result = validate(actual, control, control, predicates)
    result['backward_executed'] = True
    result['microbatches'] = len(batches)
    return result


def compiled_alias(target, api, ctx, config, call, namespace):
    import torch
    from torch._dynamo.backends.registry import lookup_backend
    forward = namespace['forward']
    reference_ctx = materialize(ctx)
    actual_ctx = materialize(ctx)
    reference = forward(reference_ctx)
    graph_executions=[]
    def backend(gm, examples):
        compiled=lookup_backend('inductor')(gm,examples)
        def invoke(*args):
            graph_executions.append([str(n.target) for n in gm.graph.nodes if n.op.startswith('call_')])
            return compiled(*args)
        return invoke
    compiled = call(target,forward,backend=backend,fullgraph=True)
    actual = compiled(actual_ctx)
    before = not differences(observation(actual),observation(reference),dict(atol=1e-5,rtol=1e-4))
    def selected(value):
        for key in ctx.get('output_path',[]): value=value[key]
        if not isinstance(value,torch.Tensor): raise ValueError('output_path must select a Tensor')
        return value
    r = selected(reference)
    aliases = sum(t.untyped_storage().data_ptr()==r.untyped_storage().data_ptr() for _,t in tensor_leaves(reference))
    delta=ctx.get('delta',1.)
    if not isinstance(delta,(int,float)) or not math.isfinite(delta): raise ValueError('finite delta required')
    with torch.no_grad():
        r.add_(delta)
        selected(actual).add_(delta)
    result = validate(actual,reference,reference,dict(initial_outputs_agree=before,returned_alias=aliases>1,
                                                      compiled_graph_executed=bool(graph_executions)))
    result['actual_metadata']=metadata_tree(actual)
    result['reference_metadata']=metadata_tree(reference)
    result['compiled_graphs']=graph_executions
    return result


def execute(namespace, ctx, config, call):
    from mirage.runtime.api_metadata import resolve
    api, family = config['api'], config['family']
    if family not in families(api):
        return dict(verdict='inapplicable', reason='Uncertified API/property combination')
    target=resolve(api)
    handlers={'materialization':materialization,'gradient_reference':gradient_reference,
              'context_restore':context_restore,'graph_accumulation':graph_accumulation}
    if family=='compiled_alias': return compiled_alias(target,api,ctx,config,call,namespace)
    return handlers[family](target,api,ctx,config,call)


def encode_initial(value):
    """Opaque MKLDNN storage needs a logical payload plus an explicit layout tag."""
    import torch
    if isinstance(value,torch.Tensor) and value.is_mkldnn:
        return {'__mirage_layout__':'mkldnn','tensor':value.to_dense().detach(),
                'requires_grad':value.requires_grad}
    if isinstance(value,torch.Tensor) and value.layout==torch.strided and value._base is not None:
        return {'__mirage_layout__':'strided_view','tensor':value}
    if isinstance(value,torch.nn.Module):
        return value.state_dict()
    if isinstance(value,dict): return {k:encode_initial(v) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(encode_initial(v) for v in value)
    if isinstance(value,list): return [encode_initial(v) for v in value]
    return value


def decode_initial(value):
    if isinstance(value,dict) and value.get('__mirage_layout__')=='strided_view':
        tensor=value['tensor']
        return tensor.as_strided(tensor.shape,tensor.stride(),tensor.storage_offset())
    if isinstance(value,dict) and value.get('__mirage_layout__')=='mkldnn':
        return value['tensor'].to_mkldnn().requires_grad_(value['requires_grad'])
    if isinstance(value,dict): return {k:decode_initial(v) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(decode_initial(v) for v in value)
    if isinstance(value,list): return [decode_initial(v) for v in value]
    return value
