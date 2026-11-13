"""Issue-derived regression fixtures. Never supplied to the LLM generator.

Upstream attribution and scientific contracts are in candidates.json. All target
operations use public APIs; metadata is collected outside compiled functions.
"""
import math
import torch
from mirage.inspector.comparison import differences


def meta(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), offset=t.storage_offset(),
                dtype=str(t.dtype), device=str(t.device), conj=t.is_conj(), neg=t.is_neg(),
                requires_grad=t.requires_grad)


def check(actual, expected, control, *, conditions, atol=0., rtol=0., extra=None):
    """A failing control or missing predicate cannot establish a violation."""
    policy = dict(atol=atol, rtol=rtol)
    failures = differences(actual, expected, policy)
    control_failures = differences(control, expected, policy)
    finite = not any('nonfinite' in d['reason'] for d in failures + control_failures)
    if not conditions or not all(v is True for v in conditions.values()):
        outcome = 'inapplicable'
    elif not finite or control_failures:
        outcome = 'oracle_disagreement'
    else:
        outcome = 'violation' if failures else 'no_violation'
    return dict(outcome=outcome, conditions=conditions, tolerance=policy,
                differences=failures, control_differences=control_failures,
                actual=actual, expected=expected, control=control, extra=extra or {})


def scatter(variant):
    width = 16 if variant == 0 else 32
    src = torch.full((2, width), float(variant + 1))
    base = torch.zeros(3, 2 * width)
    index = torch.tensor([[0], [2]]).expand(2, width)
    expected = base.clone()
    # Independent definition: destination[index[i,j],j] += source[i,j].
    for i in range(2):
        for j in range(width):
            expected[int(index[i, j]), j] += src[i, j]
    actual = base.scatter_add(0, index, src)
    control = base.scatter_add(0, index.contiguous(), src)
    return check(actual, expected, control,
                 conditions=dict(expanded=index.stride() == (1, 0), wider=base.size(1) > index.size(1),
                                 legal=bool(index.min() >= 0 and index.max() < base.size(0))),
                 extra=dict(index=meta(index), row_sums=actual.sum(1), source=src))


def slogdet(variant):
    a, b, c, d = (4. + variant, 1., 2., 3. + variant)
    z = torch.tensor([[-a*1j, -b*1j], [-c*1j, -d*1j]], dtype=torch.complex128, requires_grad=True)
    x = z.conj().imag
    loss = torch.linalg.slogdet(x).logabsdet
    actual = torch.autograd.grad(loss, z)[0].imag
    det = a*d-b*c
    expected = -torch.tensor([[d, -c], [-b, a]], dtype=torch.float64) / det
    zz = z.detach().clone().requires_grad_()
    control = torch.autograd.grad(torch.linalg.slogdet(zz.conj().imag.resolve_neg()).logabsdet, zz)[0].imag
    # Scalar determinant finite differences do not use the tested autograd path.
    vals = [-a, -b, -c, -d]
    finite = []
    for i in range(4):
        plus, minus = vals.copy(), vals.copy()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        f = lambda q: math.log(abs(q[0]*q[3]-q[1]*q[2]))
        finite.append((f(plus)-f(minus))/2e-5)
    fd = torch.tensor(finite, dtype=torch.float64).reshape(2, 2)
    return check(actual, expected, control, atol=1e-10, rtol=1e-9,
                 conditions=dict(negative_view=x.is_neg(), nonsingular=det != 0,
                                 finite_difference=bool(torch.allclose(fd, expected, atol=1e-8, rtol=1e-8))),
                 extra=dict(input=meta(x), finite_difference=fd, loss=loss.detach(), z=z.detach()))


def lu_solve(variant):
    a = torch.tensor([[[2., 1.], [1., 3.]], [[4., 1.], [2., 5.]]], dtype=torch.float64)
    a[:, 0, 0] += variant
    rhs = torch.tensor([[[1. + variant], [2.]]], dtype=torch.float64)
    lu, piv = torch.linalg.lu_factor(a)
    view = lu[:, None]
    actual = torch.linalg.lu_solve(view, piv[:, None], rhs)
    control = torch.linalg.lu_solve(view.contiguous(), piv[:, None], rhs)
    aa, bb, cc, dd = (a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1])
    det = aa*dd-bb*cc
    u, v = rhs[0, 0, 0], rhs[0, 1, 0]
    expected = torch.stack(((dd*u-bb*v)/det, (aa*v-cc*u)/det), -1).reshape(2, 1, 2, 1)
    return check(actual, expected, control, atol=1e-12, rtol=1e-12,
                 conditions=dict(singleton=view.size(-3) == 1, odd_stride=view.stride(-3) != 4,
                                 nonsingular=bool((det != 0).all())),
                 extra=dict(lu=meta(view), matrices=a, rhs=rhs, residual=a[:, None] @ actual-rhs))


def autocast(variant):
    def state():
        return dict(enabled=torch.is_autocast_enabled('cpu'), dtype=str(torch.get_autocast_dtype('cpu')),
                    cache=torch.is_autocast_cache_enabled())
    def fresh(n):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            if n:
                fresh(n-1)
    @torch.autocast('cpu', dtype=torch.bfloat16)
    def decorated(n):
        if n:
            decorated(n-1)
    with torch.autocast('cpu', enabled=False):
        before = state()
        fresh(1+variant)
        control = state()
        decorated(1+variant)
        actual = state()
    return check(actual, before, control, conditions=dict(initially_disabled=not before['enabled'], nested=True),
                 extra=dict(recursion_depth=1+variant))


def cuda_graph(variant):
    class Scale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(1., 5., device='cuda'))
        def forward(self, x):
            return x * self.weight
    x1 = torch.tensor([[1., 2., 3., 4.], [2., 3., 4., 5.]], device='cuda') + variant
    x2 = torch.tensor([[10., 20., 30., 40.], [1., 2., 3., 4.]], device='cuda') + variant
    expected = x1.sum(0) + x2.sum(0)
    plain = Scale()
    for x in (x1, x2):
        plain(x).sum().backward()
    control = plain.weight.grad.clone()
    model = Scale()
    graph = torch.cuda.make_graphed_callables(model, (torch.ones_like(x1),))
    model.weight.grad = None
    graph(x1).sum().backward()
    torch.cuda.synchronize()
    first = model.weight.grad.clone()
    graph(x2).sum().backward()
    torch.cuda.synchronize()
    actual = model.weight.grad.clone()
    return check(actual, expected, control,
                 conditions=dict(first_gradient=bool(torch.equal(first, x1.sum(0))), distinct_batches=not torch.equal(x1, x2)),
                 extra=dict(first_gradient=first, x1=x1, x2=x2, twice_second=2*x2.sum(0)))


def compiled_alias(variant):
    def fn(x):
        base = torch.complex(x, x+1)
        view = base.conj().imag
        view.add_(3)
        return base, view
    x = torch.arange(1., 5.) + variant
    ref_base, ref_view = fn(x.clone())
    base, view = torch.compile(fn, backend='inductor', fullgraph=True)(x.clone())
    initial_base = base.clone()
    initial_view = view.clone()
    delta = 100. + variant
    view.add_(delta)
    ref_view.add_(delta)
    expected = x+1-3-delta
    return check(base.imag, expected, ref_base.imag,
                 conditions=dict(initial_base=bool(torch.equal(initial_base.imag, x+1-3)),
                                 initial_view=bool(torch.equal(initial_view, -(x+1-3)))),
                 extra=dict(base=meta(base), view=meta(view), final_view=view.clone(),
                            shares_storage=base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()))


def compiled_clone(variant):
    z = torch.complex(torch.arange(1., 6.), torch.arange(2., 7.) + variant)
    x = z.conj().imag
    actual = torch.compile(torch.clone, backend='inductor', fullgraph=True)(x)
    expected = -z.imag
    control = torch.clone(x.resolve_neg())
    return check(actual, expected, control, conditions=dict(negative_view=x.is_neg()), extra=dict(input=meta(x)))


def pinv(variant):
    # Public profiling context, also documented by the upstream reporter.
    from torch.utils.flop_counter import FlopCounterMode
    entries = [[1+1j, 2-1j], [3+2j, 4+1j+variant]]
    a = torch.tensor(entries, dtype=torch.complex128)
    aa, bb = entries[0]
    cc, dd = entries[1]
    det = aa*dd-bb*cc
    # Full-rank square pseudoinverse equals the elementary 2x2 inverse.
    expected = torch.tensor([[dd, -bb], [-cc, aa]], dtype=a.dtype) / det
    control = torch.linalg.pinv(a)
    with FlopCounterMode(display=False):
        actual = torch.linalg.pinv(a)
    return check(actual, expected, control, atol=1e-10, rtol=1e-9,
                 conditions=dict(full_rank=abs(det) > 1,
                                 reference_identity=bool(torch.allclose(a @ expected, torch.eye(2, dtype=a.dtype), atol=1e-10, rtol=1e-9))),
                 extra=dict(input=a, determinant=det, reconstruction_residual=a @ actual @ a-a))


def batch_norm(variant):
    import torch.nn.functional as F
    x = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2)/10 + variant
    w = torch.tensor([1., 10., 2., 20., 3., 30., 4., 40.])[::2]
    b = torch.tensor([0., 10., 0., 20., 0., 30., 0., 40.])[::2]
    rm, rv = torch.zeros(4), torch.ones(4)
    expected = x*w.reshape(1, 4, 1, 1)/math.sqrt(1+1e-5)+b.reshape(1, 4, 1, 1)
    actual = F.batch_norm(x.to_mkldnn(), rm, rv, w, b, training=False).to_dense()
    control = F.batch_norm(x.to_mkldnn(), rm, rv, w.contiguous(), b.contiguous(), training=False).to_dense()
    return check(actual, expected, control, atol=2e-6, rtol=1e-6,
                 conditions=dict(noncontiguous=not w.is_contiguous() and not b.is_contiguous()),
                 extra=dict(weight=meta(w), bias=meta(b), input=x))


def sparse_add(variant):
    crow, col = torch.tensor([0, 1, 2]), torch.tensor([0, 1])
    def csr(values):
        return torch.sparse_csr_tensor(crow.clone(), col.clone(), values, size=(2, 2), check_invariants=True)
    lhs = csr(torch.tensor([1., 2.]) + variant)
    rhs = csr(torch.tensor([10., 20.]))
    backing = torch.tensor([-99., -88., -99., -77.])
    out = csr(backing[::2])
    stride = out.values().stride()
    torch.add(lhs, rhs, out=out)
    good = csr(torch.zeros(2))
    torch.add(lhs, rhs, out=good)
    expected = torch.tensor([11., 22.])+variant
    return check(out.values(), expected, good.values(), conditions=dict(stride_two=stride == (2,)),
                 extra=dict(backing=backing, collateral_preserved=bool(torch.equal(backing[1::2], torch.tensor([-88., -77.])))))


def std_gradient(variant):
    scale = 2.**(-100+variant)
    data = torch.tensor([-scale, scale])
    def fn(x):
        return torch.std(x, correction=0)
    def evaluate(f):
        x = data.clone().requires_grad_()
        y = f(x)
        return dict(output=y.detach(), gradient=torch.autograd.grad(y, x)[0])
    actual = evaluate(torch.compile(fn, backend='inductor', fullgraph=True))
    control = evaluate(fn)
    expected = dict(output=torch.tensor(scale), gradient=torch.tensor([-.5, .5]))
    return check(actual, expected, control, atol=0., rtol=1e-6,
                 conditions=dict(nonzero=bool((data != 0).all()), finite=bool(torch.isfinite(data).all())),
                 extra=dict(input=data, analytic_formula='std([-a,a], correction=0)=a; gradient=[-1/2,1/2]'))


def repeated_cat(variant):
    import torch.nn.functional as F
    def fn(a, b):
        t = a+b
        p = F.avg_pool2d(t, (1, 1), stride=3)
        return torch.cat((p, p), 0).chunk(2, 0)[1], torch.cat((t, t), 0).chunk(2, 0)[1]
    a = ((torch.arange(260, device='cuda', dtype=torch.float32) % 17)/8).reshape(10, 26, 1).transpose(0, 1)
    b = ((torch.arange(232, device='cuda', dtype=torch.float32) % 13)/8+variant).reshape(58, 1, 1, 4)
    actual = torch.compile(fn, fullgraph=True, backend='inductor', dynamic=False)(a, b)
    control = fn(a, b)
    t = a+b
    expected = (t[..., ::3, ::3], t)
    return check(actual, expected, control, atol=1e-6, rtol=1e-6,
                 conditions=dict(noncontiguous=not a.is_contiguous()), extra=dict(a=meta(a), b=meta(b)))


def broadcast_copy(variant):
    z = torch.tensor([complex(1+variant, 2+variant)], dtype=torch.complex64)
    src = z.conj()
    dest = torch.zeros(2, dtype=z.dtype, device='cuda')
    dest.copy_(src)
    torch.cuda.synchronize()
    control = torch.zeros_like(dest)
    control.copy_(src.resolve_conj())
    torch.cuda.synchronize()
    expected = torch.tensor([complex(1+variant, -2-variant)]*2, dtype=z.dtype)
    return check(dest.cpu(), expected, control.cpu(), conditions=dict(conjugate=src.is_conj(), broadcast=src.numel() == 1),
                 extra=dict(source=meta(src), backend='NVIDIA CUDA; upstream report initially tested ROCm only'))


def grid_gradient(variant):
    import torch.nn.functional as F
    channels = 2048 if variant == 0 else 4096
    def evaluate(dtype):
        inp = torch.tensor([[0., 1.], [0., 1.]], device='cuda', dtype=dtype).expand(1, channels, 2, 2).contiguous()
        grid = torch.zeros(1, 1, 1, 2, device='cuda', dtype=dtype, requires_grad=True)
        y = F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return torch.autograd.grad(y.sum(), grid)[0].float()
    actual = evaluate(torch.bfloat16)
    control = evaluate(torch.float32)
    expected = torch.tensor([channels, 0.], device='cuda').reshape(1, 1, 1, 2)
    return check(actual, expected, control, atol=0., rtol=.01,
                 conditions=dict(interior_grid=True, exactly_representable=channels in (2048, 4096)),
                 extra=dict(channels=channels, formula='Each channel is (x+1)/2 at pixel coordinates; align_corners=False on width 2 gives d/dgrid_x=1.'))


CASES = {'203078': scatter, '204407': slogdet, '200599': lu_solve, '202706': autocast,
         '181723': cuda_graph, '204033': compiled_alias, '145093': compiled_clone,
         '198118': pinv, '204365': batch_norm, '201268': sparse_add,
         '197089': std_gradient, '204347': repeated_cat, '200613': broadcast_copy,
         '204496': grid_gradient}

# Known inputs live only in benchmark fixtures; production generation never imports this table.
INTEGRATION_CASES = {
'203078': ('torch.Tensor.scatter_add','cpu','materialization', '''
import torch
def build(device, seed):
    return {'args': (torch.zeros(3,32), 0, torch.tensor([[0],[2]]).expand(2,16), torch.ones(2,16)), 'kwargs': {}}
'''),
'204407': ('torch.linalg.slogdet','cpu','gradient_reference', '''
import torch
def build(device, seed):
    z=torch.tensor([[-4j,-1j],[-2j,-3j]],dtype=torch.complex128,requires_grad=True)
    return {'args': (z.conj().imag,), 'kwargs': {}}
'''),
'200599': ('torch.linalg.lu_solve','cpu','materialization', '''
import torch
def build(device, seed):
    a=torch.tensor([[[2.,1.],[1.,3.]],[[4.,1.],[2.,5.]]],dtype=torch.float64)
    lu,piv=torch.linalg.lu_factor(a)
    return {'args': (lu[:,None],piv[:,None],torch.tensor([[[1.],[2.]]],dtype=torch.float64)), 'kwargs': {}}
'''),
'202706': ('torch.autocast','cpu','context_restore', '''
import torch
def build(device, seed):
    return {'args': ('cpu',), 'kwargs': {'dtype':torch.bfloat16}, 'depth':1}
'''),
'181723': ('torch.cuda.make_graphed_callables','cuda','graph_accumulation', '''
import torch
def build(device, seed):
    class Scale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.arange(1.,5.,device=device))
        def forward(self,x):
            return x*self.weight
    x1=torch.tensor([[1.,2.,3.,4.],[2.,3.,4.,5.]],device=device)
    x2=torch.tensor([[10.,20.,30.,40.],[1.,2.,3.,4.]],device=device)
    return {'module':Scale(), 'batches':[(x1,),(x2,)]}
'''),
'204033': ('torch.compile','cpu','compiled_alias', '''
import torch
def build(device,seed):
    return {'inputs':{'x':torch.arange(1.,5.)},'output_path':[1],'delta':100.}
def forward(ctx):
    x=ctx['inputs']['x']
    base=torch.complex(x,x+1)
    view=base.conj().imag
    view.add_(3)
    return base,view
'''),
'204365': ('torch.nn.functional.batch_norm','cpu','materialization', '''
import torch
def build(device,seed):
    x=torch.arange(32,dtype=torch.float32).reshape(2,4,2,2)/10
    w=torch.tensor([1.,10.,2.,20.,3.,30.,4.,40.])[::2]
    b=torch.tensor([0.,10.,0.,20.,0.,30.,0.,40.])[::2]
    return {'args':(x.to_mkldnn(),torch.zeros(4),torch.ones(4),w,b),'kwargs':{'training':False}}
'''),
'201268': ('torch.add','cpu','materialization', '''
import torch
def build(device,seed):
    crow,col=torch.tensor([0,1,2]),torch.tensor([0,1])
    lhs=torch.sparse_csr_tensor(crow,col,torch.tensor([1.,2.]),size=(2,2),check_invariants=True)
    rhs=torch.sparse_csr_tensor(crow,col,torch.tensor([10.,20.]),size=(2,2),check_invariants=True)
    backing=torch.tensor([-99.,-88.,-99.,-77.])
    out=torch.sparse_csr_tensor(crow,col,backing[::2],size=(2,2),check_invariants=True)
    return {'args':(lhs,rhs),'kwargs':{'out':out}}
'''),
'197089': ('torch.std','cpu','gradient_reference', '''
import torch
def build(device,seed):
    return {'args':(torch.tensor([-2.**-100,2.**-100],requires_grad=True),),'kwargs':{'correction':0}}
'''),
'200613': ('torch.Tensor.copy_','cuda','materialization', '''
import torch
def build(device,seed):
    return {'args':(torch.zeros(2,dtype=torch.complex64,device=device),torch.tensor([1+2j],dtype=torch.complex64).conj()),'kwargs':{}}
'''),
}
