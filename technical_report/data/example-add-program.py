import torch

def build(device, seed):
    x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([0.0, 0.5, -1.0], dtype=torch.float64, device=device)
    optimizer = torch.optim.SGD([x], lr=0.1, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False, maximize=False, foreach=False, fused=False, differentiable=False)
    return {'inputs': {'y': y}, 'parameters': {'x': x}, 'optimizer': optimizer}

def run(ctx, probe):
    x = ctx['parameters']['x']
    y = ctx['inputs']['y']
    out = probe.call(torch.add, x, y, alpha=2.0)
    loss = out.sum()
    loss.backward()
    ctx['optimizer'].step()
    return {'out': out, 'x_grad': x.grad, 'x': x}
