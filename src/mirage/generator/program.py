"""Lifecycle contracts for generated programs with arbitrary tensor operands."""

import ast
import hashlib

from mirage.generator.provider import Request, Stage


class ContractError(ValueError):
    pass


def analyze(source):
    if not isinstance(source, str) or not source.strip() or len(source) > 40000:
        raise ContractError("Expected nonempty source of at most 40000 characters")
    tree = ast.parse(source)
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.Import) and all(a.name == "math" or a.name == "torch" or a.name.startswith("torch.") for a in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and node.module and (node.module == "torch" or node.module.startswith("torch.")):
            continue
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            raise ContractError(f"Unexpected module statement at line {node.lineno}: {ast.unparse(node)[:100]}. Use functions and torch/math imports only.")
        if node.name in functions:
            raise ContractError("Duplicate function")
        functions[node.name] = node
    for name, args in {"build": ["device", "seed"], "forward": ["ctx"], "loss": ["output"]}.items():
        if name not in functions:
            raise ContractError(f"Missing {name}({', '.join(args)})")
        actual = functions[name].args
        expected = ast.parse(f"def f({', '.join(args)}): pass").body[0].args
        if ast.dump(actual) != ast.dump(expected):
            raise ContractError(f"Expected signature {name}({', '.join(args)})")
    if "sample_loss" in functions:
        if ast.dump(functions["sample_loss"].args) != ast.dump(functions["loss"].args):
            raise ContractError("sample_loss must take exactly output")
    normalized = ast.dump(tree, include_attributes=False)
    suspicious = {"id", "data_ptr", "stride", "storage_offset", "untyped_storage", "storage",
                  "is_contiguous", "is_compiling", "setattr", "getattr", "eval", "exec"}
    identity_sensitive = any(
        isinstance(n, ast.Call) and (
            isinstance(n.func, ast.Name) and n.func.id in suspicious
            or isinstance(n.func, ast.Attribute) and (n.func.attr in suspicious or n.func.attr.startswith("_")))
        for name in ("forward", "loss", "sample_loss") if name in functions
        for n in ast.walk(functions[name]))
    for name in functions:
        if name == "build":
            continue
        for node in ast.walk(functions[name]):
            if isinstance(node, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)):
                identity_sensitive = True
            if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, ast.Store):
                identity_sensitive = True
            if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr == "data"):
                identity_sensitive = True
            if isinstance(node, ast.Compare) and any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                identity_sensitive = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in functions:
                identity_sensitive = True
    return {"version": 3, "fingerprint": hashlib.sha256(normalized.encode()).hexdigest(),
            "identity_sensitive": identity_sensitive, "sample_loss": "sample_loss" in functions}


def load(source):
    import torch
    import math

    analyze(source)
    namespace = {"torch": torch, "math": math}
    exec(compile(source, "<mirage-training-program>", "exec"), namespace)
    return namespace


def request(metadata, seed, device, size, requires_backward=True):
    return Request(Stage.CANONICAL, (
        "Generate a small legal PyTorch training program for the target API using its supplied "
        "installed signature and documentation. Return JSON with exactly source. The source value "
        "contains valid Python only, with no Markdown fences or trailing JSON brace inside the string. "
        "Define build(device, seed), forward(ctx), loss(output), and optionally sample_loss(output). "
        "build creates legal operands, trainable parameters and an optimizer. Return a dict with "
        "inputs (dict of named tensors or nested values), parameters (dict of named trainable leaf tensors), "
        "optimizer (torch optimizer or None), and batch_axes (dict mapping batch input names to axis 0). "
        "Optionally register modules in a modules dictionary and every module parameter in parameters. "
        "Use torch.manual_seed(seed), create tensors on device, and keep tensor sizes small. "
        "forward accesses ctx['inputs'] and ctx['parameters'], calls the target API, and returns "
        "a Tensor or nested tensors/values. Do not detach the target's differentiable result. "
        "loss returns a scalar derived from floating target outputs; tuple results may be unpacked. "
        "For batch-separable programs optionally define sample_loss(output), returning one scalar "
        "per sample (a vector); loss must equal its mean. For matrix multiplication use a shared "
        "trainable weight and batch input. For gather use a trainable floating operand and integer indices. "
        "For Tensor in-place APIs mutate a nonleaf, not a trainable leaf. Keep the designated target "
        "API on a differentiable path: at least one floating operand must require gradients, either "
        "directly or through a registered parameter. An unrelated unused parameter does not connect "
        "the target to autograd. For parameterless operators a trainable input is appropriate. "
        "Keep its call explicit, do not replace it with a different API. Do not add instrumentation or assert "
        "an MR is valid. No file/network I/O, external inputs, hidden mutable globals, or large workloads. "
        "You choose operand shapes, API options, optimizer, topology and differentiable loss. "
        "Do not generate a full training loop; the runner executes lifecycle actions. "
        "For batch_axes only register names actually present in inputs, with axis 0; "
        "do not label parameters, reduction axes or shared weights. Use {} if no input has a sample axis. "
        + ("" if requires_backward else
           "For this explicit forward-only target, the target itself need not be differentiable: "
           "include its result in the structured output and derive the training loss from another "
           "floating output if necessary. Target backward connectivity is not required for this case.")
    ), {"contract": "training-program", "api_context": metadata,
        "configuration": {"seed": seed, "device": device, "size_hint": size,
                          "requires_backward": requires_backward}})


def analyze_property(source):
    """Bounded public-API builders; this is validation, not a security sandbox."""
    if not isinstance(source,str) or not source.strip() or len(source)>40000:
        raise ContractError('Expected property source of at most 40000 characters')
    tree=ast.parse(source)
    definitions={}
    for node in tree.body:
        if isinstance(node,ast.Import) and all(a.name=='math' or a.name=='torch' or a.name.startswith('torch.') for a in node.names): continue
        if isinstance(node,ast.ImportFrom) and node.module and (node.module=='torch' or node.module.startswith('torch.')): continue
        if not isinstance(node,ast.FunctionDef) or node.decorator_list:
            raise ContractError('Property source supports torch/math imports and functions only')
        if node.name in definitions: raise ContractError('Duplicate function')
        definitions[node.name]=node
    if 'build' not in definitions or [a.arg for a in definitions['build'].args.args]!=['device','seed']:
        raise ContractError('Define build(device, seed)')
    forbidden={'rand','randn','randint','rand_like','randn_like','randperm','normal','uniform_',
               'bernoulli','bernoulli_','dropout','manual_seed','seed','empty','empty_like',
               'eval','exec','open','compile','is_compiling','data_ptr','untyped_storage','getattr','setattr'}
    for node in ast.walk(tree):
        if isinstance(node,ast.ClassDef) and [ast.unparse(b) for b in node.bases]!=['torch.nn.Module']:
            raise ContractError('Only local torch.nn.Module classes are supported')
        if isinstance(node,ast.Assert):
            raise ContractError('The runner owns assertions and oracles')
        if isinstance(node,ast.Attribute) and isinstance(node.ctx,ast.Store) and not (isinstance(node.value,ast.Name) and node.value.id=='self'):
            raise ContractError('Do not modify imported APIs or external attributes')
        if isinstance(node,(ast.Global,ast.Nonlocal)):
            raise ContractError('No mutable globals/nonlocals')
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            if name in forbidden or name.startswith('_') and name!='__init__':
                raise ContractError(f'Unsupported or nondeterministic builder call: {name}')
        if isinstance(node,ast.Attribute) and node.attr.startswith('_') and node.attr not in ('__init__','__class__'):
            raise ContractError('Use public APIs only')
    return dict(version='property-1',fingerprint=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())


def property_request(metadata, configuration):
    from mirage.rules.properties import description
    return Request(Stage.CANONICAL,
        'Generate one small deterministic PyTorch property-test program from the installed API specification. '
        'Return JSON with exactly source. Use torch/math imports and define build(device, seed). '
        'build must return a dictionary whose named keys are specified below, never a tuple. '
        'Use explicitly initialized tensors (tensor, zeros, ones, arange, linspace), bounded dimensions '
        'and finite values. No RNG, external I/O, private APIs, assertions, or expected answers. '
        'The runner owns API invocation, state evidence, references, gradients and validation. '
        'Choose shapes, values, legal options and public representation operations yourself; vary them '
        'across attempt indices. Inputs may use ordinary views, strides, complex values, storage offsets '
        'or sparse layouts when the API supports them. Keep total storage below 2 million elements. '
        'Do not supply unrelated trainable parameters. '+description(configuration['family']),
        {'contract':'training-program','program_kind':'property-1','api_context':metadata,
         'configuration':configuration})
