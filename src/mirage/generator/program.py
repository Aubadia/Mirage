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


def request(metadata, seed, device, size, requires_backward=True, required_capabilities=()):
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
        "unmet_capability_requirements": list(required_capabilities),
        "capability_guidance": "When legal, choose a source supporting these unmet capabilities. Runtime evidence will verify them; declarations do not certify support.",
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


EDGE_CASE_PROMPT = "Consider edge cases from the assigned coverage category as well as ordinary\ninputs across the run. Choose legal, explicitly initialized inputs using the\nsupplied API specification and backend capabilities.\nFor numerical coverage, consider exact zeros, repeated and nonuniform values,\nmixed signs where legal, and small nonzero or large finite magnitudes relative\nto the dtype's representable range. Derive scales from dtype metadata rather\nthan assuming one fixed floating-point format. Record whether conversion or\nexecution changes the intended magnitude category.\nFor structural coverage, consider legal singleton dimensions, batch dimensions,\nbroadcast-compatible unequal shapes, and alternative supported layouts. Use\nempty dimensions only when permitted by both the API and the selected relation.\nWhen relationships between operands or returned values matter, expose the\nobservables required by that relation. Include nontrivial, distinguishable\nvalues where legal; do not assume that every edge case should fail.\nSelect a bounded subset for this program rather than combining every edge case.\nPreserve required domains, initial conditions, differentiation coordinates and\nsource/follow-up input relations. If a category is not applicable, report that\nlimitation instead of forcing an invalid input or inventing an expected answer.\nYou still own the training-state operations and their ordering. The framework\nmay mutate tensor representations but will not construct your training history."


def analyze_stateful(source):
    """Check the new interface without synthesizing or restricting a lifecycle."""
    from mirage.contracts.programs import initialization_call_kind
    if not isinstance(source, str) or not source.strip() or len(source) > 40000:
        raise ContractError("Expected nonempty source of at most 40000 characters")
    tree = ast.parse(source)
    definitions, aliases = {}, {"torch": "torch", "math": "math"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ("torch", "math"):
                    raise ContractError("Only torch/math imports are supported")
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("torch", "math"):
            for alias in node.names:
                if alias.name == "*": raise ContractError("Use explicit imports")
                aliases[alias.asname or alias.name] = node.module + "." + alias.name
        elif isinstance(node, ast.FunctionDef) and not node.decorator_list:
            if node.name in definitions: raise ContractError("Duplicate top-level definition")
            definitions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            if node.name in definitions: raise ContractError("Duplicate top-level definition")
            if node.decorator_list or node.keywords or node.type_params:
                raise ContractError("Top-level module classes cannot have decorators, metaclasses or type parameters")
            methods = set()
            for position, statement in enumerate(node.body):
                if isinstance(statement, ast.Pass):
                    continue
                if (position == 0 and isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)):
                    continue
                if not isinstance(statement, ast.FunctionDef) or statement.decorator_list:
                    raise ContractError("Top-level module class bodies support only undecorated methods, a docstring and pass")
                if statement.name in methods:
                    raise ContractError("Duplicate module method")
                methods.add(statement.name)
                # Defaults execute during class declaration, before build initialization.
                for default in statement.args.defaults + [d for d in statement.args.kw_defaults if d is not None]:
                    try:
                        ast.literal_eval(default)
                    except (ValueError, TypeError):
                        raise ContractError("Top-level module method defaults must be literal values") from None
            definitions[node.name] = node
        else:
            raise ContractError("Use torch/math imports, undecorated functions and inert torch.nn.Module classes")
    def resolve_name(node):
        if isinstance(node, ast.Name): return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute): return resolve_name(node.value) + "." + node.attr
        return ""
    for name, arguments in {"build": "device, seed", "run": "ctx, probe"}.items():
        expected = ast.parse(f"def f({arguments}): pass").body[0].args
        if name not in definitions or not isinstance(definitions[name], ast.FunctionDef) or ast.dump(definitions[name].args) != ast.dump(expected):
            raise ContractError(f"Define {name}({arguments})")
    seed_calls, unresolved = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.Assert)):
            raise ContractError("No mutable globals or generated oracle assertions")
        if isinstance(node, ast.ClassDef):
            if [resolve_name(base) for base in node.bases] != ["torch.nn.Module"]:
                raise ContractError("Classes must derive from torch.nn.Module (aliases allowed)")
        if isinstance(node, ast.Call):
            name = resolve_name(node.func)
            if name == 'probe.mark' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value in ('initial','final'):
                raise ContractError('initial/final checkpoints are reserved for the inspector; use a unique intermediate name')
            category = initialization_call_kind(name)
            if category == "sampling": raise ContractError(f"Explicit deterministic initialization required: {name}")
            if category == "seed_setting": seed_calls.append(node.lineno)
            if name.startswith("torch.nn."): unresolved.append(name)
            if name in ("eval", "exec", "open", "__import__"):
                raise ContractError("External execution/I/O is outside the program contract")
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            if resolve_name(node.value).split(".")[0] in ("torch", "math", "probe"):
                raise ContractError("Do not replace framework or probe attributes")
    return {"version": "stateful-1", "fingerprint": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
            "aliases": aliases, "seed_calls": seed_calls, "initialization_review": unresolved,
            "determinism": "runtime_evidence_required"}


def source_operand_bindings(source, api, selectors=()):
    """Bounded AST guidance for direct probe calls; never runtime evidence.

    Runtime prepends a bound tensor-method receiver to positional arguments.
    Dynamic call ordering and unpacked arguments are left unresolved, rather
    than guessed from the numerical operand paths.
    """
    result = {'version': 'source-operands-1', 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'scope': 'Static source guidance only; runtime verifies operands, state and backward paths.',
              'bindings': [], 'calls': [], 'omitted_bindings': 0, 'omitted_calls': 0}
    wanted = []
    for selector in selectors:
        for key in ('operand_path', 'peer_operand_path'):
            path = selector.get(key)
            if type(selector.get('call_index')) is int and isinstance(path, (list, tuple)) and path:
                coordinate = (selector['call_index'], tuple(path))
                if coordinate not in wanted:
                    wanted.append(coordinate)
    def unresolved(reason):
        result.update(status='unresolved', reason=reason)
        result['bindings'] = [{'call_index': call, 'operand_path': list(path), 'status': 'unresolved', 'reason': reason}
                              for call, path in wanted[:24]]
        result['omitted_bindings'] = max(0, len(wanted) - 24)
        return result
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError):
        return unresolved('source cannot be parsed')
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases.update({alias.asname or alias.name: node.module + '.' + alias.name for alias in node.names})
    def qualified(node):
        if isinstance(node, ast.Name): return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute): return qualified(node.value) + '.' + node.attr
        return ''
    def is_probe(node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'probe' and node.func.attr == 'call')
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    if len(functions) != 1:
        return unresolved('requires one run function')
    run = functions[0]
    if any((isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == 'probe')
           or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == 'probe')
           or (isinstance(n, ast.alias) and (n.asname or n.name.split('.')[0]) == 'probe')
           or (isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and n.name == 'probe')
           or (isinstance(n, ast.MatchMapping) and n.rest == 'probe')
           for n in ast.walk(run)):
        return unresolved('the runtime probe binding is reassigned')
    if any(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in aliases for node in ast.walk(run)):
        return unresolved('an imported callable name is reassigned')
    # Even an indirect or unused factory reference makes source-only call
    # ordinals ambiguous. Include local import aliases without trusting them
    # as evidence that any particular target actually executes.
    factories = ('torch.compile', 'torch.cuda.make_graphed_callables')
    factory_aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split('.')[0]
                value = alias.name if alias.asname else alias.name.split('.')[0]
                factory_aliases.setdefault(local, set()).add(value)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                factory_aliases.setdefault(alias.asname or alias.name, set()).add(node.module + '.' + alias.name)
    def factory_references(node):
        if isinstance(node, ast.Name): return factory_aliases.get(node.id, {node.id})
        if isinstance(node, ast.Attribute): return {prefix + '.' + node.attr for prefix in factory_references(node.value)}
        return set()
    if (api in factories or any(set(factories) & factory_references(n) for n in ast.walk(tree))
            or any(set(factories) & values for values in factory_aliases.values())):
        return unresolved('factory or compiled call boundaries require runtime call-site linkage')
    helpers = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n is not run}
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in helpers for n in ast.walk(run)):
        return unresolved('helper calls require runtime call-site linkage')
    anchors = []
    for index, statement in enumerate(run.body):
        calls = [node for node in ast.walk(statement) if is_probe(node)]
        if calls:
            if (len(calls) != 1 or not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return))
                    or statement.value is not calls[0]):
                return unresolved('nested, conditional or repeated probe call ordering is ambiguous')
            anchors.append((index, statement, calls[0]))
    if not anchors or len([n for n in ast.walk(tree) if is_probe(n)]) != len(anchors):
        return unresolved('helper or dynamic probe call ordering is unresolved')
    target_api = 'torch.' + api if api.startswith('Tensor.') else api
    for call_index, (statement_index, statement, call) in enumerate(anchors):
        call_info = {'call_index': call_index, 'statement_index': statement_index, 'line': call.lineno,
                     'source': ast.unparse(statement)[:1200], 'first_operand_setup_statement_index': anchors[0][0]}
        target = call.args[0] if call.args else None
        args = list(call.args[1:])
        kwargs = {item.arg: item.value for item in call.keywords if item.arg is not None}
        if target is None or any(isinstance(arg, ast.Starred) for arg in call.args) or any(k.arg is None for k in call.keywords):
            call_info.update(status='unresolved', reason='unpacked or missing call arguments')
        elif qualified(target) == target_api:
            call_info.update(status='mapped_static', binding_form='explicit_callable')
        elif (target_api.startswith('torch.Tensor.') and isinstance(target, ast.Attribute)
              and target.attr == target_api.rsplit('.', 1)[-1]
              and qualified(target.value).split('.')[0] not in ('torch', 'math')):
            args.insert(0, target.value)
            call_info.update(status='mapped_static', binding_form='bound_receiver_prepended')
        else:
            call_info.update(status='unresolved', reason='target alias or receiver binding is ambiguous')
        result['calls'].append(call_info)
        selected = [(number, path) for number, path in wanted if number == call_index]
        if not wanted:
            selected = [(call_index, ('args', index)) for index in range(len(args))]
            selected += [(call_index, ('kwargs', key)) for key in kwargs]
        for _, path in selected:
            item = dict(call_index=call_index, operand_path=list(path), statement_index=statement_index,
                        status=call_info['status'])
            expression = None
            if call_info['status'] == 'mapped_static' and len(path) >= 2:
                if path[0] == 'args' and type(path[1]) is int and 0 <= path[1] < len(args):
                    expression = args[path[1]]
                elif path[0] == 'kwargs' and type(path[1]) is str:
                    expression = kwargs.get(path[1])
                for key in path[2:]:
                    if expression is None or type(key) not in (str, int):
                        expression = None
                        break
                    if isinstance(expression, (ast.List, ast.Tuple)) and type(key) is int:
                        expression = expression.elts[key] if 0 <= key < len(expression.elts) else None
                    elif isinstance(expression, ast.Dict):
                        matches = [value for node, value in zip(expression.keys, expression.values)
                                   if isinstance(node, ast.Constant) and type(node.value) is type(key) and node.value == key]
                        expression = matches[0] if len(matches) == 1 else None
                    else:
                        expression = ast.Subscript(value=expression, slice=ast.Constant(value=key), ctx=ast.Load())
            if expression is None:
                item.update(status='unresolved', reason=call_info.get('reason', 'operand path has no unambiguous source expression'))
            else:
                names = sorted({n.id for n in ast.walk(expression) if isinstance(n, ast.Name)})
                definitions = {}
                for index, earlier in enumerate(run.body[:statement_index]):
                    for node in ast.walk(earlier):
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in names:
                            definitions[node.id] = {'name': node.id, 'statement_index': index,
                                'source': ast.unparse(earlier)[:800],
                                'status': 'direct_assignment' if isinstance(earlier, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                                          else 'conditional_or_nested_assignment'}
                item.update(expression=ast.unparse(expression)[:600], root_names=names[:8],
                            last_definitions=list(definitions.values())[:4], binding_form=call_info['binding_form'],
                            receiver=call_info['binding_form'] == 'bound_receiver_prepended' and path[:2] == ('args', 0),
                            redefined_after_first_setup=any(d['statement_index'] >= anchors[0][0] for d in definitions.values()))
            result['bindings'].append(item)
    existing = {(item['call_index'], tuple(item['operand_path'])) for item in result['bindings']}
    for number, path in wanted:
        if (number, path) not in existing:
            result['bindings'].append({'call_index': number, 'operand_path': list(path), 'status': 'unresolved',
                                       'reason': 'call index has no unambiguous source boundary'})
    result['omitted_bindings'] = max(0, len(result['bindings']) - 24)
    result['bindings'] = result['bindings'][:24]
    result['omitted_calls'] = max(0, len(result['calls']) - 8)
    result['calls'] = result['calls'][:8]
    result['status'] = 'mapped_static' if all(item['status'] == 'mapped_static' for item in
        result['bindings'] + result['calls']) else 'partially_unresolved'
    return result


def unmet_predicate_feedback(inspection, *, bindings=None, limit=3):
    """Small repair context; complete observations stay in immutable artifacts."""
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('Feedback limit must be between one and eight')
    def bounded(value, depth=0):
        if value is None or type(value) in (bool, int): return value
        if type(value) is float:
            return value if __import__('math').isfinite(value) else str(value)
        if isinstance(value, str): return value[:600]
        if depth >= 3: return '<omitted nested diagnostic>'
        if isinstance(value, (tuple, list)): return [bounded(v, depth + 1) for v in value[:8]]
        if isinstance(value, dict): return {str(k): bounded(v, depth + 1) for k, v in list(value.items())[:12]}
        return '<unsupported diagnostic value>'
    failed = [p for p in inspection.get('predicates', []) if p.get('attained') is not True]
    keys = ('dimension', 'field', 'checkpoint', 'call_index', 'operand_path', 'parameter_key', 'axis', 'peer_operand_path',
            'peer_axis', 'relation', 'peer_relation', 'expected', 'observed', 'evidence_status', 'reason', 'attained')
    selected = [{key: bounded(p.get(key)) for key in keys} for p in failed[:limit]]
    wanted = {(p.get('call_index'), tuple(p.get(key) or ())) for p in failed[:limit]
              for key in ('operand_path', 'peer_operand_path') if p.get(key)}
    mapped = [bounded(item) for item in (bindings or {}).get('bindings', [])
              if (item.get('call_index'), tuple(item.get('operand_path') or ())) in wanted][:2 * limit]
    return {'kind': 'unmet_state_predicates', 'api_executed': inspection.get('api_executed'),
            'required_backward_executed': inspection.get('required_backward_executed'),
            'required_backward_path': inspection.get('required_backward_path'),
            'unmet_predicates': selected, 'omitted_predicates': max(0, len(failed) - limit),
            'source_bindings': mapped, 'scope': 'Diagnostic only; no reachability or MR claim.'}


def stateful_request(metadata, configuration, *, edges=True, canonical=None, predicates=(), relation=None, mutation=None):
    import copy
    context = copy.deepcopy(metadata)
    if configuration.get('check_claim'):
        claim = configuration['check_claim']
        instructions = {
            'forward': 'One legal target return. No optimizer or backward is necessary.',
            'operand_relation': 'Construct operands with the requested relation, preserving call paths and required history. Shared storage may use disjoint legal views of one base.',
            'backward': 'Execute a backward path from the target result to its differentiable operands.',
            'accumulation': 'Execute two target-connected backward calls accumulating into the same leaf without clearing or overwriting its gradient.',
            'scope_exit': 'Construct the target context with probe.call and enter and exit that exact instance.',
            'callable_use': 'Construct the target callable with probe.call, then invoke the returned object with probe.call. Compiled/captured callables may also be invoked directly.',
            'callback_trigger': 'Register a Python function with the target using probe.call, then trigger the registered callback through its documented PyTorch event.',
            'persistent_step': 'Construct the optimizer or scheduler with probe.call. Set up valid gradients or prerequisite optimizer updates, then invoke probe.advance(product). Expose the resulting parameter or scheduler state.',
            'async_completion': 'Obtain the actual work handle through probe.call and invoke probe.wait(handle) to await completion.',
        }
        api = configuration['api']
        public_callable = 'torch.' + api if api.startswith('Tensor.') else api
        prompt = (
            f"Generate a small deterministic legal test of {api}. "
            "Return JSON with exactly source. Define build(device, seed) returning a dict with inputs "
            "and parameters dictionaries. run receives exactly that dict, e.g. x = ctx['inputs']['x']; "
            "there is no flattened ctx['x']. Only registered leaf tensors requiring grad belong in parameters. "
            "Use explicit finite small values on device, under 4096 elements. Define run(ctx, probe) returning "
            "a dictionary of tensor or primitive observations. Call the exact public target with "
            f"probe.call({public_callable}, ...), passing the real operands in place of ellipsis. "
            "Probe has no target attribute. Use its return. Optional probe.mark records a named "
            "checkpoint. Use only torch/math, no external I/O, random sampling or expected answers. "
            "The initial and final checkpoints belong to the framework. Use only history needed by this claim. "
            + instructions[claim])
        if configuration.get('state_goal'):
            goal = configuration['state_goal']
            prompt += ('\nTarget measured history: ' + goal['objective'] +
                       '\nKeep the exact target on this path. Use the shortest legal sequence. '
                       'Preserve materialized semantic input values on both devices. '
                       'When a prefix is supplied, reuse its relevant initialization and verified '
                       'operations and change only what the new history requires. A user mark or '
                       'self-reported boolean cannot satisfy an observer. Never fabricate evidence. '
                       'The framework injects make_tensor(shape, *, kind="distinct", dtype=..., '
                       'device=device, requires_grad=False, index_bound=None, sample=0) directly in '
                       'program globals. Use kind distinct, positive, index, invertible, tiny or nonzero. '
                       'Set an integer index_bound for index. Use different sample numbers for batches '
                       'or general nonsymmetric, distinguishable coordinates. Use explicit concrete '
                       'initial tensors wherever special API semantics require them.')
        if configuration.get('state_goal', {}).get('family') == 'rng':
            prompt += ' Use deterministic initial operands. Any stochastic draw must occur inside the target, optionally with an explicitly seeded generator.'
        if configuration.get('guided_prefix_mode') == 'append_only':
            prompt += ('\nKeep the parent build, all helpers, run signature, original non-return '
                       'statements and final return unchanged. Insert new statements immediately '
                       'before the final return. The harness checks this and the original target trace.')
        elif configuration.get('guided_prefix_mode') == 'reference_only':
            prompt += '\nUse the prior program as a reference for this goal.'
        if configuration.get('input_recipe'):
            prompt += ('\nA bounded input recipe is in configuration.input_recipe. In build you '
                       'can call the injected make_operand(name, *, requires_grad=False, sample=None) '
                       'for any name listed in recipe.operands. It reconstructs the recorded shape '
                       'and dtype directly on the requested device. Preserve that tested overload '
                       'and its relationships when using its operands.')
        return Request(Stage.TRANSITION if canonical else Stage.CANONICAL, prompt,
            dict(contract='training-program', program_kind='stateful-1', api_context=context,
                 canonical_source=canonical, configuration=configuration, state_predicates=list(predicates),
                 prompt_version='spec-fragment-1', prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    raw = context.get("documentation", "")
    context["documentation_annotation"] = (
        "Installed documentation is API evidence. Its RNG-based examples are not instructions "
        "to use random inputs. Seed setting is distinct from explicit deterministic initialization.")
    context["supplied_documentation_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    graph_guidance = (
        "Graph lifetime is your responsibility: backward consumes saved graph state unless you "
        "explicitly preserve it. Losses sharing an upstream differentiable intermediate also share "
        "that graph, even if their target outputs were computed separately. Choose legal retention "
        "or recomputation yourself; reconstructing only a downstream target output does not rebuild "
        "a consumed upstream graph. A backward_count goal counts actual backward/grad invocations, "
        "whereas target_backward_count counts target-connected gradient events.\n"
    ) if configuration.get("requires_backward", True) else ""
    prompt = (
        f"Target API: {configuration['api']}. Generate a small deterministic test exercising this exact API.\n"
        "Use explicit tensor/zeros/ones/arange/linspace initialization. Never sample random values, "
        "even with a seed; do not copy random examples from the API documentation.\n"
        f"Required target backward: {configuration.get('requires_backward',True)}. "
        f"Requested state predicates: {list(predicates)}.\n"
        "For an eager test, use the value RETURNED by probe.call for subsequent computation; "
        "calling the target separately and discarding probe.call output does not test its backward. "
        "If backward is not required, a legal target exercise needs no training loop or optimizer.\n"
        "Return one JSON object with exactly source (a complete Python source string). "
        "Write concise code; omit explanatory comments and redundant snapshots.\n"
        "1. Define build(device, seed) returning a dict with inputs and parameters dictionaries. "
        "Optional fields: modules, optimizer, scheduler, scaler and checkpoint_data. "
        "Use explicitly initialized tensors. A seed does not replace initialization. "
        "A torch.nn.Module class may be declared at top level or locally; top-level class bodies "
        "contain only undecorated methods, a docstring or pass, with initialization inside __init__. "
        "Replace all retained random module values. Registered module parameters must be the "
        "same objects as entries in parameters, not the tensors used to construct Parameters. "
        "For a constructed model, use parameters = dict(model.named_parameters()) and "
        "modules = {'model': model} in the returned context; never store named_parameters tuples "
        "or clone/detach those Parameter objects. "
        "Every registered parameter must be a tensor with is_leaf=True and requires_grad=True. "
        "Inputs may require gradients without being registered parameters. "
        "Omit unused optional fields. optimizer/scheduler/scaler must be real objects or None, "
        "never dictionaries. inputs/parameters/modules are dictionaries and may be empty. "
        "Functions belong in source, not checkpoint_data or returned observations.\n"
        "2. Define run(ctx, probe). You own all operations and their ordering: "
        "the framework inserts no backward, clearing, optimizer step, context, compilation or replay. "
        "Without canonical_source, write a minimal legal canonical use and the required backward. "
        "canonical_source is a starting point, not a mandatory program structure. Rewrite it as "
        "needed to reach the supplied predicates at "
        "their named checkpoints. Counts are exact, not lower bounds. Choose a different legal "
        "training transition structure from prior_transition_signatures; different values or "
        "observation copies alone do not establish diversity. No fixed action sequence is required.\n"
        + graph_guidance +
        "3. For an eager target invocation, call probe.call(target_callable, *args, **kwargs) "
        "and use its returned result. Only the configured target or an exact returned callable "
        "registered by its observed factory goes through probe.call; call other operations directly. "
        "Tensor methods require their bound receiver or an "
        "explicit receiver with the unbound method. Use a returned context manager/callable "
        "according to its public specification. For required backward, differentiable target "
        "operands need gradient history before the call and the observed result must reach "
        "backward; an unrelated downstream parameter or separate unobserved call is insufficient.\n"
        "4. A compiled_execution=True goal REQUIRES source-owned torch.compile and an actual "
        "invocation of the resulting callable. For an ordinary tensor-operator target, call it "
        "directly inside your compiled function; callable-factory targets follow their protocol below. "
        "Keep probes outside tracing/capture and invoke the compiled callable directly. "
        "For the supported compiled inspection contract, keep backward calls outside the compiled body. "
        "A compile flag or tracing-only call is not runtime evidence. Do not fake probe fields.\n"
        "5. run returns a dictionary of observable tensors/primitives, never expected answers, "
        "modules, optimizers or the whole ctx. Return tensors directly; the inspector snapshots "
        "them. Optional probe.mark(name, data) adds unique intermediate checkpoints. "
        "The inspector owns initial/final; never use those reserved names. Probe has only call "
        "and mark methods. Expose related alias partners where an effect relation needs them.\n"
        "6. No random sampling, NumPy/Tensor.numpy(), external I/O, assertions or hidden globals. "
        "Use public APIs. Keep each tensor within 4096 elements and at most four updates. "
        "Keep differentiable views in run. Generate legal values/shapes for the assigned input "
        "category. Replace canonical inputs that do not demonstrate it, consistently for both MR "
        "members; actual target values, not comments describing them as small/large, establish coverage. "
        "If a requested capability is unavailable, preserve legal coverage and do not invent evidence."
    )
    if configuration.get('representation_traversal'):
        prompt += (
            '\nFRAMEWORK REPRESENTATION TRAVERSAL: You own programs, semantic input values/shapes and '
            'training history. The framework enumerates supported value-preserving target operand '
            'representations after execution; do not add storage-layout tricks merely to obtain '
            'target stride/offset/conjugate/negative/alias coverage. Prefer ordinary explicit tensors '
            'unless views are needed for the program semantics or requested training state. '
            'Use feedback about unavailable transformations to vary semantic shapes/values; never '
            'claim a representation is covered before runtime verification.')
    if configuration.get('requires_backward', True):
        prompt += (
            '\nATTRIBUTABLE UPDATE: When the target is a differentiable eager tensor operation compatible '
            'with CPU float64, aim for a real single native SGD update. Register an actual target leaf '
            'tensor under ctx.parameters and pass that SAME object to the optimizer and probe.call. '
            'Use explicit deterministic inputs, positive finite lr, momentum in [0,1), weight_decay=0, '
            'dampening=0, nesterov=False, maximize=False, foreach=False, fused=False, differentiable=False. '
            'Construct SGD in build and return optimizer in ctx; begin with parameter.grad=None. '
            'For the reviewed source-effects fragment, use straight-line build statements: construct '
            'deterministic values with tensor literals or pure tensor factories, enable gradients on '
            'each named leaf explicitly, and pass an explicit list of those bound leaves to SGD. '
            'Indexed tensor writes, initialization loops and computed parameter collections such as '
            'mapping.values() currently retain legal coverage but lack this optimizer certificate. '
            'Use the target result for loss/backward, then call the real optimizer.step once. '
            'No closure, scheduler, AMP, user hooks or gradient overwriting belong to this bounded update '
            'contract. Do not force this protocol onto context-only or callable-factory targets, unsupported '
            'device/dtype, or APIs lacking a direct differentiable parameter operand; retain legal API coverage. '
            'A missing-update bug must not be hidden by an assertion that parameters changed.')
    if mutation is not None and mutation.get("version") != "local-regions-2":
        if canonical is None:
            raise ContractError("A local mutation requires its immutable parent source")
        prompt = prompt.replace(
            "Return one JSON object with exactly source (a complete Python source string). ",
            "Return one JSON object with exactly source (replacement Python statements for the editable region only). ")
        prompt = prompt.replace(
            "canonical_source is a starting point, not a mandatory program structure. Rewrite it as needed to reach the supplied predicates at ",
            "canonical_source is an immutable parent. Edit only mutation_contract.editable_region to reach predicates at ")
        region = mutation['editable_region']
        location = f"{region['function']} body statements [{region['start']}:{region['stop']}]"
        instructions = {
            'operand_setup': "Insert operand preparation immediately before the existing target call. "
                "The target call and all following loss/backward/return statements stay in place. "
                "Do not include probe.call, backward, grad, return or unrelated setup statements.",
            'initialization': "Replace only the selected build assignment, preserving the names needed by "
                "the remaining build statements. Do not include the build return, run body or target call.",
            'run_tail': "Replace only the suffix after the last protected target call. Its output variables "
                "already exist. Do not repeat that call or its earlier setup; include the required return."
        }
        prompt = ("LOCAL EDIT TASK: Return source containing only statements for " + location + ". "
                  "These are zero-based top-level statement indices, not line numbers. Equal bounds "
                  "mean insertion into an empty region, not replacement of the whole function. "
                  + instructions[mutation['kind']] + " The build/run rules below constrain the assembled "
                  "parent plus edit; they do not ask you to generate those functions again.\n\n" + prompt)
        prompt += ("\nLOCAL MUTATION: The build/run requirements above describe the assembled program. "
                   "Return only unindented replacement statements for mutation_contract.editable_region. "
                   "Do not return imports, a complete program, or definitions of build/run. "
                   "Mirage preserves all other statements and the original target calls. "
                   "Respect the supplied edit budget and preserved obligations. For run_tail include "
                   "a return of the required observables. The parent/child pair is exploration lineage, "
                   "not an MR pair. Each child's MR applicability is independently checked at runtime.")
    if mutation is not None and mutation.get("version") == "local-regions-2":
        prompt = (
            "Generate bounded patches implementing transition_plan, using the installed API spec. "
            "Return JSON with exactly patches: an array of region_id, preimage_sha256, source. "
            "Copy IDs/hashes exactly from mutation_contract.regions. Each source contains only "
            "unindented replacement statements, no imports or build/run definitions. "
            "All coordinates are zero-based TOP-LEVEL STATEMENT indices in the original canonical_source, "
            "not line numbers. Equal start/stop means INSERTION into an EMPTY region. "
            "region_statements is READ-ONLY surrounding context, NOT the requested replacement. "
            "Do not copy surrounding statements into a patch. Preserve names used elsewhere. "
            "device and seed are build arguments, NOT run variables; in run use an existing tensor device/dtype. "
            "For operand_setup return ONLY assignments constructing/rebinding the target operand variables "
            "already in scope. This patch MUST NOT contain probe.call, backward, grad or return: "
            "the original target/loss/backward/return remain AFTER the inserted statements. "
            "For example, an insertion patch is a short assignment, never the rest of run. "
            "initialization replaces only the selected build assignment and MUST NOT contain run code; "
            "run_tail replaces the suffix AFTER the last target and must include return. "
            "Existing target calls remain unchanged. A run_tail may add target calls when the plan "
            "requires additional boundaries. Plan graph retention before the first backward, "
            "or recompute the whole required graph. Use probe.call results for backward. "
            "The assembled program must reach every predicate at its named checkpoint. "
            "Keep edits within per-region and combined budgets. No random initialization, I/O, "
            "expected answers or fake probe evidence. Each child starts from build(). "
            "Parent/child lineage is not an MR; child MR applicability is checked independently. "
            "Registered parameters must be tensors with is_leaf=True and requires_grad=True; "
            "parameters may be empty. Inputs needing gradients need not be registered parameters."
        )
    if canonical is not None and predicates:
        prompt += (
            "\nOperand binding hints constrain only predicates selecting operand paths. "
            "Training counts and context predicates without operand paths do not require tensor bindings; "
            "their legal actions and runtime state evidence remain necessary."
        )
    if any(p.get('dimension') in ('gradient_representation', 'optimizer') for p in predicates):
        prompt += (
            '\nOPTIMIZER STATE OBLIGATIONS: optimizer.target_update_count at final counts only completed '
            'target-linked native updates. A full edit may establish registration and optimizer construction '
            'when absent. parameter_key selects the exact ctx.parameters object, while call_index/operand_path '
            'identify its actual target operand. Gradient representation predicates refer to consumed .grad '
            'at optimizer:0/step:0/pre_update; optimizer.completed refers to the corresponding post_update. '
            'These names belong to the observer: never emit probe.mark with optimizer: names. '
            'The source execution configuration optimizer_gradient_action performs a verified equal-value '
            'representation copy immediately before your real optimizer.step; do not assign p.grad or '
            'fabricate the layout in source. Preserve the target/backward/update chain and all completed-update '
            'companion predicates. A run_tail edit preserves the existing target then supplies legal backward, '
            'step and return; use full if build registration or optimizer is missing. This covers optimizer '
            'consumption of the resulting gradient, not backward accumulation into a preallocated layout.')
    if any(p.get('dimension') == 'tensor_relation' or p.get('axis') is not None for p in predicates):
        prompt += (
            "\nTENSOR RELATION GOALS: Every supplied predicate must hold at its named checkpoint. "
            "Both operand_path and peer_operand_path refer to actual operands of the SAME call_index; "
            "do not satisfy them using different calls or later tensors. Consult source_operand_bindings for "
            "the receiver/argument expression and last definitions; these are static guidance, not evidence. "
            "For a selected operand path, an unresolved binding requires a legal full edit or remains unresolved, never guessing "
            "another operand. A later reassignment can overwrite an edit before the first target. axis and peer_axis are "
            "nonnegative tensor axes, not the target API's dim argument. extent_order compares "
            "the two selected shape extents as lt/eq/gt; it does not itself establish API legality "
            "or broadcasting. Coordinate all required shapes, layouts and call options legally. "
            "shared_storage requires observed shared/independent storage identities; shared storage "
            "does not imply overlapping occupied coordinates. Selected zero_stride means extent>1 "
            "AND stride==0. singleton_stride zero/unit/other describes stride at extent==1, while "
            "not_singleton means extent!=1. Preserve required primary relation and peer_relation "
            "backward paths independently; a peer needs its own attributable gradient evidence. "
            "Respect mutation_contract regions: operand_setup may only prepare existing operands, "
            "and a full edit may restructure otherwise protected call options. Parent/child are "
            "exploration lineage, not an MR pair. The newly generated program's source/follow-up "
            "executions must preserve their own MR-required initial state and semantic inputs.")
    prompt += (
        '\nA parent_certificate_diagnostic with status unknown means missing MR proof, not an invalid program. '
        'Preserve selected states and path obligations; never delete a hard state merely to obtain a certificate. '
        'A legal case without MR proof may remain coverage-only.'
    )
    protocol = None
    if configuration['api'] == 'torch.compile':
        protocol = {
            'kind': 'compiled_callable_factory', 'factory_target': 'torch.compile',
            'lifecycle': ['define_callable', 'create_compiled_callable', 'invoke_returned_callable', 'source_owned_backward_if_required'],
            'backward_scope': 'returned_callable_boundary',
            'internal_operand_attribution': 'not_established_by_factory_boundary'}
        prompt += (
            "\nCALLABLE FACTORY PROTOCOL: The configured target is torch.compile itself. "
            "Define your function in source, then create its compiled callable in run with "
            "probe.call(torch.compile, function, backend=..., fullgraph=True). Choose eager, "
            "aot_eager or inductor from the supported execution contract. Invoke that exact "
            "returned callable on legal inputs, directly or through probe.call; do not probe "
            "an unrelated function. Factory construction alone is not executed compiled work. "
            "For a module, construct it in build and register its actual named_parameters "
            "objects plus the module: parameters = dict(model.named_parameters()), modules = {'model': model}. "
            "Do not register the tensor used to construct a Parameter. "
            "Keep probing and backward outside the compiled body. When backward is required, "
            "use a differentiable returned output. The inspector certifies the executed "
            "callable boundary, not internal operator gradients or representations. "
            "Choose the computation and training history yourself; no particular operator is required.")
    elif configuration['api'] == 'torch.cuda.make_graphed_callables':
        protocol = {
            'kind': 'captured_callable_factory', 'factory_target': 'torch.cuda.make_graphed_callables',
            'lifecycle': ['initialize_cuda_callable', 'prepare_sample_args', 'capture_factory', 'invoke_returned_callable', 'source_owned_backward_if_required'],
            'backward_scope': 'returned_callable_boundary',
            'internal_operand_attribution': 'not_established_by_factory_boundary'}
        prompt += (
            "\nCALLABLE FACTORY PROTOCOL: Prefer a small registered buffer-free CUDA module "
            "with a pure forward for the canonical test, matching the existing execution-reference "
            "contract. Functions are legal but may retain coverage-only results. Use explicit "
            "deterministic values. For a module, initialize it in build, put the module in "
            "modules and its actual named_parameters objects in parameters: "
            "parameters = dict(model.named_parameters()), modules = {'model': model}. Move parameters, "
            "buffers and sample tensors to the same CUDA device before capture; buffers must "
            "not require gradients. In run call probe.call(torch.cuda.make_graphed_callables, "
            "callable, sample_args), then invoke the exact returned callable directly or with "
            "probe.call. A single callable needs a tuple of tensor arguments; a tuple of "
            "callables needs one corresponding sample tuple per callable, in execution order. "
            "Real arguments must match sample structure, shape, dtype, device and requires_grad. "
            "The factory owns its warmup and capture; do not wrap it in another CUDA capture. "
            "Do not install module hooks before capture, replace parameters/buffers after capture, "
            "or request higher-order derivatives. Autocast capture requires cache_enabled=False. "
            "Keep probes outside the captured body. Factory construction alone is not a replay. "
            "You own gradient initialization, clearing and backward after setup; use a captured "
            "differentiable output when backward is required. The inspector certifies the "
            "returned callable's replay/backward boundary without certifying internal operators.")
    if edges: prompt += "\n\n" + EDGE_CASE_PROMPT
    prompt += '\nJSON/PYTHON ENCODING: Prefer single-quoted Python string literals inside JSON source or patch strings. Properly escape the surrounding JSON. Finish the complete requested Python program, including build/run definitions and return statements when a full program is requested, before closing the response. For a patch request, finish only the requested patch. Do not add or auto-fill missing program logic outside the generated response.'
    return Request(Stage.TRANSITION if canonical else Stage.CANONICAL, prompt,
                   {"contract": "region-patches" if mutation and mutation.get("version") == "local-regions-2" else "training-program", "program_kind": "stateful-1", "api_context": context, "configuration": configuration,
                    "canonical_source": canonical, "state_predicates": list(predicates),
                    "source_operand_bindings": source_operand_bindings(canonical, configuration["api"], predicates) if canonical else None,
                    "mutation_contract": copy.deepcopy(mutation), "callable_protocol": protocol,
                    "state_measurement_units": {
                        "extent_order": "Same-call selected operand shape[axis] compared to peer shape[peer_axis]: lt/eq/gt; API legality is separate.",
                        "shared_storage": "Compare observed storage identities; shared does not mean overlapping, and unknown does not mean independent.",
                        "zero_stride": "Selected axis requires extent>1 and stride==0; omitted axis retains any-axis semantics.",
                        "singleton_stride": "Selected extent==1: zero/unit/other stride; otherwise not_singleton.",
                        "peer_relation": "A separately verified backward relation for the peer operand at the same call/checkpoint.",
                        "target_boundary_count": "Completed eager probe boundaries and executed target-containing compiler/captured callable boundaries. Factory construction and each returned-callable execution are distinct; a compiled callable may execute multiple graphs.",
                        "backward_count": "Observed backward/grad invocations in run.",
                        "target_backward_count": "Observed target-connected output gradient events.",
                        "target_backward_node_count": "Attributable internal target backward Node execution events; output gradients or callable boundaries alone do not prove these.",
                        "callable_backward_node_count": "Observed backward Node executions at an identity-linked returned callable boundary; internal operand gradients remain unknown.",
                        "restoration_count": "Observed balanced enter/exit scopes of the target context type.",
                        "compiled_execution": "True only after a graph produced by an explicit torch.compile call actually executes. Supported backends: eager, aot_eager, inductor; choose your own legal computation and history."},
                    "input_category_definitions": {
                        "small_finite": "At least one finite nonzero target operand value with abs(value) <= sqrt(finfo(dtype).tiny) after dtype conversion.",
                        "large_finite": "At least one finite target operand value with abs(value) >= sqrt(finfo(dtype).max), with legal bounded execution.",
                        "broadcast": "Unequal actual target operand shapes that are broadcast compatible, where allowed by the API.",
                        "supported_layout": "A supported non-strided layout, opaque layout, or noncontiguous dense target operand. Framework mutation may also supply this witness."},
                    "relation_requirements": relation or {
                        "representation": "Preserve initial logical inputs and training semantics; expose target outputs and registered parameter gradients. Applicability is checked at runtime, including write roles and gradient coordinates.",
                        "context_restore": "Expose genuine entry/exit scopes; the original context contract determines restoration, not generated expected answers.",
                        "execution_mode": "Use source-owned compilation/capture when selected. Expose output, parameter gradients and related alias partners. The same authored computation is compared under certified execution configurations."}, "prompt_version": ("stateful-10-local" if mutation is not None else "stateful-10") + ("-edges" if edges else "-base"),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
