"""Version-scoped input construction and bounded validation of operand constraints."""

import ast
import hashlib
import inspect
import math
import operator
import re
import textwrap
from collections import Counter


def make_tensor(shape, *, kind="distinct", dtype=None, device="cpu", requires_grad=False,
                index_bound=None, sample=0):
    """Construct reproducible logical values on CPU before placing a tensor on its device."""
    import torch

    shape = (shape,) if type(shape) is int else tuple(shape)
    if (len(shape) > 8 or any(type(n) is not int or not 0 <= n <= 4096 for n in shape)
            or math.prod(shape) > 4096 or type(sample) is not int or abs(sample) > 4096):
        raise ValueError("Expected a small explicit shape and an integer sample")
    dtype = (torch.long if kind == "index" else torch.float32) if dtype is None else dtype
    source = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
    if kind == "index":
        if type(index_bound) is not int or index_bound <= 0 or dtype not in (torch.int32, torch.int64):
            raise ValueError("An index tensor needs an explicit positive index_bound and integer dtype")
        values = (source + sample) % index_bound
    elif kind == "invertible":
        if (len(shape) < 2 or shape[-1] != shape[-2] or shape[-1] == 0 or
                not (dtype.is_floating_point or dtype.is_complex)):
            raise ValueError("An invertible tensor needs nonempty square floating or complex matrices")
        # Strict row diagonal dominance, with distinguishable opposite off-diagonal entries.
        values = (source.to(torch.float64) + 1) / ((source.numel() + 1) * (shape[-1] + 1))
        values += (2 + sample % 5) * torch.eye(shape[-1], dtype=torch.float64)
    elif kind in ("distinct", "nonzero", "positive", "tiny"):
        values = source.to(torch.float64) + 1 + abs(sample)
        if kind != "positive":
            values *= torch.where((source + sample) % 2 == 0, -1, 1)
        if kind == "tiny":
            if not (dtype.is_floating_point or dtype.is_complex):
                raise ValueError("tiny requires a floating or complex dtype")
            values *= math.sqrt(torch.finfo(dtype).tiny) / (64 * max(1, source.numel()))
        if dtype.is_complex:
            values = torch.complex(values, values.flip([-1]) if shape else values + 1)
    else:
        raise ValueError(f"Unknown tensor construction kind: {kind}")
    return values.to(device=device, dtype=dtype).detach().requires_grad_(requires_grad)


def _source_guards(target):
    """Only unconditional top-level Python rejection paths have function-wide scope."""
    try:
        source = textwrap.dedent(inspect.getsource(target))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return []
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return []
    sha = hashlib.sha256(source.encode()).hexdigest()
    guards = []
    arguments = tree.body[0].args
    positional = [a.arg for a in arguments.posonlyargs + arguments.args]
    for item in tree.body[0].body:
        message = None
        if isinstance(item, ast.Assert):
            expr = ast.unparse(item.test)
            message = item.msg
        elif (isinstance(item, ast.If) and not item.orelse and len(item.body) == 1
              and isinstance(item.body[0], ast.Raise)):
            expr = f"not ({ast.unparse(item.test)})"
            raised = item.body[0].exc
            if isinstance(raised, ast.Call) and raised.args:
                message = raised.args[0]
        elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Call) and item.value.args:
            if ast.unparse(item.value.func) not in ("torch._check", "torch._assert"):
                continue
            expr = ast.unparse(item.value.args[0])
            if len(item.value.args) > 1:
                message = item.value.args[1]
        else:
            continue
        if isinstance(message, ast.Lambda):
            message = message.body
        guard = dict(expression=expr, line=item.lineno, source_sha256=sha,
                     source_parameters=positional,
                     scope="installed Python implementation; predicate may depend on local variables")
        if isinstance(message, ast.Constant) and isinstance(message.value, str):
            guard['diagnostic'] = message.value[:240]
        guards.append(guard)
    return guards[:24]


def evaluate_shape_guard(guard, schema, operand_shapes):
    """Three-valued interpretation on finite shape metadata; never execute source expressions."""
    if not schema or guard.get('overload') != schema:
        return dict(value=None, reason='overload_unmatched')
    expr = guard.get('expression', '')
    if not isinstance(expr, str) or len(expr) > 400:
        return dict(value=None, reason='expression_outside_bounded_grammar')
    aliases = guard.get('argument_map', {})
    comparisons = {ast.Eq:operator.eq, ast.NotEq:operator.ne, ast.Lt:operator.lt,
                   ast.LtE:operator.le, ast.Gt:operator.gt, ast.GtE:operator.ge,
                   ast.In:lambda x,y: x in y, ast.NotIn:lambda x,y: x not in y}

    def shape(node):
        if not isinstance(node, ast.Name) or node.id not in aliases:
            raise ValueError('unbound_operand')
        sizes = operand_shapes.get(aliases[node.id])
        if (not isinstance(sizes, (list, tuple)) or len(sizes) > 8 or
                any(type(n) is not int or n < 0 for n in sizes)):
            raise ValueError('unknown_operand_shape')
        return tuple(sizes)

    def read(node):
        if isinstance(node, ast.Constant) and type(node.value) in (bool, int):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(read(n) for n in node.elts)
        if isinstance(node, ast.Attribute) and node.attr in ('ndim', 'shape'):
            dims = shape(node.value)
            return len(dims) if node.attr == 'ndim' else dims
        if isinstance(node, ast.Call) and not node.keywords:
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == 'len' and len(node.args) == 1:
                arg = node.args[0]
                return shape(arg)[0] if isinstance(arg, ast.Name) else len(read(arg))
            if isinstance(fn, ast.Attribute):
                dims = shape(fn.value)
                if fn.attr in ('dim', 'ndimension') and not node.args:
                    return len(dims)
                if fn.attr == 'size' and len(node.args) <= 1:
                    if not node.args:
                        return dims
                    axis = read(node.args[0])
                    if type(axis) is int:
                        return dims[axis]
        if isinstance(node, ast.Subscript):
            dims, axis = read(node.value), read(node.slice)
            if type(dims) is tuple and type(axis) is int:
                return dims[axis]
        if isinstance(node, ast.UnaryOp):
            value = read(node.operand)
            if isinstance(node.op, ast.USub) and type(value) is int:
                return -value
            if isinstance(node.op, ast.Not) and type(value) is bool:
                return not value
        if isinstance(node, ast.Compare):
            left = read(node.left)
            for operation, item in zip(node.ops, node.comparators):
                right = read(item)
                if not comparisons[type(operation)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = read(item)
                if type(value) is not bool:
                    raise ValueError('non_boolean_operand')
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            a, b = read(node.left), read(node.right)
            if type(a) is bool and type(b) is bool:
                return a and b if isinstance(node.op, ast.BitAnd) else a or b
        raise ValueError('expression_outside_bounded_grammar')

    try:
        tree = ast.parse(expr, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 80:
            raise ValueError('expression_outside_bounded_grammar')
        value = read(tree.body)
        if type(value) is not bool:
            raise ValueError('expected_boolean_guard')
        return dict(value=value, reason='bounded_shape_evaluation')
    except (KeyError, IndexError, TypeError, ValueError, SyntaxError, RecursionError) as error:
        return dict(value=None, reason=str(error)[:120] or type(error).__name__)


def input_spec(target, spec):
    import torch
    guards = [dict(g, binding_status='public_guard_without_verified_native_mapping')
              for g in _source_guards(target)]
    schemas = spec.get("operator_schemas", [])
    sources = []
    for schema in schemas[:16]:
        sources.append(dict(schema=schema["schema"], operands=schema["arguments"]))
    # Many public wrappers are native. The installed meta implementation is a second
    # source of guards; the association is for one exact overload and one build.
    if schemas:
        try:
            from torch._meta_registrations import meta_table
            for schema in schemas[:6]:
                ns, name = schema['name'].split('::')
                packet = getattr(getattr(torch.ops, ns), name)
                op = getattr(packet, schema['overload'] or 'default')
                for guard in _source_guards(meta_table.get(op)):
                    mapped = {arg['name']: arg['name'] for arg in schema['arguments']}
                    positional = [arg['name'] for arg in schema['arguments'] if not arg['keyword_only']]
                    mapped.update(zip(guard['source_parameters'], positional))
                    guards.append(dict(guard, overload=schema['schema'], source_kind='installed_meta',
                        argument_map=mapped, binding_status='matched_meta_overload'))
        except (ImportError, AttributeError, ValueError):
            pass
    return dict(version="bounded-input-spec-1", framework_version=spec['framework_version'],
                framework_build=spec['framework_build'], bindings=sources, validation_guards=guards,
                constructor_kinds=["distinct", "positive", "index", "invertible", "tiny", "nonzero"],
                probe_status="unmeasured", observations=[], refinements=[])


def plan_guard_probes(guards, schema, names):
    """Retain ordinary controls, then add bounded inside/complement assignments per source guard."""
    baseline = [((2, 2), 'float32'), ((2, 3), 'float32'), ((3,), 'float32'),
                ((2, 2), 'float64'), ((2, 2), 'int64')]
    plans = [(dict.fromkeys(names, shape), dtype) for shape, dtype in baseline]
    shapes = [(), (1,), (2,), (1, 1), (1, 2), (2, 1), (3, 2), (1, 2, 3), (1, 2, 2, 1)]
    assignments = [dict.fromkeys(names, shape) for shape in shapes]
    assignments += [dict(dict.fromkeys(names, (2, 2)), **{name: shape}) for name in names for shape in shapes]
    for guard in [g for g in guards if g.get('overload') == schema][:6]:
        sides = {evaluate_shape_guard(guard, schema, values)['value'] for values, _ in plans}
        for assignment in assignments:
            value = evaluate_shape_guard(guard, schema, assignment)['value']
            if value is None or value in sides:
                continue
            plans.append((assignment, 'float32'))
            sides.add(value)
            if True in sides and False in sides:
                break
    return plans


def probe_input_domain(spec, device):
    """Test one small native tensor call domain. Exceptions are scoped to each concrete call."""
    import torch
    model = spec['constraints']
    # Use exact native overloads with only required tensor operands. Protocols, side
    # effects, unsupported required scalars, and user-supplied callables need real programs.
    for schema in spec.get('operator_schemas', []):
        if schema['overload'] not in ('', 'Tensor') or not schema['name'].startswith('aten::'):
            continue
        if any(arg['write'] for arg in schema['arguments']):
            continue
        name = schema['name'].split('::', 1)[1]
        op = getattr(getattr(torch.ops.aten, name), schema['overload'] or 'default')
        required = [arg for arg in op._schema.arguments if not arg.has_default_value()]
        if (not 1 <= len(required) <= 3 or any(str(arg.type) != 'Tensor' or arg.kwarg_only for arg in required)
                or not any(str(arg.type) == 'Tensor' for arg in op._schema.returns)):
            continue
        # A probe belongs to a supported native operator class; arbitrary Python
        # functions, data-dependent-output and in-place operators use source evidence only.
        excluded = {'nondeterministic_seeded', 'data_dependent_output', 'dynamic_output_shape',
                    'inplace_view', 'maybe_aliasing_or_mutating'}
        if any(tag.name in excluded for tag in op.tags):
            continue
        names = [arg.name for arg in required]
        guards = model.get('validation_guards', [])
        examples = plan_guard_probes(guards, schema['schema'], names)
        obs = []
        for shapes, dtype_name in examples:
            dtype = getattr(torch, dtype_name)
            args = [make_tensor(shapes[arg], kind='positive', dtype=dtype, device=device, sample=i)
                    for i, arg in enumerate(names)]
            entry = dict(shape=list(shapes[names[0]]), dtype=str(dtype), device=device, schema=schema['schema'],
                         operand_shapes={name:list(shape) for name, shape in shapes.items()},
                         operand_names=names,
                         guard_observations=[dict(source_sha256=g['source_sha256'], line=g['line'],
                             **evaluate_shape_guard(g, schema['schema'], shapes)) for g in guards])
            try:
                with torch.no_grad():
                    op(*args)
                if device == 'cuda':
                    torch.cuda.synchronize()
                entry['returned'] = True
            except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
                entry.update(returned=False, exception=type(exc).__name__, diagnostic=str(exc)[:240])
            obs.append(entry)
        model.update(probe_status="probed", observations=obs, refinements=refine_domains(obs) + refine_source_guards(guards, obs))
        return
    model["probe_status"] = "no_supported_bounded_probe"


def refine_domains(observations):
    """Complementary samples test hypotheses, never establish completeness over all inputs."""
    rules = []
    for name, inside in (
            ("only_square_matrices", lambda s, d: len(s) >= 2 and s[-2] == s[-1]),
            ("only_float32", lambda s, d: d == "torch.float32"),
            ("rank_at_least_two", lambda s, d: len(s) >= 2)):
        counts = Counter()
        counterexamples = []
        for o in observations:
            side = "inside" if inside(o["shape"], o["dtype"]) else "outside"
            counts[side + ("_accepted" if o["returned"] else "_rejected")] += 1
            if side == "outside" and o["returned"]:
                counterexamples.append(dict(shape=o["shape"], dtype=o["dtype"]))
        if counts["inside_accepted"]:
            rules.append(dict(proposal=name, counts=dict(counts), counterexamples=counterexamples,
                              status="overrestricted" if counterexamples else "unresolved"))
    return rules


def refine_source_guards(guards, observations):
    rules = []
    for guard in guards:
        samples = [o for o in observations if o.get('schema') == guard.get('overload') and o.get('schema')]
        counts, latest = Counter(), None
        for sample in samples:
            result = evaluate_shape_guard(guard, sample['schema'], sample['operand_shapes'])
            side = {True:'inside', False:'outside', None:'unknown'}[result['value']]
            counts[side + ('_accepted' if sample['returned'] else '_rejected')] += 1
            if side == 'outside' and sample['returned']:
                latest = {k: sample.get(k) for k in ('schema','device','operand_shapes','dtype')}
        rules.append(dict(proposal=guard['expression'][:400], overload=guard.get('overload'),
            source_sha256=guard['source_sha256'], line=guard['line'], counts=dict(counts),
            status=('overrestricted' if latest else 'observed_both_sides' if counts['inside_accepted'] and counts['outside_rejected']
                    else 'unresolved'), latest_counterexample=latest))
    return rules

def trace_input_observations(path):
    """Capture only errors raised by the target and successful returns at that boundary."""
    import json
    if not path.is_file():
        return []
    pending, observed = {}, []
    with path.open() as lines:
        for line in lines:
            if not line.endswith("\n"):
                continue
            event = json.loads(line)
            call = event.get("call")
            if event["event"] == "api_enter":
                pending[call] = [{k: d.get(k) for k in ("path", "shape", "dtype", "device")}
                                 for d in event["metadata"][:12]]
            elif event["event"] in ("api_exit", "target_exception") and call in pending:
                item = dict(operands=pending.pop(call), returned=event["event"] == "api_exit",
                            schema=event.get('native_schema'), call=call)
                if not item["returned"]:
                    item.update(exception=event["exception"], diagnostic=event["diagnostic"])
                observed.append(item)
            if len(observed) == 8:
                break
    return observed


def refine_trace_constraints(spec, observations, *, device=None):
    """Relate target exceptions to scoped shape hypotheses; unrelated errors only change one condition."""
    records = [o for o in observations[-128:] if o.get('operands')]
    groups = {}
    for o in records:
        placements = tuple(sorted({d.get('device') or 'unknown' for d in o['operands']}))
        key = (o.get('schema'), placements)
        groups.setdefault(key, []).append(o)
    refinements = []
    for (schema, placements), samples in groups.items():
        model = spec.get('constraints', {})
        binding = next((b for b in model.get('bindings', []) if b['schema'] == schema), None)
        guards = [g for g in model.get('validation_guards', []) if schema and g.get('overload') == schema]
        if binding and guards:
            arguments = binding['operands']
            positional = [a['name'] for a in arguments if not a.get('keyword_only')]
            names = {a['name'] for a in arguments}
            shapes = []
            for example in samples:
                dims = {}
                for d in example['operands']:
                    path = d.get('path') or []
                    if len(path) != 2:
                        continue
                    parent, key = path
                    if parent == 'args' and type(key) is int and 0 <= key < len(positional):
                        dims[positional[key]] = d.get('shape')
                    if parent == 'kwargs' and key in names:
                        dims[key] = d.get('shape')
                shapes.append(dict(example, operand_shapes=dims, device=device))
            measured = refine_source_guards(guards, shapes)
            by_source = {(g['source_sha256'], g['line']): g for g in guards}
            for rule in measured:
                original = by_source[(rule['source_sha256'], rule['line'])]
                rule.update(api=spec['api'], framework_build=spec.get('framework_build'), device=device,
                            operand_devices=list(placements), scope='installed_guard_on_observed_native_overload',
                            diagnostic=original.get('diagnostic'))
                if rule.get('latest_counterexample'):
                    latest = next(s for s in reversed(shapes) if s['returned'] and
                        evaluate_shape_guard(original, schema, s['operand_shapes'])['value'] is False)
                    rule['latest_counterexample'].update({k: latest.get(k) for k in ('evidence','call','attempt')})
            refinements.extend(measured)
        errors = {}
        for example in samples:
            if example['returned']:
                continue
            message = str(example.get('diagnostic', ''))[:240]
            normalized = re.sub(r'(?<![a-zA-Z])[-+]?\d+\b', '#', message).strip()
            errors[(example.get('exception'), normalized)] = example
        for (exception, signature), rejected in list(errors.items())[-6:]:
            diagnostic = str(rejected.get('diagnostic', ''))[:240]
            operand = rejected['operands'][0]
            path = operand.get('path')
            if 'square' in diagnostic.lower():
                proposal, test = 'last_two_extents_equal', lambda s: len(s) >= 2 and s[-2] == s[-1]
            elif re.search(r'\b(rank|dimension|size|shape|dimensional)\b', diagnostic.lower()):
                # A provisional boundary hypothesis. One same-rank success can overturn it.
                rank = len(operand.get('shape') or ())
                proposal, test = f'rank_differs_from_rejected_{rank}', lambda s: len(s) != rank
            else:
                proposal, test = 'unresolved_input_error', None
            counts, latest = Counter(), None
            for example in samples:
                desc = next((d for d in example['operands'] if d.get('path') == path), None)
                if test is None or desc is None or not isinstance(desc.get('shape'), list):
                    counts['unclassified'] += 1
                    continue
                inside = test(desc['shape'])
                side = 'inside' if inside else 'outside'
                other = re.sub(r'(?<![a-zA-Z])[-+]?\d+\b', '#', str(example.get('diagnostic', ''))[:240]).strip()
                same_error = (not example['returned'] and example.get('exception') == exception and other == signature)
                outcome = 'accepted' if example['returned'] else 'same_error' if same_error else 'different_error'
                counts[side + '_' + outcome] += 1
                if not inside and example['returned']:
                    latest = {k: example.get(k) for k in ('attempt','call','evidence','operands')}
            status = ('overrestricted' if latest else 'insufficient_to_avoid_error' if counts['inside_same_error']
                      else 'outside_changed_error_only' if counts['outside_different_error'] else 'unresolved')
            refinements.append(dict(api=spec['api'], framework_build=spec.get('framework_build'),
                device=device, operand_devices=list(placements), overload=schema,
                scope='observed_public_call' if not schema else 'observed_native_overload',
                operand_path=path, proposal=proposal, exception=exception, diagnostic=diagnostic,
                error_signature=signature, status=status, counts=dict(counts), latest_counterexample=latest,
                rejected_example={k: rejected.get(k) for k in ('attempt','call','evidence','operands')}))
    return refinements

def choose_input_recipe(model, proposal):
    """Solve the finite, observed part of a versioned overload's bounded input domain."""
    feasible = [o for o in model.get("observations", []) if o.get("returned")]
    if not feasible:
        return None
    chosen = feasible[proposal % len(feasible)]
    binding = next((b for b in model.get("bindings", []) if b["schema"] == chosen["schema"]), None)
    if not binding:
        return None
    return dict(shape=chosen["shape"], operand_shapes=chosen.get('operand_shapes', {}),
                dtype=chosen["dtype"], schema=chosen["schema"],
                operands=chosen["operand_names"], device=chosen["device"],
                constructor="positive", scope="one previously accepted bounded native shape/dtype assignment")


OPERAND_CONSTRUCTOR = "make_operand"


def recipe_digest(recipe):
    import json
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def operand_builder(recipe, device, expected_digest, evidence):
    import torch
    identity = recipe_digest(recipe)
    if expected_digest is not None and identity != expected_digest:
        raise ValueError("Input recipe changed after selection")

    def build(name, *, requires_grad=False, sample=None):
        value = make_operand(name, recipe=recipe, device=device, requires_grad=requires_grad, sample=sample)
        reference = make_operand(name, recipe=recipe, device='cpu', requires_grad=False, sample=sample)
        shape = recipe.get('operand_shapes', {}).get(name, recipe['shape'])
        dtype = getattr(torch, recipe['dtype'].removeprefix('torch.'))
        if (tuple(value.shape) != tuple(shape) or value.dtype != dtype or not torch.equal(value.detach().cpu(), reference)
                or value.dtype != reference.dtype
                or value.device.type != torch.device(device).type or value.stride() != reference.stride()
                or value.storage_offset() != reference.storage_offset() or value.requires_grad != requires_grad
                or not value.is_leaf):
            raise ValueError('Bound operand differs from its frozen recipe')
        evidence.append(dict(name=name, sample=sample, requires_grad=requires_grad, recipe_sha256=identity,
                             shape=list(value.shape), dtype=str(value.dtype), device=str(value.device)))
        return value
    return build


def make_operand(name, *, recipe, device, requires_grad=False, sample=None):
    """Reconstruct a solver-selected operand while keeping projection and history explicit."""
    import torch
    if not recipe or name not in recipe["operands"]:
        raise ValueError("No bounded input recipe for this named operand")
    if sample is None:
        sample = recipe["operands"].index(name)
    dtype = getattr(torch, recipe["dtype"].removeprefix("torch."))
    shape = recipe.get('operand_shapes', {}).get(name, recipe['shape'])
    return make_tensor(shape, kind=recipe["constructor"], dtype=dtype, device=device,
                       requires_grad=requires_grad, sample=sample)
