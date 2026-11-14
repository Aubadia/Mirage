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
            if node.name in definitions: raise ContractError("Duplicate function")
            definitions[node.name] = node
        else:
            raise ContractError("Use torch/math imports and undecorated top-level functions")
    def resolve_name(node):
        if isinstance(node, ast.Name): return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute): return resolve_name(node.value) + "." + node.attr
        return ""
    for name, arguments in {"build": "device, seed", "run": "ctx, probe"}.items():
        expected = ast.parse(f"def f({arguments}): pass").body[0].args
        if name not in definitions or ast.dump(definitions[name].args) != ast.dump(expected):
            raise ContractError(f"Define {name}({arguments})")
    seed_calls, unresolved = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.Assert)):
            raise ContractError("No mutable globals or generated oracle assertions")
        if isinstance(node, ast.ClassDef):
            if [resolve_name(base) for base in node.bases] != ["torch.nn.Module"]:
                raise ContractError("Local classes must derive from torch.nn.Module (aliases allowed)")
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


def stateful_request(metadata, configuration, *, edges=True, canonical=None, predicates=(), relation=None, mutation=None):
    import copy
    context = copy.deepcopy(metadata)
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
        "Replace all retained random module values. Registered module parameters must be the "
        "same objects as entries in parameters, not the tensors used to construct Parameters. "
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
        "and use its returned result. ONLY the configured target goes through probe.call; "
        "call other operations directly. Tensor methods require their bound receiver or an "
        "explicit receiver with the unbound method. Use a returned context manager/callable "
        "according to its public specification. For required backward, differentiable target "
        "operands need gradient history before the call and the observed result must reach "
        "backward; an unrelated downstream parameter or separate unobserved call is insufficient.\n"
        "4. A compiled_execution=True goal REQUIRES source-owned torch.compile and an actual "
        "invocation of the resulting callable. Call the target directly inside your compiled function; "
        "keep probes outside tracing/capture and invoke the compiled callable directly. "
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
    if mutation is not None:
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
    if edges: prompt += "\n\n" + EDGE_CASE_PROMPT
    return Request(Stage.TRANSITION if canonical else Stage.CANONICAL, prompt,
                   {"contract": "training-program", "program_kind": "stateful-1", "api_context": context, "configuration": configuration,
                    "canonical_source": canonical, "state_predicates": list(predicates),
                    "mutation_contract": copy.deepcopy(mutation),
                    "state_measurement_units": {
                        "target_boundary_count": "Completed eager probe target boundaries or executed target-containing compiler graph boundaries. For a factory, construction is one boundary; using a returned context is measured separately.",
                        "backward_count": "Observed backward/grad invocations in run.",
                        "target_backward_count": "Observed target-connected output gradient events.",
                        "target_backward_node_count": "Attributable backward Node execution events; output gradients alone do not prove target backward.",
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
                        "execution_mode": "Use source-owned compilation/capture when selected. Expose output, parameter gradients and related alias partners. The same authored computation is compared under certified execution configurations."}, "prompt_version": ("stateful-5-local" if mutation is not None else ("stateful-3" if graph_guidance else "stateful-2")) + ("-edges" if edges else "-base"),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
