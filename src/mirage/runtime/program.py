"""Generic generated-program lifecycle with runner-owned execution evidence."""

from contextlib import nullcontext
import weakref

from mirage.generator.program import ContractError, analyze, load
from mirage.runtime.api_metadata import resolve
from mirage.runtime.state import rng_state, restore_rng


def leaves(value, prefix=""):
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}.{index}")


def snapshot(value):
    """Normalize structured observations; this is not a replay-state serializer."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ContractError(f"Unsupported observation type: {type(value).__name__}")


def tensor_metadata(value):
    from mirage.runtime.representations import metadata

    tensors = dict(leaves(value))
    return metadata(tensors)


def build(namespace, config):
    import torch

    ctx = namespace["build"](config["device"], config["seed"])
    if not isinstance(ctx, dict) or not {"inputs", "parameters", "optimizer", "batch_axes"} <= ctx.keys():
        raise ContractError("build must return inputs, parameters, optimizer and batch_axes")
    if any(not isinstance(ctx[k], dict) for k in ("inputs", "parameters", "batch_axes")):
        raise ContractError("inputs, parameters and batch_axes must be dictionaries")
    if any(not isinstance(p, torch.Tensor) or not p.is_leaf or not p.requires_grad
           for p in ctx["parameters"].values()):
        raise ContractError("parameters must contain named trainable leaf tensors")
    if any(t.layout != torch.strided for _, t in leaves((ctx["inputs"], ctx["parameters"]))):
        raise ContractError("This execution adapter currently supports dense strided tensors")
    modules = ctx.setdefault("modules", {})
    if not isinstance(modules, dict) or any(not isinstance(m, torch.nn.Module) for m in modules.values()):
        raise ContractError("modules must be a dictionary of torch.nn.Module instances")
    registered = {id(p) for p in ctx["parameters"].values()}
    if any(id(p) not in registered for module in modules.values() for p in module.parameters()):
        raise ContractError("Register all module parameters in parameters for state capture and replay")
    if sum(t.numel() for _, t in leaves((ctx["inputs"], ctx["parameters"]))) > 2_000_000:
        raise ContractError("Initial tensor budget exceeded")
    for _, t in leaves(ctx["inputs"]):
        if t.grad_fn is not None:
            raise ContractError("Build derived parameter views in forward so replay preserves autograd history")
    optimizer = ctx["optimizer"]
    if optimizer is not None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ContractError("optimizer must be a torch optimizer or None")
        known = {id(p) for p in ctx["parameters"].values()}
        if any(id(p) not in known for group in optimizer.param_groups for p in group["params"]):
            raise ContractError("Register every optimizer parameter in parameters")
    for name, axis in ctx["batch_axes"].items():
        if name not in ctx["inputs"] or axis != 0 or not isinstance(ctx["inputs"][name], torch.Tensor):
            raise ContractError(f"batch_axes key {name!r} is invalid; use only tensor input names {list(ctx['inputs'])}, axis 0. Use an empty dict when no input is a sample batch; parameters are not batch inputs.")
    return ctx


def initialize(source, config, path):
    import torch

    ctx = build(load(source), config)
    optimizer = ctx["optimizer"]
    # Saving the connected tensor tree together preserves storage sharing.
    value = {"inputs": ctx["inputs"], "parameters": ctx["parameters"],
             "optimizer": optimizer.state_dict() if optimizer else None,
             "modules": {name: module.state_dict() for name, module in ctx["modules"].items()},
             "gradients": {name: tensor.grad for name, tensor in ctx["parameters"].items()},
             "input_gradients": {name: tensor.grad for name, tensor in leaves(ctx["inputs"])
                                 if tensor.is_leaf and tensor.requires_grad},
             "extras": {name: item for name, item in ctx.items()
                        if name not in ("inputs", "parameters", "optimizer", "modules", "batch_axes")},
             "rng": rng_state(config["device"]), "batch_axes": ctx["batch_axes"]}
    snapshot(value["extras"])
    torch.save(value, path)
    return {"metadata": tensor_metadata((ctx["inputs"], ctx["parameters"])),
            "parameter_names": list(ctx["parameters"]), "batch_axes": ctx["batch_axes"]}


def restore(namespace, config, path):
    import torch

    ctx = build(namespace, config)
    initial = torch.load(path, map_location=config["device"], weights_only=True)
    optimizer = ctx["optimizer"]
    if set(ctx["parameters"]) != set(initial["parameters"]):
        raise ContractError("Initialization parameter names changed during replay")
    if optimizer:
        names = {id(p): name for name, p in ctx["parameters"].items()}
        for group in optimizer.param_groups:
            group["params"] = [initial["parameters"][names[id(p)]] for p in group["params"]]
        optimizer.load_state_dict(initial["optimizer"])
    names = {id(parameter): name for name, parameter in ctx["parameters"].items()}
    for name, module in ctx["modules"].items():
        for path, parameter in list(module.named_parameters()):
            parent = module
            parts = path.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], initial["parameters"][names[id(parameter)]])
        module.load_state_dict(initial["modules"][name])
    ctx.update(inputs=initial["inputs"], parameters=initial["parameters"], batch_axes=initial["batch_axes"])
    ctx.update(initial.get("extras", {}))
    for name, parameter in ctx["parameters"].items():
        parameter.grad = initial.get("gradients", {}).get(name)
    for name, tensor in leaves(ctx["inputs"]):
        if tensor.is_leaf and tensor.requires_grad:
            tensor.grad = initial.get("input_gradients", {}).get(name)
    restored_rng = initial["rng"]
    restored_rng["cpu"] = restored_rng["cpu"].cpu()
    if restored_rng["cuda"] is not None:
        restored_rng["cuda"] = restored_rng["cuda"].cpu()
    restore_rng(restored_rng, config["device"])
    return ctx


def state(ctx, scaler=None):
    return {"parameters": snapshot(ctx["parameters"]),
            "modules": {name: snapshot(module.state_dict()) for name, module in ctx.get("modules", {}).items()},
            "gradients": {k: snapshot(p.grad) for k, p in ctx["parameters"].items()},
            "input_gradients": {name: snapshot(tensor.grad) for name, tensor in leaves(ctx["inputs"])
                                if tensor.requires_grad and tensor.is_leaf},
            "extras": snapshot({name: item for name, item in ctx.items()
                                if name not in ("inputs", "parameters", "optimizer", "modules", "batch_axes")}),
            "optimizer": snapshot(ctx["optimizer"].state_dict()) if ctx["optimizer"] else None,
            "scaler": snapshot(scaler.state_dict()) if scaler else None}


def transform(ctx, representation, name):
    import torch

    if representation == "contiguous":
        return
    if name not in ctx["inputs"] or not isinstance(ctx["inputs"][name], torch.Tensor):
        raise ContractError("Representation target must name a tensor input")
    tensor = ctx["inputs"][name]
    if representation == "clone":
        result = tensor.clone()
    elif representation == "strided":
        if not tensor.ndim or tensor.numel() < 2:
            raise ContractError("A scalar or singleton cannot demonstrate noncontiguity")
        buffer = torch.empty((*tensor.shape, 2), dtype=tensor.dtype, device=tensor.device)
        result = buffer[..., 0]
        result.copy_(tensor)
    elif representation == "offset":
        buffer = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
        result = buffer[1:].view(tensor.shape)
        result.copy_(tensor)
    elif representation == "shared":
        partners = [(key, value) for key, value in ctx["inputs"].items()
                    if key != name and isinstance(value, torch.Tensor)
                    and value.shape == tensor.shape and value.dtype == tensor.dtype
                    and value.device == tensor.device and torch.equal(value, tensor)]
        if not partners:
            raise ContractError("Shared storage needs a compatible equal-valued input partner")
        result = tensor.view_as(tensor)
        ctx["inputs"][partners[0][0]] = result
    else:
        raise ContractError("Unknown representation")
    ctx["inputs"][name] = result


def run(source, config, initial_path, recorder):
    import torch
    from torch.overrides import TorchFunctionMode
    from mirage.runtime.environment import configure, describe
    from mirage.runtime.mr_actions import SavedTensors, accumulation_supported

    configure()
    namespace = load(source)
    graph = analyze(source)
    ctx = restore(namespace, config, initial_path)
    target = resolve(config["api"])
    optimizer = ctx["optimizer"]
    precision = config.get("precision", "fp32")
    scaler = torch.amp.GradScaler(config["device"], enabled=precision == "fp16")
    initial = {"state": state(ctx, scaler), "inputs": snapshot(ctx["inputs"]),
               "metadata": tensor_metadata((ctx["inputs"], ctx["parameters"])),
               "rng": rng_state(config["device"])}
    target_calls = []
    target_gradients = []
    operation_names = []
    operation_records = []
    mutated_operations = []
    current_phase = "history"
    representation_changed = False
    update_count = 0
    backward_count = 0
    sample_loss_matches = True
    input_gradients = {}
    for path, tensor in leaves(ctx["inputs"]):
        if tensor.requires_grad:
            def input_hook(gradient, name=path):
                if current_phase == "training":
                    value = snapshot(gradient) / scaler.get_scale()
                    input_gradients[name] = input_gradients.get(name, 0) + value
                return gradient
            tensor.register_hook(input_hook)
    saved = SavedTensors(config.get("saved_tensors", False))
    origins = {}

    def operand_origins(tensor):
        item = origins.get(id(tensor))
        return item[1] if item is not None and item[0]() is tensor else set()

    def remember(tensor, names):
        origins[id(tensor)] = (weakref.ref(tensor), set(names))

    class Observe(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = getattr(func, "__name__", str(func))
            operation_names.append(name)
            operation_records.append({"name": name,
                                      "shapes": [list(t.shape) for _, t in leaves(args)],
                                      "dim": kwargs.get("dim", args[1] if name in ("mean", "sum", "softmax", "log_softmax") and len(args) > 1 and isinstance(args[1], int) else None)})
            hit = func is target
            versions = [(tensor, tensor._version) for _, tensor in leaves((args, kwargs))]
            dependencies = set().union(*(operand_origins(tensor) for _, tensor in leaves((args, kwargs))))
            before = snapshot((args, kwargs)) if hit else None
            meta = tensor_metadata((args, kwargs)) if hit else None
            if hit:
                recorder.emit("api_enter", api=config["api"], metadata=meta, phase=current_phase)
            result = func(*args, **kwargs)
            if any(tensor._version != version for tensor, version in versions):
                mutated_operations.append(name)
            for _, tensor in leaves(result):
                remember(tensor, dependencies)
            if hit:
                index = len(target_calls)
                selected = ctx["inputs"].get(config.get("input_name", ""))
                rep = config.get("representation", "contiguous")
                operand_reached = rep == "contiguous"
                if isinstance(selected, torch.Tensor):
                    actual = [tensor for _, tensor in leaves((args, kwargs))
                              if config.get("input_name") in operand_origins(tensor)]
                    if rep == "strided":
                        operand_reached = any(not tensor.is_contiguous() for tensor in actual)
                    elif rep == "offset":
                        operand_reached = any(tensor.storage_offset() > 0 for tensor in actual)
                    elif rep == "clone":
                        operand_reached = bool(actual) and representation_changed
                    elif rep == "shared":
                        storages = [tensor.untyped_storage().data_ptr() for tensor in actual]
                        operand_reached = any(storages.count(storage) >= 2 for storage in storages)
                target_calls.append({"phase": current_phase, "inputs": before,
                                     "metadata": meta, "output": snapshot(result),
                                     "input_names": sorted(dependencies),
                                     "representation_reached": operand_reached})
                recorder.emit("api_exit", api=config["api"], call=index, phase=current_phase)
                for path, tensor in leaves(result):
                    if tensor.requires_grad:
                        def hook(gradient, i=index, p=path):
                            target_gradients.append({"call": i, "path": p})
                            recorder.emit("target_backward", call=i, path=p)
                            return gradient
                        tensor.register_hook(hook)
            return result

    backend = config.get("backend", "eager")
    forward = namespace["forward"]
    compiled_evidence = []
    compiled_operations = []
    if backend != "eager":
        from torch._dynamo.backends.registry import lookup_backend

        def compile_backend(fx, examples):
            nodes = [str(n.target) for n in fx.graph.nodes if n.op.startswith("call_")]
            names = [getattr(n.target, "__name__", str(n.target)) for n in fx.graph.nodes if n.op.startswith("call_")]
            matches = [n for n in fx.graph.nodes if n.op.startswith("call_")
                       and (n.target is target or n.op == "call_method" and n.target == config["api"].split(".")[-1])]
            output_node = next(node for node in fx.graph.nodes if node.op == "output")
            from torch.utils._pytree import tree_flatten

            output_leaves, _ = tree_flatten(output_node.args[0])

            def reaches_leaf(node, destination):
                pending, visited = [node], set()
                while pending:
                    item = pending.pop()
                    if item in visited:
                        continue
                    visited.add(item)
                    example = item.meta.get("example_value", item.meta.get("val"))
                    if not any(tensor.requires_grad for _, tensor in leaves(example)):
                        continue
                    if item is destination:
                        return True
                    pending.extend(user for user in item.users if user.op != "output")
                return False

            differentiated_outputs = [any(reaches_leaf(node, output) for node in matches)
                                      for output in output_leaves]
            fn = lookup_backend("eager" if backend == "compile_eager" else backend)(fx, examples)

            def invoke(*args):
                compiled_evidence.append(nodes)
                compiled_operations.extend(names)
                output = fn(*args)
                if matches:
                    index = len(target_calls)
                    target_calls.append({"phase": current_phase, "evidence": "executed FX target"})
                    actual, _ = tree_flatten(output)
                    # Observe only the corresponding target-dependent FX output.
                    # Another returned tensor may be the sole input to the loss.
                    if len(actual) == len(differentiated_outputs):
                        for position, (tensor, connected) in enumerate(zip(actual, differentiated_outputs)):
                            if connected and isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                                def hook(gradient, i=index, p=position):
                                    target_gradients.append({"call": i, "path": str(p)})
                                    recorder.emit("target_backward", call=i, path=str(p))
                                    return gradient
                                tensor.register_hook(hook)
                return output
            return invoke
        forward = torch.compile(forward, backend=compile_backend, fullgraph=True)

    def evaluate(context, use_samples=False):
        nonlocal sample_loss_matches
        amp = (torch.autocast(config["device"], dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
               if precision != "fp32" else nullcontext())
        checkpointing = config.get("checkpoint", False) and current_phase == "training"
        def observed_forward(value):
            for input_name, item in value["inputs"].items():
                for _, tensor in leaves(item):
                    remember(tensor, operand_origins(tensor) | {input_name})
            with Observe() if backend == "eager" else nullcontext():
                return forward(value)

        with saved.context(), amp:
            if checkpointing:
                from torch.utils.checkpoint import checkpoint
                output = checkpoint(observed_forward, context, use_reentrant=False, preserve_rng_state=True)
                recorder.emit("checkpoint_forward")
            else:
                output = observed_forward(context)
            with Observe():
                loss = namespace["sample_loss" if use_samples else "loss"](output)
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            raise ContractError("Loss must be a differentiable Tensor")
        if not use_samples and loss.ndim != 0:
            raise ContractError("loss must return a scalar")
        if use_samples:
            count = len(operation_records)
            with Observe():
                ordinary = namespace["loss"](output)
            # The scalar objective is checked for agreement, not certified as a
            # per-sample operation (its final batch reduction is intentional).
            del operation_records[count:]
            sample_loss_matches &= bool(ordinary.ndim == 0 and torch.allclose(
                ordinary.detach(), loss.detach().mean(), atol=1e-6, rtol=1e-5))
        return output, loss

    def backward(loss, retain=False):
        nonlocal backward_count
        scaler.scale(loss).backward(retain_graph=retain)
        backward_count += 1
        recorder.emit("backward", retained=retain)

    def update():
        nonlocal update_count
        if optimizer is None:
            raise ContractError("Optimizer update requires an optimizer")
        completed = []
        handle = optimizer.register_step_post_hook(lambda *_: completed.append(True))
        try:
            scaler.step(optimizer)
            scaler.update()
        finally:
            handle.remove()
        update_count += len(completed)
        recorder.emit("optimizer_update", completed=bool(completed))

    def zero(none=True):
        trainables = {id(tensor): tensor for tensor in ctx["parameters"].values()}
        trainables.update({id(tensor): tensor for _, tensor in leaves(ctx["inputs"])
                           if tensor.is_leaf and tensor.requires_grad})
        for parameter in trainables.values():
            if none:
                parameter.grad = None
            else:
                parameter.grad = torch.zeros_like(parameter)
        recorder.emit("zero_grad", set_to_none=none)

    # History is executed, not replaced by a synthetic optimizer-state snapshot.
    recorder.phase = "history"
    history = config.get("history")
    if history is not None:
        from mirage.contracts.exploration import validate_history

        for index, action in enumerate(validate_history(history, config)):
            if action.startswith("zero:"):
                zero(action == "zero:none")
            elif action == "forward":
                _, loss = evaluate(ctx)
            elif action.startswith("backward"):
                backward(loss, retain=action.endswith(":retain"))
            elif action == "step":
                update()
            recorder.emit("plan_action", index=index, name=action)
    else:
        for _ in range(config.get("optimizer_steps", 0)):
            zero()
            _, loss = evaluate(ctx)
            backward(loss)
            update()
        zero(config.get("gradient_state", "none") != "zero")
        if config.get("gradient_state") == "accumulated":
            _, loss = evaluate(ctx)
            backward(loss)
    common = {"state": state(ctx, scaler), "inputs": snapshot(ctx["inputs"]),
              "metadata": tensor_metadata((ctx["inputs"], ctx["parameters"])),
              "rng": rng_state(config["device"])}
    history_updates = update_count
    current_phase = "training"
    recorder.phase = "training"
    selected = ctx["inputs"].get(config.get("input_name", ""))
    previous_storage = selected.untyped_storage().data_ptr() if isinstance(selected, torch.Tensor) else None
    transform(ctx, config.get("representation", "contiguous"), config.get("input_name", ""))
    transformed = ctx["inputs"].get(config.get("input_name", ""))
    representation_changed = (isinstance(transformed, torch.Tensor)
                              and transformed.untyped_storage().data_ptr() != previous_storage)
    transformed_metadata = tensor_metadata((ctx["inputs"], ctx["parameters"]))
    recorder.emit("representation", metadata=transformed_metadata)
    output = None
    loss_value = 0.0
    chunks = config.get("microbatches", 1)
    operation_names.clear()
    operation_records.clear()
    mutated_operations.clear()
    if chunks > 1 or config.get("accumulation_source", False):
        if precision != "fp32" or "sample_loss" not in namespace or not ctx["batch_axes"]:
            raise ContractError("Accumulation requires FP32, sample_loss and batch axes")
        lengths = {ctx["inputs"][name].shape[0] for name in ctx["batch_axes"]}
        if len(lengths) != 1:
            raise ContractError("Batch axes must have the same sample count")
        count = lengths.pop()
        if count < chunks:
            raise ContractError("Each microbatch must be nonempty")
        for indices in torch.tensor_split(torch.arange(count), chunks):
            a, b = int(indices[0]), int(indices[-1]) + 1
            context = dict(ctx, inputs=dict(ctx["inputs"]))
            for name in ctx["batch_axes"]:
                context["inputs"][name] = ctx["inputs"][name][a:b]
            output, values = evaluate(context, True)
            if values.shape != (b - a,):
                raise ContractError("sample_loss must return exactly one value per sample")
            loss = values.sum() / count
            loss_value += float(loss.detach())
            backward(loss)
    else:
        output, loss = evaluate(ctx)
        loss_value = float(loss.detach())
        graph_mode = config.get("graph_mode", "fresh")
        backward(loss, retain=graph_mode == "retained")
        if graph_mode == "retained":
            backward(loss)
        elif graph_mode == "recomputed":
            output, loss = evaluate(ctx)
            backward(loss)
    if optimizer is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
    before_update = state(ctx, scaler)
    if optimizer is not None:
        update()
    final = state(ctx, scaler)
    train_calls = [i for i, call in enumerate(target_calls) if call["phase"] == "training"]
    api_seen = bool(train_calls)
    backward_seen = any(item["call"] in train_calls for item in target_gradients)
    if backend != "eager":
        operation_names.extend(compiled_operations)
    from mirage.runtime.mr_actions import value_operations
    pure = not graph["identity_sensitive"] and not mutated_operations and value_operations(operation_names) and not any(
        name.endswith("_") and not name.startswith("__") for name in operation_names)
    stochastic = any(name.rstrip("_") in ("normal", "uniform", "exponential", "poisson", "cauchy", "geometric", "log_normal")
                     or any(word in name.lower() for word in ("rand", "dropout", "bernoulli", "multinomial"))
                     for name in operation_names)
    representation = config.get("representation", "contiguous")
    representation_reached = bool(train_calls) and all(
        target_calls[index].get("representation_reached", representation == "contiguous")
        for index in train_calls)
    gradients_at_target = {**common["state"]["gradients"],
                           **{f"input:{k}": v for k, v in common["state"]["input_gradients"].items()}}
    gradient_state = config.get("gradient_state", "none")
    gradient_reached = (
        all(g is None for g in gradients_at_target.values()) if gradient_state == "none"
        else bool(gradients_at_target) and all(g is not None for g in gradients_at_target.values()))
    if gradient_state == "zero" and gradient_reached:
        gradient_reached = all(bool((g == 0).all()) for g in gradients_at_target.values())
    transition = []
    for event in recorder.events:
        if event["event"] == "plan_action":
            transition.append(event["name"])
        elif event["event"] == "representation":
            transition.append(f"representation:{config.get('input_name', '')}:{representation}")
        elif event["event"] in ("api_exit", "backward", "optimizer_update", "checkpoint_forward"):
            transition.append(f"{event.get('phase', '')}:{event['event']}")
    result = {
        "version": 3, "graph": graph, "environment": describe(config["device"]),
        "initial": initial, "common": common, "transformed_metadata": transformed_metadata,
        "api_executed": api_seen, "backward_executed": backward_seen,
        "target_state_reached": (representation_reached and gradient_reached
                                 and history_updates == config.get("optimizer_steps", 0)),
        "history_updates": history_updates,
        "target_calls": target_calls, "target_gradients": target_gradients,
        "operations": operation_names, "operation_records": operation_records,
        "mutated_operations": mutated_operations,
        "pure": pure, "stochastic": stochastic, "saved_tensors": saved.evidence(),
        "batch_separable": accumulation_supported(source, ctx, operation_records),
        "sample_loss_matches": sample_loss_matches,
        "observations": {**final, "loss": loss_value, "output": snapshot(output),
                         "rng": rng_state(config["device"]),
                         "gradients": {"parameters": before_update["gradients"], "inputs": input_gradients}},
        "before_update": before_update, "updates": update_count,
        "backwards": backward_count, "compiled_graphs": compiled_evidence,
        "config": config, "completed": True,
        "transition": transition,
    }
    return result
