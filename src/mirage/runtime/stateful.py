"""Execute LLM-owned training histories; probes only observe and call APIs."""
from dataclasses import asdict
import json
import math

from mirage.generator.program import analyze_stateful, ContractError
from mirage.contracts.programs import validate_checkpoint_data, bind_call
from mirage.contracts.capabilities import (EvidenceStatus, MutationObligations,
    GRADIENT_EVIDENCE_VERSION, GRADIENT_EVIDENCE_MAX_EVENTS, GRADIENT_EVIDENCE_MAX_ELEMENTS)
from mirage.backends.pytorch import PyTorchBackend, context_state
from mirage.runtime.operands import discover, candidates, apply, canonicalize
from mirage.runtime.api_metadata import resolve
from mirage.runtime.program import leaves


def snapshot(value):
    """Portable logical observations; representation facts are recorded separately."""
    import torch
    if isinstance(value, torch.Tensor):
        logical = value.to_dense() if value.layout != torch.strided or value.is_mkldnn else value
        return logical.detach().cpu().clone()
    if isinstance(value, (torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler)):
        return {'stateful_type':type(value).__qualname__, 'state_dict':snapshot(value.state_dict())}
    if isinstance(value, torch.Generator):
        return {'generator_device': str(value.device), 'generator_state': value.get_state().clone()}
    if isinstance(value, (torch.dtype, torch.device)):
        return {'framework_literal':str(value)}
    if isinstance(value, dict): return {str(k):snapshot(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [snapshot(v) for v in value]
    if value is None or type(value) in (bool, int, float, str): return value
    if callable(value): return {'callable':getattr(value, '__qualname__', type(value).__qualname__)}
    raise ContractError(f'Unsupported observation type: {type(value).__name__}')


def operand_generators(value, path=()):
    import torch
    if isinstance(value, torch.Generator):
        yield path, value
    elif isinstance(value, (dict, list, tuple)):
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in items:
            yield from operand_generators(child, (*path, key))


def output_structure(value):
    """Keep Python return containers that a portable tensor snapshot normalizes."""
    import torch
    if isinstance(value, dict):
        return ['mapping', {str(k):output_structure(v) for k,v in value.items()}]
    if isinstance(value, (list,tuple)):
        return [type(value).__module__ + '.' + type(value).__qualname__, [output_structure(v) for v in value]]
    return 'tensor' if isinstance(value, torch.Tensor) else type(value).__qualname__


def training_snapshot(ctx):
    from mirage.backends.pytorch import context_state
    known={'inputs','parameters','modules','optimizer','scheduler','scaler','checkpoint_data'}
    return snapshot({'inputs':ctx['inputs'], 'parameters':ctx['parameters'],
                     'input_gradients':{str(path):t.grad for path,t in leaves(ctx['inputs']) if t.is_leaf and t.requires_grad},
                     'checkpoint_data':ctx.get('checkpoint_data'),
                     'extra_state':{key:value for key,value in ctx.items() if key not in known},
                     'gradients':{k:p.grad for k,p in ctx['parameters'].items()},
                     'modules':{k:m.state_dict() for k,m in ctx.get('modules',{}).items()},
                     'optimizer':ctx['optimizer'].state_dict() if ctx.get('optimizer') is not None else None,
                     'scheduler':ctx['scheduler'].state_dict() if ctx.get('scheduler') is not None else None,
                     'scaler':ctx['scaler'].state_dict() if ctx.get('scaler') is not None else None,
                     'context':context_state()})


def tensor_version(tensor):
    try: return tensor._version
    except RuntimeError: return None


def _broadcastable(left, right):
    from itertools import zip_longest
    return all(a == b or a == 1 or b == 1 for a, b in zip_longest(reversed(left), reversed(right), fillvalue=1))


def evidence_record(value):
    # Artifacts contain primitive records, not custom enum objects.
    return json.loads(json.dumps(asdict(value)))


def observational(method):
    """Do not credit observer-created tensor operations to the generated history."""
    from functools import wraps
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        previous = self._observing
        self._observing = True
        try:
            return method(self, *args, **kwargs)
        finally:
            self._observing = previous
    return wrapped


class Probe:
    def __init__(self, target, ctx, config, recorder):
        self._target, self._ctx, self._config, self._recorder = target, ctx, config, recorder
        self._backend = PyTorchBackend()
        self._vjp_operand_versions = {}
        self.calls, self.checkpoints, self.backward = [], [], []
        self.backward_nodes, self.operand_gradients, self.backward_invocations = [], [], []
        self.callable_backward_nodes = []
        self._callable_registrations = []
        self.gradient_evidence = {'version': GRADIENT_EVIDENCE_VERSION, 'events': [],
                                  'limitations': [], 'dropped_events': 0, 'dropped_limitations': 0}
        self._backward_invocation = None
        self._hook_handles = []
        self.products = []
        self.product_events = []
        self.accumulations = []
        self.saved_tensor_events = []
        self.state_operations = []
        self._goal_output = None
        self._goal_leaves = []
        self.recompute_frames = []
        self._recompute_id = 0
        self._executing_target = None
        self._leaf_calls = {}
        self._previous_grads = {}
        self._continuing_grads = {}
        self.mutations, self.candidates = [], []
        self.operations = []
        self.context_events = []
        self.observed_products = set()
        self.factory_requires_invocation = False
        self.context_observer = None
        self.compiled_graphs = []
        self.capture_replays = []
        self._observing = False
        self._optimizer_evidence = None

    def _register_callable(self, value, *, factory_target, factory_id, callable_id):
        """Keep live identity locally; artifacts use deterministic factory/callable IDs."""
        self._callable_registrations.append((value, {
            'factory_target': factory_target, 'factory_id': factory_id, 'callable_id': callable_id}))

    def _validate_callable_model(self, model):
        import torch
        if not callable(model):
            raise ContractError('Callable factory requires a callable function or module')
        if isinstance(model, torch.nn.Module):
            registered = {id(p) for p in self._ctx['parameters'].values()}
            if any(id(p) not in registered for p in model.parameters()):
                raise ContractError('Callable module parameters must be registered as the same objects in parameters')

    def _validate_capture_callables(self, callables, sample_args):
        """Diagnose explicit capture-contract violations before invoking the factory."""
        import torch
        from torch.utils._pytree import tree_leaves
        models = callables if isinstance(callables, tuple) else (callables,)
        samples = sample_args if isinstance(callables, tuple) else (sample_args,)
        if not models or not isinstance(samples, tuple) or len(models) != len(samples):
            raise ContractError('Capture requires one sample_args tuple for each callable')
        for model, arguments in zip(models, samples):
            self._validate_callable_model(model)
            if not isinstance(arguments, tuple):
                raise ContractError('Capture sample_args for each callable must be a tuple')
            tensors = tree_leaves(arguments)
            if any(not isinstance(t, torch.Tensor) for t in tensors):
                raise ContractError('Capture sample_args must contain only tensor leaves')
            if isinstance(model, torch.nn.Module):
                if any(b.requires_grad for b in model.buffers()):
                    raise ContractError('Capture module buffers must not require gradients')
                if model._backward_hooks or model._forward_hooks or model._forward_pre_hooks:
                    raise ContractError('Capture module must not have hooks installed before capture')
                tensors += list(model.parameters()) + list(model.buffers())
            devices = {t.device for t in tensors}
            if tensors and (any(d.type != 'cuda' for d in devices) or len(devices) != 1):
                raise ContractError('Capture samples, module parameters and buffers must share one CUDA device')
        if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
            raise ContractError('Capture under autocast requires cache_enabled=False')

    @observational
    def mark(self, name, values=None, *, _optimizer_observer=None):
        import torch
        if not isinstance(name, str) or not name:
            raise ContractError("Checkpoint requires a nonempty name")
        if name.startswith("optimizer:") and (_optimizer_observer is None or _optimizer_observer is not self._optimizer_evidence):
            raise ContractError("Optimizer lifecycle checkpoints are reserved for runtime observation")
        data = {"name": name, "values": snapshot(values), "grad_enabled": torch.is_grad_enabled(),
                "inference_enabled": torch.is_inference_mode_enabled(),
                "gradients": {key: snapshot(value.grad) for key, value in self._ctx['parameters'].items()},
                "target_backward_events": len(self.backward)}
        gradient_values = [value.grad for value in self._ctx['parameters'].values()]
        data['state_facts'] = {
            'context.grad_enabled': data['grad_enabled'],
            'context.inference_enabled': data['inference_enabled'],
            'context.restoration_count': sum(e['balanced'] for e in self.context_events),
            'gradient.backward_count': sum(e['name'] in ('backward', 'grad') for e in self.operations),
            'gradient.none': bool(gradient_values) and all(g is None for g in gradient_values),
            'gradient.zero': bool(gradient_values) and all(g is not None and bool((g == 0).all()) for g in gradient_values),
            'graph_lifecycle.target_backward_count': len(self.backward),
            'graph_lifecycle.target_backward_node_count': len(self.backward_nodes),
            'graph_lifecycle.callable_backward_node_count': len(self.callable_backward_nodes),
            'graph_lifecycle.target_call_count': len(self.calls),
            'graph_lifecycle.target_boundary_count': len(self.calls),
        }
        if self._optimizer_evidence is not None:
            data['state_facts']['optimizer.target_update_count'] = self._optimizer_evidence.target_update_count()
        if any(g.get('target_execution_certified') for g in self.compiled_graphs):
            data['state_facts']['execution_mode.compiled_execution'] = True
        if self.capture_replays:
            data['state_facts']['execution_mode.captured_execution'] = True
        data['completed_backward_invocations'] = [e['index'] for e in self.backward_invocations if e['completed']]
        data['evidence_offsets'] = {'calls': len(self.calls), 'outputs': len(self.backward),
                                    'nodes': len(self.backward_nodes), 'operands': len(self.operand_gradients),
                                    'callable_nodes': len(self.callable_backward_nodes),
                                    'invocations': len(self.backward_invocations)}
        data['observable_metadata'] = [asdict(d) for d, _ in discover(values, self._backend)]
        self.checkpoints.append(data)
        self._recorder.emit("checkpoint", name=name, grad_enabled=data['grad_enabled'])

    def _gradient_limitation(self, call, reason):
        limitation = {'call': call, 'reason': reason}
        limits = self.gradient_evidence['limitations']
        if limitation in limits:
            return
        if len(limits) < GRADIENT_EVIDENCE_MAX_EVENTS:
            limits.append(limitation)
        else:
            self.gradient_evidence['dropped_limitations'] += 1

    @observational
    def _capture_gradients(self, event, grad_inputs, grad_outputs, edges, outputs, reason):
        """Snapshot one Node invocation without retaining its graph or tensor storage."""
        import torch
        events = self.gradient_evidence['events']
        if len(events) >= GRADIENT_EVIDENCE_MAX_EVENTS:
            self.gradient_evidence['dropped_events'] += 1
            self._gradient_limitation(event['call'], 'gradient event budget exceeded')
            return
        item = dict(event, inputs=[], outputs=[], status='unknown', reason=reason)
        if self._config.get('oracle_improvements'):
            versions = self._vjp_operand_versions.get(event['call'], ())
            item['operands_unchanged_at_backward'] = bool(versions) and all(
                version is not None and tensor_version(tensor) == version for tensor, version in versions)
        if reason:
            events.append(item)
            return
        input_slots = {slot for slot, _ in edges}
        output_slots = {slot for slot, _ in outputs}
        if any(g is not None and slot not in input_slots for slot, g in enumerate(grad_inputs)):
            item['reason'] = 'unbound differentiable input slot'
        elif any(g is not None and slot not in output_slots for slot, g in enumerate(grad_outputs)):
            item['reason'] = 'unbound differentiable output slot'
        elif any(slot >= len(grad_inputs) for slot, _ in edges) or any(slot >= len(grad_outputs) for slot, _ in outputs):
            item['reason'] = 'node gradient arity differs from bindings'
        else:
            values = [grad_inputs[slot] for slot, _ in edges] + [grad_outputs[slot] for slot, _ in outputs]
            if any(g is not None and (not isinstance(g, torch.Tensor) or g.layout != torch.strided) for g in values):
                item['reason'] = 'unsupported gradient value or layout'
            elif sum(g.numel() for g in values if g is not None) > GRADIENT_EVIDENCE_MAX_ELEMENTS:
                item['reason'] = 'gradient element budget exceeded'
            else:
                try:
                    item['inputs'] = [{'path': list(path), 'slot': slot, 'gradient': snapshot(grad_inputs[slot])}
                                      for slot, path in edges]
                    item['outputs'] = [{'path': path, 'slot': slot, 'gradient': snapshot(grad_outputs[slot])}
                                       for slot, path in outputs]
                    item.update(status='complete', reason='observed direct Node input/output gradients')
                except (RuntimeError, ValueError, TypeError) as error:
                    item.update(inputs=[], outputs=[], reason=f'gradient snapshot failed: {type(error).__name__}')
        events.append(item)

    @observational
    def observe_outputs(self, index, outputs, input_edges, *, node_admissible=True, boundary=False):
        """Keep numerical Node evidence separate from semantic execution events."""
        bindings, nodes = [], {}
        for path, tensor in outputs:
            if not tensor.requires_grad:
                continue
            def observe(gradient, i=index, p=path):
                event = {"call": i, "path": p, "invocation": self._backward_invocation}
                self.backward.append(event)
                self._recorder.emit("target_output_gradient", **event)
                return gradient
            self._hook_handles.append(tensor.register_hook(observe))
            node, links, reason = self._backend.backward_binding(tensor, input_edges, boundary=boundary)
            if not node_admissible:
                node, links, reason = None, [], "target backward attribution unsupported for this graph"
            bindings.append({'output_path': path, 'status': 'verified' if node is not None else 'unknown',
                             'scope': 'callable' if boundary else 'target',
                             'reason': reason, 'operand_paths': [list(p) for _, p in links]})
            if node is None:
                self._gradient_limitation(index, reason)
                continue
            data = nodes.setdefault(node, {'edges': tuple(links), 'outputs': [], 'reason': None})
            data['outputs'].append((tensor.output_nr, path))
            if data['edges'] != tuple(links):
                data['reason'] = 'inconsistent operand bindings for shared output node'
        # All output slots must be known before registering a shared Node hook.
        for node_index, (node, data) in enumerate(nodes.items()):
            edges, output_bindings = data['edges'], tuple(data['outputs'])
            reason = data['reason']
            numerical_edges, numerical_binding = edges, None
            if (self._config.get('oracle_improvements') and not boundary and reason is None
                    and {path for _, path in edges} != {path for path, _ in input_edges}):
                aggregate = self._backend.local_vjp_coordinate_edges(
                    node, input_edges, self.calls[index].get('metadata', []))
                if aggregate is not None:
                    numerical_edges, numerical_binding = aggregate, 'exact_object_sum'
            if boundary:
                reason = 'factory boundary has no direct operand VJP certificate'
            elif {path for _, path in numerical_edges} != {path for path, _ in input_edges} or not numerical_edges:
                reason = 'missing or ambiguous differentiable operand bindings'
            elif len({slot for slot, _ in output_bindings}) != len(output_bindings):
                reason = 'multiple output paths share one differentiation coordinate'
            def node_executed(grad_inputs, grad_outputs, i=index, n=node_index,
                              edges=edges, numerical_edges=numerical_edges, numerical_binding=numerical_binding,
                              outputs=output_bindings, reason=reason, boundary=boundary):
                event = {'call': i, 'node': n, 'invocation': self._backward_invocation}
                if boundary:
                    self.callable_backward_nodes.append(event)
                    self._recorder.emit('callable_backward', **event)
                else:
                    self.backward_nodes.append(event)
                    self._recorder.emit('target_backward', **event)
                for slot, operand_path in edges:
                    if slot < len(grad_inputs) and grad_inputs[slot] is not None:
                        contribution = dict(event, path=list(operand_path), slot=slot)
                        self.operand_gradients.append(contribution)
                        self._recorder.emit('target_operand_gradient', **contribution)
                capture_event = dict(event, coordinate_binding=numerical_binding) if numerical_binding else event
                self._capture_gradients(capture_event, grad_inputs, grad_outputs, numerical_edges, outputs, reason)
            self._hook_handles.append(node.register_hook(node_executed))
        self.calls[index]['backward_bindings'] = bindings

    def close(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._callable_registrations.clear()
        self._vjp_operand_versions.clear()

    def advance(self, product, *args, **kwargs):
        import torch
        owner = next((index for value, index in self.products if value is product), None)
        if owner is None or not isinstance(product, (torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler)):
            raise ContractError("State advance requires a supported target-produced optimizer or scheduler")
        def state():
            values = [p for group in product.param_groups for p in group['params']] if isinstance(product, torch.optim.Optimizer) else []
            return snapshot((product.state_dict(), values))
        before = state()
        scheduler = isinstance(product, torch.optim.lr_scheduler.LRScheduler)
        rates = snapshot([g['lr'] for g in product.optimizer.param_groups]) if scheduler else None
        result = product.step(*args, **kwargs)
        effect = self._backend.compare(before, state(), 'exact')
        if effect['equal'] is False:
            self.product_events.append(dict(call=owner, event='persistent_step_effect',
                product_kind='scheduler' if scheduler else 'optimizer',
                persistent_state=bool(product.state) if not scheduler else False,
                learning_rate_changed=(self._backend.compare(
                    rates, snapshot([g['lr'] for g in product.optimizer.param_groups]), 'exact')['equal'] is False) if scheduler else False))
        return result

    def recompute(self, fn, *args):
        """Link initial invocation and an actual non-reentrant checkpoint recomputation."""
        from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
        group = self._recompute_id
        self._recompute_id += 1
        pass_number = 0
        def invoke(*values):
            nonlocal pass_number
            self.recompute_frames.append((group, pass_number))
            pass_number += 1
            try:
                return fn(*values)
            finally:
                self.recompute_frames.pop()
        with set_checkpoint_early_stop(False):
            return checkpoint(invoke, *args, use_reentrant=False, preserve_rng_state=True)

    def saved_on_cpu(self):
        """Pack to independent CPU storage; only target-issued saves are credited."""
        import torch
        def pack(tensor):
            owner = self._executing_target
            payload = tensor.detach().to('cpu', copy=True)
            return owner, tensor.device, payload
        def unpack(saved):
            owner, device, payload = saved
            restored = payload.to(device, copy=True)
            if owner is not None and self._backward_invocation is not None and len(self.saved_tensor_events) < 32:
                self.saved_tensor_events.append(dict(call=owner, invocation=self._backward_invocation,
                    source_device=str(device), stored_device=str(payload.device), restored_device=str(restored.device)))
            return restored
        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

    def _target_resources(self, operands):
        """Identity links to registered batch inputs and module buffers at the actual boundary."""
        import torch
        inputs = {id(t): path for path, t in leaves(self._ctx['inputs'])}
        buffers = {id(t): (key, name, module.training) for key, module in self._ctx.get('modules', {}).items()
                   for name, t in module.named_buffers()}
        resources = []
        for descriptor, tensor in operands:
            item = dict(path=list(descriptor.path))
            key = id(tensor)
            if key in buffers:
                module, name, training = buffers[key]
                item['module_buffer'] = dict(module=module, name=name, training=training)
            if key in inputs and tensor.layout == torch.strided and not tensor.is_quantized and not tensor.is_nested:
                item['input_key'] = inputs[key]
            if len(item) > 1:
                resources.append(item)
        return resources

    def _goal_update(self):
        import torch
        result = self._goal_output
        if (not isinstance(result, torch.Tensor) or not result.requires_grad or not result.is_floating_point()
                or not 0 < result.numel() <= 4096 or not self._goal_leaves):
            raise ContractError('State update requires a bounded real differentiable target result and a registered leaf operand')
        optimizer = self._ctx.get('optimizer')
        if type(optimizer) not in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
            raise ContractError('State update observer supports native SGD, Adam and AdamW')
        registered = [p for group in optimizer.param_groups for p in group['params']]
        leaves = [entry for entry in self._goal_leaves if any(entry[3] is p for p in registered)]
        if not leaves:
            raise ContractError('State update requires optimizer identity binding to the target leaf')
        return optimizer, result.sum(), leaves

    def scaled_overflow_step(self):
        """Exercise an enabled scaler; prove skipped step via an actual optimizer hook."""
        import torch
        optimizer, loss, bound = self._goal_update()
        scaler = self._ctx.get('scaler')
        if not isinstance(scaler, torch.amp.GradScaler) or not scaler.is_enabled():
            raise ContractError('Scaled step needs an enabled torch.amp.GradScaler on the target device')
        optimizer.zero_grad(set_to_none=True)
        initial_scale = scaler.get_scale()
        invocation = len(self.backward_invocations)
        scaler.scale(loss * float('inf')).backward()
        nonfinite = [(idx, path, key) for idx, path, key, p in bound if p.grad is not None and not bool(p.grad.isfinite().all())]
        actual_steps = []
        hook = optimizer.register_step_post_hook(lambda *args: actual_steps.append(True))
        try:
            scaler.step(optimizer)
            scaler.update()
        finally:
            hook.remove()
        if not actual_steps and scaler.get_scale() < initial_scale:
            self.state_operations.extend(dict(event='skipped_scaled_step', call=idx, path=path, parameter=key,
                                               invocation=invocation) for idx, path, key in nonfinite)

    def restore_and_step(self):
        """Restore a nonempty optimizer checkpoint and observe its state tensors in the next update."""
        import torch
        from copy import deepcopy
        from torch.utils._python_dispatch import TorchDispatchMode
        optimizer, loss, bound = self._goal_update()
        watched = [entry for entry in bound if optimizer.state.get(entry[3])]
        if not watched:
            raise ContractError('Prepare a real warmup update with persistent optimizer state before restoring')
        checkpoint = deepcopy(optimizer.state_dict())
        for _, _, _, p in watched:
            optimizer.state.pop(p, None)
        optimizer.load_state_dict(checkpoint)
        if self._backend.compare(snapshot(optimizer.state_dict()), snapshot(checkpoint), 'exact')['equal'] is not True:
            raise ContractError('Saved optimizer state was not reconstructed')
        state_ids = {key: {id(t) for _, t in leaves(optimizer.state[p])} for _, _, key, p in watched}
        touched = set()
        class ObserveState(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                values = {id(t) for _, t in leaves((args, kwargs or {}))}
                # These operations consume prior tensor values in the supported native optimizers.
                if func._schema.name in ('aten::mul_', 'aten::add_', 'aten::lerp_', 'aten::addcmul_', 'aten::sqrt'):
                    touched.update(key for key, ids in state_ids.items() if ids & values)
                return func(*args, **(kwargs or {}))
        optimizer.zero_grad(set_to_none=True)
        invocation = len(self.backward_invocations)
        loss.backward()
        with ObserveState():
            optimizer.step()
        self.state_operations.extend(dict(event='used_restored_state', call=idx, path=path, parameter=key,
                                           invocation=invocation) for idx, path, key, _ in watched if key in touched)

    def synchronize_gradient(self):
        """Complete a real collective on target-produced gradients in an installed process group."""
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() < 2:
            raise ContractError('An initialized multi-rank process group is required')
        _, loss, bound = self._goal_update()
        invocation = len(self.backward_invocations)
        loss.backward()
        for idx, path, key, p in bound:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad)
            self.state_operations.append(dict(event='synchronized_gradient', call=idx, path=path, parameter=key,
                invocation=invocation, world_size=dist.get_world_size(), rank=dist.get_rank()))

    def wait(self, product):
        import torch
        index = next((i for value, i in self.products if value is product), None)
        types = (torch.futures.Future, torch.distributed.Work) if torch.distributed.is_available() else (torch.futures.Future,)
        if index is None or not isinstance(product, types):
            raise ContractError("Async completion requires a supported target-produced handle")
        result = product.wait()
        if isinstance(product, torch.futures.Future) and not product.done():
            raise ContractError("Future did not complete")
        if isinstance(result, bool) and not result:
            raise ContractError("Work wait returned false")
        self.product_events.append(dict(call=index, event='wait_completed'))
        return result

    @observational
    def gradient_boundary(self, invocation, before):
        if self._config.get('check_claim') != 'accumulation':
            return
        for parameter, calls in self._leaf_calls.values():
            grad = parameter.grad
            previous = self._previous_grads.get(id(parameter))
            if before:
                if (grad is not None and previous and previous[0] is grad and previous[1] is not None
                        and previous[1] == tensor_version(grad)):
                    self._continuing_grads[id(parameter)] = previous
            else:
                carried = self._continuing_grads.pop(id(parameter), None)
                if carried and grad is carried[0] and tensor_version(grad) != carried[1]:
                    self.accumulations.append(dict(first=carried[2], second=invocation, leaf_calls=list(calls)))
                if grad is not None:
                    self._previous_grads[id(parameter)] = (grad, tensor_version(grad), invocation)

    @observational
    def call(self, target, *args, **kwargs):
        import torch
        # Calling a probe while tracing is not a runtime target witness.
        if torch.compiler.is_compiling():
            raise ContractError("Probe inside compiled tracing is not a certified execution boundary")
        registered = next((entry for value, entry in self._callable_registrations if value is target), None)
        if registered is not None and registered['factory_target'] == self._config.get('api'):
            # The factory wrapper owns invocation evidence; do not add a second target boundary.
            self._observing = False
            try:
                return target(*args, **kwargs)
            finally:
                self._observing = True
        if self._config.get('check_claim') == 'callable_use':
            owner = next((i for value, i in self.products if target is value), None)
            if owner is not None:
                self._observing = False
                try:
                    result = target(*args, **kwargs)
                finally:
                    self._observing = True
                self.product_events.append(dict(call=owner, event='callable_return'))
                return result
        receiver = getattr(target, "__self__", None)
        actual = getattr(type(receiver), getattr(target, "__name__", ""), None) if receiver is not None else target
        if target is not self._target and actual is not self._target and getattr(target, '__wrapped__', None) is not self._target:
            raise ContractError("probe.call must invoke the configured target")
        if receiver is not None and actual is self._target:
            target, args = self._target, (receiver,) + args
        binding = bind_call(target, args, kwargs)
        if binding.status == EvidenceStatus.UNKNOWN:
            binding = self._backend.bind_native(target,args,kwargs)
        if binding.status == EvidenceStatus.REJECTED:
            raise ContractError(binding.reason)
        tree = {"args": args, "kwargs": kwargs}
        obligations = self._backend.operand_obligations(target, tree)
        if 'writable' in self._config:
            obligations = MutationObligations(writable=self._config['writable'])
        plans = candidates(tree, self._backend, obligations)
        self.candidates.extend(dict(evidence_record(p), call_index=len(self.calls)) for p in plans)
        if self._config.get('canonicalize_inputs'):
            tree, facts = canonicalize(tree, self._backend, obligations)
            self.mutations.extend(evidence_record(fact) for fact in facts)
        mutation = self._config.get("mutation")
        if mutation and len(self.calls) == mutation.get("call_index", 0):
            selected = next((p for p in plans if p.path == tuple(mutation['path'])
                             and p.action == mutation['action']), None)
            if selected is None: raise ContractError("Requested mutation does not address an actual operand")
            tree, evidence = apply(tree, selected, self._backend, obligations)
            fact = evidence_record(evidence)
            if (self._config.get('representation_candidate') is not None
                    or self._config.get('representation_traversal')):
                fact.update(action=selected.action, path=list(selected.path), call_index=len(self.calls))
            self.mutations.append(fact)
        operands = discover(tree, self._backend)
        if sum(t.numel() for _, t in operands) > 2_000_000:
            raise ContractError("Operand element budget exceeded")
        index = len(self.calls)
        if self._optimizer_evidence is not None:
            self._optimizer_evidence.bind_target(index, operands)
        before = snapshot(tree)
        versions = [(t, tensor_version(t)) for _,t in operands]
        writable_paths = [list(d.path) for d,_ in operands if isinstance(obligations, dict) and obligations[d.path].writable]
        descriptors = [asdict(d) for d, _ in operands]
        properties = [{'path': list(d.path), 'facts': self._backend.input_properties(t)} for d, t in operands]
        shapes = [d.shape for d, _ in operands]
        for item, shape in zip(properties, shapes):
            item['facts']['broadcast'] = any(shape != other and _broadcastable(shape, other) for other in shapes)
        # Factory construction returns a callable, not an operand-differentiable result.
        # Creating sample-input AccumulateGrad nodes here can disturb CUDA capture's
        # stream ownership; observe edges only when the returned callable executes.
        factory = self.factory_requires_invocation
        input_edges = [] if factory else self._backend.gradient_edges(operands)
        entry_context = dict(context_state(), inference_enabled=torch.is_inference_mode_enabled())
        # User-supplied MR obligations cannot certify a numerical replay as read-only.
        native_roles = self._backend.operand_obligations(target, tree)
        schema_read_only = (all(native_roles[d.path].writable is False for d, _ in operands)
                            if isinstance(native_roles, dict) and operands else False)
        if self._config.get('check_claim') == 'accumulation':
            for descriptor, tensor in operands:
                if tensor.is_leaf and tensor.requires_grad:
                    self._leaf_calls.setdefault(id(tensor), (tensor, []))[1].append(
                        [index, list(descriptor.path)])
        if self._config.get('check_claim') == 'callback_trigger':
            import types
            # Wrap only the function actually passed at the target boundary.
            def observe_callback(fn):
                if not isinstance(fn, types.FunctionType):
                    return fn
                def observed(*values, **named):
                    result = fn(*values, **named)
                    if any(owner == index for _, owner in self.products):
                        self.product_events.append(dict(call=index, event='callback_return'))
                    return result
                return observed
            tree = {'args': tuple(observe_callback(v) for v in tree['args']),
                    'kwargs': {k: observe_callback(v) for k, v in tree['kwargs'].items()}}
        guard = None
        if mutation and mutation.get('call_index', 0) == index and mutation['path'] in writable_paths:
            value = next(t for d, t in operands if list(d.path) == mutation['path'])
            base = value._base
            # The representation adapter explicitly zero-initializes this backing tensor.
            if base is not None and mutation['action'] == 'offset':
                guard = base.reshape(-1)[:1]
            elif base is not None and mutation['action'] == 'strided':
                guard = base[..., 1]
        guard_before = snapshot(guard)
        dispatches = []
        from contextlib import nullcontext
        native_observer = nullcontext()
        if (self._config.get('check_claim') in ('forward', 'operand_relation') or
                self._config.get('device_observation') and self._config.get('check_claim') in ('backward', 'accumulation')):
            from torch.utils._python_dispatch import TorchDispatchMode
            class ObserveDispatch(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    if len(dispatches) < 2:
                        dispatches.append(dict(schema=str(func._schema), tags=[tag.name for tag in func.tags]))
                    return func(*args, **(kwargs or {}))
            native_observer = ObserveDispatch()
        self.operations.append({'name': 'target_enter'})
        self._recorder.emit("api_enter", call=index, metadata=descriptors)
        self._observing = False
        guided = isinstance(self._config.get('state_goal'), dict)
        resources = self._target_resources(operands) if guided else []
        family = self._config.get('state_goal', {}).get('family')
        rng_before = None
        if guided and (index == 0 or family == 'rng'):
            from mirage.runtime.state import rng_state
            rng_before = rng_state(self._config['device'])
            explicit_rng = [(path, gen, gen.get_state()) for path, gen in operand_generators(tree)]
        self._executing_target = index
        try:
            with native_observer:
                output = target(*tree['args'], **tree['kwargs'])
        except (ValueError, TypeError, RuntimeError, IndexError) as error:
            if (self._config.get('device_observation') and getattr(self._recorder, 'path', None) is not None
                    and not isinstance(error, (torch.OutOfMemoryError, NotImplementedError))):
                partial = dict(index=index, inputs=before, metadata=descriptors, entry_context=entry_context,
                               execution_mode='eager', writable_paths=writable_paths,
                               target_exception=type(error).__name__)
                torch.save(dict(self._device_initial, config=self._config, completed=False,
                                api_executed=False, target_calls=[*self.calls, partial]),
                           self._recorder.path.parent / 'target-exception.pt')
            if self._config.get('check_claim'):
                self._recorder.emit('target_exception', call=index, exception=type(error).__name__,
                                    native_schema=dispatches[0]['schema'] if len(dispatches) == 1 else None,
                                    diagnostic=str(error)[:320])
            raise
        finally:
            self._executing_target = None
            self._observing = True
        product_kinds = set()
        if guided:
            if hasattr(type(output), '__enter__') and hasattr(type(output), '__exit__'):
                product_kinds.add('context')
                if self.context_observer:
                    self.context_observer.register_product(output)
            if callable(output):
                product_kinds.add('callable')
            if isinstance(output, torch.optim.Optimizer):
                product_kinds.add('optimizer')
            if isinstance(output, torch.optim.lr_scheduler.LRScheduler):
                product_kinds.add('scheduler')
            self.observed_products.update(product_kinds)
        if product_kinds or self._config.get('check_claim') in ('scope_exit', 'callable_use', 'callback_trigger', 'async_completion', 'persistent_step'):
            self.products.append((output, index))
        self.calls.append({"index": index, "inputs": before, "metadata": descriptors,
                           "output": snapshot(output) if isinstance(output, (torch.Tensor, dict, list, tuple, int, float, str, bool, type(None)))
                                     else {'opaque_type':type(output).__qualname__},
                           "binding": evidence_record(binding), "input_properties": properties,
                           "execution_mode": "eager", "entry_context": entry_context,
                           "schema_read_only": schema_read_only,
                           "write_effects": [list(d.path) for (d,t),(_,v) in zip(operands,versions) if v is not None and tensor_version(t) != v],
                           "writable_paths": writable_paths, "native_dispatch": dispatches,
                           "destination_values": [snapshot(t) for d, t in operands if list(d.path) in writable_paths],
                           "guard_preserved": torch.equal(guard_before, snapshot(guard)) if guard is not None else None})
        if self._config.get('device_observation'):
            self.calls[-1]['output_structure'] = output_structure(output)
        if guided:
            self.calls[-1]['registered_resources'] = resources
            if family in ('loss_scaling', 'training_checkpoint', 'distributed'):
                self._goal_output = output
                self._goal_leaves = [(index, list(d.path), key, p) for d, t in operands
                                     for key, p in self._ctx['parameters'].items() if t is p]
            if self.recompute_frames:
                self.calls[-1]['recompute_frame'] = list(self.recompute_frames[-1])
                self.calls[-1]['backward_invocation'] = self._backward_invocation
            if rng_before is not None:
                rng_after = rng_state(self._config['device'])
                advanced = [key for key in ('cpu', 'cuda') if rng_before[key] is not None
                            and not torch.equal(rng_before[key], rng_after[key])]
                advanced.extend(list(path) for path, gen, state in explicit_rng
                                if not torch.equal(state, gen.get_state()))
                self.calls[-1].update(target_rng_consumed=bool(advanced), rng_sources_advanced=advanced)
        if writable_paths:
            written_groups = {d.storage_group for d, _ in operands if list(d.path) in writable_paths
                              and d.storage_group is not None and d.layout == 'torch.strided'}
            self.calls[-1]['alias_partner_values'] = [
                {'path': list(d.path), 'value': snapshot(t)} for d, t in operands
                if list(d.path) not in writable_paths and d.layout == 'torch.strided'
                and d.storage_group in written_groups]
        if self._config.get('check_claim'):
            partners = discover({'operands': tree, 'result': output}, self._backend)
            self.calls[-1]['returned_alias_observations'] = [
                dict(output_path=list(result.path), operand_path=list(operand.path),
                     shared_storage=result.storage_group == operand.storage_group)
                for result, _ in partners if result.path[0] == 'result' and result.layout == 'torch.strided'
                and result.storage_group is not None
                for operand, _ in partners if operand.path[0] == 'operands' and operand.layout == 'torch.strided'
                and operand.storage_group is not None][:64]
        if self._config.get('oracle_improvements'):
            self._vjp_operand_versions[index] = versions
            self.calls[-1]['local_vjp_snapshot'] = {
                'version': 'local-vjp-snapshot-1', 'independent_logical_copies': True,
                'target_read_only': schema_read_only and not self.calls[-1]['write_effects']}
        self._recorder.emit("api_exit", call=index,
                            native_schema=dispatches[0]['schema'] if len(dispatches) == 1 else None)
        self.operations.append({'name': 'target_return'})
        import inspect
        self.observe_outputs(index, leaves(output), input_edges,
                             node_admissible=inspect.isbuiltin(target) or inspect.ismethoddescriptor(target))
        return output


def execute(source, config, recorder, initial_reference=None, *, restore_prefix_rng=False):
    import torch
    from copy import deepcopy
    config = deepcopy(config)
    from mirage.runtime.environment import configure, describe
    from mirage.runtime.state import rng_state
    configure()
    graph = analyze_stateful(source)
    from mirage.runtime.input_constraints import OPERAND_CONSTRUCTOR, operand_builder, make_tensor
    constructed = []
    from functools import partial
    namespace = {"torch": torch, "math": math,
                 "make_tensor": partial(make_tensor, device=config["device"]),
                 OPERAND_CONSTRUCTOR: operand_builder(config.get("input_recipe"), config["device"],
                     config.get("input_recipe_sha256"), constructed)}
    exec(compile(source, "<mirage-stateful-program>", "exec"), namespace)
    before_rng = torch.random.get_rng_state().clone()
    build_rng = rng_state(config['device']) if config.get('check_claim') else None
    ctx = namespace['build'](config['device'], config.get('seed', 0))
    if not isinstance(ctx, dict) or not isinstance(ctx.get('inputs'), dict) or not isinstance(ctx.get('parameters'), dict):
        raise ContractError("build must return inputs and parameters dictionaries")
    for name, parameter in ctx['parameters'].items():
        if not isinstance(parameter, torch.Tensor):
            raise ContractError(f"parameters[{name!r}] must be a Tensor; got {type(parameter).__name__}")
        if not parameter.is_leaf or not parameter.requires_grad:
            raise ContractError(f"parameters[{name!r}] requires is_leaf=True and requires_grad=True; "
                                f"observed is_leaf={parameter.is_leaf}, requires_grad={parameter.requires_grad}. "
                                "Use inputs for non-parameter tensors; parameters may be empty.")
    if not isinstance(ctx.get('modules',{}),dict): raise ContractError('modules must be a dictionary of actual modules')
    for key in ('optimizer','scheduler','scaler'):
        if ctx.get(key) is not None and not callable(getattr(ctx[key],'state_dict',None)):
            raise ContractError(f'{key} must be an actual stateful object or None; omit unused optional fields, not empty dictionaries')
    for module in ctx.get('modules', {}).values():
        if not isinstance(module, torch.nn.Module): raise ContractError("Unsupported registered module")
        if any(id(x) not in {id(p) for p in ctx['parameters'].values()} for x in module.parameters()):
            raise ContractError("Register all module parameters")
    known={'inputs','parameters','modules','optimizer','scheduler','scaler','checkpoint_data'}
    try:
        validate_checkpoint_data(ctx.get('checkpoint_data'), lambda x: isinstance(x, torch.Tensor))
        validate_checkpoint_data({k:v for k,v in ctx.items() if k not in known}, lambda x:isinstance(x,torch.Tensor))
    except ValueError as error:
        raise ContractError(str(error)) from error
    if sum(x.numel() for _, x in leaves((ctx['inputs'], ctx['parameters']))) > 2_000_000:
        raise ContractError("Initialization element budget exceeded")
    initial_metadata = [asdict(d) for d, _ in discover((ctx['inputs'], ctx['parameters']), PyTorchBackend())]
    initial = training_snapshot(ctx)
    if initial_reference is not None:
        if (PyTorchBackend().compare(initial, initial_reference['initial'], 'exact')['equal'] is not True
                or initial_metadata != initial_reference['initial_metadata']):
            raise ContractError('Saved-source replay did not reconstruct the recorded initial values/representation')
    environment = describe(config['device'])
    rng_changed = not torch.equal(before_rng, torch.random.get_rng_state())
    if build_rng is not None:
        rng_changed |= PyTorchBackend().compare(build_rng, rng_state(config['device']), 'exact')['equal'] is not True
    probe = Probe(resolve(config['api']), ctx, config, recorder)
    if config.get('device_observation'):
        probe._device_initial = dict(initial=initial, initial_metadata=initial_metadata, environment=environment)
    from mirage.runtime.optimizer import OptimizerEvidence
    optimizer_evidence = OptimizerEvidence(probe, source)
    probe._optimizer_evidence = optimizer_evidence
    if restore_prefix_rng:
        from mirage.runtime.state import restore_rng
        restore_rng(initial_reference['initial_rng'], config['device'])
    initial_rng = rng_state(config['device'])
    probe.mark('initial')
    from torch.overrides import TorchFunctionMode
    class ObserveActions(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, '__name__', str(func))
            backward = not probe._observing and name in ('backward', 'grad') and probe._backward_invocation is None
            if backward:
                invocation = {'index': len(probe.backward_invocations), 'kind': name, 'completed': False}
                probe.backward_invocations.append(invocation)
                probe._backward_invocation = invocation['index']
                optimizer_evidence.backward_start(invocation['index'])
                probe.gradient_boundary(invocation['index'], before=True)
            try:
                result = func(*args, **(kwargs or {}))
                if backward:
                    invocation['completed'] = True
                    probe.gradient_boundary(invocation['index'], before=False)
            finally:
                if backward:
                    probe._backward_invocation = None
            if not probe._observing and name in ('backward', 'grad', 'zero_', 'detach', 'detach_', 'requires_grad_'):
                if len(probe.operations) >= 4096:
                    raise ContractError('Observed action budget exceeded')
                probe.operations.append({'name': name})
                recorder.emit('training_action', name=name)
            return result
    from mirage.backends.pytorch import ContextObserver, CompilerObserver, CaptureObserver, OptimizerObserver, capture_source_certificate, execution_source_certificate
    try:
        with ContextObserver(probe._target, probe.context_events, lambda name: probe.operations.append({'name':name})) as contexts:
            probe.context_observer = contexts
            with CompilerObserver(probe, namespace, config) as compiler, CaptureObserver(probe, namespace, config) as capture:
                probe.compiled_graphs = compiler.graphs
                probe.capture_replays = capture.replays
                with optimizer_evidence, ObserveActions() as action_observer, OptimizerObserver(
                        ctx.get('optimizer'), lambda name: probe.operations.append({'name':name}),
                        before_step=optimizer_evidence.before_step, after_step=optimizer_evidence.after_step) as updates:
                    optimizer_evidence.bind_observer(updates)
                    probe._action_observer = action_observer
                    output = namespace['run'](ctx, probe)
            probe.mark('final', output)
            final_training_state = training_snapshot(ctx)
    finally:
        probe.close()
    final_rng = rng_state(config['device'])
    rng_unchanged = PyTorchBackend().compare(initial_rng, final_rng, 'exact')['equal'] is True
    for event in probe.context_events:
        group = event.get('instance_group')
        if group is not None and group < len(contexts.instances):
            event['factory_call'] = next((index for value, index in probe.products
                                          if value is contexts.instances[group]), None)
            event['autocast_scope'] = isinstance(contexts.instances[group], torch.autocast)
            owner = event.get('wrapper_owner_group')
            if owner is not None:
                event['wrapper_factory_call'] = next((index for value, index in probe.products
                                                      if value is contexts.instances[owner]), None)
    result = {"observed_products":sorted(probe.observed_products),
            "factory_requires_invocation":probe.factory_requires_invocation, "product_events": probe.product_events, "accumulations": probe.accumulations,
            "saved_tensor_events": probe.saved_tensor_events, "state_operations": probe.state_operations,
            "version": "stateful-1", "environment": environment, "run_rng_unchanged": rng_unchanged,
            "initial_rng": initial_rng, "final_rng": final_rng, "graph": graph, "config": config, "completed": True,
            "initial": initial, "initial_metadata": initial_metadata, "final_training_state": final_training_state,
            "program_certificate": PyTorchBackend().program_certificate(source), "observations": snapshot(output), "checkpoints": probe.checkpoints,
            "numerical_source_certificate": PyTorchBackend().numerical_source_certificate(source),
            "optimizer_source_certificate": optimizer_evidence.certificate,
            "optimizer_evidence": optimizer_evidence.record,
            "target_parameter_bindings": optimizer_evidence.bindings,
            "target_calls": probe.calls, "target_gradients": probe.backward,
            "gradient_evidence": probe.gradient_evidence,
            "target_backward_nodes": probe.backward_nodes, "target_operand_gradients": probe.operand_gradients,
            "callable_backward_nodes": probe.callable_backward_nodes,
            "backward_invocations": probe.backward_invocations,
            "compiled_graphs": probe.compiled_graphs, "compilation_factories": compiler.factory_records,
            "capture_replays": capture.replays,
            "capture_factories": capture.factory_records, "reference_substitutions": capture.substitutions,
            "capture_source_certificate": capture_source_certificate(source),
            "execution_source_certificate": execution_source_certificate(source),
            "context_events": probe.context_events, "context_contract": contexts.supported,
            "mutation_candidates": probe.candidates, "mutations": probe.mutations,
            "api_executed": bool(probe.calls) and not bool(capture.substitutions),
            "initialization_rng_changed": rng_changed,
            "determinism": "unresolved" if rng_changed or graph['initialization_review'] else "not_certified",
            "correctness": "unassessed", "verdict": None,
            "reason": "Stateful runtime evidence collected; relation validation required"}

    # Probe only a witnessed direct re-entry with an available restoration contract.
    # The generated program and its final observations have already finished.
    if (contexts.supported and config.get('state_goal') and not config.get('device_observation')
            and not config.get('representation_candidate')):
        from mirage.backends.pytorch import probe_context_decorator
        direct = {e.get('factory_call') for e in probe.context_events if e.get('balanced')
                  and e.get('reentrant_same_instance') and not e.get('decorated_entry')}
        for value, index in probe.products:
            if index in direct:
                result['decorator_probes'] = [dict(call=index, identity=probe_context_decorator(value))]
                break
    result["constructed_operands"] = constructed
    if config.get('oracle_improvements'):
        result['scoped_vjp_source_certificate'] = PyTorchBackend().scoped_vjp_source_certificate(source)

    from mirage.inspector.stateful import inspect_state, input_coverage, callable_execution_indices
    callable_calls = callable_execution_indices(result)
    result['backward_executed'] = any(e.get('invocation') is not None and
        probe.backward_invocations[e['invocation']]['completed']
        for e in probe.backward_nodes + [e for e in probe.callable_backward_nodes if e['call'] in callable_calls])
    result['observed_actions'] = [event['name'] for event in probe.operations]
    # Detached copies used only for observations must not manufacture diversity.
    # Raw events remain available; this conservative signature omits all detach events.
    result['transition_signature'] = [name for name in result['observed_actions'] if name not in ('detach','detach_')]
    result['inspection'] = inspect_state(result, config.get('predicates', ()))
    result['input_coverage'] = input_coverage(result)
    result['target_state_reached'] = result['inspection']['target_state_reached']
    if not result['inspection']['valid']:
        result.update(verdict='invalid', reason='Required runtime target/backward/state evidence missing')
    return result
