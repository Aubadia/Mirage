"""Execute LLM-owned training histories; probes only observe and call APIs."""
from dataclasses import asdict
import json
import math

from mirage.generator.program import analyze_stateful, ContractError
from mirage.contracts.programs import validate_checkpoint_data, bind_call
from mirage.contracts.capabilities import EvidenceStatus, MutationObligations
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
    if isinstance(value, (torch.dtype, torch.device)):
        return {'framework_literal':str(value)}
    if isinstance(value, dict): return {str(k):snapshot(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [snapshot(v) for v in value]
    if value is None or type(value) in (bool, int, float, str): return value
    if callable(value): return {'callable':getattr(value, '__qualname__', type(value).__qualname__)}
    raise ContractError(f'Unsupported observation type: {type(value).__name__}')


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
        self.calls, self.checkpoints, self.backward = [], [], []
        self.backward_nodes, self.operand_gradients, self.backward_invocations = [], [], []
        self._backward_invocation = None
        self._hook_handles = []
        self.mutations, self.candidates = [], []
        self.operations = []
        self.context_events = []
        self.compiled_graphs = []
        self.capture_replays = []
        self._observing = False

    @observational
    def mark(self, name, values=None):
        import torch
        if not isinstance(name, str) or not name:
            raise ContractError("Checkpoint requires a nonempty name")
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
            'graph_lifecycle.target_call_count': len(self.calls),
            'graph_lifecycle.target_boundary_count': len(self.calls),
        }
        if any(g.get('target_execution_certified') for g in self.compiled_graphs):
            data['state_facts']['execution_mode.compiled_execution'] = True
        if self.capture_replays:
            data['state_facts']['execution_mode.captured_execution'] = True
        data['completed_backward_invocations'] = [e['index'] for e in self.backward_invocations if e['completed']]
        data['evidence_offsets'] = {'calls': len(self.calls), 'outputs': len(self.backward),
                                    'nodes': len(self.backward_nodes), 'operands': len(self.operand_gradients),
                                    'invocations': len(self.backward_invocations)}
        data['observable_metadata'] = [asdict(d) for d, _ in discover(values, self._backend)]
        self.checkpoints.append(data)
        self._recorder.emit("checkpoint", name=name, grad_enabled=data['grad_enabled'])

    @observational
    def observe_outputs(self, index, outputs, input_edges, *, node_admissible=True, boundary=False):
        """Keep output-gradient and Node-execution evidence independent."""
        bindings, seen = [], set()
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
                             'reason': reason, 'operand_paths': [list(p) for _, p in links]})
            if node is None or node in seen:
                continue
            seen.add(node)
            node_index = len(seen) - 1
            def node_executed(grad_inputs, grad_outputs, i=index, n=node_index, edges=tuple(links)):
                event = {'call': i, 'node': n, 'invocation': self._backward_invocation}
                self.backward_nodes.append(event)
                self._recorder.emit('target_backward', **event)
                for slot, operand_path in edges:
                    if slot < len(grad_inputs) and grad_inputs[slot] is not None:
                        contribution = dict(event, path=list(operand_path), slot=slot)
                        self.operand_gradients.append(contribution)
                        self._recorder.emit('target_operand_gradient', **contribution)
            self._hook_handles.append(node.register_hook(node_executed))
        self.calls[index]['backward_bindings'] = bindings

    def close(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @observational
    def call(self, target, *args, **kwargs):
        import torch
        # Calling a probe while tracing is not a runtime target witness.
        if torch.compiler.is_compiling():
            raise ContractError("Probe inside compiled tracing is not a certified execution boundary")
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
            self.mutations.append(evidence_record(evidence))
        operands = discover(tree, self._backend)
        if sum(t.numel() for _, t in operands) > 2_000_000:
            raise ContractError("Operand element budget exceeded")
        index = len(self.calls)
        before = snapshot(tree)
        versions = [(t, tensor_version(t)) for _,t in operands]
        writable_paths = [list(d.path) for d,_ in operands if isinstance(obligations, dict) and obligations[d.path].writable]
        descriptors = [asdict(d) for d, _ in operands]
        properties = [{'path': list(d.path), 'facts': self._backend.input_properties(t)} for d, t in operands]
        shapes = [d.shape for d, _ in operands]
        for item, shape in zip(properties, shapes):
            item['facts']['broadcast'] = any(shape != other and _broadcastable(shape, other) for other in shapes)
        input_edges = self._backend.gradient_edges(operands)
        entry_context = dict(context_state(), inference_enabled=torch.is_inference_mode_enabled())
        self.operations.append({'name': 'target_enter'})
        self._recorder.emit("api_enter", call=index, metadata=descriptors)
        self._observing = False
        try:
            output = target(*tree['args'], **tree['kwargs'])
        finally:
            self._observing = True
        self.calls.append({"index": index, "inputs": before, "metadata": descriptors,
                           "output": snapshot(output) if isinstance(output, (torch.Tensor, dict, list, tuple, int, float, str, bool, type(None)))
                                     else {'opaque_type':type(output).__qualname__},
                           "binding": evidence_record(binding), "input_properties": properties,
                           "execution_mode": "eager", "entry_context": entry_context,
                           "write_effects": [list(d.path) for (d,t),(_,v) in zip(operands,versions) if v is not None and tensor_version(t) != v],
                           "writable_paths": writable_paths})
        self._recorder.emit("api_exit", call=index)
        self.operations.append({'name': 'target_return'})
        import inspect
        self.observe_outputs(index, leaves(output), input_edges,
                             node_admissible=inspect.isbuiltin(target) or inspect.ismethoddescriptor(target))
        return output


def execute(source, config, recorder, initial_reference=None):
    import torch
    from copy import deepcopy
    config = deepcopy(config)
    from mirage.runtime.environment import configure, describe
    from mirage.runtime.state import rng_state
    configure()
    graph = analyze_stateful(source)
    namespace = {"torch": torch, "math": math}
    exec(compile(source, "<mirage-stateful-program>", "exec"), namespace)
    before_rng = torch.random.get_rng_state().clone()
    ctx = namespace['build'](config['device'], config.get('seed', 0))
    if not isinstance(ctx, dict) or not isinstance(ctx.get('inputs'), dict) or not isinstance(ctx.get('parameters'), dict):
        raise ContractError("build must return inputs and parameters dictionaries")
    if any(not isinstance(x, torch.Tensor) or not x.is_leaf or not x.requires_grad for x in ctx['parameters'].values()):
        raise ContractError("Register trainable leaf tensors as parameters")
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
    probe = Probe(resolve(config['api']), ctx, config, recorder)
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
            try:
                result = func(*args, **(kwargs or {}))
                if backward:
                    invocation['completed'] = True
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
            with CompilerObserver(probe, namespace, config) as compiler, CaptureObserver(probe, namespace, config) as capture:
                probe.compiled_graphs = compiler.graphs
                probe.capture_replays = capture.replays
                with ObserveActions() as action_observer, OptimizerObserver(ctx.get('optimizer'), lambda name: probe.operations.append({'name':name})):
                    probe._action_observer = action_observer
                    output = namespace['run'](ctx, probe)
            probe.mark('final', output)
            final_training_state = training_snapshot(ctx)
    finally:
        probe.close()
    final_rng = rng_state(config['device'])
    rng_unchanged = PyTorchBackend().compare(initial_rng, final_rng, 'exact')['equal'] is True
    result = {"version": "stateful-1", "environment": environment, "run_rng_unchanged": rng_unchanged,
            "initial_rng": initial_rng, "final_rng": final_rng, "graph": graph, "config": config, "completed": True,
            "initial": initial, "initial_metadata": initial_metadata, "final_training_state": final_training_state,
            "program_certificate": PyTorchBackend().program_certificate(source), "observations": snapshot(output), "checkpoints": probe.checkpoints,
            "target_calls": probe.calls, "target_gradients": probe.backward,
            "target_backward_nodes": probe.backward_nodes, "target_operand_gradients": probe.operand_gradients,
            "backward_invocations": probe.backward_invocations,
            "compiled_graphs": probe.compiled_graphs, "capture_replays": capture.replays,
            "capture_factories": capture.factory_records, "reference_substitutions": capture.substitutions,
            "capture_source_certificate": capture_source_certificate(source),
            "execution_source_certificate": execution_source_certificate(source),
            "context_events": probe.context_events, "context_contract": contexts.supported,
            "mutation_candidates": probe.candidates, "mutations": probe.mutations,
            "api_executed": bool(probe.calls) and not bool(capture.substitutions), "backward_executed": any(e["invocation"] is not None and
                probe.backward_invocations[e["invocation"]]["completed"] for e in probe.backward_nodes),
            "initialization_rng_changed": rng_changed,
            "determinism": "unresolved" if rng_changed or graph['initialization_review'] else "not_certified",
            "correctness": "unassessed", "verdict": None,
            "reason": "Stateful runtime evidence collected; relation validation required"}

    from mirage.inspector.stateful import inspect_state, input_coverage
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
