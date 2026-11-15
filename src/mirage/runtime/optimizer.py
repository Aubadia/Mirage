"""Bounded evidence for target-produced gradients consumed by native updates.

The source owns backward and step. Only an explicitly configured, value-preserving
copy of parameter.grad is performed at the observed pre-update boundary.
"""
from dataclasses import asdict
from functools import wraps
import math
from itertools import product

from mirage.generator.program import ContractError

VERSION = 'optimizer-update-1'
CERTIFICATE = 'explicit_optimizer_gradient_update_v1'
MAX_EVENTS = 16
MAX_ELEMENTS = 4096


def observed(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        previous = self.probe._observing
        self.probe._observing = True
        try:
            return method(self, *args, **kwargs)
        finally:
            self.probe._observing = previous
    return wrapped


def nonoverlapping(tensor):
    # A bounded exact coordinate check also proves stride-two views for which
    # the native overlap heuristic returns an inconclusive value.
    if tensor.numel() > MAX_ELEMENTS or any(s < 0 for s in tensor.stride()):
        return False
    # product eagerly materializes its pools, including huge axes of empty tensors.
    if tensor.numel() == 0:
        return True
    positions = {sum(i * stride for i, stride in zip(index, tensor.stride()))
                 for index in product(*(range(n) for n in tensor.shape))}
    return len(positions) == tensor.numel()


class OptimizerEvidence:
    def __init__(self, probe, source):
        import torch
        self.probe, self.ctx = probe, probe._ctx
        self.optimizer = self.ctx.get('optimizer')
        self.certificate = probe._backend.optimizer_source_certificate(source)
        self.record = {'version': VERSION, 'registry': {'optimizer_id': 'optimizer:0',
                       'native_type': None, 'parameter_keys': [], 'parameters': [], 'groups': []},
                       'steps': [], 'accumulations': [], 'limitations': [], 'dropped_events': 0}
        self.parameters = dict(self.ctx['parameters'])
        self.bindings = []
        self.handles = []
        self.accumulated = {}
        self.backward_before = {}
        self.pending = None
        self.expected_hooks = None
        self.initial_groups = None
        self.initial_parameter_objects = dict(self.parameters)
        self._initial_native_methods = None
        self.action = probe._config.get('optimizer_gradient_action')
        if self.action is not None:
            if (not isinstance(self.action, dict) or set(self.action) != {'step_index', 'parameter_key', 'action'}
                    or type(self.action['step_index']) is not int or self.action['step_index'] != 0
                    or type(self.action['parameter_key']) is not str
                    or self.action['action'] not in ('canonical', 'strided', 'offset')):
                raise ContractError('optimizer_gradient_action requires step_index=0, parameter_key and canonical/strided/offset action')
        if isinstance(self.optimizer, torch.optim.Optimizer):
            known = {id(p) for p in self.parameters.values()}
            if any(id(p) not in known for group in self.optimizer.param_groups for p in group['params']):
                raise ContractError('Register every optimizer parameter as the same object in parameters')
        self.domain_reason = self._registration_domain()
        if self.domain_reason:
            if self.optimizer is not None:
                self.limit(self.domain_reason)
        else:
            self.initial_groups = self._groups()
            self._initial_native_methods = (self.optimizer.step.__func__, self.optimizer.zero_grad.__func__)

    def limit(self, reason):
        if reason not in self.record['limitations']:
            if len(self.record['limitations']) < MAX_EVENTS:
                self.record['limitations'].append(reason)
            else:
                self.record['dropped_events'] += 1

    def _groups(self):
        from mirage.runtime.stateful import snapshot
        names = {id(p): key for key, p in self.parameters.items()}
        return [{'group_index': index, 'parameter_keys': [names.get(id(p)) for p in group['params']],
                 'options': snapshot({k: v for k, v in group.items() if k != 'params'})}
                for index, group in enumerate(self.optimizer.param_groups)]

    def _registration_domain(self):
        import torch
        from torch.optim import optimizer as optimizer_module
        optimizer = self.optimizer
        if optimizer is None:
            return 'no registered optimizer'
        self.record['registry']['native_type'] = type(optimizer).__module__ + '.' + type(optimizer).__qualname__
        if type(optimizer) is not torch.optim.SGD:
            return 'only exact native SGD is reviewed'
        if not self.parameters or len(self.parameters) > 4 or any(type(k) is not str or not k for k in self.parameters):
            return 'one to four named parameters with string keys are required'
        if len({id(p) for p in self.parameters.values()}) != len(self.parameters):
            return 'ambiguous duplicate parameter registration'
        native_types = (torch.Tensor, torch.nn.Parameter)
        if any(type(p) not in native_types or not p.is_leaf or not p.requires_grad for p in self.parameters.values()):
            return 'native trainable leaf parameters are required'
        if sum(p.numel() for p in self.parameters.values()) > MAX_ELEMENTS:
            return 'optimizer parameter element budget exceeded'
        if any(p.device.type != 'cpu' or p.dtype != torch.float64 or p.layout != torch.strided
               or p.is_mkldnn or not torch.isfinite(p).all() for p in self.parameters.values()):
            return 'optimizer domain requires finite CPU float64 dense parameters'
        identities = [self.probe._backend.storage_identity(p) for p in self.parameters.values()]
        if len(set(identities)) != len(identities) or any(not nonoverlapping(p) for p in self.parameters.values()):
            return 'parameter storage sharing or overlap is unsupported'
        if any(p.grad is not None for p in self.parameters.values()):
            return 'optimizer chain starts with parameter.grad=None'
        if any(p._backward_hooks or getattr(p, '_post_accumulate_grad_hooks', None) for p in self.parameters.values()):
            return 'foreign parameter gradient hooks are unsupported'
        if any(self.ctx.get(k) is not None for k in ('scheduler', 'scaler')):
            return 'scheduler and AMP/scaler state are outside this optimizer domain'
        if (optimizer._optimizer_step_pre_hooks or optimizer._optimizer_step_post_hooks
                or optimizer_module._global_optimizer_pre_hooks or optimizer_module._global_optimizer_post_hooks):
            return 'foreign optimizer hooks are unsupported'
        if ('step' in optimizer.__dict__ or 'zero_grad' in optimizer.__dict__
                or getattr(optimizer.step, '__func__', None) is not torch.optim.SGD.step
                or getattr(optimizer.zero_grad, '__func__', None) is not torch.optim.SGD.zero_grad):
            return 'optimizer method overrides are unsupported'
        if getattr(optimizer, 'grad_scale', None) is not None or getattr(optimizer, 'found_inf', None) is not None:
            return 'AMP optimizer attributes are unsupported'
        if not 1 <= len(optimizer.param_groups) <= 4:
            return 'one to four parameter groups are required'
        names = {id(p): key for key, p in self.parameters.items()}
        entries = []
        seen = set()
        for group_index, group in enumerate(optimizer.param_groups):
            if not group['params']:
                return 'empty optimizer group is unsupported'
            for group_position, p in enumerate(group['params']):
                if id(p) in seen:
                    return 'duplicate optimizer parameter membership'
                seen.add(id(p))
                entries.append({'parameter_key': names[id(p)], 'parameter_id': 'parameter:' + names[id(p)],
                                'group_index': group_index, 'group_position': group_position})
            if (type(group.get('lr')) not in (int, float) or not math.isfinite(group['lr']) or group['lr'] <= 0
                    or type(group.get('momentum')) not in (int, float) or not math.isfinite(group['momentum'])
                    or not 0 <= group['momentum'] < 1):
                return 'finite positive scalar lr and momentum in [0,1) are required'
            if any(group.get(k) is not False for k in ('foreach', 'fused', 'differentiable', 'maximize', 'nesterov')):
                return 'SGD dispatch flags must explicitly select the reviewed native path'
            if any(type(group.get(k)) not in (int, float) or group[k] != 0 for k in ('weight_decay', 'dampening')):
                return 'weight decay and dampening are outside this optimizer domain'
            if set(group) != {'params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'foreach', 'differentiable', 'fused'}:
                return 'unreviewed optimizer group options'
        if seen != set(names):
            return 'all registered parameters must belong to this optimizer'
        if optimizer.state:
            return 'only a cold-start native optimizer state is reviewed'
        self.record['registry']['parameter_keys'] = list(self.parameters)
        self.record['registry']['parameters'] = entries
        self.record['registry']['groups'] = self._groups()
        if self.certificate != CERTIFICATE:
            return 'source effects lack the optimizer-specific certificate'
        return None

    @observed
    def __enter__(self):
        if self.domain_reason is None:
            for key, parameter in self.parameters.items():
                self.handles.append(parameter.register_post_accumulate_grad_hook(
                    lambda value, key=key: self._accumulated(key, value)))
        return self

    def bind_observer(self, observer):
        if self.domain_reason is None:
            self.expected_hooks = (set(self.optimizer._optimizer_step_pre_hooks), set(self.optimizer._optimizer_step_post_hooks))

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.accumulated.clear()
        self.backward_before.clear()

    def bind_target(self, index, operands):
        for descriptor, tensor in operands:
            names = [key for key, p in self.parameters.items() if p is tensor and type(key) is str]
            if len(names) == 1:
                key = names[0]
                self.bindings.append({'parameter_key': key, 'parameter_id': 'parameter:' + key,
                                      'call_index': index, 'operand_path': list(descriptor.path)})

    @observed
    def backward_start(self, invocation):
        if self.domain_reason is None:
            self.backward_before[invocation] = {key: p.grad is None for key, p in self.parameters.items()}

    @observed
    def _accumulated(self, key, parameter):
        from mirage.runtime.stateful import snapshot, tensor_version
        invocation = self.probe._backward_invocation
        gradient = parameter.grad
        if gradient is None or len(self.record['accumulations']) >= MAX_EVENTS:
            self.record['dropped_events'] += 1
            return
        event = {'index': len(self.record['accumulations']), 'parameter_key': key,
                 'parameter_id': 'parameter:' + key, 'invocation': invocation,
                 'started_none': self.backward_before.get(invocation, {}).get(key) is True,
                 'gradient': snapshot(gradient)}
        self.record['accumulations'].append(event)
        self.accumulated[key] = (gradient, tensor_version(gradient), event)

    def _snapshot(self):
        from mirage.runtime.stateful import snapshot
        parameters, gradients, metadata, slots = {}, {}, {}, {}
        for key, p in self.parameters.items():
            parameters[key], gradients[key] = snapshot(p), snapshot(p.grad)
            if p.grad is not None:
                metadata[key] = asdict(self.probe._backend.describe_tensor(p.grad, ('gradients', key), 0, 0))
            slots[key] = snapshot(dict(self.optimizer.state.get(p, {})))
        return {'parameters': parameters, 'gradients': gradients, 'gradient_metadata': metadata, 'slots': slots}

    def _pre_domain(self, args, kwargs):
        import torch
        from torch.optim import optimizer as optimizer_module
        if self.domain_reason:
            return self.domain_reason
        if len(self.record['steps']) != 1:
            return 'only one completed optimizer step is reviewed'
        if self.probe._backend.compare(self._groups(), self.initial_groups, 'exact')['equal'] is not True:
            return 'optimizer parameter groups changed'
        if any(self.ctx['parameters'].get(k) is not p for k, p in self.initial_parameter_objects.items()):
            return 'registered parameter identity changed'
        if (getattr(self.optimizer.step, '__func__', None), getattr(self.optimizer.zero_grad, '__func__', None)) != self._initial_native_methods:
            return 'optimizer method identity changed'
        if (set(self.optimizer._optimizer_step_pre_hooks), set(self.optimizer._optimizer_step_post_hooks)) != self.expected_hooks:
            return 'optimizer hooks changed after registration'
        if optimizer_module._global_optimizer_pre_hooks or optimizer_module._global_optimizer_post_hooks:
            return 'foreign global optimizer hooks'
        if args != (self.optimizer,) or kwargs:
            return 'optimizer closures or step arguments are unsupported'
        if (torch.is_inference_mode_enabled() or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')
                or self.probe.compiled_graphs or self.probe.capture_replays):
            return 'optimizer chain requires ordinary eager execution without AMP'
        if len(self.probe.backward_invocations) != 1 or not self.probe.backward_invocations[0]['completed']:
            return 'exactly one completed backward invocation is required'
        if self.optimizer.state:
            return 'optimizer slots changed before the first update'
        all_tensors = list(self.parameters.values())
        for key, p in self.parameters.items():
            g = p.grad
            if (g is None or type(g) is not torch.Tensor or g.device.type != 'cpu' or g.dtype != torch.float64
                    or g.layout != torch.strided or g.is_mkldnn or g.requires_grad or not bool(torch.isfinite(g).all())
                    or not nonoverlapping(g)):
                return 'consumed gradients must be finite native nonoverlapping dense CPU float64 values'
            all_tensors.append(g)
        identities = [self.probe._backend.storage_identity(t) for t in all_tensors]
        if len(set(identities)) != len(identities):
            return 'gradient/parameter storage sharing is unsupported'
        return None

    def _provenance(self, key):
        import torch
        from mirage.runtime.stateful import tensor_version
        result = {'parameter_key': key, 'parameter_id': 'parameter:' + key, 'status': 'unknown',
                  'reason': 'no unique attributable target-to-parameter gradient', 'unchanged_to_pre_step': False}
        bindings = [b for b in self.bindings if b['parameter_key'] == key]
        if len(bindings) != 1:
            return result
        binding = bindings[0]
        result.update(binding)
        if key not in self.accumulated:
            return result
        gradient, version, event = self.accumulated[key]
        result['post_accumulate_event'] = event['index']
        records = [e for e in self.record['accumulations'] if e['parameter_key'] == key]
        if len(records) != 1 or not event['started_none'] or event['invocation'] != 0:
            result['reason'] = 'ambiguous accumulation or nonempty initial gradient'
            return result
        p = self.parameters[key]
        if (p.grad is not gradient or tensor_version(gradient) != version or version is None
                or not torch.equal(p.grad.detach().cpu(), event['gradient'])):
            result['reason'] = 'parameter gradient was replaced or changed after accumulation'
            return result
        result['unchanged_to_pre_step'] = True
        events = [e for e in self.probe.gradient_evidence['events'] if e.get('call') == binding['call_index']
                  and e.get('invocation') == event['invocation'] and e.get('status') == 'complete']
        if len(events) != 1:
            return result
        direct = [v for v in events[0]['inputs'] if v['path'] == binding['operand_path']]
        if (len(direct) != 1 or direct[0]['gradient'] is None
                or not torch.equal(direct[0]['gradient'], event['gradient'])):
            result['reason'] = 'consumed gradient differs from the direct target contribution'
            return result
        result.update(status='verified', reason='direct target Node contribution reached this leaf and remained unchanged until update',
                      node=events[0]['node'], invocation=event['invocation'],
                      informative_nonzero=bool(event['gradient'].ne(0).any()))
        return result

    def _mutate(self, step):
        import torch
        from mirage.runtime.representations import connected_representation
        from mirage.runtime.stateful import snapshot
        action = self.action
        if action is None:
            return
        mutation = {'status': 'unknown', 'parameter_key': action['parameter_key'], 'action': action['action'], 'logical_equal': False}
        step['mutation'] = mutation
        if action['step_index'] != step['index']:
            mutation['reason'] = 'configured optimizer step was not reached'
            return
        key = action['parameter_key']
        if step['domain']['status'] != 'verified' or not any(p['parameter_key'] == key and p['status'] == 'verified' for p in step['gradient_provenance']):
            mutation['reason'] = 'gradient mutation requires verified native domain and target provenance'
            return
        parameter = self.parameters[key]
        before = parameter.grad
        mutation['before'] = asdict(self.probe._backend.describe_tensor(before, ('gradients', key), 0, 0))
        replacement = connected_representation(before, action['action'])
        equal = (replacement.shape == before.shape and replacement.dtype == before.dtype and replacement.device == before.device
                 and torch.equal(replacement, before) and not replacement.requires_grad)
        metadata = asdict(self.probe._backend.describe_tensor(replacement, ('gradients', key), 0, 0))
        attained = (metadata['properties']['contiguous'] and metadata['offset'] == 0 if action['action'] == 'canonical'
                    else not metadata['properties']['contiguous'] if action['action'] == 'strided'
                    else metadata['offset'] > 0)
        others = list(self.parameters.values()) + [p.grad for p in self.parameters.values() if p.grad is not None]
        separate = all(self.probe._backend.storage_identity(replacement) != self.probe._backend.storage_identity(t) for t in others)
        if not equal or not attained or not separate or not nonoverlapping(replacement):
            mutation.update(reason='gradient action did not establish the requested exact-value independent representation', after=metadata)
            return
        parameter.grad = replacement
        mutation.update(status='verified', logical_equal=True, before_values=snapshot(before), after=metadata,
                        reason='harness-owned exact-value gradient replacement immediately before native update')

    @observed
    def before_step(self, optimizer, args, kwargs):
        index = len(self.record['steps'])
        if index >= MAX_EVENTS:
            self.record['dropped_events'] += 1
            return
        step = {'index': index, 'optimizer_id': 'optimizer:0', 'pre_checkpoint': f'optimizer:0/step:{index}/pre_update',
                'post_checkpoint': f'optimizer:0/step:{index}/post_update', 'completed': False,
                'domain': {'status': 'unknown', 'reason': 'not checked'}, 'pre': {}, 'post': {},
                'gradient_provenance': [], 'mutation': None}
        self.record['steps'].append(step)
        self.pending = step
        reason = self._pre_domain(args, kwargs)
        step['domain'] = {'status': 'unknown' if reason else 'verified', 'reason': reason or 'reviewed native single-step SGD domain'}
        if reason:
            self.limit(reason)
        else:
            step['gradient_provenance'] = [self._provenance(k) for k in self.parameters]
            self._mutate(step)
            step['pre'] = self._snapshot()
            self.probe.mark(step['pre_checkpoint'], _optimizer_observer=self)

    @observed
    def after_step(self, optimizer, args, kwargs):
        if self.pending is None:
            return
        step = self.pending
        step['completed'] = True
        self.pending = None
        if step['domain']['status'] == 'verified':
            step['post'] = self._snapshot()
            step['parameter_changed'] = self.probe._backend.compare(step['pre']['parameters'], step['post']['parameters'], 'exact')['equal'] is False
            self.probe.mark(step['post_checkpoint'], _optimizer_observer=self)

    def target_update_count(self):
        if self.record['limitations'] or self.record['dropped_events'] or len(self.record['steps']) > 1:
            return 0
        return sum(s['completed'] and s['domain']['status'] == 'verified'
                   and (s['mutation'] is None or s['mutation']['status'] == 'verified' and s['mutation']['logical_equal'])
                   and any(p['status'] == 'verified' for p in s['gradient_provenance'])
                   for s in self.record['steps'])
