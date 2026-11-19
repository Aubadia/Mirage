"""Recursive numerical comparison with explicit finite-value and dtype checks."""

import math
from pathlib import Path


def differences(left, right, tolerance: dict | None = None, path: str = "value") -> list[dict]:
    import torch

    tolerance = tolerance or {"atol": 0.0, "rtol": 0.0}
    if path.endswith(".step") or ".optimizer.param_groups" in path or path.startswith("optimizer.param_groups"):
        tolerance = {"atol": 0.0, "rtol": 0.0}
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return [{"path": path, "reason": "tensor/type mismatch"}]
        if left.shape != right.shape or left.dtype != right.dtype:
            return [{"path": path, "reason": "shape/dtype mismatch"}]
        if left.is_floating_point() or left.is_complex():
            if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                return [{"path": path, "reason": "nonfinite tensor"}]
            dtype = torch.complex128 if left.is_complex() else torch.float64
            a, b = left.to(dtype), right.to(dtype)
            error = (a - b).abs()
            allowed = tolerance["atol"] + tolerance["rtol"] * b.abs()
            if bool((error <= allowed).all()):
                return []
            return [{"path": path, "reason": "numerical mismatch",
                     "max_absolute_error": float(error.max()), "tolerance": tolerance}]
        return [] if torch.equal(left, right) else [{"path": path, "reason": "discrete tensor mismatch"}]
    if type(left) is not type(right):
        return [{"path": path, "reason": "type mismatch"}]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [{"path": path, "reason": "mapping keys mismatch"}]
        return [item for key in left for item in differences(left[key], right[key], tolerance, f"{path}.{key}")]
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return [{"path": path, "reason": "sequence length mismatch"}]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in differences(a, b, tolerance, f"{path}[{index}]")]
    if isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            return [{"path": path, "reason": "nonfinite scalar"}]
        if abs(left - right) <= tolerance["atol"] + tolerance["rtol"] * abs(right):
            return []
    elif left == right:
        return []
    return [{"path": path, "reason": "scalar mismatch", "left": left, "right": right}]


def equal(left, right) -> bool:
    return not differences(left, right)



DEVICE_POLICY = 'eager-pointwise-v1'
PAIRED_DEVICE_POLICY = 'paired-runtime-v2'
OBSERVED_DEVICE_POLICY = 'observed-runtime-v3'


def device_admission(record, schemas):
    """One direct, deterministic, finite, single-valued operation on its declared device."""
    import torch
    from mirage.backends.pytorch import _REPRESENTATION_SEMANTIC_CLASSES
    config = record['config']
    if not record['inspection']['valid'] or config.get('check_claim') not in ('forward', 'operand_relation'):
        return 'claim_history_or_observables_unsupported'
    if config.get('requires_backward') or record.get('backward_invocations'):
        return 'cotangent_alignment_unavailable'
    if record.get('initialization_rng_changed') or record.get('run_rng_unchanged') is not True:
        return 'random_inputs_or_history'
    calls = record.get('target_calls', [])
    if len(calls) != 1 or calls[0].get('execution_mode') != 'eager':
        return 'single_eager_call_required'
    call = calls[0]
    native = call.get('native_dispatch', [])
    if len(native) != 1 or native[0]['schema'] not in schemas:
        return 'exact_native_overload_unavailable'
    tags = native[0]['tags']
    op = native[0]['schema'].split('(')[0].split('::')[-1].split('.')[0]
    if 'pointwise' not in tags or any('nondeterministic' in tag for tag in tags):
        return 'single_valued_pointwise_policy_unavailable'
    if not call.get('schema_read_only') or call.get('write_effects'):
        return 'write_or_alias_effect_policy_unavailable'
    context = call['entry_context']
    if context['cpu_enabled'] or context['cuda_enabled'] or context['inference_enabled']:
        return 'precision_or_execution_context_unsupported'
    tensors = call.get('metadata', [])
    device = config['device'].split(':')[0]
    if any(t['device'].split(':')[0] != device for t in record['initial_metadata']):
        return 'mixed_initial_device_placement'
    if not tensors or any(t['device'].split(':')[0] != device or t['layout'] != 'torch.strided'
                          or t['storage_group'] is None for t in tensors):
        return 'operand_placement_or_representation_unsupported'
    output = call['output']
    if not isinstance(output, torch.Tensor) or not output.numel() or not torch.isfinite(output).all():
        return 'output_domain_unsupported'
    if any(t['properties']['complex'] for t in tensors) or output.is_complex():
        return 'complex_domain_unsupported'
    if output.is_floating_point() or any(t['properties']['floating'] for t in tensors):
        if output.dtype not in (torch.float32, torch.float64) or op not in _REPRESENTATION_SEMANTIC_CLASSES['pointwise']:
            return 'floating_point_policy_unavailable'
        from mirage.runtime.program import leaves
        for _, x in leaves(call['inputs']):
            if x.dtype != output.dtype or not x.numel() or not torch.isfinite(x).all():
                return 'input_domain_unsupported'
            if not ((x.abs() >= 0.125) & (x.abs() <= 8)).all():
                return 'outside_frozen_numerical_domain'
    elif not op.startswith(('bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
                            'logical_and', 'logical_or', 'logical_xor', 'logical_not')):
        return 'discrete_semantics_unreviewed'
    return None


def aligned_device_records(original, candidate):
    """Require the frozen logical state, schema, alias partitions and target checkpoint."""
    from mirage.backends.pytorch import PyTorchBackend
    def equal(a, b):
        return PyTorchBackend().compare(a, b, 'exact')['equal'] is True
    def descriptors(items):
        return [{k: v for k, v in item.items() if k != 'device'} for item in items]
    a, b = original['target_calls'][0], candidate['target_calls'][0]
    for key in ('torch', 'cuda_runtime', 'deterministic_algorithms', 'matmul_precision', 'mirage_code_sha256'):
        if original['environment'][key] != candidate['environment'][key]:
            return 'environment_changed'
    if (not equal(original['initial'], candidate['initial']) or
            descriptors(original['initial_metadata']) != descriptors(candidate['initial_metadata'])):
        return 'initial_state_or_layout_changed'
    if a['native_dispatch'] != b['native_dispatch'] or a['entry_context'] != b['entry_context']:
        return 'overload_or_context_changed'
    if descriptors(a['metadata']) != descriptors(b['metadata']) or not equal(a['inputs'], b['inputs']):
        return 'target_inputs_or_alias_topology_changed'
    if (original['transition_signature'] != candidate['transition_signature'] or
            [c['name'] for c in original['checkpoints']] != [c['name'] for c in candidate['checkpoints']]):
        return 'semantic_checkpoint_changed'
    return None


def device_comparison(left, right):
    import torch
    if left.dtype != right.dtype or left.shape != right.shape:
        return dict(status='incomparable', reason='output_contract_changed')
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        return dict(status='incomparable', reason='nonfinite_output')
    if not left.is_floating_point():
        return dict(status='agreement' if torch.equal(left, right) else 'mismatch',
                    policy=DEVICE_POLICY, unequal_elements=int((left != right).sum()), tolerance='exact')
    a, b = left.double(), right.double()
    delta = (a - b).abs()
    eps = torch.finfo(left.dtype).eps
    limit = 8 * eps * torch.maximum(a.abs(), b.abs()).clamp_min(1)
    return dict(status='agreement' if (delta <= limit).all() else 'mismatch', policy=DEVICE_POLICY,
                max_abs_delta=float(delta.max()), unequal_elements=int((delta > limit).sum()),
                tolerance='8 eps * max(1, abs(cpu), abs(cuda))', epsilon=eps)


def paired_alignment(a, b, *, frozen=False):
    """Only semantic inputs and history are alignment obligations. Outputs remain observable."""
    from mirage.backends.pytorch import PyTorchBackend
    def equal(x, y):
        return PyTorchBackend().compare(x, y, 'exact')['equal'] is True
    def descriptors(items):
        return [{k:v for k,v in item.items() if k != 'device'} for item in items]
    for key in ('torch','cuda_runtime','deterministic_algorithms','matmul_precision','mirage_code_sha256'):
        if a['environment'][key] != b['environment'][key]:
            return 'environment_changed'
    if not equal(a['initial'], b['initial']) or descriptors(a['initial_metadata']) != descriptors(b['initial_metadata']):
        return 'initial_state_or_layout_changed'
    calls, other = a['target_calls'], b['target_calls']
    if not calls or len(calls) != len(other):
        return 'target_call_count_changed'
    for left, right in zip(calls, other):
        if left.get('execution_mode') != right.get('execution_mode') or left.get('entry_context') != right.get('entry_context'):
            return 'overload_or_context_changed'
        if (left.get('native_dispatch') or not frozen) and left.get('native_dispatch') != right.get('native_dispatch'):
            return 'overload_or_context_changed'
        if (descriptors(left.get('metadata', ())) != descriptors(right.get('metadata', ())) or
                not equal(left.get('inputs'), right.get('inputs'))):
            return 'target_inputs_or_alias_topology_changed'
    if ([c['name'] for c in a['checkpoints']] != [c['name'] for c in b['checkpoints']] or
            a['transition_signature'] != b['transition_signature'] or
            a.get('backward_invocations') != b.get('backward_invocations')):
        return 'semantic_history_changed'
    def scopes(record):
        return [tuple(e.get(k) for k in ('factory_call','instance_group','entry_depth','balanced',
                                          'decorated_entry','reentrant_same_instance','before'))
                for e in record.get('context_events', [])]
    if scopes(a) != scopes(b):
        return 'context_entry_history_changed'
    # Cotangents are semantic inputs to local VJPs. Keep gradient values for comparison.
    grads, other_grads = (r.get('gradient_evidence', {}).get('events', []) for r in (a,b))
    if len(grads) != len(other_grads):
        return 'backward_event_count_changed'
    for left, right in zip(grads, other_grads):
        if any(left.get(k) != right.get(k) for k in ('call','node','invocation','status')):
            return 'backward_history_changed'
        if not equal(left['outputs'], right['outputs']):
            return 'cotangent_alignment_unavailable'
    return None


def device_observables(record, schemas):
    """Bounded pointwise output/effect and local VJP policies over aligned executions."""
    import torch
    from mirage.backends.pytorch import _REPRESENTATION_SEMANTIC_CLASSES
    config = record['config']
    if not record.get('completed') or not record.get('api_executed'):
        return {}, 'target_execution_unverified'
    if not record['inspection']['valid']:
        return {}, 'requested_state_not_attained'
    claim = config.get('check_claim')
    if claim == 'scope_exit':
        events = record.get('context_events', [])
        if (not events or not record.get('context_contract') or not all(
                e.get('balanced') and e.get('factory_call') is not None and
                isinstance(e.get('before'), dict) and isinstance(e.get('after'), dict) and
                e['before'].keys() == e['after'].keys() and
                (not e.get('reentrant_same_instance') or e.get('decorated_entry')) for e in events)):
            return {}, 'scope_restoration_evidence_unavailable'
        return {f'scope-{i}/restored/{k}':torch.tensor(e['before'][k] == e['after'][k])
                for i,e in enumerate(events) for k in e['before']}, None
    if claim not in ('forward','operand_relation','backward','accumulation'):
        return {}, 'claim_history_or_observables_unsupported'
    if record.get('initialization_rng_changed') or record.get('run_rng_unchanged') is not True:
        return {}, 'random_inputs_or_history'
    result = {}
    device = str(config['device']).split(':')[0]
    if any(item['device'].split(':')[0] != device for item in record['initial_metadata']):
        return {}, 'mixed_initial_device_placement'
    calls = record['target_calls']
    if not calls:
        return {}, 'target_execution_unverified'
    for call in calls:
        opcodes = call.get('native_dispatch', [])
        if (call.get('execution_mode') != 'eager' or len(opcodes) != 1 or
                opcodes[0]['schema'] not in schemas):
            return {}, 'exact_native_overload_unavailable'
        op = opcodes[0]['schema'].split('(')[0].split('::')[-1].split('.')[0]
        writable = call.get('writable_paths', [])
        copy = op == 'copy_'
        tags = opcodes[0]['tags']
        if any('nondeterministic' in tag for tag in tags) or ('pointwise' not in tags and not copy):
            return {}, 'single_valued_pointwise_policy_unavailable'
        if not writable and (not call.get('schema_read_only') or call.get('write_effects')):
            return {}, 'write_or_alias_effect_policy_unavailable'
        if writable and (op not in _REPRESENTATION_SEMANTIC_CLASSES['write_values'] |
                         _REPRESENTATION_SEMANTIC_CLASSES['pointwise'] or
                         len(writable) != len(call.get('destination_values', []))):
            return {}, 'write_or_alias_effect_policy_unavailable'
        context = call['entry_context']
        if context['cpu_enabled'] or context['cuda_enabled'] or context['inference_enabled']:
            return {}, 'precision_or_execution_context_unsupported'
        metadata = call.get('metadata', [])
        if not metadata or any(d['device'].split(':')[0] != device or d['layout'] != 'torch.strided'
                               or d['storage_group'] is None for d in metadata):
            return {}, 'operand_placement_or_representation_unsupported'
        output = call['output']
        if not isinstance(output, torch.Tensor) or not output.numel() or not torch.isfinite(output).all():
            return {}, 'output_domain_unsupported'
        if output.is_complex() or any(d['properties']['complex'] for d in metadata):
            return {}, 'complex_domain_unsupported'
        if output.is_floating_point() or any(d['properties']['floating'] for d in metadata):
            from mirage.runtime.program import leaves
            if (output.dtype not in (torch.float32, torch.float64) or
                    op.rstrip('_') not in _REPRESENTATION_SEMANTIC_CLASSES['pointwise'] and not copy):
                return {}, 'floating_point_policy_unavailable'
            for _, value in leaves(call['inputs']):
                if (value.dtype != output.dtype or not value.numel() or
                        not ((value.abs() >= 0.125) & (value.abs() <= 8)).all()):
                    return {}, 'outside_frozen_numerical_domain'
        elif not copy and not op.startswith(('bitwise_and','bitwise_or','bitwise_xor','bitwise_not',
                                              'logical_and','logical_or','logical_xor','logical_not')):
            return {}, 'discrete_semantics_unreviewed'
        prefix = 'call-' + str(call['index'])
        result[prefix + '/output'] = output
        for path, value in zip(writable, call.get('destination_values', [])):
            result[prefix + '/destination/' + str(path)] = value
        for item in call.get('alias_partner_values', []):
            result[prefix + '/alias/' + str(item['path'])] = item['value']
        if call.get('guard_preserved') is not None:
            result[prefix + '/guard_preserved'] = torch.tensor(call['guard_preserved'])
    if claim in ('backward','accumulation') or record.get('backward_invocations'):
        events = record.get('gradient_evidence', {})
        if not events.get('events') or events.get('dropped_events') or events.get('limitations'):
            return {}, 'local_vjp_evidence_unavailable'
        for event in events['events']:
            if (event['status'] != 'complete' or not event.get('inputs') or not event.get('outputs') or
                    event.get('operands_unchanged_at_backward') is not True):
                return {}, 'local_vjp_evidence_unavailable'
            # Broadcast-backward includes a reduction whose error requires a separate policy.
            output = calls[event['call']]['output']
            for slot in event['inputs'] + event['outputs']:
                value = slot['gradient']
                if (not isinstance(value, torch.Tensor) or value.shape != output.shape or
                        value.dtype != output.dtype or not torch.isfinite(value).all()):
                    return {}, 'local_vjp_domain_unavailable'
            for slot in event['inputs']:
                key = (event['call'], event['node'], event['invocation'], slot['path'])
                result['vjp/' + str(key)] = slot['gradient']
        if claim == 'accumulation':
            # Local contributions alone do not certify storage in the accumulated leaf.
            from mirage.runtime.program import leaves
            final = record['final_training_state']
            values = list(leaves((final['gradients'], final['input_gradients'])))
            if not record.get('accumulations') or not values:
                return {}, 'accumulated_leaf_observations_unavailable'
            for path, value in values:
                result['accumulated/' + str(path)] = value
    return result, None


def differential_worker(request, directory):
    import torch
    from mirage.runtime.stateful import execute
    from mirage.runtime.probes import Recorder
    from mirage.artifacts.bundle import load_tensor_record
    original = load_tensor_record(Path(request['original']))
    if request.get('policy') in (PAIRED_DEVICE_POLICY, OBSERVED_DEVICE_POLICY):
        device = request['device']
        side = dict(device=device, execution_attempted=False, completed=False, target_returned=False)
        if device == 'cuda' and not torch.cuda.is_available():
            return dict(side, status='device_unavailable', reason='cuda_unavailable')
        original_device = str(original['config']['device']).split(':')[0]
        placements = {d['device'].split(':')[0] for d in original['initial_metadata']}
        placements.update(d['device'].split(':')[0] for c in original['target_calls'] for d in c.get('metadata', []))
        side['topology'] = 'mixed' if len(placements | {original_device}) > 1 else 'single_device'
        if side['topology'] == 'mixed' and device != original_device:
            return dict(side, status='incomparable', reason='mixed_device_program_has_no_single_device_projection')
        side['execution_attempted'] = True
        try:
            config = dict(original['config'], device=device, device_observation=True)
            record = execute(request['source'], config, Recorder(directory/'trace.jsonl'))
            if device == 'cuda':
                torch.cuda.synchronize()
        except torch.OutOfMemoryError as error:
            return dict(side, status='incomplete', reason='backend_resource_exhausted', diagnostic=str(error))
        except NotImplementedError as error:
            return dict(side, status='backend_unavailable', reason='backend_operation_unavailable', diagnostic=str(error))
        except (RuntimeError, ValueError, TypeError, IndexError, AssertionError) as error:
            partial = directory/'target-exception.pt'
            if request.get('policy') == OBSERVED_DEVICE_POLICY and partial.exists():
                partial.replace(directory/'member.pt')
                return dict(side, status='ready', target_exception=type(error).__name__, diagnostic=str(error))
            return dict(side, status='execution_error', reason='program_or_backend_rejected_replay',
                        exception=type(error).__name__, diagnostic=str(error))
        side.update(completed=record['completed'], target_returned=record['api_executed'],
                    requested_state_attained=record['inspection']['valid'])
        torch.save(record, directory/'member.pt')
        return dict(side, status='ready', reason=None, synchronized=device == 'cuda',
                    alignment_reason=paired_alignment(original, record, frozen=True))
    reason = device_admission(original, request['schemas'])
    if reason:
        return dict(status='incomparable', reason=reason)
    device = request['device']
    if device == 'cuda' and not torch.cuda.is_available():
        return dict(status='incomparable', reason='cuda_unavailable')
    try:
        record = execute(request['source'], dict(original['config'], device=device), Recorder(directory/'trace.jsonl'))
    except NotImplementedError as error:
        return dict(status='incomparable', reason='backend_operation_unavailable', diagnostic=str(error))
    if device == 'cuda':
        torch.cuda.synchronize()
    reason = device_admission(record, request['schemas']) or aligned_device_records(original, record)
    torch.save(record, directory/'member.pt')
    return dict(status='incomparable' if reason else 'ready', reason=reason, device=device,
                synchronized=device == 'cuda', environment=record['environment'])


def compare_saved_devices(request):
    from mirage.artifacts.bundle import load_tensor_record
    observing = request.get('policy') == OBSERVED_DEVICE_POLICY
    if observing or request.get('policy') == PAIRED_DEVICE_POLICY:
        records = [load_tensor_record(Path(path)) for path in request['members']]
        def compare(a, b):
            reason = paired_alignment(a, b)
            if reason:
                return dict(status='incomparable', reason=reason)
            left, left_reason = device_observables(a, request['schemas'])
            right, right_reason = device_observables(b, request['schemas'])
            if left_reason or right_reason:
                return dict(status='incomparable', reason=left_reason or right_reason)
            if left.keys() != right.keys():
                return dict(status='incomparable', reason='observation_paths_changed')
            results = {path: device_comparison(value, right[path]) for path, value in left.items()}
            status = ('incomparable' if any(v['status'] == 'incomparable' for v in results.values()) else
                      'mismatch' if any(v['status'] == 'mismatch' for v in results.values()) else 'agreement')
            return dict(status=status, policy=PAIRED_DEVICE_POLICY, observables=results)
        if observing:
            original = load_tensor_record(Path(request['original'])) if request.get('original') else None
            return observed_comparison(records, request['schemas'], original)
        first = compare(*records[:2])
        if len(records) == 2:
            return first
        second = compare(*records[2:])
        within = [compare(records[i], records[i + 2]) for i in (0, 1)]
        same_mismatch = any(item['status'] == 'mismatch' and
                            second.get('observables', {}).get(path, {}).get('status') == 'mismatch'
                            for path, item in first.get('observables', {}).items())
        stable = same_mismatch and all(item['status'] == 'agreement' for item in within)
        return dict(status='stable_mismatch' if stable else 'unstable_or_unconfirmed',
                    first=first, repeat=second, within_device=within, confirmed_defect=False)
    values = [load_tensor_record(Path(path))['target_calls'][0]['output'] for path in request['members']]
    first = device_comparison(*values[:2])
    if len(values) == 4:
        second = device_comparison(*values[2:])
        within = [device_comparison(values[i], values[i + 2]) for i in (0, 1)]
        stable = first['status'] == second['status'] == 'mismatch' and all(x['status'] == 'agreement' for x in within)
        return dict(status='stable_mismatch' if stable else 'unstable_or_unconfirmed',
                    first=first, repeat=second, within_device=within, confirmed_defect=False)
    return first


def observed_comparison(records, schemas, original=None):
    from mirage.inspector.device_observations import observe_pair

    def assess(a, b):
        raw = observe_pair(a, b, original)
        if not a.get('completed') or not b.get('completed'):
            reason = 'target_execution_incomplete'
        else:
            reason = None
            if original is not None:
                reason = paired_alignment(original, a, frozen=True) or paired_alignment(original, b, frozen=True)
            reason = reason or paired_alignment(a, b)
        status = 'incomparable'
        checks = {}
        if not reason:
            lhs, lreason = device_observables(a, schemas)
            rhs, rreason = device_observables(b, schemas)
            reason = lreason or rreason or ('observation_paths_changed' if lhs.keys() != rhs.keys() else None)
            if not reason and lhs:
                checks = {k:device_comparison(lhs[k], rhs[k]) for k in lhs}
                statuses = {o['status'] for o in checks.values()}
                status = ('incomparable' if 'incomparable' in statuses else
                          'mismatch' if 'mismatch' in statuses else 'agreement')
        return dict(status=status, reason=reason, observables=checks, raw=raw, policy=OBSERVED_DEVICE_POLICY)

    first = assess(*records[:2])
    if len(records) == 2:
        return first
    repeat = assess(*records[2:])
    within = [observe_pair(records[i], records[i+2], original) for i in (0,1)]
    stable_fields = [key for key in first['raw']['differences'] if key in repeat['raw']['differences']
                     and all(x['fields'].get(key, {}).get('status') == 'equal' for x in within)]
    status = first['status']
    if status == 'mismatch':
        local = [assess(records[i], records[i+2]) for i in (0,1)]
        stable = any(v['status'] == repeat['observables'].get(k, {}).get('status') == 'mismatch'
                     for k,v in first['observables'].items()) and all(v['status']=='agreement' for v in local)
        status = 'stable_mismatch' if stable else 'unstable_or_unconfirmed'
    return dict(first, status=status, repeat=repeat, within_device=within,
                stable_observed_fields=stable_fields, confirmed_defect=False)
