"""Runtime-evidence aggregation independent of the tensor backend."""
import json

from mirage.contracts.capabilities import (Evidence, EvidenceStatus, StatePredicate,
                                         PathEvidence, SEMANTIC_COVERAGE_VERSION,
                                         OPTIMIZER_COVERAGE_VERSION,
                                         JOINT_COVERAGE_MAX_OPERANDS, JOINT_COVERAGE_MAX_AXES)

FUNNEL = ("state_plan", "generation", "parse", "static_contract", "initialization", "snapshot",
          "mutation_eligibility", "mutation_application", "target_entry", "target_return",
          "backward", "state", "relation", "audit")


def failure_record(first_blocking, reason, secondary=()):
    if first_blocking not in FUNNEL:
        raise ValueError("Unknown execution funnel stage")
    return {"first_blocking_stage": first_blocking, "reason": reason,
            "secondary_diagnostics": list(secondary)}


def callable_execution_indices(record):
    """Return only observed invocations linked to this configured callable factory."""
    target = record.get('config', {}).get('api')
    registrations = {(kind, f['factory_id'], product)
                     for kind, field in [('compiled_callable_execution','compilation_factories'),
                                         ('factory_result_execution','capture_factories')]
                     for f in record.get(field, ()) if 'factory_id' in f and f.get('factory_target') == target
                     for product in f.get('callable_ids', ())}
    return {c['index'] for c in record.get('target_calls', ())
            if c.get('factory_target') == target and c.get('callable_identity_verified') is True
            and (c.get('kind'), c.get('factory_id'), c.get('callable_id')) in registrations}


def product_use_feedback(record):
    """Keep target-linked use evidence independent of selected goal attainment."""
    calls = {c['index'] for c in record.get('target_calls', ())}
    usages = set()
    gaps = []
    probes = {p['call']: p['identity'] for p in record.get('decorator_probes', ())}
    for event in record.get('context_events', ()):
        call = event.get('factory_call')
        owner = event.get('wrapper_factory_call')
        if owner in calls and event.get('balanced') and event.get('wrapper_owner_group') != event.get('instance_group'):
            usages.add((owner, 'decorated_different_instance'))
        if call not in calls or not event.get('balanced'):
            continue
        mode = 'scope_exit'
        if event.get('reentrant_same_instance'):
            mode = 'decorated_same_instance' if event.get('decorated_entry') else 'direct_same_instance'
        elif event.get('decorated_entry'):
            mode = 'decorated_scope'
        usages.add((call, mode))
        if mode == 'direct_same_instance' and record.get('context_contract') and probes.get(call) == 'same_instance':
            gap = dict(oracle='context_restore', state='scope_reentry', required_use='decorated_same_instance',
                       observed_use=mode, call=call)
            if gap not in gaps:
                gaps.append(gap)
    for event in record.get('product_events', ()):
        if event.get('call') in calls and event.get('event'):
            usages.add((event['call'], event['event']))
    return dict(uses=[dict(call=c, mode=m) for c, m in sorted(usages)], oracle_gaps=gaps)


def inspect_state(record, predicates=()):
    """Caller-supplied expectations are compared only with inspector-owned fields."""
    predicates=[p if isinstance(p,StatePredicate) else StatePredicate(**p) for p in predicates]
    checkpoints = {}
    for checkpoint in record.get('checkpoints', ()):
        name = checkpoint['name']
        # Reusing a semantic name is ambiguous unless the caller specifies an occurrence.
        if name in checkpoints:
            checkpoints[name] = None
        else:
            checkpoints[name] = checkpoint
    attained = []
    for raw in predicates:
        predicate = raw if isinstance(raw, StatePredicate) else StatePredicate(**raw)
        checkpoint = checkpoints.get(predicate.checkpoint)
        fact = Evidence()
        if checkpoint:
            facts = checkpoint.get('state_facts', {})
            key = predicate.dimension + '.' + predicate.field
            if key in facts:
                fact = Evidence(EvidenceStatus.VERIFIED, 'runtime checkpoint observation',
                                facts[key], predicate.checkpoint)
        if predicate.dimension in ('representation', 'data', 'tensor_relation'):
            fact = operand_fact(record, predicate)
        if predicate.dimension in ('gradient_representation', 'optimizer'):
            fact = optimizer_fact(record, predicate)
        attained.append({'call_index': predicate.call_index, 'operand_path': predicate.operand_path,
                         'relation': predicate.relation, 'axis': predicate.axis,
                         'peer_operand_path': predicate.peer_operand_path, 'peer_axis': predicate.peer_axis,
                         'peer_relation': predicate.peer_relation, 'dimension': predicate.dimension, 'field': predicate.field,
                         'parameter_key': predicate.parameter_key,
                         'checkpoint': predicate.checkpoint, 'expected': predicate.expected,
                         'observed': fact.value, 'evidence_status': fact.status.value, 'reason': fact.reason,
                         'attained': predicate.holds(fact)})
    callable_calls = callable_execution_indices(record)
    factory_target = record.get('factory_requires_invocation')
    if factory_target is None:  # The original archives predate runtime adapter identity.
        factory_target = record.get('config', {}).get('api') in ('torch.compile', 'torch.cuda.make_graphed_callables')
    api = bool(record.get('api_executed')) and (not factory_target or bool(callable_calls))
    backward_required=record.get('config',{}).get('requires_backward',True)
    completed = {e['index'] for e in record.get('backward_invocations', ()) if e['completed']}
    nodes = [e for e in record.get('target_backward_nodes', ()) if e.get('invocation') in completed]
    nodes += [e for e in record.get('callable_backward_nodes', ())
              if e.get('call') in callable_calls and e.get('invocation') in completed]
    backward = not backward_required or bool(nodes)
    required_path='callable' if factory_target else 'target'
    for field,kind in (('compiled_execution','executed FX graph'),('captured_execution','factory_result_execution')):
        if any(p.dimension=='execution_mode' and p.field==field and p.expected is True for p in predicates):
            indices={c['index'] for c in record.get('target_calls',()) if c.get('evidence')==kind or c.get('kind')==kind}
            backward=backward and (not backward_required or any(e.get('call') in indices for e in nodes))
            required_path=field+('_callable' if factory_target else '_target')
    claim = record.get('config', {}).get('check_claim')
    history = claim_history(record, claim, callable_calls, completed)
    legal_source = bool(record.get('completed') and record.get('api_executed') and record.get('target_calls'))
    goal = record.get('config', {}).get('state_goal')
    witnesses = training_state_evidence(record, completed, callable_calls, inspect_optimizer(record)) if goal and legal_source else {}
    combinations = []
    for left, right in (('target_backward', 'same_graph_twice'), ('target_backward', 'saved_target_unpacked'),
                        ('target_backward', 'target_update'), ('target_backward', 'restored_update'),
                        ('target_backward', 'overflow_step_skipped'), ('target_backward', 'synchronized_gradient'),
                        ('autocast_target', 'compiled_target'), ('target_recomputed', 'autocast_target')):
        a, b = witnesses.get(left), witnesses.get(right)
        if a and b and a['call'] == b['call']:
            combinations.append([left, right])
    if 'distinct_batch_accumulation' in witnesses:
        combinations.append(['distinct_batches', 'leaf_accumulation'])
    transitions = [unit for unit, witness in witnesses.items() if witness['checkpoint'] in (
        'completed_backward', 'post_update', 'context_exit', 'post_used_restored_state',
        'post_skipped_scaled_step', 'post_synchronized_gradient') or unit in ('target_recomputed', 'distinct_batches')]
    needed = [goal['id']] if goal and goal['id'] != 'bootstrap' else []
    measured = all(name in witnesses for name in needed)
    feedback = product_use_feedback(record)
    required_use = (goal or {}).get('required_use')
    use_reached = not required_use or any(u['mode'] == required_use for u in feedback['uses'])
    goal_reached = bool(use_reached and api and backward and history['attained'] and all(p['attained'] for p in attained) and measured)
    valid = bool(record.get('completed') and goal_reached)
    return {'valid': valid, 'legal_source': legal_source, 'product_use': feedback, 'claim_history': history,
            'observed_products': record.get('observed_products', []),
            'state_goal': dict(id=goal['id'], family=goal['family'], attained=valid,
                               missing=[name for name in needed if name not in witnesses] + ([required_use] if not use_reached else [])) if goal else None,
            'training_state_units': sorted(witnesses), 'training_state_evidence': witnesses,
            'training_state_combinations': combinations, 'training_state_transitions': sorted(transitions),
            'api_executed': api, 'required_backward_executed': backward, 'required_backward_path':required_path,
            'callable_execution_required': factory_target, 'required_callable_executed': not factory_target or bool(callable_calls),
            'predicates': attained, 'target_state_reached': goal_reached if goal else all(p['attained'] for p in attained),
            'transition_signature': record.get('transition_signature', []),
            'correctness': 'unassessed', 'semantic_coverage': semantic_coverage(record, valid=valid),
            'optimizer_coverage': inspect_optimizer(record, valid=valid)}


def claim_history(record, claim, callable_calls, completed):
    calls = {c['index'] for c in record.get('target_calls', ())}
    events = [e for e in record.get('product_events', ()) if e['call'] in calls]
    if claim == 'scope_exit':
        events = [e for e in record.get('context_events', ()) if e.get('balanced') and e.get('factory_call') in calls]
    elif claim == 'callable_use':
        events = [e for e in events if e['event'] == 'callable_return'] + sorted(callable_calls)
    elif claim in ('callback_trigger', 'async_completion', 'persistent_step'):
        kind = {'callback_trigger':'callback_return', 'async_completion':'wait_completed',
                'persistent_step':'persistent_step_effect'}[claim]
        events = [e for e in events if e['event'] == kind]
    elif claim == 'accumulation':
        contributions = {(e['invocation'], e['call'], tuple(e['path'])) for e in record.get('target_operand_gradients', ())}
        events = [e for e in record.get('accumulations', ())
                  if e.get('leaf_calls') and {e['first'], e['second']} <= completed and e['second'] == e['first'] + 1
                  and all(any((inv, call, tuple(path)) in contributions for call, path in e['leaf_calls'])
                          for inv in (e['first'], e['second']))]
    else:
        return dict(claim=claim, attained=True, reason='target, backward and predicates validated separately')
    return dict(claim=claim, attained=bool(events), events=events, checkpoint='final',
                reason='linked runtime history observed' if events else 'required linked runtime history missing')


def training_state_evidence(record, completed, callable_calls, optimizer):
    """Return target-linked units. Raw observation indices are evidence, never novelty keys."""
    calls = {c['index']: c for c in record.get('target_calls', ())}
    facts = {}
    def mark(unit, call, checkpoint='target_return', **evidence):
        facts.setdefault(unit, dict(call=call, checkpoint=checkpoint, **evidence))

    device_kind = str(record['config'].get('device', 'cpu')).split(':')[0]
    samples, graph_nodes, recomputes = {}, {}, {}
    for index, call in calls.items():
        context = call.get('entry_context', {})
        if context.get('inference_enabled') is True:
            mark('inference_target', index, 'target_entry')
        if context.get(device_kind + '_enabled') is True:
            mark('autocast_target', index, 'target_entry')
        if call.get('target_rng_consumed') is True:
            mark('target_rng_consumed', index)
        for resource in call.get('product_resources', call.get('registered_resources', ())):
            if 'module_buffer' in resource:
                mark('module_buffer_operand', index, 'target_entry', path=resource['path'], **resource['module_buffer'])
            if 'input_key' in resource:
                path = tuple(resource['path'])
                value = call.get('product_inputs', call.get('inputs'))
                for component in path:
                    value = value[str(component)] if isinstance(value, dict) else value[component]
                slot = (call.get('factory_id'), call.get('callable_id'), path)
                prior = samples.get(slot)
                # Compare small, same-domain *registered inputs* at the same call slot.
                # A second call over identical values is insufficient evidence.
                if prior is not None and prior[1] != resource['input_key'] and index == prior[0] + 1:
                    old = prior[2]
                    supported = str(value.dtype) in ('torch.bool','torch.uint8','torch.int8','torch.int16',
                        'torch.int32','torch.int64','torch.float16','torch.bfloat16','torch.float32','torch.float64',
                        'torch.complex64','torch.complex128')
                    if (supported and value.shape == old.shape and value.dtype == old.dtype and value.numel() <= 4096
                            and bool(value.isfinite().all()) and bool(old.isfinite().all())
                            and not value.equal(old)):
                        mark('distinct_batches', index, 'target_entry', previous_call=prior[0], path=list(path))
                samples[slot] = index, resource['input_key'], value
        frame = call.get('recompute_frame')
        if frame:
            group, number = frame
            if number == 0:
                recomputes[group] = index
            elif group in recomputes and call.get('backward_invocation') in completed:
                mark('target_recomputed', index, 'completed_backward', original_call=recomputes[group],
                     invocation=call['backward_invocation'])
                if context.get(device_kind + '_enabled') is True:
                    mark('autocast_recomputed_target', index, 'completed_backward',
                         original_call=recomputes[group], invocation=call['backward_invocation'])
        if (call.get('execution_mode') == 'compiled' and call.get('evidence') == 'executed FX graph'
                and (call.get('kind') != 'compiled_callable_execution' or index in callable_calls)):
            mark('compiled_target', index)
            if context.get(device_kind + '_enabled') is True:
                mark('autocast_compiled_target', index)
        if index in callable_calls and call.get('execution_mode') == 'captured' and call.get('completed_replays', 0) > 0:
            mark('captured_target_replay', index)

    nodes = list(record.get('target_backward_nodes', ())) + [
        n for n in record.get('callable_backward_nodes', ()) if n.get('call') in callable_calls]
    for node in nodes:
        idx, invocation = node.get('call'), node.get('invocation')
        if idx in calls and invocation in completed:
            mark('target_backward', idx, 'completed_backward', invocation=invocation)
            if type(node.get('node')) is not int:
                continue
            group = graph_nodes.setdefault((idx, node['node']), set())
            group.add(invocation)
            if len(group) > 1:
                mark('same_graph_twice', idx, 'completed_backward', invocations=sorted(group))
    history = claim_history(record, 'accumulation', callable_calls, completed)
    batches = facts.get('distinct_batches')
    for event in history.get('events', ()):
        owner = event['leaf_calls'][0][0]
        if owner in calls:
            mark('leaf_accumulation', owner, 'completed_backward', first=event['first'], second=event['second'])
        if batches:
            keys = {(g['invocation'], g['call'], tuple(g['path'])) for g in record.get('target_operand_gradients', ())}
            leaf_calls = {(idx, tuple(path)) for idx, path in event['leaf_calls']}
            if all(any((inv, call, path) in keys for idx, path in leaf_calls if call == idx)
                   for inv, call in [(event['first'], batches['previous_call']), (event['second'], batches['call'])]):
                mark('distinct_batch_accumulation', batches['call'], 'completed_backward',
                     previous_call=batches['previous_call'], first=event['first'], second=event['second'])
    for event in record.get('saved_tensor_events', ()):
        index, invocation = event.get('call'), event.get('invocation')
        if (index in calls and invocation in completed and event.get('stored_device') == 'cpu'
                and event.get('source_device') == event.get('restored_device')
                and any(n.get('call') == index and n.get('invocation') == invocation for n in nodes)):
            mark('saved_target_unpacked', index, 'completed_backward', invocation=invocation)
    operations = {'skipped_scaled_step': 'overflow_step_skipped', 'used_restored_state': 'restored_update',
                  'synchronized_gradient': 'synchronized_gradient'}
    for event in record.get('state_operations', ()):
        idx, invocation, kind = event.get('call'), event.get('invocation'), event.get('event')
        if idx not in calls or invocation not in completed or kind not in operations or not event.get('parameter'):
            continue
        if kind == 'synchronized_gradient' and event.get('world_size', 0) < 2:
            continue
        if (any(node.get('call') == idx and node.get('invocation') == invocation for node in nodes)
                and any(g.get('call') == idx and g.get('invocation') == invocation and g.get('path') == event.get('path')
                        for g in record.get('target_operand_gradients', ()))):
            mark(operations[kind], idx, 'post_' + kind, parameter=event['parameter'], invocation=invocation)
    for event in record.get('context_events', ()):
        index = event.get('factory_call')
        if (index in calls and event.get('balanced') and event.get('reentrant_same_instance')
                and event.get('instance_group') is not None):
            mark('scope_reentry', index, 'context_exit', group=event.get('instance_group'))
    for event in record.get('product_events', ()):
        idx = event.get('call')
        if idx not in calls or event.get('event') != 'persistent_step_effect':
            continue
        if event.get('product_kind') == 'optimizer' and event.get('persistent_state'):
            mark('produced_optimizer_state', idx, 'post_update')
        if event.get('product_kind') == 'scheduler' and event.get('learning_rate_changed'):
            mark('produced_scheduler_step', idx, 'post_update')
    for witness in optimizer.get('witnesses', ()):
        if witness.get('completed') and witness['call_index'] in calls:
            mark('target_update', witness['call_index'], witness['post_checkpoint'], parameter=witness['parameter_key'])
    return facts


def _optimizer_path(value):
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(part) not in (str, int) for part in value)):
        return None
    return tuple(value)


def inspect_optimizer(record, *, valid=False):
    """Verify the bounded target-gradient/update chain without tensor computation.

    Runtime native-domain/effect certificates and observed event identities are
    required together. This is consumed-gradient representation, not evidence
    that backward wrote into the transformed representation.
    """
    evidence = record.get('optimizer_evidence') or {}
    steps = evidence.get('steps', [])
    limitations = list(evidence.get('limitations', []))[:16]
    witnesses = []
    supported = evidence.get('version') == 'optimizer-update-1'
    bounded = (isinstance(steps, list) and len(steps) <= 1
               and evidence.get('dropped_events', 0) == 0)
    registry = evidence.get('registry') or {}
    parameters = registry.get('parameters') or []
    keys = [p.get('parameter_key') for p in parameters]
    registry_ok = (registry.get('optimizer_id') == 'optimizer:0' and bool(parameters)
                   and all(isinstance(key, str) and key for key in keys) and len(set(keys)) == len(keys)
                   and all(p.get('parameter_id') == 'parameter:' + p['parameter_key'] for p in parameters))
    completed = {e.get('index') for e in record.get('backward_invocations', []) if e.get('completed') is True}
    calls = record.get('target_calls', [])
    nodes = record.get('target_backward_nodes', [])
    bindings = record.get('target_parameter_bindings', [])
    source_ok = record.get('optimizer_source_certificate') == 'explicit_optimizer_gradient_update_v1'
    if (supported and bounded and registry_ok and source_ok and not limitations
            and record.get('completed') is True and record.get('api_executed') is True):
        for step in steps:
            pre_name, post_name = 'optimizer:0/step:0/pre_update', 'optimizer:0/step:0/post_update'
            if (type(step.get('index')) is not int or step['index'] != 0
                    or step.get('optimizer_id') != 'optimizer:0' or step.get('completed') is not True
                    or step.get('pre_checkpoint') != pre_name or step.get('post_checkpoint') != post_name
                    or step.get('domain', {}).get('status') != 'verified'):
                continue
            pre, post = step.get('pre') or {}, step.get('post') or {}
            if any(not isinstance(state.get(field), dict) or set(state[field]) != set(keys)
                   for state in (pre, post) for field in ('parameters', 'gradients', 'gradient_metadata', 'slots')):
                continue
            mutation = step.get('mutation')
            if mutation is not None and (mutation.get('status') != 'verified'
                    or mutation.get('logical_equal') is not True or mutation.get('parameter_key') not in keys
                    or mutation.get('action') not in ('canonical', 'strided', 'offset')):
                continue
            origins = step.get('gradient_provenance', [])
            for origin in origins:
                key, call, path = origin.get('parameter_key'), origin.get('call_index'), origin.get('operand_path')
                path_key = _optimizer_path(path)
                if (key not in keys or origin.get('parameter_id') != 'parameter:' + key
                        or origin.get('status') != 'verified' or type(call) is not int
                        or origin.get('unchanged_to_pre_step') is not True
                        or path_key is None
                        or any(type(origin.get(field)) is not int or origin[field] < 0 for field in ('node', 'invocation'))
                        or origin.get('invocation') not in completed
                        or sum(p.get('parameter_key') == key for p in origins) != 1):
                    continue
                identity = {k: origin.get(k) for k in ('node', 'invocation')}
                accumulations = [e for e in evidence.get('accumulations', [])
                    if e.get('index') == origin.get('post_accumulate_event') and e.get('parameter_key') == key
                    and e.get('parameter_id') == origin['parameter_id'] and e.get('invocation') == origin['invocation']
                    and e.get('started_none') is True and e.get('gradient') is not None]
                gradients = record.get('gradient_evidence') or {}
                direct = [e for e in gradients.get('events', []) if e.get('call') == call
                    and all(e.get(k) == v for k, v in identity.items()) and e.get('status') == 'complete']
                if (len(accumulations) != 1 or gradients.get('version') != 'gradient-evidence-1'
                        or len(direct) != 1 or sum(_optimizer_path(v.get('path')) == path_key and v.get('gradient') is not None
                            for v in direct[0].get('inputs', [])) != 1):
                    continue
                matching_nodes = [i for i, event in enumerate(nodes) if event.get('call') == call
                                  and all(event.get(k) == v for k, v in identity.items())]
                matching_calls = [c for c in calls if c.get('index') == call]
                matching_bindings = [b for b in bindings if b.get('parameter_key') == key
                    and b.get('parameter_id') == origin['parameter_id'] and b.get('call_index') == call
                    and _optimizer_path(b.get('operand_path')) == path_key]
                operand_events = [e for e in record.get('target_operand_gradients', [])
                    if e.get('call') == call and _optimizer_path(e.get('path')) == path_key
                    and all(e.get(k) == v for k, v in identity.items())]
                if (len(matching_nodes) != 1 or len(matching_calls) != 1 or len(matching_bindings) != 1
                        or not operand_events or sum(_optimizer_path(d.get('path')) == path_key
                            for d in matching_calls[0].get('metadata', [])) != 1):
                    continue
                metadata = pre['gradient_metadata'].get(key)
                if (not isinstance(metadata, dict) or metadata.get('layout') != 'torch.strided'
                        or metadata.get('dtype') != 'torch.float64' or metadata.get('device') != 'cpu'
                        or _strides(metadata) is None or type(metadata.get('offset')) is not int
                        or metadata['offset'] < 0 or pre['gradients'][key] is None):
                    continue
                witnesses.append({'parameter_key': key, 'parameter_id': origin['parameter_id'],
                    'call_index': call, 'operand_path': list(path), 'step_index': 0,
                    'pre_checkpoint': pre_name, 'post_checkpoint': post_name, 'metadata': metadata,
                    'completed': True, 'provenance': dict(origin),
                    'relations': {'target_gradient_update': {'status': 'verified', 'events': matching_nodes}}})
    verified_steps = sorted({w['step_index'] for w in witnesses})
    eligible = bool(valid and record.get('completed') is True and witnesses)
    signatures = []
    if eligible:
        for witness in witnesses:
            metadata = witness['metadata']
            unit = {'version': OPTIMIZER_COVERAGE_VERSION, 'api': record.get('config', {}).get('api'),
                    'operand_path': witness['operand_path'], 'optimizer': registry.get('native_type'),
                    'phase': 'consumed_gradient_then_completed_update',
                    'gradient_representation': {'contiguous': metadata.get('properties', {}).get('contiguous'),
                                                'offset_nonzero': metadata['offset'] != 0,
                                                'dtype': metadata['dtype'], 'layout': metadata['layout']},
                    'relations': ['target_gradient_update']}
            if unit not in signatures:
                signatures.append(unit)
    return {'version': OPTIMIZER_COVERAGE_VERSION, 'eligible': eligible, 'witnesses': witnesses,
            'eligible_chains': witnesses if eligible else [], 'signatures': signatures,
            'observed_units': len(signatures), 'verified_steps': verified_steps,
            'target_update_count': len(verified_steps),
            'count_verified': bool(supported and bounded and not limitations and len(verified_steps) == len(steps)),
            'limitations': limitations, 'denominator': None, 'correctness': 'unassessed'}


def optimizer_witness_fact(witnesses, predicate):
    predicate = predicate if isinstance(predicate, StatePredicate) else StatePredicate(**predicate)
    path = _optimizer_path(predicate.operand_path)
    if path is None:
        return Evidence(reason='optimizer witness requires a scoped operand path')
    phase = 'pre_checkpoint' if predicate.dimension == 'gradient_representation' else 'post_checkpoint'
    selected = [w for w in witnesses if w.get(phase) == predicate.checkpoint
                and w.get('parameter_key') == predicate.parameter_key and w.get('call_index') == predicate.call_index
                and _optimizer_path(w.get('operand_path')) == path
                and w.get('completed') is True
                and w.get('relations', {}).get('target_gradient_update', {}).get('status') == 'verified']
    if len(selected) != 1:
        return Evidence(reason='missing or ambiguous target-gradient update witness')
    metadata = selected[0].get('metadata') or {}
    value = (True if predicate.dimension == 'optimizer' else
             metadata.get('properties', {}).get('contiguous') if predicate.field == 'contiguous' else
             metadata['offset'] != 0 if type(metadata.get('offset')) is int and metadata['offset'] >= 0 else None)
    if type(value) is not bool:
        return Evidence(reason='consumed gradient representation is unobserved')
    return Evidence(EvidenceStatus.VERIFIED, 'native optimizer boundary and linked completed backward',
                    value, predicate.checkpoint)


def optimizer_fact(record, predicate):
    predicate = predicate if isinstance(predicate, StatePredicate) else StatePredicate(**predicate)
    coverage = inspect_optimizer(record)
    if predicate.dimension == 'optimizer' and predicate.field == 'target_update_count':
        if (coverage['count_verified'] and record.get('completed') is True
                and sum(c.get('name') == 'final' for c in record.get('checkpoints', [])) == 1):
            return Evidence(EvidenceStatus.VERIFIED, 'count of complete verified target-gradient update chains',
                            coverage['target_update_count'], 'final')
        return Evidence(reason='complete target-linked optimizer count is unavailable')
    return optimizer_witness_fact(coverage['witnesses'], predicate)


def input_coverage(record):
    requested = record.get('config', {}).get('input_categories', ())
    observed = {}; measurements=[]
    for call in record.get('target_calls', ()):
        for operand in call.get('input_properties', ()):
            measurements.append({'call':call['index'],'path':operand['path'],'facts':operand['facts']})
            for key, value in operand['facts'].items():
                if value is True:
                    observed.setdefault(key, []).append({'call': call['index'], 'path': operand['path']})
    return {'requested': list(requested), 'target_observed': observed, 'operand_measurements':measurements,
            'relation_eligible': {}, 'oracle_eligibility': 'not_established_by_input_coverage'}


def input_category_feedback(outcome):
    """Refine coverage independently of correctness; never repair a suspect to Pass."""
    if not outcome.get('inspection',{}).get('valid') or outcome.get('verdict')=='suspected_failure': return None
    coverage=outcome.get('input_coverage',{})
    measurements=coverage.get('operand_measurements',[])
    if not measurements: return None
    observed=set(coverage.get('target_observed',{}))
    if outcome.get('verdict')=='pass': observed.update(outcome.get('followup_input_coverage',{}).get('target_observed',{}))
    missing=[c for c in coverage.get('requested',[]) if c not in observed]
    # No floating magnitude predicate is asserted applicable to integer operands.
    missing=[c for c in missing if c not in ('small_finite','large_finite') or
             any('small_threshold' in m['facts'] for m in measurements)]
    if not missing: return None
    return {'missing_categories':missing,'actual_operand_measurements':measurements,
            'purpose':'Input coverage only. The existing correctness verdict is unchanged; generate a new legal initialization shared by both MR members. No expected answer or training sequence is supplied.'}


def semantic_coverage(record, *, valid=False):
    """Join runtime witnesses only within a call, operand and checkpoint boundary.

    Signatures describe boundary representation and proven path relations. They
    never assert that a representation was saved or consumed during backward.
    Counts are observed units, not a fraction of a theoretical state space.
    """
    events = {'outputs': record.get('target_gradients', []),
              'nodes': record.get('target_backward_nodes', []),
              'operands': record.get('target_operand_gradients', []),
              'callable_nodes': record.get('callable_backward_nodes', [])}
    callable_calls = callable_execution_indices(record)
    environment = {k: record.get('environment', {}).get(k) for k in
                   ('torch', 'cuda_runtime', 'device', 'deterministic_algorithms', 'matmul_precision')}
    witnesses, signatures, unary_units = [], set(), {}
    checkpoints = record.get('checkpoints', [])
    names = [c['name'] for c in checkpoints]
    for checkpoint in checkpoints:
        if (checkpoint['name'].startswith('optimizer:') or names.count(checkpoint['name']) != 1
                or 'evidence_offsets' not in checkpoint):
            continue
        # New lifecycle instrumentation has its own coverage version. Adding it
        # must not manufacture ordinary operand-v2 novelty for an unchanged call.
        state_facts = {key: value for key, value in checkpoint.get('state_facts', {}).items()
                       if not key.startswith('optimizer.')}
        offsets = checkpoint['evidence_offsets']
        completed = set(checkpoint.get('completed_backward_invocations', []))
        for call in record.get('target_calls', [])[:offsets['calls']]:
            index = call['index']
            scoped = {kind: [(i, event) for i, event in enumerate(items[:offsets.get(kind, 0)])
                             if event['call'] == index and event.get('invocation') in completed]
                      for kind, items in events.items()}
            for operand in call.get('metadata', []) or [None]:
                path = list(operand['path']) if operand else None
                operand_events = [(i, e) for i, e in scoped['operands'] if e['path'] == path]
                relations = {}
                for name, selected in (('output_gradient', scoped['outputs']),
                                       ('target_backward_node', scoped['nodes']),
                                       ('operand_gradient_contribution', operand_events),
                                       ('callable_backward_node', scoped['callable_nodes'] if index in callable_calls else [])):
                    evidence = PathEvidence(EvidenceStatus.VERIFIED if selected else EvidenceStatus.UNKNOWN,
                                            'completed backward runtime events' if selected else 'no attributable runtime event',
                                            tuple(i for i, _ in selected))
                    relations[name] = dict(evidence.record(), scope='operand' if name == 'operand_gradient_contribution' else ('callable' if name == 'callable_backward_node' else 'call'))
                relations['backward_representation'] = PathEvidence(
                    EvidenceStatus.UNSUPPORTED, 'saved tensor representation is not instrumented').record()
                witness = {'call': index, 'operand_path': path, 'checkpoint': checkpoint['name'],
                           'metadata': operand, 'execution_mode': call.get('execution_mode', 'unknown'),
                           'backend': call.get('backend'), 'relations': relations,
                           'entry_context': call.get('entry_context'),
                           'state_facts': state_facts,
                           'backward_bindings': call.get('backward_bindings', []),
                           'input_properties': next((p['facts'] for p in call.get('input_properties', [])
                               if list(p['path']) == path), {})}
                witnesses.append(witness)
                if not valid:
                    continue
                # Per-call IDs, raw sizes/strides and exact event counts are not novelty.
                representation = None
                data = None
                if operand:
                    properties = operand.get('properties', {})
                    representation = {k: properties.get(k) for k in ('negative', 'conjugate', 'contiguous', 'mkldnn')}
                    representation.update(layout=operand.get('layout'), device=operand.get('device'),
                        offset='unknown' if operand.get('offset') is None else ('zero' if operand['offset'] == 0 else 'nonzero'),
                        zero_stride=_zero_stride(operand),
                        aliases=None if operand.get('storage_group') is None else [list(other['path']) for other in call.get('metadata', [])
                                 if other['path'] != operand['path'] and operand.get('storage_group') is not None
                                 and other.get('storage_group') == operand['storage_group']],
                        requires_grad=operand.get('requires_grad'), leaf=operand.get('leaf'),
                        axis_classes=[dict(axis=axis, **fields) for axis in _selected_axes(operand)
                                      if (fields := _axis_fields(operand, axis)) is not None])
                    facts = next((p['facts'] for p in call.get('input_properties', [])
                                  if list(p['path']) == path), {})
                    shape = _shape(operand)
                    data = {'dtype': operand.get('dtype'), 'rank': None if shape is None else len(shape),
                            'empty': None if shape is None else 0 in shape,
                            'singleton': None if shape is None else 1 in shape,
                            'categories': sorted(k for k, v in facts.items() if v is True)}
                state = {k: ('multiple' if v > 1 else v) if type(v) is int and k.endswith('_count') else v
                         for k, v in state_facts.items()}
                unit = {'version': SEMANTIC_COVERAGE_VERSION, 'unit_kind': 'unary', 'api': record.get('config', {}).get('api'),
                        'operand_path': path, 'checkpoint': checkpoint['name'], 'representation': representation,
                        'data': data, 'state': state, 'execution_mode': witness['execution_mode'],
                        'backend': witness['backend'], 'environment': environment, 'entry_context': witness['entry_context'],
                        'relations': sorted(k for k, v in relations.items() if v['status'] == EvidenceStatus.VERIFIED)}
                signatures.add(json.dumps(unit, sort_keys=True, separators=(',', ':')))
                unary_units[(checkpoint['name'], index, tuple(path or ()))] = unit
    joint_units, measurement_counts = _joint_coverage(witnesses, unary_units, valid)
    signatures.update(json.dumps(unit, sort_keys=True, separators=(',', ':')) for unit in joint_units)
    return {'version': SEMANTIC_COVERAGE_VERSION, 'instrumentation': 'output-and-direct-node-hooks',
            'environment': environment, 'events': events, 'witnesses': witnesses,
            'signatures': [json.loads(s) for s in sorted(signatures)], 'eligible': valid,
            'observed_units': len(signatures),
            'observed_unary_units': sum(json.loads(s)['unit_kind'] == 'unary' for s in signatures),
            'observed_joint_units': sum(json.loads(s)['unit_kind'] == 'joint' for s in signatures),
            'measurement_counts': measurement_counts,
            'joint_bounds': {'operands_per_call_checkpoint': JOINT_COVERAGE_MAX_OPERANDS,
                             'axes_per_operand': JOINT_COVERAGE_MAX_AXES,
                             'selection': 'typed-path order; first and last axes'},
            'denominator': None, 'correctness': 'unassessed'}


def _shape(metadata):
    shape = metadata.get('shape')
    if not isinstance(shape, (tuple, list)) or any(type(n) is not int or n < 0 for n in shape):
        return None
    return shape


def _strides(metadata):
    shape, strides = _shape(metadata), metadata.get('strides')
    if (shape is None or not isinstance(strides, (tuple, list)) or len(strides) != len(shape)
            or any(type(stride) is not int for stride in strides)):
        return None
    return strides


def _zero_stride(metadata):
    strides = _strides(metadata)
    return None if strides is None else any(
        n > 1 and stride == 0 for n, stride in zip(metadata['shape'], strides))


def _axis_fields(metadata, axis):
    shape, strides = _shape(metadata), _strides(metadata)
    if shape is None or type(axis) is not int or not 0 <= axis < len(shape) or strides is None:
        return None
    return {'zero_stride': shape[axis] > 1 and strides[axis] == 0,
            'singleton_stride': 'not_singleton' if shape[axis] != 1 else (
                'zero' if strides[axis] == 0 else 'unit' if strides[axis] == 1 else 'other')}


def _selected_axes(metadata):
    shape = _shape(metadata)
    return tuple(dict.fromkeys((0, len(shape) - 1)))[:JOINT_COVERAGE_MAX_AXES] if shape else ()


def _joint_groups(witnesses):
    groups = {}
    for witness in witnesses:
        path = witness.get('operand_path')
        if not isinstance(witness.get('metadata'), dict) or not path:
            continue
        groups.setdefault((witness['checkpoint'], witness['call']), []).append(witness)
    for key, group in groups.items():
        paths = [tuple(w['operand_path']) for w in group]
        unique = [w for w in group if paths.count(tuple(w['operand_path'])) == 1]
        unique.sort(key=lambda w: tuple((0, key) if type(key) is int else (1, key)
                                      for key in w['operand_path']))
        yield key, unique[:JOINT_COVERAGE_MAX_OPERANDS], max(0, len(unique) - JOINT_COVERAGE_MAX_OPERANDS), len(group) - len(unique)


def joint_candidates(witnesses):
    """Bounded same-call/checkpoint pairs, shared by planning and measurement.

    Distinct paths may name the same object. No pair implies independent operand
    gradients, occupied-coordinate overlap, legality or MR applicability.
    """
    from itertools import combinations, product
    for _, group, _, _ in _joint_groups(witnesses):
        for primary, peer in combinations(group, 2):
            axes, peer_axes = _selected_axes(primary['metadata']), _selected_axes(peer['metadata'])
            coordinates = product(axes, peer_axes) if axes and peer_axes else ((None, None),)
            for axis, peer_axis in coordinates:
                yield {'primary': primary, 'peer': peer, 'axis': axis, 'peer_axis': peer_axis}


def witness_fact(witnesses, predicate):
    """Evaluate a predicate using metadata and independently scoped path evidence."""
    predicate = predicate if isinstance(predicate, StatePredicate) else StatePredicate(**predicate)
    def select(path, relation):
        matches = [w for w in witnesses if w['checkpoint'] == predicate.checkpoint
                   and w['call'] == predicate.call_index and w['operand_path'] == list(path)]
        if len(matches) != 1 or not isinstance(matches[0].get('metadata'), dict):
            return None, 'missing or ambiguous operand witness'
        witness = matches[0]
        if relation and witness.get('relations', {}).get(relation, {}).get('status') != 'verified':
            return None, 'required path relation not established'
        return witness, None
    witness, reason = select(predicate.operand_path, predicate.relation)
    if reason:
        return Evidence(reason=reason)
    metadata = witness['metadata']
    if predicate.dimension == 'tensor_relation':
        peer, reason = select(predicate.peer_operand_path, predicate.peer_relation)
        if reason:
            return Evidence(reason='peer ' + reason)
        other = peer['metadata']
        if predicate.field == 'shared_storage':
            # Sparse identities may name only values storage, not index buffers.
            if any(md.get('layout') != 'torch.strided' or md.get('properties', {}).get('mkldnn') is not False
                   for md in (metadata, other)):
                return Evidence(reason='whole-operand storage relation requires observed dense strided tensors')
            groups = (metadata.get('storage_group'), other.get('storage_group'))
            if any(type(group) is not int or group < 0 for group in groups):
                return Evidence(reason='storage identity not observed for both operands')
            value = groups[0] == groups[1]
        else:
            shape, peer_shape = _shape(metadata), _shape(other)
            if (shape is None or peer_shape is None or predicate.axis >= len(shape)
                    or predicate.peer_axis >= len(peer_shape)):
                return Evidence(reason='selected extent not observed')
            left, right = shape[predicate.axis], peer_shape[predicate.peer_axis]
            value = 'lt' if left < right else 'gt' if left > right else 'eq'
    else:
        properties = metadata.get('properties', {})
        fields = dict(properties, **{k: metadata.get(k) for k in
                      ('dtype', 'device', 'layout', 'shape', 'strides', 'requires_grad', 'leaf')})
        fields['offset_nonzero'] = None if metadata.get('offset') is None else metadata['offset'] != 0
        fields['zero_stride'] = _zero_stride(metadata)
        if predicate.axis is not None:
            selected = _axis_fields(metadata, predicate.axis)
            if selected is None:
                return Evidence(reason='selected stride not observed')
            fields.update(selected)
        if predicate.field == 'aliases' and metadata.get('storage_group') is not None:
            fields['aliases'] = [w['operand_path'] for w in witnesses
                if w['checkpoint'] == predicate.checkpoint and w['call'] == predicate.call_index
                and w['operand_path'] != list(predicate.operand_path)
                and (w.get('metadata') or {}).get('storage_group') == metadata['storage_group']]
        if predicate.dimension == 'data':
            fields = dict(witness.get('input_properties', {}), dtype=metadata.get('dtype'), shape=metadata.get('shape'))
        value = fields.get(predicate.field)
        if value is None:
            return Evidence(reason='operand field not instrumented')
    return Evidence(EvidenceStatus.VERIFIED, 'call-bound runtime operand observation', value, predicate.checkpoint)


def operand_fact(record, predicate):
    return witness_fact(semantic_coverage(record)['witnesses'], predicate)


def _joint_coverage(witnesses, unary_units, valid):
    from itertools import combinations
    counts = {'joint_candidate_units': 0, 'joint_measured_units': 0, 'joint_unknown_units': 0,
              'omitted_operands': 0, 'ambiguous_operands': 0, 'omitted_axis_pairs': 0}
    for _, group, omitted, ambiguous in _joint_groups(witnesses):
        counts['omitted_operands'] += omitted
        counts['ambiguous_operands'] += ambiguous
        for left, right in combinations(group, 2):
            shapes = (_shape(left['metadata']), _shape(right['metadata']))
            if all(shapes):
                counts['omitted_axis_pairs'] += len(shapes[0]) * len(shapes[1]) - len(
                    _selected_axes(left['metadata'])) * len(_selected_axes(right['metadata']))
    units = []
    for candidate in joint_candidates(witnesses):
        counts['joint_candidate_units'] += 1
        left, right = candidate['primary'], candidate['peer']
        axis, peer_axis = candidate['axis'], candidate['peer_axis']
        common = dict(dimension='tensor_relation', checkpoint=left['checkpoint'],
                      call_index=left['call'], operand_path=left['operand_path'], peer_operand_path=right['operand_path'])
        storage = witness_fact(witnesses, StatePredicate(field='shared_storage', expected=True, **common))
        measured = {}
        if storage.established:
            measured['shared_storage'] = storage.value
        if axis is not None and peer_axis is not None:
            extent = witness_fact(witnesses, StatePredicate(field='extent_order', expected='eq',
                                                           axis=axis, peer_axis=peer_axis, **common))
            if extent.established:
                measured['extent_order'] = extent.value
            for key, witness, selected_axis in (('primary_axis', left, axis), ('peer_axis', right, peer_axis)):
                fields = _axis_fields(witness['metadata'], selected_axis)
                if fields is not None:
                    measured[key] = dict(axis=selected_axis, **fields)
        if not measured:
            counts['joint_unknown_units'] += 1
            continue
        counts['joint_measured_units'] += 1
        if not valid:
            continue
        selected = [unary_units[(w['checkpoint'], w['call'], tuple(w['operand_path']))] for w in (left, right)]
        unit = {k: selected[0][k] for k in ('version', 'api', 'checkpoint', 'state', 'execution_mode',
                                         'backend', 'environment', 'entry_context')}
        unit.update(unit_kind='joint', operands=[{k: u[k] for k in
                    ('operand_path', 'representation', 'data', 'relations')} for u in selected],
                    axis=axis, peer_axis=peer_axis, tensor_relations=measured)
        units.append(unit)
    return units, counts
