"""Conditional semantic-family relations; API admission is independent of this catalog."""
from mirage.contracts.capabilities import Evidence, EvidenceStatus, RelationSpec

RELATIONS = {
    'optimizer_gradient_representation': RelationSpec(
        'optimizer_gradient_representation', 1,
        ('valid_executions', 'native_optimizer_domain', 'optimizer_source_effects',
         'target_gradient_update', 'verified_gradient_transformation', 'complete_update_observables'),
        ('same_program', 'same_initial_state', 'same_environment', 'same_optimizer_registry',
         'same_pre_update_state', 'aligned_optimizer_step', 'permitted_case_difference'),
        ('updated_parameters', 'optimizer_slots'), ('gradient_representation',),
        'comparison', 'dtype_scaled_v1'),
    'representation': RelationSpec(
        'representation', 1,
        ('valid_executions', 'logical_value_semantics', 'deterministic', 'read_only_target',
         'preserved_coordinates', 'finite_observations', 'numerical_domain'),
        ('same_initial_state', 'same_program', 'same_target_inputs', 'aligned_checkpoints'),
        ('target_outputs', 'gradients'), ('representation',), 'comparison', 'dtype_scaled_v1'),
    'materialized_value_effect': RelationSpec(
        'materialized_value_effect', 2,
        ('valid_executions', 'logical_value_semantics', 'deterministic', 'matching_write_roles',
         'preserved_coordinates', 'finite_observations', 'numerical_domain'),
        ('same_initial_state', 'same_program', 'same_target_inputs', 'aligned_checkpoints'),
        ('target_outputs', 'destinations', 'preserved_guard'), ('representation', 'output_storage_identity'), 'comparison', 'dtype_scaled_v1'),
    'context_restore': RelationSpec(
        'context_restore', 1, ('valid_executions', 'context_restore_contract', 'balanced_scopes'),
        ('aligned_checkpoints',), ('context_state',), ('nested_contexts',), 'comparison', 'exact'),
    'differentiation': RelationSpec(
        'differentiation', 1, ('valid_executions', 'differentiable_domain', 'reference_stable'),
        ('same_coordinates', 'same_scalarization'), ('directional_gradient',),
        ('finite_difference_perturbation',), 'comparison', 'convergence_checked_v1'),
    'accumulation': RelationSpec(
        'accumulation', 1, ('valid_executions', 'additive_loss', 'sample_independence'),
        ('same_initial_state', 'same_weighted_examples', 'aligned_checkpoints'), ('gradients',),
        ('microbatch_partition',), 'comparison', 'dtype_scaled_v1'),
    'state_restore': RelationSpec(
        'state_restore', 1, ('valid_executions', 'complete_persistent_state'),
        ('same_initial_state', 'same_remaining_actions', 'aligned_checkpoints'), ('persistent_state',),
        ('checkpoint_roundtrip',), 'comparison', 'exact'),
    'execution_mode': RelationSpec(
        'execution_mode', 1, ('valid_executions', 'executed_mode', 'semantic_execution_contract'),
        ('same_initial_state', 'same_computation', 'aligned_checkpoints'), ('outputs', 'gradients'),
        ('execution_mode',), 'comparison', 'dtype_scaled_v1'),
    'alias_effect': RelationSpec(
        'alias_effect', 1, ('valid_executions', 'observable_alias_partners', 'legal_effects'),
        ('same_logical_inputs', 'same_effect_sequence', 'aligned_checkpoints'), ('alias_propagation',),
        ('execution_mode',), 'comparison', 'exact'),
}


def optimizer_gradient_candidates(record):
    """Only a source-valid, attributed native update can supply a pair candidate."""
    from mirage.inspector.stateful import inspect_optimizer
    valid = record.get('completed') is True and record.get('inspection', {}).get('valid') is True
    return inspect_optimizer(record, valid=valid).get('eligible_chains', []) if valid else []


def optimizer_followup_predicates(predicates, *, parameter_key, pre_checkpoint, post_checkpoint):
    """Relax only the selected step's permitted representation difference."""
    return [p for p in predicates if not (p.get('dimension') == 'gradient_representation' and
            p.get('parameter_key') == parameter_key and
            p.get('checkpoint') in (pre_checkpoint, post_checkpoint))]


def optimizer_gradient_pair(left, right, backend, *, step_index, parameter_key):
    """Compare one observed native update; arbitrary final source state is not an output."""
    name = 'optimizer_gradient_representation'
    facts = {}
    def establish(key, condition, reason):
        facts[key] = Evidence(EvidenceStatus.VERIFIED if condition else EvidenceStatus.UNKNOWN,
                              reason, True if condition else None, 'comparison', 'adapter')
    def exact(a, b):
        return backend.compare(a, b, 'exact').get('equal') is True
    def step(record):
        matches = [s for s in record.get('optimizer_evidence', {}).get('steps', [])
                   if s.get('index') == step_index]
        return matches[0] if len(matches) == 1 else {}
    def chains(record):
        return [w for w in optimizer_gradient_candidates(record)
                if w.get('step_index') == step_index and w.get('parameter_key') == parameter_key]
    steps = [step(record) for record in (left, right)]
    witnesses = [chains(record) for record in (left, right)]
    establish('valid_executions', all(r.get('completed') is True and
              r.get('inspection', {}).get('valid') is True for r in (left, right)),
              'each member must satisfy its own runtime obligations')
    establish('native_optimizer_domain', all(s.get('domain', {}).get('status') == 'verified' for s in steps),
              'reviewed native optimizer domain at the actual update')
    establish('optimizer_source_effects', all(r.get('optimizer_source_certificate') ==
              'explicit_optimizer_gradient_update_v1' for r in (left, right)),
              'separate restricted optimizer source-effect certificate')
    identity_fields = ('parameter_key', 'parameter_id', 'call_index', 'operand_path', 'step_index')
    provenance = bool(all(len(w) == 1 for w in witnesses))
    if provenance:
        provenance = all(exact(witnesses[0][0].get(k), witnesses[1][0].get(k)) for k in identity_fields)
    establish('target_gradient_update', provenance,
              'same registered parameter and target path through a completed backward to the consumed gradient')
    establish('same_program', bool(left.get('graph')) and exact(left.get('graph'), right.get('graph')),
              'identical analyzed source computation')
    establish('same_initial_state', 'initial' in left and 'initial_metadata' in left and
              exact(left.get('initial'), right.get('initial')) and
              exact(left.get('initial_metadata'), right.get('initial_metadata')),
              'concrete initial values, optimizer state and representation')
    establish('same_environment', bool(left.get('environment')) and
              exact(left.get('environment'), right.get('environment')),
              'identical recorded execution environment')
    registries = [r.get('optimizer_evidence', {}).get('registry') for r in (left, right)]
    establish('same_optimizer_registry', bool(registries[0]) and exact(*registries),
              'native optimizer, stable group order, parameter bindings and options')
    pre = [s.get('pre', {}) for s in steps]
    establish('same_pre_update_state', all(all(k in p for k in ('parameters', 'gradients', 'slots')) for p in pre)
              and all(exact(pre[0].get(k), pre[1].get(k)) for k in ('parameters', 'gradients', 'slots')),
              'exact logical parameters, every gradient and optimizer slot immediately before the step')
    establish('aligned_optimizer_step', all(s.get('completed') is True for s in steps)
              and bool(steps[0].get('pre_checkpoint')) and bool(steps[0].get('post_checkpoint'))
              and all(steps[0].get(k) == steps[1].get(k) for k in
                      ('index', 'optimizer_id', 'pre_checkpoint', 'post_checkpoint')),
              'same completed native step and reserved semantic boundaries')
    configs = [dict(r.get('config', {})) for r in (left, right)]
    source_predicates = configs[0].pop('predicates', [])
    followup_predicates = configs[1].pop('predicates', [])
    actions = [c.pop('optimizer_gradient_action', None) for c in configs]
    permitted_predicates = optimizer_followup_predicates(source_predicates, parameter_key=parameter_key,
        pre_checkpoint=steps[0].get('pre_checkpoint'), post_checkpoint=steps[0].get('post_checkpoint'))
    action_valid = all(a is None or (a.get('step_index') == step_index and
                       a.get('parameter_key') == parameter_key and a.get('action') in
                       ('canonical', 'strided', 'offset')) for a in actions)
    establish('permitted_case_difference', exact(*configs) and action_valid and actions[1] is not None
              and exact(followup_predicates, permitted_predicates),
              'only the selected gradient action and representation-specific followup goals may differ')
    metadata = [p.get('gradient_metadata', {}) for p in pre]
    selected = [m.get(parameter_key, {}) for m in metadata]
    # Shape, dtype, device and other gradients are invariant. Actual strides or offset must change.
    changed = bool(selected[0]) and bool(selected[1]) and any(
        selected[0].get(k) != selected[1].get(k) for k in ('strides', 'offset'))
    def invariant_metadata(item):
        value = {k: v for k, v in item.items() if k not in ('strides', 'offset', 'object_group', 'storage_group')}
        if 'properties' in value:
            value['properties'] = {k: v for k, v in value['properties'].items() if k != 'contiguous'}
        return value
    invariant = exact(invariant_metadata(selected[0]), invariant_metadata(selected[1]))
    unchanged_others = metadata[0].keys() == metadata[1].keys() and all(
        exact(metadata[0][k], metadata[1][k]) for k in metadata[0] if k != parameter_key)
    mutation = steps[1].get('mutation') or {}
    source_mutation = steps[0].get('mutation') or {}
    mutation_valid = (mutation.get('status') == 'verified' and mutation.get('logical_equal') is True and
                      mutation.get('parameter_key') == parameter_key and actions[1] is not None and
                      mutation.get('action') == actions[1].get('action'))
    if actions[0] is not None:
        mutation_valid = (mutation_valid and source_mutation.get('status') == 'verified' and
                          source_mutation.get('logical_equal') is True and
                          source_mutation.get('parameter_key') == parameter_key and
                          source_mutation.get('action') == actions[0].get('action'))
    else:
        mutation_valid = mutation_valid and not source_mutation
    establish('verified_gradient_transformation', changed and invariant and unchanged_others and mutation_valid,
              'measured, exact-value-preserving, sole gradient representation difference')
    outputs = [{'updated_parameters': s.get('post', {}).get('parameters'),
                'optimizer_slots': s.get('post', {}).get('slots')} for s in steps]
    establish('complete_update_observables', all(isinstance(o['updated_parameters'], dict) and
              bool(o['updated_parameters']) and isinstance(o['optimizer_slots'], dict) and
              o['updated_parameters'].keys() == p.get('parameters', {}).keys() and
              o['optimizer_slots'].keys() == p.get('slots', {}).keys() for o, p in zip(outputs, pre)),
              'post-update parameters and slots captured for every registered optimizer parameter')
    result = evaluate(name, facts, *outputs, backend.compare)
    result.update(step_index=step_index, parameter_key=parameter_key,
                  source_goal_valid=left.get('inspection', {}).get('valid') is True,
                  followup_goal_valid=right.get('inspection', {}).get('valid') is True)
    if not RELATIONS[name].missing(facts):
        result['exact_output_comparison'] = backend.compare(*outputs, 'exact')
        changes = []
        for s in steps:
            amounts = {}
            for key, before in s['pre']['parameters'].items():
                after = s.get('post', {}).get('parameters', {}).get(key)
                try:
                    import math
                    amount = float((after - before).abs().max()) if before.numel() else 0.0
                    amounts[key] = amount if math.isfinite(amount) else None
                except (AttributeError, RuntimeError, TypeError):
                    amounts[key] = None
            changes.append(amounts)
        result['max_abs_parameter_update'] = dict(zip(('source', 'followup'), changes))
        result['update_amount_scope'] = 'secondary observation; a nonzero change is not an applicability requirement'
    return result


def select(evidence):
    """Rank only established relations; unknown capabilities do not reject an API."""
    return [name for name, relation in RELATIONS.items() if not relation.missing(evidence)]


def evaluate(name, evidence, left, right, compare):
    if name not in RELATIONS:
        return {'verdict': None, 'correctness': 'unassessed', 'reason': 'unsupported relation',
                'confirmed_defect': False}
    relation = RELATIONS[name]
    missing = relation.missing(evidence)
    if missing:
        return {'verdict': 'inapplicable', 'correctness': 'unassessed',
                'reason': 'relation obligations not established', 'missing': list(missing),
                'relation': name, 'confirmed_defect': False}
    outcome = compare(left, right, relation.numerical_policy)
    if outcome.get('equal') is None:
        return {'verdict': None, 'correctness': 'unassessed', 'reason': outcome['reason'],
                'relation': name, 'comparison': outcome, 'confirmed_defect': False}
    return {'verdict': 'pass' if outcome['equal'] else 'suspected_failure',
            'correctness': 'assessed', 'relation': name, 'comparison': outcome,
            'reason': 'MR output relation holds' if outcome['equal'] else 'MR output relation violated',
            'confirmed_defect': False}


def representation_pair(left, right, backend):
    """An adapter establishes measured facts; LLM assertions never supply certificates."""
    evidence = backend.representation_evidence(left, right)
    observations = []
    for record in (left, right):
        observations.append({'target_outputs': [c['output'] for c in record.get('target_calls', ())],
                             'gradients': record.get('checkpoints', [{}])[-1].get('gradients', {}),
                             'input_gradients':record.get('final_training_state',{}).get('input_gradients',{})})
    write_paths = [[c.get('writable_paths',[]) for c in r.get('target_calls',())] for r in (left,right)]
    if any(write_paths[0]) or any(write_paths[1]):
        matched = bool(write_paths[0]) and write_paths[0] == write_paths[1]
        matched = matched and all(
            isinstance(c.get('destination_values'), list) and len(c['destination_values']) == len(c.get('writable_paths', []))
            for record in (left, right) for c in record['target_calls'])
        partner_paths = [[[v['path'] for v in c.get('alias_partner_values', [])]
                          for c in record['target_calls']] for record in (left, right)]
        matched = matched and partner_paths[0] == partner_paths[1]
        evidence['matching_write_roles'] = Evidence(EvidenceStatus.VERIFIED if matched else EvidenceStatus.UNKNOWN,
            'matching schema write roles and observed destinations', True if matched else None,
            'comparison','adapter')
        for observed, record in zip(observations, (left, right)):
            observed['destinations'] = [c.get('destination_values') for c in record['target_calls']]
            observed['partners'] = [c.get('alias_partner_values', []) for c in record['target_calls']]
        outcome = evaluate('materialized_value_effect', evidence,
            *[{k: item[k] for k in ('target_outputs', 'destinations', 'partners')} for item in observations], backend.compare)
        if outcome['verdict'] in ('pass', 'suspected_failure') and any(
                c.get('guard_preserved') is False for record in (left, right) for c in record['target_calls']):
            outcome.update(verdict='suspected_failure', reason='write changed an initialized position outside the destination')
        return outcome
    return evaluate('representation', evidence, *observations, backend.compare)



def context_restoration(record, backend):
    events = record.get('context_events', ())
    facts = {}
    for key, valid in {
        'valid_executions': record.get('inspection', {}).get('valid', False),
        'context_restore_contract': record.get('context_contract', False) and all(
            not e.get('reentrant_same_instance') or e.get('decorated_entry') for e in events),
        'balanced_scopes': bool(events) and all(e.get('balanced') for e in events),
        'aligned_checkpoints': bool(events) and all('before' in e and 'after' in e for e in events),
    }.items():
        facts[key] = Evidence(EvidenceStatus.VERIFIED if valid else EvidenceStatus.UNKNOWN,
                              'original context enter/exit observation', True if valid else None,
                              'comparison', 'adapter')
    return evaluate('context_restore', facts, [e.get('before') for e in events],
                    [e.get('after') for e in events], backend.compare)



def execution_mode_pair(left, right, backend):
    """Compare the same source's explicit compiler calls across supported backends."""
    facts = {}
    graphs = [r.get('compiled_graphs', ()) for r in (left,right)]
    aligned = bool(graphs[0]) and len(graphs[0]) == len(graphs[1])
    same_graph = aligned and all(a['fingerprint'] == b['fingerprint']
                                and backend.compare(a.get('inputs'), b.get('inputs'), 'exact')['equal'] is True
                                for a,b in zip(*graphs))
    supported = all(g['backend'] in ('eager','inductor','aot_eager') for group in graphs for g in group)
    for key, valid in {
        'valid_executions': all(r.get('inspection',{}).get('valid') for r in (left,right)),
        'executed_mode': aligned,
        'semantic_execution_contract': supported and all(r.get('run_rng_unchanged') and r.get('execution_source_certificate') for r in (left,right)),
        'same_initial_state': backend.compare(left.get('initial'),right.get('initial'),'exact')['equal'] is True
                              and left.get('initial_metadata') == right.get('initial_metadata'),
        'same_computation': same_graph and left.get('graph',{}).get('fingerprint') == right.get('graph',{}).get('fingerprint'),
        'aligned_checkpoints': [x['name'] for x in left.get('checkpoints',())] == [x['name'] for x in right.get('checkpoints',())],
    }.items():
        facts[key] = Evidence(EvidenceStatus.VERIFIED if valid else EvidenceStatus.UNKNOWN,
                              'executed compiler graph and source evidence', True if valid else None,
                              'comparison','adapter')
    def observations(record):
        return {'outputs':record['observations'], 'gradients':record['checkpoints'][-1]['gradients'],
                'parameters':record.get('final_training_state',{}).get('parameters'),
                'input_gradients':record.get('final_training_state',{}).get('input_gradients',{}),
                'optimizer':record.get('final_training_state',{}).get('optimizer')}
    return evaluate('execution_mode', facts, observations(left), observations(right), backend.compare)



def captured_execution_pair(actual, reference, backend):
    """Factory identity control preserves the authored training history; no steps added."""
    events = actual.get('capture_factories', ())
    calls = [[c for c in r.get('target_calls',()) if c.get('kind') == 'factory_result_execution'] for r in (actual,reference)]
    aligned = bool(calls[0]) and len(calls[0]) == len(calls[1])
    unchanged = bool(events) and all(e['buffer_free'] and backend.compare(e['before'],e['after'],'exact')['equal'] is True for e in events)
    facts = {}
    checks = {
        'valid_executions': actual.get('inspection',{}).get('valid') and reference.get('completed')
                            and reference.get('backward_executed') and reference.get('reference_substitutions',0)>0,
        'executed_mode': bool(actual.get('capture_replays')) and any(c.get('completed_replays',0)>0 for c in calls[0]),
        'semantic_execution_contract': unchanged and actual.get('capture_source_certificate')
                                       and all(r.get('run_rng_unchanged') for r in (actual,reference)),
        'same_initial_state': backend.compare(actual.get('initial'),reference.get('initial'),'exact')['equal'] is True,
        'same_computation': actual.get('graph',{}).get('fingerprint') == reference.get('graph',{}).get('fingerprint')
                            and aligned and all(backend.compare(a['inputs'],b['inputs'],'exact')['equal'] is True for a,b in zip(*calls)),
        'aligned_checkpoints': [c['name'] for c in actual.get('checkpoints',())] == [c['name'] for c in reference.get('checkpoints',())],
    }
    for key,valid in checks.items():
        facts[key] = Evidence(EvidenceStatus.VERIFIED if valid else EvidenceStatus.UNKNOWN,
                              'observed callable capture and source-owned control',True if valid else None,'comparison','adapter')
    def observable(r): return {'outputs':r['observations'],'gradients':r['checkpoints'][-1]['gradients']}
    return evaluate('execution_mode',facts,observable(actual),observable(reference),backend.compare)
