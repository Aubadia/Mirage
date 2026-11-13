"""Conditional semantic-family relations; API admission is independent of this catalog."""
from mirage.contracts.capabilities import Evidence, EvidenceStatus, RelationSpec

RELATIONS = {
    'representation': RelationSpec(
        'representation', 1,
        ('valid_executions', 'logical_value_semantics', 'deterministic', 'read_only_target',
         'preserved_coordinates', 'finite_observations', 'numerical_domain'),
        ('same_initial_state', 'same_program', 'same_target_inputs', 'aligned_checkpoints'),
        ('target_outputs', 'gradients'), ('representation',), 'comparison', 'dtype_scaled_v1'),
    'materialized_value_effect': RelationSpec(
        'materialized_value_effect', 1,
        ('valid_executions', 'logical_value_semantics', 'deterministic', 'matching_write_roles',
         'preserved_coordinates', 'finite_observations', 'numerical_domain'),
        ('same_initial_state', 'same_program', 'same_target_inputs', 'aligned_checkpoints'),
        ('target_outputs',), ('representation', 'output_storage_identity'), 'comparison', 'dtype_scaled_v1'),
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
        evidence['matching_write_roles'] = Evidence(EvidenceStatus.VERIFIED if matched else EvidenceStatus.UNKNOWN,
            'schema-derived write roles; comparison observes returned logical values only', True if matched else None,
            'comparison','adapter')
        return evaluate('materialized_value_effect', evidence,
                        {'target_outputs':observations[0]['target_outputs']},
                        {'target_outputs':observations[1]['target_outputs']}, backend.compare)
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
