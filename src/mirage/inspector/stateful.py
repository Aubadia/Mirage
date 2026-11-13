"""Runtime-evidence aggregation independent of the tensor backend."""
from mirage.contracts.capabilities import Evidence, EvidenceStatus, StatePredicate

FUNNEL = ("generation", "parse", "static_contract", "initialization", "snapshot",
          "mutation_eligibility", "mutation_application", "target_entry", "target_return",
          "backward", "state", "relation", "audit")


def failure_record(first_blocking, reason, secondary=()):
    if first_blocking not in FUNNEL:
        raise ValueError("Unknown execution funnel stage")
    return {"first_blocking_stage": first_blocking, "reason": reason,
            "secondary_diagnostics": list(secondary)}


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
        attained.append({'dimension': predicate.dimension, 'field': predicate.field,
                         'checkpoint': predicate.checkpoint, 'expected': predicate.expected,
                         'observed': fact.value, 'evidence_status': fact.status.value,
                         'attained': predicate.holds(fact)})
    api = bool(record.get('api_executed'))
    backward_required=record.get('config',{}).get('requires_backward',True)
    backward = not backward_required or bool(record.get('backward_executed'))
    required_path='target'
    for field,kind in (('compiled_execution','executed FX graph'),('captured_execution','factory_result_execution')):
        if any(p.dimension=='execution_mode' and p.field==field and p.expected is True for p in predicates):
            indices={c['index'] for c in record.get('target_calls',()) if c.get('evidence')==kind or c.get('kind')==kind}
            backward=backward and (not backward_required or any(e.get('call') in indices for e in record.get('target_gradients',())))
            required_path=field+'_target'
    return {'valid': bool(record.get('completed') and api and backward and all(p['attained'] for p in attained)),
            'api_executed': api, 'required_backward_executed': backward, 'required_backward_path':required_path,
            'predicates': attained, 'target_state_reached': all(p['attained'] for p in attained),
            'transition_signature': record.get('transition_signature', []),
            'correctness': 'unassessed'}


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
