"""Runtime-evidence aggregation independent of the tensor backend."""
import json

from mirage.contracts.capabilities import (Evidence, EvidenceStatus, StatePredicate,
                                         PathEvidence, SEMANTIC_COVERAGE_VERSION)

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
    completed = {e['index'] for e in record.get('backward_invocations', ()) if e['completed']}
    nodes = [e for e in record.get('target_backward_nodes', ()) if e.get('invocation') in completed]
    backward = not backward_required or bool(nodes)
    required_path='target'
    for field,kind in (('compiled_execution','executed FX graph'),('captured_execution','factory_result_execution')):
        if any(p.dimension=='execution_mode' and p.field==field and p.expected is True for p in predicates):
            indices={c['index'] for c in record.get('target_calls',()) if c.get('evidence')==kind or c.get('kind')==kind}
            backward=backward and (not backward_required or any(e.get('call') in indices for e in nodes))
            required_path=field+'_target'
    valid = bool(record.get('completed') and api and backward and all(p['attained'] for p in attained))
    return {'valid': valid,
            'api_executed': api, 'required_backward_executed': backward, 'required_backward_path':required_path,
            'predicates': attained, 'target_state_reached': all(p['attained'] for p in attained),
            'transition_signature': record.get('transition_signature', []),
            'correctness': 'unassessed', 'semantic_coverage': semantic_coverage(record, valid=valid)}


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
              'operands': record.get('target_operand_gradients', [])}
    environment = {k: record.get('environment', {}).get(k) for k in
                   ('torch', 'cuda_runtime', 'device', 'deterministic_algorithms', 'matmul_precision')}
    witnesses, signatures = [], set()
    checkpoints = record.get('checkpoints', [])
    names = [c['name'] for c in checkpoints]
    for checkpoint in checkpoints:
        if names.count(checkpoint['name']) != 1 or 'evidence_offsets' not in checkpoint:
            continue
        offsets = checkpoint['evidence_offsets']
        completed = set(checkpoint.get('completed_backward_invocations', []))
        for call in record.get('target_calls', [])[:offsets['calls']]:
            index = call['index']
            scoped = {kind: [(i, event) for i, event in enumerate(items[:offsets[kind]])
                             if event['call'] == index and event.get('invocation') in completed]
                      for kind, items in events.items()}
            for operand in call.get('metadata', []) or [None]:
                path = list(operand['path']) if operand else None
                operand_events = [(i, e) for i, e in scoped['operands'] if e['path'] == path]
                relations = {}
                for name, selected in (('output_gradient', scoped['outputs']),
                                       ('target_backward_node', scoped['nodes']),
                                       ('operand_gradient_contribution', operand_events)):
                    evidence = PathEvidence(EvidenceStatus.VERIFIED if selected else EvidenceStatus.UNKNOWN,
                                            'completed backward runtime events' if selected else 'no attributable runtime event',
                                            tuple(i for i, _ in selected))
                    relations[name] = dict(evidence.record(), scope='operand' if name == 'operand_gradient_contribution' else 'call')
                relations['backward_representation'] = PathEvidence(
                    EvidenceStatus.UNSUPPORTED, 'saved tensor representation is not instrumented').record()
                witness = {'call': index, 'operand_path': path, 'checkpoint': checkpoint['name'],
                           'metadata': operand, 'execution_mode': call.get('execution_mode', 'unknown'),
                           'backend': call.get('backend'), 'relations': relations,
                           'entry_context': call.get('entry_context'),
                           'state_facts': checkpoint.get('state_facts', {}),
                           'backward_bindings': call.get('backward_bindings', [])}
                witnesses.append(witness)
                if not valid:
                    continue
                # Per-call IDs, raw sizes/strides and exact event counts are not novelty.
                representation = None
                data = None
                if operand:
                    properties = operand.get('properties', {})
                    representation = {k: properties.get(k) for k in ('negative', 'conjugate', 'contiguous', 'mkldnn')}
                    representation.update(layout=operand['layout'], device=operand['device'],
                        offset='unknown' if operand.get('offset') is None else ('zero' if operand['offset'] == 0 else 'nonzero'),
                        zero_stride=any(n > 1 and stride == 0 for n, stride in
                                        zip(operand['shape'], operand.get('strides') or ())),
                        aliases=[list(other['path']) for other in call.get('metadata', [])
                                 if other['path'] != operand['path'] and operand.get('storage_group') is not None
                                 and other.get('storage_group') == operand['storage_group']],
                        requires_grad=operand.get('requires_grad'), leaf=operand.get('leaf'))
                    facts = next((p['facts'] for p in call.get('input_properties', [])
                                  if list(p['path']) == path), {})
                    data = {'dtype': operand['dtype'], 'rank': len(operand['shape']),
                            'empty': 0 in operand['shape'], 'singleton': 1 in operand['shape'],
                            'categories': sorted(k for k, v in facts.items() if v is True)}
                state = {k: ('multiple' if v > 1 else v) if type(v) is int and k.endswith('_count') else v
                         for k, v in checkpoint.get('state_facts', {}).items()}
                unit = {'version': SEMANTIC_COVERAGE_VERSION, 'api': record.get('config', {}).get('api'),
                        'operand_path': path, 'checkpoint': checkpoint['name'], 'representation': representation,
                        'data': data, 'state': state, 'execution_mode': witness['execution_mode'],
                        'backend': witness['backend'], 'environment': environment, 'entry_context': witness['entry_context'],
                        'relations': sorted(k for k, v in relations.items() if v['status'] == EvidenceStatus.VERIFIED)}
                signatures.add(json.dumps(unit, sort_keys=True, separators=(',', ':')))
    return {'version': SEMANTIC_COVERAGE_VERSION, 'instrumentation': 'output-and-direct-node-hooks',
            'environment': environment, 'events': events, 'witnesses': witnesses,
            'signatures': [json.loads(s) for s in sorted(signatures)], 'eligible': valid,
            'observed_units': len(signatures), 'denominator': None, 'correctness': 'unassessed'}
