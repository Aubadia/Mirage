"""LLM proposals over measured goals; proposals never certify reachability."""
from dataclasses import asdict
import hashlib
import json

from mirage.contracts.capabilities import StatePredicate
from mirage.generator.provider import Request, Stage
from mirage.generator.program import ContractError

PLAN_SCHEMA = {
    'type': 'object', 'properties': {'plans': {'type': 'array', 'maxItems': 3,
        'items': {'type': 'object', 'properties': {
            'goal_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 3},
            'actions': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 6},
            'preconditions': {'type': 'array', 'items': {'type': 'string'}},
            'regions': {'type': 'array', 'items': {'type': 'string', 'enum':
                ['initialization', 'operand_setup', 'run_tail', 'full']}, 'minItems': 1},
            'rationale': {'type': 'string'}},
            'required': ['goal_ids', 'actions', 'preconditions', 'regions', 'rationale'],
            'additionalProperties': False}}}, 'required': ['plans'], 'additionalProperties': False}


def goal_catalog(entry, explicit=None):
    """Bounded candidate vocabulary from source-member observations only."""
    witnesses = entry.get('evidence', {}).get('witnesses', [])
    goals = []
    def add(predicate, regions, observed=None):
        predicate = json.loads(json.dumps(predicate))
        # Unknown is not already attained and is never labelled feasible.
        if observed is not None and type(observed) is type(predicate['expected']) and observed == predicate['expected']:
            return
        key = json.dumps(predicate, sort_keys=True)
        goal_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        if any(g['id'] == goal_id for g in goals):
            return
        goals.append({'id': goal_id, 'predicate': predicate, 'required_regions': regions,
                      'observed': observed, 'feasibility': 'requires_static_plan_and_runtime_validation'})
    finals = [w for w in witnesses if w['checkpoint'] == 'final']
    facts = finals[0].get('state_facts', {}) if finals else {}
    if explicit is not None:
        for raw in explicit:
            p = asdict(StatePredicate(**raw))
            key = p['dimension'] + '.' + p['field']
            regions = ['full'] if p['dimension'] == 'execution_mode' else (
                ['operand_setup'] if p['dimension'] in ('representation', 'data') else ['run_tail'])
            observed = facts.get(key)
            if p['dimension'] in ('representation', 'data'):
                matching = [w for w in witnesses if w['checkpoint'] == p['checkpoint'] and w['call'] == p['call_index']
                            and w['operand_path'] == list(p['operand_path'])]
                if len(matching) == 1:
                    w = matching[0]
                    md = w.get('metadata') or {}
                    observed = dict(md.get('properties', {}), **md).get(p['field'])
                    if p['field'] == 'offset_nonzero':
                        observed = None if md.get('offset') is None else md['offset'] != 0
                    if p.get('relation') and w.get('relations', {}).get(p['relation'], {}).get('status') != 'verified':
                        observed = None
            add(p, regions, observed)
        return goals
    for dimension, field in [('gradient', 'backward_count'), ('graph_lifecycle', 'target_backward_count'),
                             ('graph_lifecycle', 'target_boundary_count'), ('context', 'restoration_count')]:
        current = facts.get(dimension + '.' + field)
        # Semantic coverage buckets counts above one as 'multiple'.
        if type(current) is int and current == 1:
            add(asdict(StatePredicate(dimension, field, current + 1, 'final')), ['run_tail'], current)
    if finals and facts.get('execution_mode.compiled_execution') is not True:
        add(asdict(StatePredicate('execution_mode', 'compiled_execution', True, 'final')), ['full'], facts.get('execution_mode.compiled_execution'))
    # At most two operand locations keep the planning prompt bounded.
    for witness in finals[:2]:
        metadata = witness.get('metadata')
        if not metadata:
            continue
        properties = metadata.get('properties', {})
        relation = 'operand_gradient_contribution' if witness.get('relations', {}).get(
            'operand_gradient_contribution', {}).get('status') == 'verified' else None
        candidates = [('contiguous', False, properties.get('contiguous')),
                      ('offset_nonzero', True, None if metadata.get('offset') is None else metadata['offset'] != 0),
                      ('negative', True, properties.get('negative'))]
        if 'complex' in metadata.get('dtype', ''):
            candidates.append(('conjugate', True, properties.get('conjugate')))
        if properties.get('floating'):
            alternate_dtype = 'torch.float64' if metadata['dtype'] == 'torch.float32' else 'torch.float32'
            add(asdict(StatePredicate('data', 'dtype', alternate_dtype, 'final',
                call_index=witness['call'], operand_path=tuple(witness['operand_path']), relation=relation)), ['operand_setup'], metadata['dtype'])
        for field, expected, observed in candidates:
            add(asdict(StatePredicate('representation', field, expected, 'final',
                call_index=witness['call'], operand_path=tuple(witness['operand_path']), relation=relation)), ['operand_setup'], observed)
    return goals


def planning_request(metadata, source, entry, catalog, configuration):
    return Request(Stage.STATE_GOAL,
        'Propose up to three DISTINCT final-state transition plans from goal_catalog. '
        'Each plan chooses one to three goal_ids and supplies a legal ordered action sequence, '
        'preconditions and required editable regions. Actions must be performed in those regions: '
        'operand_setup means assignments in run immediately before the existing target, never build; '
        'initialization means the selected build assignment; run_tail means after the target. '
        'Use installed API documentation and actual '
        'operand/path witnesses. Skip unsupported goals; return plans=[] if none can be planned. '
        'Do not emit Python code. New execution starts at build(), not at the parent final state. '
        'Preserve existing target invocations and required backward paths. Counts are exact. '
        'For repeated backward, retain or rebuild shared graphs before their first consumption. '
        'For repeated target boundaries add a real target invocation; changing initialization cannot '
        'change call counts. full permits restructuring for compilation/context wrapping. '
        'A plan is provisional: runtime must establish predicates and MR applicability independently.',
        {'contract': 'state-goal-plan', 'api_context': metadata, 'configuration': configuration,
         'canonical_source': source, 'initial_conditions': entry.get('configuration'),
         'measured_parent_witnesses': entry.get('evidence', {}).get('witnesses', [])[:8],
         'known_coverage': entry.get('coverage', [])[:8], 'goal_catalog': catalog,
         'measurement_contract': {
             'negative': 'Tensor.is_neg() lazy NEGATIVE VIEW BIT, NOT negative numerical values. torch.neg alone does not establish it.',
             'conjugate': 'Tensor.is_conj() lazy conjugate view bit, not complex values alone.',
             'contiguous': 'Actual is_contiguous at the target boundary. Transposing a 1-D tensor cannot change it.',
             'offset_nonzero': 'Actual storage_offset != 0 on the selected target operand.',
             'target_boundary_count': 'Number of forward target boundaries. Extra backward does NOT increment this count.',
             'target_backward_count': 'Target output gradient events, distinct from backward invocation count.',
             'backward_count': 'Actual backward/autograd.grad invocations; graph must be retained before first consumption or rebuilt.',
             'dtype': 'Actual operand dtype at target entry, not a planned dtype.',
             'gradient_preservation': 'New floating/complex operands must retain gradient history or require gradients before the call when backward is required.'}})


def validate_plans(content, catalog):
    try:
        value = json.loads(content)
    except (ValueError, TypeError) as error:
        raise ContractError('Invalid state-plan JSON') from error
    if not isinstance(value, dict) or set(value) != {'plans'} or not isinstance(value['plans'], list) or len(value['plans']) > 3:
        raise ContractError('Expected at most three state plans')
    known = {g['id']: g for g in catalog}
    accepted, rejected, seen = [], [], set()
    for raw in value['plans']:
        try:
            if not isinstance(raw, dict) or set(raw) != {'goal_ids', 'actions', 'preconditions', 'regions', 'rationale'}:
                raise ValueError('Invalid plan fields')
            ids, regions = raw['goal_ids'], raw['regions']
            if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or any(not isinstance(i, str) or i not in known for i in ids):
                raise ValueError('Unknown or empty goal IDs')
            if len(set(ids)) != len(ids) or tuple(sorted(ids)) in seen:
                raise ValueError('Duplicate plan or goal')
            if not isinstance(regions, list) or not regions or any(r not in ('initialization','operand_setup','run_tail','full') for r in regions):
                raise ValueError('Unknown edit regions')
            required = {r for i in ids for r in known[i]['required_regions']}
            if 'full' not in regions and not required <= set(regions):
                raise ValueError('Goal/edit-region mismatch: required ' + ','.join(sorted(required)))
            for key in ('actions', 'preconditions'):
                if not isinstance(raw[key], list) or any(not isinstance(s, str) or not s.strip() or len(s) > 1000 for s in raw[key]):
                    raise ValueError('Invalid actions/preconditions')
            if not 1 <= len(raw['actions']) <= 6 or not isinstance(raw['rationale'], str):
                raise ValueError('Invalid action count/rationale')
            predicates = [known[i]['predicate'] for i in ids]
            coordinates = [(p['dimension'], p['field'], p['checkpoint'], p.get('call_index'), tuple(p.get('operand_path') or ())) for p in predicates]
            if len(set(coordinates)) != len(coordinates):
                raise ValueError('Conflicting predicates')
            family = '+'.join(sorted({p['dimension'] + '.' + p['field'] for p in predicates}))
            accepted.append(dict(raw, predicates=predicates, family=family,
                                 status='planned_not_verified', expected_gain='unattained_predicates'))
            seen.add(tuple(sorted(ids)))
        except (ValueError, TypeError) as error:
            rejected.append({'plan': raw, 'reason': str(error)})
    return accepted, rejected
