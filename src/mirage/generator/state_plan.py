"""LLM proposals over measured goals; proposals never certify reachability."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

from mirage.contracts.capabilities import StatePredicate, JOINT_COVERAGE_MAX_OPERANDS
from mirage.generator.provider import Request, Stage
from mirage.generator.program import ContractError, source_operand_bindings

def _proposal_schema(status):
    planned = status == 'planned'
    return {'type': 'object', 'properties': {
        'status': {'type': 'string', 'enum': [status]},
        'goal_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 3},
        'actions': {'type': 'array', 'items': {'type': 'string'},
                    'minItems': 1 if planned else 0, 'maxItems': 6 if planned else 0},
        'preconditions': {'type': 'array', 'items': {'type': 'string'}},
        'regions': {'type': 'array', 'items': {'type': 'string', 'enum':
                    ['initialization', 'operand_setup', 'run_tail', 'full']},
                    'minItems': 1 if planned else 0, 'maxItems': 4 if planned else 0},
        'rationale': {'type': 'string'},
        'blockers': {'type': 'array', 'items': {'type': 'string'},
                     'minItems': 0 if planned else 1, 'maxItems': 0 if planned else 6}},
        'required': ['status', 'goal_ids', 'actions', 'preconditions', 'regions', 'rationale', 'blockers'],
        'additionalProperties': False}


PLAN_SCHEMA = {
    'type': 'object', 'properties': {
        'plans': {'type': 'array', 'maxItems': 3,
                  'items': {'anyOf': [_proposal_schema('planned'), _proposal_schema('unresolved')]}},
        'examined_goal_ids': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 34}},
    'required': ['plans', 'examined_goal_ids'], 'additionalProperties': False}


def planning_schema(catalog):
    """Constrain spelling and proposal shape, without claiming plan feasibility."""
    identifiers = [goal['id'] for goal in catalog]
    if (any(not isinstance(value, str) or not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('Planning schema requires unique nonempty goal IDs')
    schema = deepcopy(PLAN_SCHEMA)
    if identifiers:
        for branch in schema['properties']['plans']['items']['anyOf']:
            branch['properties']['goal_ids']['items']['enum'] = list(identifiers)
        schema['properties']['examined_goal_ids']['items']['enum'] = list(identifiers)
    else:
        # An empty enum is not a valid schema; only the empty reply is permitted.
        schema['properties']['plans']['maxItems'] = 0
        schema['properties']['examined_goal_ids']['maxItems'] = 0
    return schema


CATALOG_LIMITS = {'ordinary_operands': 2, 'joint_candidates': 2, 'axis_slots': 4,
                  'optimizer_parameters': 2, 'goals': 34}

PLANNING_FAMILIES = ('training_state', 'tensor_state', 'tensor_relation')


def goal_family(goal):
    dimension = goal['predicate']['dimension']
    if dimension == 'tensor_relation': return 'tensor_relation'
    return 'tensor_state' if dimension in ('representation', 'data', 'gradient_representation') else 'training_state'


def optimizer_action_for_predicates(predicates):
    """A source-case representation action; this never schedules a training step."""
    selected = [p for p in predicates if p.get('dimension') == 'gradient_representation']
    if not selected:
        return None
    possibilities = {'canonical', 'strided', 'offset'}
    coordinates = set()
    for raw in selected:
        p = StatePredicate(**raw)
        coordinates.add((p.parameter_key, p.checkpoint, p.call_index, tuple(p.operand_path)))
        choices = ({'canonical', 'offset'} if p.expected else {'strided'}) if p.field == 'contiguous' else (
            {'offset'} if p.expected else {'canonical', 'strided'})
        possibilities &= choices
    if len(coordinates) != 1 or not possibilities:
        raise ValueError('Gradient goals require one parameter/update and compatible representation actions')
    return {'step_index': 0, 'parameter_key': selected[0]['parameter_key'],
            'action': next(action for action in ('canonical', 'strided', 'offset') if action in possibilities)}


def _optimizer_companion(predicate):
    return asdict(StatePredicate('optimizer', 'completed', True,
        predicate['checkpoint'].rsplit('/', 1)[0] + '/post_update',
        call_index=predicate['call_index'], operand_path=predicate['operand_path'],
        relation='target_gradient_update', parameter_key=predicate['parameter_key']))



def library_configuration_plans(catalog):
    """Instantiate reviewed actions; source eligibility and attainment remain external."""
    accepted, rejected = [], []
    if len(catalog) > CATALOG_LIMITS['goals']:
        return [], [{'goal_ids': [], 'reason': 'Catalog exceeds the bounded vocabulary'}]
    try:
        planning_schema(catalog)
    except (KeyError, TypeError, ValueError) as error:
        return [], [{'goal_ids': [], 'reason': str(error)}]
    for goal in catalog:
        if 'source_case_action' not in goal:
            continue
        identifiers = [goal.get('id')]
        try:
            predicate = asdict(StatePredicate(**goal['predicate']))
            companion = _optimizer_companion(predicate)
            normalized = json.loads(json.dumps(goal.get('required_companions', [])))
            if (predicate['dimension'] != 'gradient_representation'
                    or normalized != json.loads(json.dumps([companion]))
                    or goal['source_case_action'] != optimizer_action_for_predicates([predicate])
                    or goal.get('required_regions') != ['run_tail']):
                raise ValueError('Known action requires an exact gradient goal and completed-update companion')
            raw = {'status': 'planned', 'goal_ids': identifiers,
                   'actions': ['Reuse the existing source-owned target backward and optimizer step; apply the configured equal-value gradient representation at pre_update'],
                   'preconditions': ['A verified parent chain with the same parameter, target path and completed update'],
                   'regions': ['run_tail'], 'blockers': [],
                   'rationale': 'The transition library supplies the execution configuration; no source edit is needed'}
            plans, errors = validate_plans(json.dumps({'plans': [raw], 'examined_goal_ids': identifiers}), [goal])
            if errors or len(plans) != 1:
                raise ValueError('Known action failed typed plan validation: ' + str(errors))
            accepted.append(dict(plans[0], proposal_origin='library_known_action', configuration_only=True))
        except (KeyError, TypeError, ValueError, ContractError) as error:
            rejected.append({'goal_ids': identifiers, 'reason': str(error),
                             'status': 'rejected_library_proposal', 'reachability': 'unassessed'})
    return accepted, rejected


def _coverage_context(coverage):
    groups = {'unary': [], 'joint': [], 'unknown': []}
    for unit in coverage:
        try:
            value = json.loads(unit) if isinstance(unit, str) else unit
            kind = value.get('unit_kind', 'unknown') if isinstance(value, dict) else 'unknown'
        except (TypeError, ValueError):
            kind = 'unknown'
        groups[kind if kind in groups else 'unknown'].append(unit)
    selected = {kind: values[:4] if kind != 'unknown' else [] for kind, values in groups.items()}
    return [unit for values in selected.values() for unit in values], {
        kind: {'total': len(values), 'shown': len(selected[kind]), 'omitted': len(values) - len(selected[kind])}
        for kind, values in groups.items()}


def _rotate(values, limit, offset):
    if not values:
        return []
    start = offset % len(values)
    return (values[start:] + values[:start])[:limit]


def _rotation(entry):
    # A parent's catalog stays stable across repeated planning requests. New
    # proposals rotate across the observer's bounded operand/edge-axis vocabulary.
    config = entry.get('configuration', {})
    value = config.get('proposal_index', config.get('attempt', 0))
    return value if type(value) is int and value >= 0 else 0


def _gradient_relation(witness):
    return 'operand_gradient_contribution' if witness.get('relations', {}).get(
        'operand_gradient_contribution', {}).get('status') == 'verified' else None


def goal_catalog(entry, explicit=None):
    """Bounded candidate vocabulary from source-member observations only."""
    from mirage.inspector.stateful import joint_candidates, witness_fact, optimizer_witness_fact
    witnesses = entry.get('evidence', {}).get('witnesses', [])
    optimizer_witnesses = entry.get('evidence', {}).get('optimizer_witnesses', [])
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
        if predicate['dimension'] == 'gradient_representation':
            goals[-1]['required_companions'] = [_optimizer_companion(predicate)]
            goals[-1]['source_case_action'] = optimizer_action_for_predicates([predicate])
    finals = [w for w in witnesses if w['checkpoint'] == 'final']
    facts = finals[0].get('state_facts', {}) if finals else {}
    if explicit is not None:
        for raw in explicit:
            predicate = StatePredicate(**raw)
            p = asdict(predicate)
            key = p['dimension'] + '.' + p['field']
            scoped = p['dimension'] in ('representation', 'data', 'tensor_relation')
            regions = ['full'] if p['dimension'] in ('execution_mode', 'optimizer') else (
                ['operand_setup'] if scoped else ['run_tail'])
            selected = next((w.get('state_facts', {}) for w in witnesses
                             if w['checkpoint'] == p['checkpoint']), {})
            observed = selected.get(key)
            if scoped:
                fact = witness_fact(witnesses, predicate)
                observed = fact.value if fact.established else None
            elif p['dimension'] in ('gradient_representation', 'optimizer'):
                if p['field'] == 'target_update_count':
                    observed = entry.get('evidence', {}).get('optimizer_target_update_count')
                else:
                    fact = optimizer_witness_fact(optimizer_witnesses, predicate)
                    observed = fact.value if fact.established else None
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
    current_updates = entry.get('evidence', {}).get('optimizer_target_update_count')
    if entry.get('configuration', {}).get('requires_backward', True) and current_updates != 1:
        add(asdict(StatePredicate('optimizer', 'target_update_count', 1, 'final')), ['full'], current_updates)
    rotation = _rotation(entry)
    counts = {}
    for witness in finals:
        key = (witness['call'], tuple(witness['operand_path'] or ()))
        counts[key] = counts.get(key, 0) + 1
    operands = sorted((w for w in finals if w.get('metadata') and counts[
        (w['call'], tuple(w['operand_path'] or ()))] == 1),
        key=lambda w: (w['call'], tuple((0, key) if type(key) is int else (1, key)
                                      for key in w['operand_path'])))
    per_call, bounded_operands = {}, []
    for witness in operands:
        count = per_call.get(witness['call'], 0)
        if count < JOINT_COVERAGE_MAX_OPERANDS:
            bounded_operands.append(witness)
        per_call[witness['call']] = count + 1
    operands = bounded_operands
    selected_operands = _rotate(operands, CATALOG_LIMITS['ordinary_operands'], rotation)
    for witness in selected_operands:
        metadata = witness['metadata']
        properties = metadata.get('properties', {})
        relation = _gradient_relation(witness)
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
    selected_pairs = _rotate(list(joint_candidates(finals)), CATALOG_LIMITS['joint_candidates'], rotation)
    axis_slots = []
    def add_axis(witness, axis):
        if axis is None:
            return
        key = (witness['call'], tuple(witness['operand_path']), axis)
        if not any(existing == key for existing, _, _ in axis_slots):
            axis_slots.append((key, witness, axis))
    for candidate in selected_pairs:
        primary, peer = candidate['primary'], candidate['peer']
        common = dict(dimension='tensor_relation', checkpoint=primary['checkpoint'],
                      call_index=primary['call'], operand_path=primary['operand_path'],
                      peer_operand_path=peer['operand_path'], relation=_gradient_relation(primary),
                      peer_relation=_gradient_relation(peer))
        storage = StatePredicate(field='shared_storage', expected=True, **common)
        observed = witness_fact(witnesses, storage)
        if observed.established:
            add(asdict(StatePredicate(field='shared_storage', expected=not observed.value, **common)),
                ['operand_setup'], observed.value)
        axis, peer_axis = candidate['axis'], candidate['peer_axis']
        if axis is not None and peer_axis is not None:
            for expected in ('lt', 'eq', 'gt'):
                predicate = StatePredicate(field='extent_order', expected=expected, axis=axis, peer_axis=peer_axis, **common)
                observed = witness_fact(witnesses, predicate)
                if observed.established:
                    add(asdict(predicate), ['operand_setup'], observed.value)
            add_axis(primary, axis)
            add_axis(peer, peer_axis)
    # Unary APIs still explore the observer's edge axes; relational slots take
    # precedence so a plan can conjoin a pair relation with its representation.
    for witness in selected_operands:
        shape = witness['metadata'].get('shape')
        if isinstance(shape, (list, tuple)) and shape and all(type(n) is int and n >= 0 for n in shape):
            for axis in sorted({0, len(shape)-1}):
                add_axis(witness, axis)
    for index, (_, witness, axis) in enumerate(axis_slots[:CATALOG_LIMITS['axis_slots']]):
        common = dict(dimension='representation', checkpoint=witness['checkpoint'],
                      call_index=witness['call'], operand_path=witness['operand_path'],
                      relation=_gradient_relation(witness), axis=axis)
        predicate = StatePredicate(field='zero_stride', expected=True, **common)
        observed = witness_fact(witnesses, predicate)
        if observed.established:
            add(asdict(StatePredicate(field='zero_stride', expected=not observed.value, **common)),
                ['operand_setup'], observed.value)
        predicate = StatePredicate(field='singleton_stride', expected='unit', **common)
        observed = witness_fact(witnesses, predicate)
        if observed.established:
            alternatives = [value for value in ('zero', 'unit', 'other', 'not_singleton') if value != observed.value]
            expected = alternatives[(rotation + index) % len(alternatives)]
            add(asdict(StatePredicate(field='singleton_stride', expected=expected, **common)),
                ['operand_setup'], observed.value)
    selected_updates = _rotate(optimizer_witnesses, CATALOG_LIMITS['optimizer_parameters'], rotation)
    for witness in selected_updates:
        common = dict(dimension='gradient_representation', checkpoint=witness['pre_checkpoint'],
            call_index=witness['call_index'], operand_path=witness['operand_path'],
            relation='target_gradient_update', parameter_key=witness['parameter_key'])
        for field in ('contiguous', 'offset_nonzero'):
            fact = optimizer_witness_fact(optimizer_witnesses, StatePredicate(field=field, expected=True, **common))
            if fact.established:
                shape = witness.get('metadata', {}).get('shape')
                if (field == 'contiguous' and fact.value is True and isinstance(shape, (list, tuple))
                        and all(type(n) is int and n >= 0 for n in shape)
                        and (0 in shape or all(n == 1 for n in shape))):
                    # Zero/one-element dense tensors cannot supply noncontiguity.
                    continue
                add(asdict(StatePredicate(field=field, expected=not fact.value, **common)), ['run_tail'], fact.value)
    return goals


def planning_request(metadata, source, entry, catalog, configuration, *, planning_family=None):
    if planning_family not in (None, *PLANNING_FAMILIES):
        raise ValueError("Unknown planning family")
    # Retain the exact witnesses named by goals, including peers beyond an old
    # positional prompt cutoff. Automatic catalogs select at most six operands.
    required = set()
    for goal in catalog:
        p = goal['predicate']
        for key in ('operand_path', 'peer_operand_path'):
            if p.get(key):
                required.add((p['checkpoint'], p.get('call_index'), tuple(p[key])))
    witnesses = entry.get('evidence', {}).get('witnesses', [])
    measured = [w for w in witnesses if (w['checkpoint'], w['call'], tuple(w['operand_path'] or ())) in required]
    if not measured:
        measured = witnesses[:1]
    requested_goal_ids = [goal['id'] for goal in catalog if planning_family is None or goal_family(goal) == planning_family]
    families = {family: [g['id'] for g in catalog if goal_family(g) == family] for family in PLANNING_FAMILIES}
    bindings = source_operand_bindings(source, configuration['api'], [g['predicate'] for g in catalog])
    coverage, coverage_counts = _coverage_context(entry.get('coverage', []))
    return Request(Stage.STATE_GOAL,
        'Propose up to three DISTINCT final-state transition plans from goal_catalog. '
        'Each plan chooses one to three goal_ids and supplies a legal ordered action sequence, '
        'preconditions and required editable regions. Actions must be performed in those regions: '
        'operand_setup means assignments in run immediately before the existing target, never build; '
        'initialization means the selected build assignment; run_tail means after the target. '
        'Use installed API documentation and actual operand/path witnesses. This request is one '
        'planning-family round. Each planned proposal must include at least one requested_goal_id; '
        'compatible goals from other families may be conjoined. Do not fill slots merely to prefer '
        'tensor relations over training history or ordinary representations. Up to three proposals '
        'are allowed, including unresolved outcomes; fewer proposals are fine. '
        'Set status=planned only for a concrete executable action sequence with nonempty actions '
        'and regions and empty blockers. Otherwise set status=unresolved with empty actions and '
        'regions and specific blockers. Never place a refusal or an inability to plan in actions. '
        'Unresolved is a planning limitation, not proof of global unreachability. '
        'examined_goal_ids must be exactly the union of IDs addressed by returned proposals; '
        'do not claim the rest of the catalog was examined. Return plans=[] and examined_goal_ids=[] '
        'when no proposal was examined. A single-goal plan is sufficient; do not force a conjunction. '
        'For goals selecting operand_path or peer_operand_path, use source_operand_bindings to identify '
        'the actual receiver, positional or keyword expression and its last assignment. These static '
        'hints do not prove runtime linkage. Unknown selected operand bindings must remain unresolved '
        'or be handled through a legal full edit. Training counts and context predicates without operand '
        'paths do not require tensor bindings; plan from their measured state and legal actions. An assignment after the '
        'first operand_setup insertion point can overwrite an edit aimed at a later target. '
        'Do not emit Python code. New execution starts at build(), not at the parent final state. '
        'Preserve existing target invocations and required backward paths. Counts are exact. '
        'For repeated backward, retain or rebuild shared graphs before their first consumption. '
        'For repeated target boundaries add a real target invocation; changing initialization cannot '
        'change call counts. full permits restructuring for compilation/context wrapping. '
        'Tensor relations bind both actual operand paths in the same target call and checkpoint. '
        'Axis selectors are nonnegative tensor axes, not the API dim argument. Choose coordinated '
        'legal shape/layout edits and preserve each selected operand backward relation separately. '
        'All one-to-three chosen goals are a conjunction; combine a relation with its relevant '
        'representation only when both can hold. Use full if target arguments or surrounding code '
        'must change outside operand_setup. Parent/child lineage is exploration, not an MR pair: '
        'each new source/follow-up pair must share its own required initial values and semantic inputs. '
        'A parent_certificate_diagnostic with status unknown means missing MR proof, not an invalid program. '
        'Preserve selected states and path obligations; never delete a hard state merely to obtain a certificate. '
        'A legal case without MR proof may remain coverage-only. '
        'OPTIMIZER GOALS: target_update_count counts only complete target-connected native updates, '
        'not arbitrary step calls. A full edit may register the actual differentiable target leaf in '
        'ctx.parameters and construct its real SGD optimizer in build; never invent a parameter identity. '
        'The bounded contract uses one eager CPU float64 native SGD update, positive finite lr, '
        'momentum in [0,1), weight_decay=0, dampening=0, nesterov=False, maximize=False, '
        'foreach=False, fused=False, differentiable=False, no closure, scheduler or AMP. '
        'Start with grad=None; source owns target, backward and optimizer.step. '
        'Gradient representation goals select the consumed parameter.grad at the reserved pre_update '
        'checkpoint, not the forward operand. required_companions remain mandatory even if previously '
        'attained. source_case_action is performed by the harness at the actual source-owned step; '
        'do not assign, clear or rewrite parameter.grad after backward. Never add probe.mark with an '
        'optimizer: name. This tests optimizer consumption, not backward writing into that layout. '
        'A plan is provisional: runtime must establish predicates and MR applicability independently.',
        {'contract': 'state-goal-plan', 'api_context': metadata, 'configuration': configuration,
         'canonical_source': source, 'initial_conditions': entry.get('configuration'),
         'measured_parent_witnesses': measured,
         'measured_optimizer_witnesses': [w for w in entry.get('evidence', {}).get('optimizer_witnesses', [])
             if any(g['predicate'].get('parameter_key') == w.get('parameter_key') for g in catalog)][:2],
         'catalog_limits': dict(CATALOG_LIMITS),
         'catalog_rotation': {'offset': _rotation(entry),
                              'basis': 'parent proposal_index, falling back to attempt or zero; fixed for this parent',
                              'scope': 'rotates within the observer first-four operands and first/last-axis joint slice; not exhaustive'},
         'known_coverage': coverage, 'known_coverage_counts': coverage_counts, 'goal_catalog': catalog,
         'planning_family': planning_family, 'requested_goal_ids': requested_goal_ids,
         'goal_families': families, 'source_operand_bindings': bindings,
         'plan_protocol': 'state-goal-plan-4',
         'measurement_contract': {
             'negative': 'Tensor.is_neg() lazy NEGATIVE VIEW BIT, NOT negative numerical values. torch.neg alone does not establish it.',
             'conjugate': 'Tensor.is_conj() lazy conjugate view bit, not complex values alone.',
             'contiguous': 'Actual is_contiguous at the target boundary. Transposing a 1-D tensor cannot change it.',
             'offset_nonzero': 'Actual storage_offset != 0 on the selected target operand.',
             'extent_order': 'Compare shape[axis] of operand_path with shape[peer_axis] of peer_operand_path in the same call/checkpoint: lt, eq or gt. This alone does not establish API legality or broadcast compatibility.',
             'shared_storage': 'True/false only from two observed storage identities. Shared storage need not mean occupied-coordinate overlap; unknown identity is not independence.',
             'zero_stride': 'With axis, shape[axis] > 1 and stride[axis] == 0. Without axis, preserve the existing any-axis meaning.',
             'singleton_stride': 'At the selected axis: zero/unit/other classifies the stride only when extent is one; otherwise not_singleton. zero_stride=True and a singleton class cannot hold at the same axis.',
             'peer_relation': 'The peer has its own required backward-path evidence. Primary relation evidence does not establish peer contribution.',
             'target_boundary_count': 'Number of forward target boundaries. Extra backward does NOT increment this count.',
             'target_backward_count': 'Target output gradient events, distinct from backward invocation count.',
             'backward_count': 'Actual backward/autograd.grad invocations; graph must be retained before first consumption or rebuilt.',
             'target_update_count': 'Completed native updates with verified direct target-to-registered-parameter gradient provenance; arbitrary optimizer calls do not count.',
             'gradient_representation': 'The selected parameter.grad metadata at optimizer pre_update, after a verified value-preserving source-case action; its companion completed update at post_update is mandatory.',
             'dtype': 'Actual operand dtype at target entry, not a planned dtype.',
             'gradient_preservation': 'New floating/complex operands must retain gradient history or require gradients before the call when backward is required.'}})


def validate_plans(content, catalog, *, planning_family=None):
    if planning_family not in (None, *PLANNING_FAMILIES):
        raise ValueError("Unknown planning family")
    try:
        value = json.loads(content)
    except (ValueError, TypeError) as error:
        raise ContractError('Invalid state-plan JSON') from error
    if (not isinstance(value, dict) or set(value) not in ({'plans'}, {'plans', 'examined_goal_ids'})
            or not isinstance(value.get('plans'), list) or len(value['plans']) > 3):
        raise ContractError('Expected at most three state plans')
    known = {g['id']: g for g in catalog}
    version_two = 'examined_goal_ids' in value
    if version_two:
        examined = value['examined_goal_ids']
        if (not isinstance(examined, list) or len(examined) > CATALOG_LIMITS['goals']
                or any(not isinstance(i, str) or i not in known for i in examined) or len(set(examined)) != len(examined)):
            raise ContractError('Invalid examined goal IDs')
        named = {i for raw in value['plans'] if isinstance(raw, dict) and isinstance(raw.get('goal_ids'), list)
                 for i in raw['goal_ids'] if isinstance(i, str)}
        if set(examined) != named:
            raise ContractError('Examined goal IDs must match the returned proposals')
    accepted, rejected, seen = [], [], set()
    for raw in value['plans']:
        try:
            fields = {'goal_ids', 'actions', 'preconditions', 'regions', 'rationale'}
            if version_two: fields |= {'status', 'blockers'}
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError('Invalid plan fields')
            ids, regions = raw['goal_ids'], raw['regions']
            if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or any(not isinstance(i, str) or i not in known for i in ids):
                raise ValueError('Unknown or empty goal IDs')
            if len(set(ids)) != len(ids) or tuple(sorted(ids)) in seen:
                raise ValueError('Duplicate plan or goal')
            if planning_family is not None and not any(goal_family(known[i]) == planning_family for i in ids):
                raise ValueError('Plan does not address the requested planning family')
            status = raw.get('status', 'planned')
            if status not in ('planned', 'unresolved'):
                raise ValueError('Unknown proposal status')
            for key in ('actions', 'preconditions', 'blockers'):
                items = raw.get(key, [])
                if not isinstance(items, list) or any(not isinstance(s, str) or not s.strip() or len(s) > 1000 for s in items):
                    raise ValueError('Invalid actions/preconditions/blockers')
            if (not isinstance(regions, list) or any(r not in ('initialization','operand_setup','run_tail','full') for r in regions)
                    or len(raw['actions']) > 6 or len(raw.get('blockers', [])) > 6 or not isinstance(raw['rationale'], str)):
                raise ValueError('Invalid regions/action count/rationale')
            if status == 'unresolved':
                if raw['actions'] or regions or not raw.get('blockers'):
                    raise ValueError('Unresolved proposals require blockers and no actions or regions')
                rejected.append({'plan': raw, 'reason': 'Unresolved proposal; reachability unassessed',
                                 'status': 'unresolved', 'examined_goal_ids': ids, 'blockers': raw['blockers']})
                seen.add(tuple(sorted(ids)))
                continue
            if not raw['actions'] or not regions or raw.get('blockers'):
                raise ValueError('Planned proposals require actions/regions and no blockers')
            required = {r for i in ids for r in known[i]['required_regions']}
            if 'full' not in regions and not required <= set(regions):
                raise ValueError('Goal/edit-region mismatch: required ' + ','.join(sorted(required)))
            predicates = [known[i]['predicate'] for i in ids]
            for i in ids:
                for companion in known[i].get('required_companions', []):
                    if companion not in predicates:
                        predicates.append(companion)
            optimizer_action = optimizer_action_for_predicates(predicates)
            measurements, selectors = {}, set()
            for p in predicates:
                StatePredicate(**p)
                coordinate = (p['dimension'], p['field'], p['checkpoint'], p.get('call_index'),
                              tuple(p.get('operand_path') or ()), p.get('axis'),
                              tuple(p.get('peer_operand_path') or ()), p.get('peer_axis'), p.get('parameter_key'))
                # Evidence obligations remain attached to their authored paths,
                # while reversed pairs describe the same physical measurement.
                selector = coordinate + (p.get('relation'), p.get('peer_relation'))
                expected_value = p['expected']
                if p['dimension'] == 'tensor_relation':
                    path_key = lambda path: tuple((0, key) if type(key) is int else (1, key) for key in path)
                    if path_key(coordinate[4]) > path_key(coordinate[6]):
                        coordinate = coordinate[:4] + (coordinate[6], coordinate[7], coordinate[4], coordinate[5], coordinate[8])
                        if p['field'] == 'extent_order':
                            expected_value = {'lt':'gt', 'eq':'eq', 'gt':'lt'}[expected_value]
                expected = (type(expected_value), json.dumps(expected_value, sort_keys=True))
                if coordinate in measurements and measurements[coordinate] != expected:
                    raise ValueError('Conflicting predicates at the same selectors')
                measurements[coordinate] = expected
                if selector in selectors:
                    raise ValueError('Duplicate predicate selectors')
                selectors.add(selector)
            for p in predicates:
                if p['dimension'] != 'representation' or p['field'] != 'zero_stride' or p['expected'] is not True:
                    continue
                if p.get('axis') is not None and any(q['dimension'] == 'representation'
                    and q['field'] == 'singleton_stride' and q['expected'] != 'not_singleton'
                    and tuple(q.get('operand_path') or ()) == tuple(p.get('operand_path') or ())
                    and all(q.get(key) == p.get(key) for key in ('checkpoint', 'call_index', 'axis'))
                    for q in predicates):
                    raise ValueError('Conflicting zero-stride and singleton-axis predicates')
            family = '+'.join(sorted({p['dimension'] + '.' + p['field'] for p in predicates}))
            accepted.append(dict(raw, predicates=predicates, family=family,
                                 optimizer_gradient_action=optimizer_action,
                                 status='planned_not_verified', proposal_status='planned',
                                 planning_family=planning_family, examined_goal_ids=ids,
                                 expected_gain='unattained_predicates'))
            seen.add(tuple(sorted(ids)))
        except (ValueError, TypeError) as error:
            rejected.append({'plan': raw, 'reason': str(error), 'status': 'rejected', 'examined_goal_ids': []})
    return accepted, rejected
