"""Verified stateful execution seeds with immutable replay/lineage references."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from mirage.artifacts.bundle import digest, verify
from mirage.contracts.capabilities import SEMANTIC_COVERAGE_VERSION, OPTIMIZER_COVERAGE_VERSION
from mirage.generator.program import analyze_stateful

VERSION = 'stateful-corpus-2'


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def guided_prefix_witness(parent_record, child_record, comparator):
    """Compare the original initialization and observed target/backward history before a suffix.

    A positive witness establishes these observations only. The requested
    training state and the child's correctness oracles have independent checks.
    """
    parent, child = parent_record, child_record
    evidence = {'version': 'guided-prefix-1', 'verified': False}

    def fail(reason):
        return dict(evidence, reason=reason)

    def equal(left, right):
        try:
            return comparator.compare(left, right, 'exact')['equal'] is True
        except (RuntimeError, TypeError, ValueError):
            return False

    if not all(record.get('completed') and record.get('api_executed') for record in (parent, child)):
        return fail('missing_completed_target_execution')
    for field in ('api', 'seed', 'device', 'input_recipe'):
        config = parent.get('config', {})
        if (field != 'input_recipe' and field not in config
                or config.get(field) != child.get('config', {}).get(field)):
            return fail('configuration_mismatch:' + field)
    for field in ('environment', 'initial', 'initial_metadata', 'initial_rng'):
        if field not in parent or field not in child or not equal(parent[field], child[field]):
            return fail('initialization_mismatch:' + field)

    calls, replayed = parent.get('target_calls', []), child.get('target_calls', [])
    evidence['parent_target_calls'] = len(calls)
    if not calls or len(replayed) < len(calls):
        return fail('missing_target_calls')
    required = ('index', 'inputs', 'output', 'metadata', 'entry_context', 'execution_mode')
    optional = ('kind', 'evidence', 'write_effects', 'writable_paths', 'destination_values',
                'alias_partner_values', 'returned_alias_observations', 'guard_preserved',
                'schema_read_only', 'backward_bindings', 'factory_target', 'factory_id',
                'callable_id', 'invocation_index', 'callable_identity_verified')
    for index, (call, replay) in enumerate(zip(calls, replayed)):
        measured = required + tuple(field for field in optional if field in call)
        if (any(field not in call or field not in replay for field in measured)
                or not equal({k: call[k] for k in measured}, {k: replay[k] for k in measured})):
            return fail('target_call_mismatch:' + str(index))

    invocations = parent.get('backward_invocations', [])
    completed = {event['index'] for event in invocations if event.get('completed')}
    backwards = parent.get('target_backward_nodes', [])
    evidence['parent_backward_invocations'] = len(invocations)
    if not any(event.get('call') in range(len(calls)) and event.get('invocation') in completed
               for event in backwards):
        return fail('no_parent_target_linked_backward')
    child_invocations = child.get('backward_invocations', [])
    if not equal(invocations, child_invocations[:len(invocations)]):
        return fail('backward_history_mismatch:backward_invocations')
    original_ids = {event['index'] for event in invocations}
    for field in ('target_backward_nodes', 'target_gradients', 'target_operand_gradients',
                  'callable_backward_nodes'):
        before, after = parent.get(field), child.get(field)
        if (not isinstance(before, list) or not isinstance(after, list)
                or not equal(before, [event for event in after if event.get('invocation') in original_ids])):
            return fail('backward_history_mismatch:' + field)
    before, after = parent.get('gradient_evidence', {}), child.get('gradient_evidence', {})
    if 'events' in before:
        events = before['events']
        replayed_events = [event for event in after.get('events', []) if event.get('invocation') in original_ids]
        if before.get('version') != after.get('version') or not equal(events, replayed_events):
            return fail('recorded_gradient_mismatch')
    return dict(evidence, verified=True, reason='frozen_initialization_and_target_history_replayed')


class Corpus:
    """Round-robin seeds; all attempt artifacts survive admission/deduplication."""
    def __init__(self, root, *, coverage_basis=None):
        self.root = Path(root).resolve()
        self.path = self.root / 'corpus.json'
        self.entries, self.quarantine, self.selections = [], [], {}
        self.prefixes = []
        self.coverage_basis = coverage_basis or 'semantic'
        if self.coverage_basis not in ('semantic', 'transition'):
            raise ValueError('Unknown corpus coverage basis')
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved.get('version') != VERSION:
                raise ValueError('Unsupported corpus version; revalidate into a fresh corpus')
            if (saved.get('semantic_coverage_version') != SEMANTIC_COVERAGE_VERSION
                    or any(entry.get('coverage_version') != SEMANTIC_COVERAGE_VERSION
                           or any(unit.get('version', SEMANTIC_COVERAGE_VERSION) != SEMANTIC_COVERAGE_VERSION
                                  for unit in entry.get('coverage', ()))
                           for entry in saved.get('entries', ()))):
                raise ValueError('Incompatible semantic coverage version; revalidate into a fresh corpus')
            if any(entry.get('optimizer_coverage_version') not in (None, OPTIMIZER_COVERAGE_VERSION)
                   or any(unit.get('version') != OPTIMIZER_COVERAGE_VERSION
                          for unit in entry.get('optimizer_coverage', ()))
                   for entry in saved.get('entries', ())):
                raise ValueError('Incompatible optimizer coverage version; revalidate into a fresh corpus')
            stored_basis = saved.get('coverage_basis', 'semantic')
            if coverage_basis is not None and stored_basis != coverage_basis:
                raise ValueError('Cannot change a persisted corpus coverage basis')
            self.coverage_basis = stored_basis
            self.entries = saved['entries']
            self.prefixes = saved.get('witnessed_prefixes', [])
            self.quarantine = saved['quarantine']
            self.selections = saved['selections']

    def save(self):
        temporary = self.root / '.corpus-pending.json'
        temporary.write_text(json.dumps(self.summary(), indent=2, allow_nan=False) + '\n')
        temporary.replace(self.path)

    def summary(self):
        return deepcopy({'version': VERSION, 'selection_policy': 'round-robin unless an explicit seed is requested',
                         'coverage_basis':self.coverage_basis,
                         'semantic_coverage_version': SEMANTIC_COVERAGE_VERSION,
                         'optimizer_coverage_version': OPTIMIZER_COVERAGE_VERSION,
                         'entries': self.entries, 'witnessed_prefixes': self.prefixes, 'quarantine': self.quarantine,
                         'selections': self.selections,
                         'active_seeds': sum(e['active'] for e in self.entries)})

    def source(self, entry):
        """A seed is executable evidence, not just a source string or signature."""
        manifest_path = self.root / entry['manifest']
        if not manifest_path.resolve().is_relative_to(self.root) or digest(manifest_path) != entry['manifest_sha256']:
            raise ValueError('Corpus replay manifest checksum mismatch')
        manifest = json.loads(manifest_path.read_text())
        verify(manifest_path.parent, manifest['files'])
        verify(Path(manifest['engine']), manifest['engine_inventory'])
        source = (manifest_path.parent / manifest['source']).read_text()
        if hashlib.sha256(source.encode()).hexdigest() != entry['source_sha256']:
            raise ValueError('Corpus source checksum mismatch')
        return source

    def select(self, api, *, seed_id=None):
        candidates = [e for e in self.entries if e['active'] and e['configuration']['api'] == api]
        if not candidates:
            if seed_id is not None:
                raise ValueError('Requested seed is not active for this API')
            return None
        position = self.selections.get(api, 0)
        if seed_id is None:
            entry = candidates[position % len(candidates)]
        else:
            entry = next((e for e in candidates if e['id'] == seed_id), None)
            if entry is None:
                raise ValueError('Requested seed is not active for this API')
        self.source(entry)
        self.selections[api] = position + 1
        self.save()
        return deepcopy(entry)

    def admit(self, source, row, manifest_path, *, parent_id=None, edit=None):
        coverage = row.get('inspection', {}).get('semantic_coverage', {})
        optimizer = row.get('inspection', {}).get('optimizer_coverage', {})
        optimizer_measured = optimizer.get('version') == OPTIMIZER_COVERAGE_VERSION
        optimizer_units = (deepcopy(optimizer.get('signatures', []))
                           if optimizer_measured and optimizer.get('eligible') else [])
        if any(unit.get('version') != OPTIMIZER_COVERAGE_VERSION for unit in optimizer_units):
            optimizer_units = []
        relative = str(Path(manifest_path).resolve().relative_to(self.root))
        if row.get('verdict') == 'suspected_failure':
            self.quarantine.append({'manifest': relative, 'parent_id': parent_id,
                                    'reason': 'suspected_failure; excluded from mutation and repair'})
            self.save()
            return {'admitted': False, 'reason': 'quarantined_suspected_failure'}
        inspection = row.get('inspection', {})
        units = inspection.get('training_state_evidence') or {}
        if (row.get('configuration', {}).get('check_claim') and inspection.get('legal_source')
                and (not inspection.get('valid') or inspection.get('product_use', {}).get('oracle_gaps'))
                and units and row.get('source_completed')
                and not row.get('quarantined') and row.get('verdict') != 'invalid'
                and row.get('mutation_obligations_valid') is not False):
            # This index is deliberately outside mutation admission and MR coverage.
            manifest_path = self.root / relative
            manifest = json.loads(manifest_path.read_text())
            if manifest['configuration'] != row['configuration']:
                raise ValueError('Prefix configuration changed')
            checksum = hashlib.sha256(source.encode()).hexdigest()
            entry = dict(id='prefix-' + digest(manifest_path)[:24], active=True, manifest=relative,
                         manifest_sha256=digest(manifest_path), source_sha256=checksum,
                         configuration=deepcopy(row['configuration']), witnesses=deepcopy(units),
                         initial_reference=str((manifest_path.parent/manifest['initial_reference']).relative_to(self.root)),
                         cost=dict(seconds=row['finished_seconds']-row['started_seconds']))
            self.source(entry)
            if not any(e['id'] == entry['id'] for e in self.prefixes):
                self.prefixes.append(entry)
                self.save()
            if not inspection.get('valid'):
                return dict(admitted=False, reason='legal_witnessed_prefix', prefix_id=entry['id'])
            row['oracle_prefix_id'] = entry['id']
        if (row.get('mutation_obligations_valid') is False or not row.get('inspection', {}).get('valid') or not coverage.get('eligible')
                or coverage.get('version') != SEMANTIC_COVERAGE_VERSION or not coverage.get('signatures')
                or any(unit.get('version', SEMANTIC_COVERAGE_VERSION) != SEMANTIC_COVERAGE_VERSION
                       for unit in coverage.get('signatures', ()))
                or row.get('verdict') == 'invalid'):
            return {'admitted': False, 'reason': 'unverified_execution_or_coverage'}
        if parent_id is not None and not any(e['id'] == parent_id for e in self.entries + self.prefixes):
            raise ValueError('Unknown corpus parent')
        manifest_path = self.root / relative
        manifest = json.loads(manifest_path.read_text())
        if manifest['configuration'] != row['configuration'] or not (manifest_path.parent / manifest['initial_reference']).is_file():
            raise ValueError('Seed requires its execution configuration and saved initialization')
        source_digest = hashlib.sha256(source.encode()).hexdigest()
        entry = {'id': hashlib.sha256((relative + source_digest).encode()).hexdigest()[:24],
                 'source_sha256': source_digest, 'fingerprint': analyze_stateful(source)['fingerprint'],
                 'configuration': deepcopy(row['configuration']),
                 'manifest': relative, 'manifest_sha256': digest(manifest_path),
                 'initial_reference': str((manifest_path.parent / manifest['initial_reference']).relative_to(self.root)),
                 'parent_id': parent_id, 'edit': deepcopy(edit), 'active': True,
                 'coverage': deepcopy(coverage['signatures']), 'coverage_version': coverage['version'],
                 'optimizer_coverage': optimizer_units,
                 'optimizer_coverage_version': OPTIMIZER_COVERAGE_VERSION if optimizer_measured else None,
                 'evidence': {k: deepcopy(row.get(k)) for k in ('backward_executed', 'transition_signature')},
                 'relation': row.get('relation'), 'verdict': row.get('verdict'),
                 'correctness': row.get('correctness', 'unassessed'),
                 'mr_applicable': row.get('verdict') == 'pass',
                 'cost': {'seconds': row['finished_seconds'] - row['started_seconds'],
                          'usage': deepcopy(row.get('generation_usage'))}}
        entry['evidence']['backward_calls'] = sorted({w['call'] for w in coverage.get('witnesses', [])
            if w['checkpoint'] == 'final' and w['relations']['target_backward_node']['status'] == 'verified'})
        entry['evidence']['witnesses'] = deepcopy(coverage.get('witnesses', []))
        entry['evidence']['optimizer_witnesses'] = deepcopy(optimizer.get('witnesses', [])) if optimizer_measured else []
        entry['evidence']['optimizer_target_update_count'] = (
            optimizer.get('target_update_count') if optimizer_measured and optimizer.get('count_verified') else None)
        entry['evidence']['target_parameter_bindings'] = deepcopy(row.get('target_parameter_bindings', []))
        self.source(entry)
        def units_for(seed):
            if self.coverage_basis == 'semantic':
                return {_key(s) for s in seed['coverage'] + seed.get('optimizer_coverage', [])}
            return {_key({'api':seed['configuration']['api'], 'transition':seed['evidence']['transition_signature']})}
        units = units_for(entry)
        active = [e for e in self.entries if e['active'] and e['configuration']['api'] == row['api']]
        known = {u for e in active for u in units_for(e)}
        new_relation = entry['mr_applicable'] and any(
            not any(e['mr_applicable'] and e['relation'] == entry['relation']
                    and signature in units_for(e) for e in active)
            for signature in units)
        equivalent = [e for e in active if units_for(e) == units
                      and (e['relation'], e['mr_applicable']) == (entry['relation'], entry['mr_applicable'])]
        identical = [e for e in equivalent if e['fingerprint'] == entry['fingerprint']]
        cheaper = identical and entry['cost']['seconds'] < min(e['cost']['seconds'] for e in identical)
        if units - known:
            reason = 'new_semantic_coverage' if self.coverage_basis == 'semantic' else 'new_transition_coverage'
        elif new_relation:
            reason = 'new_applicable_relation'
        elif not identical and len(equivalent) < 2:
            reason = 'alternative_program_structure'
        elif cheaper:
            reason = 'lower_observed_cost'
            for previous in identical:
                previous['active'] = False
        else:
            return {'admitted': False, 'reason': 'duplicate_coverage', 'new_units': 0}
        entry['admission_reason'] = reason
        self.entries.append(entry)
        self.save()
        return {'admitted': True, 'seed_id': entry['id'], 'reason': reason, 'new_units': len(units - known)}
