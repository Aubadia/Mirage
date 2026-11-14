"""Verified stateful execution seeds with immutable replay/lineage references."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from mirage.artifacts.bundle import digest, verify
from mirage.contracts.capabilities import SEMANTIC_COVERAGE_VERSION
from mirage.generator.program import analyze_stateful

VERSION = 'stateful-corpus-1'


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Corpus:
    """Round-robin seeds; all attempt artifacts survive admission/deduplication."""
    def __init__(self, root, *, coverage_basis=None):
        self.root = Path(root).resolve()
        self.path = self.root / 'corpus.json'
        self.entries, self.quarantine, self.selections = [], [], {}
        self.coverage_basis = coverage_basis or 'semantic'
        if self.coverage_basis not in ('semantic', 'transition'):
            raise ValueError('Unknown corpus coverage basis')
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved.get('version') != VERSION:
                raise ValueError('Unsupported corpus version')
            stored_basis = saved.get('coverage_basis', 'semantic')
            if coverage_basis is not None and stored_basis != coverage_basis:
                raise ValueError('Cannot change a persisted corpus coverage basis')
            self.coverage_basis = stored_basis
            self.entries = saved['entries']
            self.quarantine = saved['quarantine']
            self.selections = saved['selections']

    def save(self):
        temporary = self.root / '.corpus-pending.json'
        temporary.write_text(json.dumps(self.summary(), indent=2, allow_nan=False) + '\n')
        temporary.replace(self.path)

    def summary(self):
        return deepcopy({'version': VERSION, 'selection_policy': 'round-robin unless an explicit seed is requested',
                         'coverage_basis':self.coverage_basis,
                         'entries': self.entries, 'quarantine': self.quarantine,
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
        relative = str(Path(manifest_path).resolve().relative_to(self.root))
        if row.get('verdict') == 'suspected_failure':
            self.quarantine.append({'manifest': relative, 'parent_id': parent_id,
                                    'reason': 'suspected_failure; excluded from mutation and repair'})
            self.save()
            return {'admitted': False, 'reason': 'quarantined_suspected_failure'}
        if (row.get('mutation_obligations_valid') is False or not row.get('inspection', {}).get('valid') or not coverage.get('eligible')
                or coverage.get('version') != SEMANTIC_COVERAGE_VERSION or not coverage.get('signatures')
                or row.get('verdict') == 'invalid'):
            return {'admitted': False, 'reason': 'unverified_execution_or_coverage'}
        if parent_id is not None and not any(e['id'] == parent_id for e in self.entries):
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
                 'evidence': {k: deepcopy(row.get(k)) for k in ('backward_executed', 'transition_signature')},
                 'relation': row.get('relation'), 'verdict': row.get('verdict'),
                 'correctness': row.get('correctness', 'unassessed'),
                 'mr_applicable': row.get('verdict') == 'pass',
                 'cost': {'seconds': row['finished_seconds'] - row['started_seconds'],
                          'usage': deepcopy(row.get('generation_usage'))}}
        entry['evidence']['backward_calls'] = sorted({w['call'] for w in coverage.get('witnesses', [])
            if w['checkpoint'] == 'final' and w['relations']['target_backward_node']['status'] == 'verified'})
        self.source(entry)
        def units_for(seed):
            if self.coverage_basis == 'semantic':
                return {_key(s) for s in seed['coverage']}
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
