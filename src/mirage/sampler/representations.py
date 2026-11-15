"""Finite per-source representation queues built from observed operand candidates."""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from mirage.artifacts.bundle import digest

VERSION = 'representation-traversal-1'


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def program_goal_catalog(catalog):
    """LLM planning owns semantic data and training state, not target storage actions."""
    return [g for g in catalog if g['predicate']['dimension'] not in ('representation', 'tensor_relation')]


def previously_reached(plan, row):
    """Reuse exact runtime mutation evidence, independently of the MR verdict."""
    followup = row.get('followup_inspection', {})
    if (row.get('followup_completed') is not True or followup.get('valid') is not True
            or row.get('mutation_obligations_valid') is not True
            or row.get('mutation') != plan):
        return None
    for relation in ('target_backward_node', 'callable_backward_node'):
        def calls(inspection):
            return {w['call'] for w in inspection.get('semantic_coverage', {}).get('witnesses', [])
                    if w.get('checkpoint') == 'final'
                    and w.get('relations', {}).get(relation, {}).get('status') == 'verified'}
        if not calls(row.get('inspection', {})) <= calls(followup):
            return None
    matches = [fact for fact in row.get('followup_mutations', [])
               if fact.get('status') == 'verified'
               and all(fact.get(key) == value for key, value in plan.items())]
    return matches[0] if len(matches) == 1 else None


class RepresentationTraversal:
    """Round-robin source queues; planned eligibility never establishes attainment."""
    def __init__(self):
        self.sources = []
        self.waiting_sources = deque()

    def register(self, source, row):
        if (row.get('proposal_origin') == 'framework_representation'
                or row.get('verdict') == 'suspected_failure'
                or row.get('inspection', {}).get('valid') is not True
                or row.get('mutation_obligations_valid') is not True):
            return None
        candidates = row.get('mutation_candidates')
        if not isinstance(candidates, list):
            return None
        config = {k: deepcopy(v) for k, v in row['configuration'].items()
                  if k not in ('attempt', 'proposal_index', 'canonical_reference', 'oracle_seen_units')}
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        identity = hashlib.sha256(_key([source_hash, config]).encode()).hexdigest()[:24]
        if any(s['source_id'] == identity for s in self.sources):
            return identity
        directory = Path(row['directory']).resolve()
        initial = directory / 'execution/source.pt'
        if not initial.is_file():
            return None
        plans, seen = [], set()
        for candidate in candidates:
            plan = {k: deepcopy(candidate[k]) for k in ('path', 'action', 'call_index')}
            key = _key(plan)
            if key in seen:
                continue
            seen.add(key)
            eligibility = deepcopy(candidate.get('eligibility', {}))
            plans.append({'id': hashlib.sha256(key.encode()).hexdigest()[:16], 'plan': plan,
                          'eligibility': eligibility,
                          'status': 'pending' if eligibility.get('status') == 'verified' else 'ineligible'})
            observed = previously_reached(plan, row) if plans[-1]['status'] == 'pending' else None
            if observed is not None:
                plans[-1].update(status='previously_reached', evidence={
                    'directory': str(directory), 'member': 'followup', 'mutation': deepcopy(observed)})
        seconds = row.get('process', {}).get('elapsed_seconds')
        measured = seconds if type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 else None
        self.sources.append({'source_id': identity, 'source_sha256': source_hash,
            'source_directory': str(directory), 'configuration': config,
            'initial_reference': str(initial), 'initial_sha256': digest(initial),
            'required_backward_calls': sorted({w['call'] for w in
                row['inspection'].get('semantic_coverage', {}).get('witnesses', [])
                if w.get('checkpoint') == 'final' and
                w.get('relations', {}).get('target_backward_node', {}).get('status') == 'verified'}),
            'required_callable_backward_calls': sorted({w['call'] for w in
                row['inspection'].get('semantic_coverage', {}).get('witnesses', [])
                if w.get('checkpoint') == 'final' and
                w.get('relations', {}).get('callable_backward_node', {}).get('status') == 'verified'}),
            'measured_worker_seconds': measured, 'candidates': plans})
        if any(p['status'] == 'pending' for p in plans):
            self.waiting_sources.append(len(self.sources) - 1)
        return identity

    def choose(self, remaining, fallback):
        """Serve existing waiters before newly arriving sources can pass them."""
        for _ in range(len(self.waiting_sources)):
            index = self.waiting_sources.popleft()
            source = self.sources[index]
            candidate = next((p for p in source['candidates'] if p['status'] == 'pending'), None)
            if candidate is None:
                continue
            measured = source['measured_worker_seconds']
            reserve = max(10., 1.5 * measured + 5.) if measured is not None else fallback
            if remaining <= reserve:
                self.waiting_sources.append(index)
                continue
            path = Path(source['source_directory']) / 'program.py'
            if digest(path) != source['source_sha256'] or digest(Path(source['initial_reference'])) != source['initial_sha256']:
                self.waiting_sources.appendleft(index)
                raise ValueError('Representation parent source or initialization checksum changed')
            candidate['status'] = 'dispatched'
            # New sources join after these waiters; a growing catalog cannot
            # move an older source's next turn indefinitely into the future.
            if any(p['status'] == 'pending' for p in source['candidates']):
                self.waiting_sources.append(index)
            return deepcopy(source), deepcopy(candidate), path.read_text(), reserve
        return None

    def complete(self, source_id, candidate_id, row):
        source = next(s for s in self.sources if s['source_id'] == source_id)
        candidate = next(c for c in source['candidates'] if c['id'] == candidate_id)
        candidate.update(status='reached' if row.get('representation_candidate_attained') is True
                         and row.get('mutation_obligations_valid') is True else 'not_verified',
                         directory=row['directory'], reason=row.get('reason'),
                         seconds=row['finished_seconds'] - row['started_seconds'])
        return candidate['status']

    def summary(self):
        return deepcopy({'version': VERSION, 'enabled': True, 'sources': self.sources,
            'scope': 'Observed single-action candidates per source/configuration; no exhaustive representation or combination claim',
            'policy': 'Every other nonrepair opportunity, FIFO source turns with new arrivals behind existing waiters and backend candidate order; other slots retain fresh/state exploration'})
