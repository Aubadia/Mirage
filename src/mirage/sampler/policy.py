"""Seeded inverse-count sampling with a uniform exploration component."""

import math
import random
from collections.abc import Mapping

from typing import Hashable

Candidate = Hashable


class Sampler:
    def __init__(self, candidates: tuple[Candidate, ...], *, seed: int, epsilon: float = 0.1) -> None:
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("Candidates must be nonempty and unique")
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")

        self.candidates = tuple(candidates)
        self.epsilon = epsilon
        self.random = random.Random(seed)

    def probabilities(self, counts: Mapping[Candidate, int]) -> tuple[float, ...]:
        if set(counts).difference(self.candidates):
            raise ValueError("Counts contain an unknown candidate")
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("Counts must be nonnegative integers")

        weights = [1 / (1 + counts.get(candidate, 0)) for candidate in self.candidates]
        total = sum(weights)
        uniform = self.epsilon / len(weights)
        return tuple(uniform + (1 - self.epsilon) * weight / total for weight in weights)

    def sample(self, counts: Mapping[Candidate, int]) -> Candidate:
        return self.random.choices(self.candidates, weights=self.probabilities(counts), k=1)[0]


class FeedbackSampler:
    """Seeded exploration with smoothed validity × novelty / measured seconds."""
    def __init__(self, *, seed=0, epsilon=0.1, coverage_basis='semantic'):
        if isinstance(epsilon, bool) or not 0 <= epsilon <= 1 or not math.isfinite(epsilon):
            raise ValueError('epsilon must be finite and in [0, 1]')
        if coverage_basis not in ('semantic', 'transition'):
            raise ValueError('Unknown coverage basis')
        self.random = random.Random(seed)
        self.epsilon, self.coverage_basis = epsilon, coverage_basis
        self.families, self.parents = {}, {}
        self.seen, self.relations = set(), set()

    @staticmethod
    def _empty():
        return {'attempts':0, 'valid':0, 'gain':0., 'seconds':0.}

    def _estimate(self, candidate):
        family = self.families.get((candidate['api'], candidate['kind']), self._empty())
        own = self.parents.get((candidate['api'], candidate.get('parent'), candidate['kind']), self._empty())
        # Family evidence transfers across freshly admitted seeds; two prior trials
        # keep an untested parent from being treated as certainly good or bad.
        validity = (family['valid'] + 1) / (family['attempts'] + 2)
        gain = (family['gain'] + 1) / (family['valid'] + 1)
        cost = (family['seconds'] + 10) / (family['attempts'] + 1)
        probability = (own['valid'] + 2 * validity) / (own['attempts'] + 2)
        expected_gain = (own['gain'] + 2 * gain) / (own['valid'] + 2)
        seconds = (own['seconds'] + 2 * cost) / (own['attempts'] + 2)
        return {'validity_estimate':probability, 'gain_estimate':expected_gain,
                'seconds_estimate':seconds, 'score':probability * expected_gain / max(seconds, .001)}

    def choose(self, candidates, *, reason='adaptive'):
        if not candidates:
            raise ValueError('No statically plannable candidates')
        if len({(c['api'], c.get('parent'), c['kind']) for c in candidates}) != len(candidates):
            raise ValueError('Candidates must be distinct')
        estimates = [self._estimate(c) for c in candidates]
        total = sum(e['score'] for e in estimates)
        probabilities = [self.epsilon/len(candidates) + (1-self.epsilon)*e['score']/total for e in estimates]
        index = self.random.choices(range(len(candidates)), weights=probabilities, k=1)[0]
        return dict(candidates[index]), {'policy':'adaptive-v1', 'reason':reason,
                'coverage_basis':self.coverage_basis, 'candidate_count':len(candidates),
                'selected_index':index, 'selected_probability':probabilities[index],
                'candidates':[dict(c, **estimate, probability=p) for c, estimate, p in zip(candidates, estimates, probabilities)]}

    def observe(self, candidate, row):
        from mirage.artifacts.report import verified_search_units
        seconds = row['finished_seconds'] - row['started_seconds']
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Feedback requires measured nonnegative elapsed seconds')
        units = verified_search_units(row, self.coverage_basis)
        pairs = {(u, row['relation']) for u in units
                 if row.get('relation') and row.get('verdict') in ('pass', 'suspected_failure')}
        new_units, new_pairs = units - self.seen, pairs - self.relations
        self.seen.update(units); self.relations.update(pairs)
        gain = float(bool(new_units)) + float(bool(new_pairs))
        for table, key in ((self.families, (candidate['api'], candidate['kind'])),
                           (self.parents, (candidate['api'], candidate.get('parent'), candidate['kind']))):
            stats = table.setdefault(key, self._empty())
            stats['attempts'] += 1
            stats['valid'] += bool(units)
            stats['gain'] += gain
            stats['seconds'] += seconds
        return {'verified':bool(units), 'new_units':len(new_units), 'new_applicable_units':len(new_pairs),
                'bounded_gain':gain, 'measured_seconds':seconds, 'coverage_basis':self.coverage_basis}
