"""Seeded inverse-count sampling with a uniform exploration component."""

import math
import random
from collections.abc import Mapping

from mirage.contracts.models import Candidate


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
