"""Framework-neutral limits for independent API sessions sharing resources."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirage.contracts.programs import RunOptions


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class SchedulingOptions:
    workers: int
    generation_concurrency: int = 2
    active_api_limit: int = 1
    per_api_worker_limit: int | None = None
    per_api_generation_limit: int | None = None

    def __post_init__(self):
        for name in ('workers', 'generation_concurrency', 'active_api_limit'):
            _positive(getattr(self, name), name)
        for name, total in (('per_api_worker_limit', self.workers),
                            ('per_api_generation_limit', self.generation_concurrency)):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, total)
            else:
                _positive(value, name)
                if value > total:
                    raise ValueError(f'{name} cannot exceed shared capacity')


@dataclass(frozen=True)
class APIJob:
    identifier: str
    api: str
    options: 'RunOptions'
    stateful_options: dict = field(default_factory=dict)
    check: bool = False

    def __post_init__(self):
        if not self.identifier or not self.api:
            raise ValueError('Session identifier and API are required')
        if self.check and (self.options.strategy != 'stateful' or not self.options.generation_pipeline
                           or self.stateful_options):
            raise ValueError('API check sessions require a stateful generation pipeline and a resolved name')
        allowed = {'edges', 'predicates', 'path_policy', 'coverage_refinement', 'search_policy',
                   'generation_mode', 'coverage_basis', 'representation_traversal', 'oracle_improvements'}
        if set(self.stateful_options) - allowed or self.stateful_options and self.options.strategy != 'stateful':
            raise ValueError('Invalid stateful session options')
        if self.options.strategy not in ('training', 'stateful'):
            raise ValueError('Shared API sessions support training and stateful')
