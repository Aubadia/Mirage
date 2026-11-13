"""Structural operand discovery and bounded, API-independent mutation plans."""
from itertools import combinations
from mirage.contracts.capabilities import (
    Evidence, EvidenceStatus, MutationCandidate, MutationObligations,
)


class MutationInapplicable(ValueError):
    """A diagnosed transformation precondition failure, not a target defect."""
    pass


def discover(tree, backend):
    """Keep paths typed; dotted dictionary keys must not collide with nested keys."""
    objects, storages, active = {}, {}, set()
    found = []

    def visit(value, path):
        if backend.is_tensor(value):
            group = objects.setdefault(id(value), len(objects))
            identity = backend.storage_identity(value)
            storage = None if identity is None else storages.setdefault(identity, len(storages))
            found.append((backend.describe_tensor(value, path, group, storage), value))
        elif isinstance(value, (dict, list, tuple)):
            if id(value) in active:
                raise ValueError("Cyclic operand containers are unsupported")
            active.add(id(value))
            for key, child in (value.items() if isinstance(value, dict) else enumerate(value)):
                if type(key) not in (str, int):
                    raise ValueError("Operand keys must be strings or integers")
                visit(child, path + (key,))
            active.remove(id(value))
    visit(tree, ())
    return found


def reconstruct(tree, replacements, backend):
    """Replace by object identity, retaining repeated tensor/container references."""
    memo = {}
    def visit(value):
        if backend.is_tensor(value):
            return replacements.get(id(value), value)
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, dict):
            result = {key: visit(child) for key, child in value.items()}
        elif isinstance(value, list):
            result = [visit(child) for child in value]
        elif isinstance(value, tuple):
            values = [visit(child) for child in value]
            result = type(value)(*values) if hasattr(value, "_fields") else tuple(values)
        else:
            return value
        memo[id(value)] = result
        return result
    # Discovery rejects cycles before reconstruction.
    discover(tree, backend)
    return visit(tree)


def candidates(tree, backend, obligations=None):
    obligations = obligations or MutationObligations()
    operands = discover(tree, backend)
    plans, seen = [], set()
    for descriptor, tensor in operands:
        if descriptor.object_group in seen:
            continue
        seen.add(descriptor.object_group)
        aliases = [d for d, _ in operands if d.storage_group is not None
                   and d.storage_group == descriptor.storage_group
                   and d.object_group != descriptor.object_group]
        current = obligations.get(descriptor.path, MutationObligations()) if isinstance(obligations, dict) else obligations
        for action in backend.mutation_actions(descriptor):
            reason = None
            if current.preserve_identity:
                reason = "replacement would change object identity"
            elif current.preserve_aliases and aliases:
                reason = "distinct shared-storage views require a joint alias transformation"
            elif current.writable is None:
                reason = "operand write effects are unknown"
            fact = (Evidence(EvidenceStatus.REJECTED, reason) if reason else
                    backend.mutation_eligibility(tensor, action, current))
            plans.append(MutationCandidate(descriptor.path, action, fact))
    return plans


def compositions(plans, limit=2):
    """Bounded combinations do not mix two mutations of the same operand."""
    if type(limit) is not int or not 1 <= limit <= 2:
        raise ValueError("Composition bound must be one or two")
    eligible = [p for p in plans if p.eligibility.established]
    for size in range(1, limit + 1):
        for group in combinations(eligible, size):
            if len({p.path for p in group}) == size:
                yield group


def apply(tree, plan, backend, obligations=None):
    obligations = obligations or MutationObligations()
    selected = next((p for p in candidates(tree, backend, obligations)
                     if p.path == plan.path and p.action == plan.action), None)
    if selected is None or not selected.eligibility.established:
        raise MutationInapplicable(selected.eligibility.reason if selected else "Unknown mutation candidate")
    tensor = dict((d.path, t) for d, t in discover(tree, backend))[plan.path]
    current = obligations.get(plan.path, MutationObligations()) if isinstance(obligations, dict) else obligations
    replacement = backend.mutate(tensor, plan.action, current)
    result = reconstruct(tree, {id(tensor): replacement}, backend)
    evidence = backend.verify_mutation(tensor, replacement, plan.action, current)
    if not evidence.established:
        raise MutationInapplicable(evidence.reason)
    return result, evidence



def canonicalize(tree, backend, obligations):
    """Canonical materialization of actual operands, independent of target identity."""
    changed, evidence = tree, []
    original = discover(tree, backend)
    seen = set()
    for descriptor, tensor in original:
        if descriptor.object_group in seen: continue
        seen.add(descriptor.object_group)
        actions = backend.mutation_actions(descriptor)
        action = next((a for a in ('canonical','values_canonical','logical_materialize') if a in actions), None)
        if action is None: raise MutationInapplicable('No canonical adapter for operand layout')
        plan = next((p for p in candidates(changed, backend, obligations)
                     if p.path == descriptor.path and p.action == action), None)
        if plan is None or not plan.eligibility.established:
            raise MutationInapplicable(plan.eligibility.reason if plan else 'No canonical candidate')
        changed, fact = apply(changed, plan, backend, obligations)
        evidence.append(fact)
    return changed, evidence
