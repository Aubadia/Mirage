"""Framework-neutral candidate admission and bounded primitive path catalogues."""
from hashlib import sha256
import json

from mirage.contracts.candidates import Candidate, path_key
from mirage.contracts.capabilities import Evidence, EvidenceStatus


def legal_paths(config, *, backward=True, optimizer=True):
    """Finite diverse complete paths, never a retained graph as a final state.

    The catalogue is deliberately bounded, not exhaustive. Every complete path
    is checked by the same state machine used immediately before execution.
    """
    from mirage.contracts.exploration import validate_history
    steps = config.get("optimizer_steps", 0)
    if (steps and not optimizer) or ((steps or config["gradient_state"] == "accumulated") and not backward):
        return ()
    minimal = ["zero:none", "forward", "backward", "step"] * steps
    minimal += ["zero:zero" if config["gradient_state"] == "zero" else "zero:none"]
    if config["gradient_state"] == "accumulated":
        minimal += ["forward", "backward"]
    alternatives = [minimal]
    if backward:
        alternatives += [["forward", "backward:retain", "backward", *minimal],
                         ["forward", "backward", "forward", "backward", *minimal]]
    alternatives.append([*minimal, "zero:zero" if config["gradient_state"] == "zero" else "zero:none"]
                        if config["gradient_state"] != "accumulated" else [*minimal, "forward", "backward"])
    result = []
    for actions in alternatives:
        try:
            path = validate_history(actions, config)
        except ValueError:
            continue
        if path not in result:
            result.append(path)
    return tuple(result)


def registry(base, snapshot):
    """Retain inadmissible goals; only affirmative runtime evidence admits work."""
    variants = (
        ("layout", {"representation": "strided"}, ("strided",)),
        ("saved_tensor", {"saved_tensors": True}, ("backward",)),
        ("accumulation", {"microbatches": 2}, ("backward", "microbatch")),
        ("storage", {"representation": "clone", "gradient_state": "zero"}, ("clone", "gradients")),
        ("checkpoint", {"checkpoint": True}, ("backward",)),
        ("layout", {"representation": "offset", "optimizer_steps": 1}, ("offset", "optimizer")),
        (None, {"gradient_state": "accumulated"}, ("backward", "gradients", "accumulated_gradients")),
        (None, {"graph_mode": "retained"}, ("backward",)),
        (None, {"graph_mode": "recomputed"}, ("backward",)),
        ("layout", {"representation": "strided", "gradient_state": "accumulated"}, ("strided", "backward", "gradients", "accumulated_gradients")),
        ("storage", {"representation": "clone", "optimizer_steps": 2}, ("clone", "optimizer")),
        (None, {"representation": "shared"}, ("shared",)),
        ("mutation", {"backend": base["backend"] if base["backend"] != "eager" else "compile_eager"}, ("mutation",)),
    )
    candidates = []
    # An absent subject keeps representation goals unresolved instead of erasing
    # them from the global goal pool. Nonrepresentation states need no operand.
    subjects = snapshot.subjects or ({"path": None, "facts": {}},)
    for subject in subjects:
        for rule, change, requirements in variants:
            path = subject["path"]
            facts = dict(snapshot.facts)
            facts.update({k: Evidence(**dict(v, status=EvidenceStatus(v["status"])))
                          for k, v in subject.get("facts", {}).items()})
            if "representation" in change:
                requirements = (*requirements, "target_operand", "alias_preserved")
            required = {key: facts.get(key, Evidence()) for key in requirements}
            rejected = any(f.status in (EvidenceStatus.REJECTED, EvidenceStatus.UNSUPPORTED)
                           for f in required.values())
            status = "rejected" if rejected else "admitted" if all(f.established for f in required.values()) else "unresolved"
            reasons = tuple(f"{key}: {f.reason}" for key, f in required.items() if not f.established)
            config = dict(base, **change)
            if path is not None:
                config.update(input_path=list(path), input_name=path_key(path))
            key = json.dumps(config, sort_keys=True)
            candidates.append(Candidate(sha256(key.encode()).hexdigest()[:16],
                rule or "training_state", (path,) if path is not None else (), "pre_transform",
                change, tuple(requirements), tuple(change), tuple(change),
                ("target_api", "target_state", "checkpoint_prerequisites"), status, reasons,
                rule, config, {k: vars(v) for k, v in required.items()}))
    return tuple(candidates)


def unmet_requirements(candidates):
    return sorted({key for candidate in candidates if candidate.status != "admitted"
                   for key, evidence in candidate.evidence.items()
                   if evidence["status"] != EvidenceStatus.VERIFIED or evidence["provenance"] == "llm"})
