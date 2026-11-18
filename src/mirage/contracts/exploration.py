"""Finite state targets and checked primitive history plans for generated programs."""

from mirage.generator.program import ContractError
from mirage.generator.provider import Request, Stage


def targets(base, metadata, limit):
    tensor_names = [name.removeprefix(".0.") for name in metadata
                    if name.startswith(".0.") and "." not in name.removeprefix(".0.")]
    names = sorted(tensor_names, key=lambda name: "float" not in metadata[f".0.{name}"]["dtype"])
    variants = [
        ("layout", {"representation": "strided"}),
        ("saved_tensor", {"saved_tensors": True}),
        ("accumulation", {"microbatches": 2}),
        ("storage", {"representation": "clone", "gradient_state": "zero"}),
        ("checkpoint", {"checkpoint": True}),
        ("layout", {"representation": "offset", "optimizer_steps": 1}),
        (None, {"gradient_state": "accumulated"}),
        (None, {"graph_mode": "retained"}),
        (None, {"graph_mode": "recomputed"}),
        ("layout", {"representation": "strided", "gradient_state": "accumulated"}),
        ("storage", {"representation": "clone", "optimizer_steps": 2}),
        (None, {"representation": "shared"}),
    ]
    if not names:
        names = [""]
        variants = [(rule, change) for rule, change in variants if "representation" not in change]
    # Every registered tensor can be explored without adding an API-specific recipe.
    result = [(rule, dict(base, input_name=name, **changes)) for name in names for rule, changes in variants]
    return result if limit is None else result[:limit]


def source_config(rule, followup):
    result = dict(followup)
    if rule in ("layout", "storage"):
        result["representation"] = "contiguous"
    elif rule == "checkpoint":
        result["checkpoint"] = False
    elif rule == "saved_tensor":
        result["saved_tensors"] = False
    elif rule == "accumulation":
        result.update(microbatches=1, accumulation_source=True)
    elif rule == "mutation":
        result["backend"] = "eager"
    return result


def validate_history(actions, config):
    if not isinstance(actions, (list, tuple)) or any(not isinstance(a, str) for a in actions):
        raise ContractError("actions must be an array of primitive names")
    if len(actions) > 8 * (config["optimizer_steps"] + 1):
        raise ContractError("History exceeds its action budget")
    graph = False
    gradients = "none"
    steps = 0
    for action in actions:
        if action in ("zero:none", "zero:zero"):
            gradients = "none" if action == "zero:none" else "zero"
        elif action == "forward":
            graph = True
        elif action in ("backward", "backward:retain"):
            if not graph:
                raise ContractError("backward requires a live forward graph; insert forward before this backward")
            graph = action.endswith(":retain")
            gradients = "accumulated"
        elif action == "step":
            if gradients != "accumulated" or graph:
                raise ContractError("step requires accumulated gradients and no live graph; backward:retain must be followed by a consuming backward before step")
            steps += 1
        else:
            raise ContractError(f"Unknown history action: {action}")
    if graph or steps != config["optimizer_steps"] or gradients != config["gradient_state"]:
        raise ContractError(
            f"Required final state: optimizer_steps={config['optimizer_steps']}, "
            f"gradient_state={config['gradient_state']}, live_graph=False. "
            f"Your plan produces optimizer_steps={steps}, gradient_state={gradients}, live_graph={graph}. "
            "Only step increments optimizer_steps. step does not clear gradients; "
            "finish with zero:none or zero:zero when that final gradient state is required."
        )
    return tuple(actions)


def history_request(config, source, path, paths=None):
    if paths is not None:
        return Request(Stage.TRANSITION,
            "Select one offered legal complete history. Return JSON with exactly actions, "
            "equal to one offered path. Do not invent, extend or repair actions. "
            "The catalogue already enforces graph lifetime, update count and final gradient state. "
            "Use the path hint to rank diversity among these valid choices.",
            {"contract": "state-plan", "configuration": config,
             "offered_paths": [list(p) for p in paths],
             "path_hint": "minimal history" if path == 0 else "alternative legal accumulation/history"})
    return Request(Stage.TRANSITION, (
        "Plan a legal history before the target training operation. Return JSON with exactly actions. "
        "Available actions: zero:none clears gradients to None; zero:zero allocates zero gradients; "
        "forward builds a fresh graph; backward consumes it and accumulates gradients; "
        "backward:retain accumulates gradients and retains the graph; step performs an optimizer update. "
        "Do not step with a retained graph. End without a live graph, with exactly optimizer_steps "
        "updates and gradient_state none/zero/accumulated as requested. Never restore trained state. "
        "Use at most 8*(optimizer_steps+1) actions. Prefer distinct legal histories for different path hints. "
        f"This request requires exactly {config['optimizer_steps']} step actions, final gradients "
        f"{config['gradient_state']}, and no live graph."
    ), {"contract": "state-plan", "configuration": config, "source": source,
        "path_hint": "minimal history" if path == 0 else "alternative legal accumulation/history"})


def property_configuration(api, device, attempt, family):
    """Deterministic coverage cycle over general representation/execution choices."""
    representations=('original','strided','offset','negative','conjugate','singleton_stride','expanded')
    return dict(api=api,device=device,seed=attempt,attempt=attempt,family=family,
                representation=representations[attempt%len(representations)] if family in ('materialization','gradient_reference') else 'original',
                subject=attempt//len(representations),
                backend=('eager' if attempt%2 else 'inductor') if family=='gradient_reference' else 'eager')


def stateful_input_profile(attempt):
    """Finite, deterministic category schedule with no target identity input."""
    from mirage.contracts.capabilities import INPUT_CATEGORIES, InputCoverage
    if type(attempt) is not int or attempt < 0:
        raise ValueError("attempt must be nonnegative")
    return InputCoverage((INPUT_CATEGORIES[attempt % len(INPUT_CATEGORIES)],))


def stateful_state_predicates(observed, attempt):
    """Select goals from observed capabilities, without target API identifiers.

    These are exploration requests, not reachability certificates or transitions.
    The generated program must implement and reach its selected goal.
    """
    from dataclasses import asdict
    from mirage.contracts.capabilities import StatePredicate
    choices=[]
    if observed.get('backward_executed'):
        choices.extend((StatePredicate('gradient','backward_count',2,'final'),
                        StatePredicate('graph_lifecycle','target_backward_count',2,'final'),
                        StatePredicate('execution_mode','compiled_execution',True,'final')))
    if 'context_exit' in observed.get('transition_signature',()):
        choices.append(StatePredicate('context','restoration_count',2,'final'))
    return [asdict(choices[attempt % len(choices)])] if choices else []


def stateful_diversity_predicates(primary, attempt):
    """Explore joint abstract states, not source templates or prescribed actions."""
    from copy import deepcopy
    goals=deepcopy(list(primary))
    if goals and not any(p['dimension']=='graph_lifecycle' and p['field']=='target_boundary_count' for p in goals):
        goals.append(dict(dimension='graph_lifecycle',field='target_boundary_count',
                          expected=1+attempt%2,checkpoint='final'))
    return goals
