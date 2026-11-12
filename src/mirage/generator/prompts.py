"""Stage-specific generation requests with explicit program and MR contracts."""

from dataclasses import asdict

from mirage.contracts.apis import PROFILES
from mirage.generator.provider import Request, Stage
from mirage.rules.registry import RULES


INSTRUCTIONS = {
    Stage.CANONICAL: "Generate the canonical forward and loss for the target API.",
    Stage.SKELETON: "Derive the reusable controlled training program from the validated canonical program.",
    Stage.TRANSITION: "Fill the ordered state-transition actions at the harness injection points.",
    Stage.FOLLOWUP: "Produce the follow-up program preserving the specified MR input relation and computation.",
}


def request(stage, config, previous=None, mr=None):
    profile = PROFILES[config.api]
    return Request(stage, INSTRUCTIONS[stage] + (
        " Return only a JSON object with exactly source (Python text) and actions (array of strings)."
        " The supported grammar is the supplied forward/loss AST; whitespace, comments and docstrings may vary."
        " Preserve all operations and signatures. During canonical generation only, you may choose literal"
        " dropout p in (0,1), softmax dim in (-2,-1,0,1), gelu approximate='none' or 'tanh',"
        " or layer_norm eps in [1e-8,1e-3]. Later stages must preserve the canonical computation."
        " Do not add side effects or instrumentation."
        " The harness owns setup, representation construction, warm-up, RNG restoration, checkpointing,"
        " forward/loss, backward, AMP, Adam and state probes. Return the ordered actions exactly."
    ), {
        "api": config.api, "profile": asdict(profile), "grammar_example": profile.source,
        "configuration": asdict(config), "actions": list(config.actions),
        "previous_program": previous.source if previous else None,
        "mr": RULES[mr].as_dict() if mr else None,
    })
