"""Deterministic controller fixture; never counted as real-model validation."""

import json

from mirage.generator.provider import Response, Usage
from test_program import SOURCE


class FixtureProvider:
    def generate(self, request, *, deadline):
        if request.context["contract"] == "training-program":
            value = {"source": SOURCE}
        else:
            config = request.context["configuration"]
            actions = []
            for _ in range(config["optimizer_steps"]):
                actions += ["zero:none", "forward", "backward", "step"]
            actions.append("zero:zero" if config["gradient_state"] == "zero" else "zero:none")
            if config["gradient_state"] == "accumulated":
                actions += ["forward", "backward"]
            if request.context["path_hint"].startswith("alternative") and config["gradient_state"] != "accumulated":
                actions.append(actions[-1])
            value = {"actions": actions}
        return Response(json.dumps(value), Usage(0, 0), "deterministic-test-fixture")


def create():
    return FixtureProvider()
