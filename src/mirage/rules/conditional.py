"""Conditional relations over verified generated executions."""

from mirage.inspector.comparison import differences, equal
import math


RULE_IDS = ("layout", "storage", "checkpoint", "mutation", "saved_tensor", "accumulation")
RULE_VERSION = 3


def inspect(rule, left, right):
    if rule is None:
        return {"verdict": None, "correctness": "unassessed", "reason": "coverage-only"}
    if rule not in RULE_IDS:
        raise ValueError("Unknown MR")
    base = {"rule": rule, "rule_version": RULE_VERSION, "confirmed_defect": False}
    if not left or not right or not left.get("completed") or not right.get("completed"):
        return dict(base, verdict=None, reason="Incomplete execution or missing evidence")
    if not all(r.get("api_executed") and (not r["config"].get("requires_backward", True) or r.get("backward_executed"))
               and r.get("target_state_reached") for r in (left, right)):
        return dict(base, verdict="invalid", reason="Required runtime path/state evidence missing")
    reasons = []
    if any(record["graph"]["identity_sensitive"] for record in (left, right)):
        reasons.append("Program uses effects or introspection outside the MR certificate")
    if left["graph"]["fingerprint"] != right["graph"]["fingerprint"]:
        reasons.append("Different canonical computations")
    if not equal(left["initial"], right["initial"]):
        reasons.append("Different initial inputs, state, alias topology or RNG")
    if not equal(left["common"], right["common"]):
        reasons.append("Different state/input at the transformation checkpoint")
    if not equal(left["environment"], right["environment"]):
        reasons.append("Different execution environments")
    a, b = dict(left["config"]), dict(right["config"])
    allowed = {
        "layout": ("representation",), "storage": ("representation",),
        "checkpoint": ("checkpoint",), "mutation": ("backend",),
        "saved_tensor": ("saved_tensors",),
        "accumulation": ("microbatches", "accumulation_source"),
    }[rule]
    for config in (a, b):
        for key in allowed:
            config.pop(key, None)
    if a != b:
        reasons.append("Configurations differ outside the MR input relation")
    if rule != "mutation" and not all(r["pure"] for r in (left, right)):
        reasons.append("Purity/representation-independence was not established")
    if rule not in ("mutation", "accumulation"):
        actual = [[call for call in record["target_calls"] if call["phase"] == "training"]
                  for record in (left, right)]
        if not all(calls and "inputs" in calls[0] for calls in actual):
            reasons.append("Actual target operands were not captured")
        elif not equal(actual[0][0]["inputs"], actual[1][0]["inputs"]):
            reasons.append("Logical target operands differ after the transformation")
    if rule not in ("checkpoint",) and any(r["stochastic"] for r in (left, right)):
        reasons.append("Stochastic computation outside this MR")
    if rule == "layout" and (left["config"]["representation"] != "contiguous"
                             or right["config"]["representation"] not in ("strided", "offset")):
        reasons.append("Not a declared layout transformation")
    if rule == "storage" and right["config"]["representation"] not in ("clone", "shared"):
        reasons.append("Not a declared storage transformation")
    if rule == "storage" and right["config"]["representation"] == "shared":
        metadata = right["common"]["metadata"]
        parameters = {value["storage_group"] for key, value in metadata.items() if key.startswith(".1.")}
        if any(value["requires_grad"] or value["storage_group"] in parameters
               for key, value in metadata.items() if key.startswith(".0.")):
            reasons.append("Shared-storage MR excludes trainable or parameter-aliasing inputs")
    if rule == "checkpoint" and (left["config"].get("checkpoint") or not right["config"].get("checkpoint")):
        reasons.append("Not a checkpoint pair")
    if rule == "saved_tensor":
        evidence = right["saved_tensors"]
        if (left["saved_tensors"]["enabled"] or not evidence["enabled"]
                or not evidence["packed"] or not evidence["unpacked"] or not evidence["roundtrip_valid"]):
            reasons.append("No verified lossless saved-tensor roundtrip")
    if rule == "accumulation":
        if not all(r["batch_separable"] and r.get("sample_loss_matches") for r in (left, right)):
            reasons.append("Batch separability was not certified")
        if (left["config"].get("precision") != "fp32" or right["config"].get("microbatches", 1) <= 1
                or not left["config"].get("accumulation_source")
                or left["config"].get("gradient_state") != "none"):
            reasons.append("Accumulation requires FP32, empty initial gradients and a full/microbatch pair")
    if rule == "mutation" and (left["config"].get("backend") != "eager"
                               or right["config"].get("backend") == "eager"):
        reasons.append("Mutation requires eager/compiled execution")
    if reasons:
        return dict(base, verdict="inapplicable", reasons=reasons)
    import torch

    def nonfinite(value):
        if isinstance(value, torch.Tensor):
            return (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return any(nonfinite(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(nonfinite(item) for item in value)
        return isinstance(value, float) and not math.isfinite(value)

    for record in (left, right):
        if nonfinite(record["observations"]):
            return dict(base, verdict=None, correctness="unassessed",
                        reason="Nonfinite observations require an API-specific numerical policy")
    precision = left["config"].get("precision", "fp32")
    atol, rtol = {"fp32": (1e-5, 1e-4), "fp16": (2e-3, 1e-2), "bf16": (1e-2, 5e-2)}[precision]
    tolerance = {"atol": atol, "rtol": rtol}
    observables = ["loss", "gradients", "parameters", "modules", "optimizer", "scaler"]
    if rule != "accumulation":
        observables.append("output")
    if rule == "checkpoint":
        observables.append("rng")
    mismatches = []
    for name in observables:
        mismatches.extend(differences(left["observations"][name], right["observations"][name],
                                      None if name in ("scaler", "rng") else tolerance, name))
    # Successful update counts are outputs, not an applicability filter.
    mismatches.extend(differences(left["updates"], right["updates"], path="updates"))
    return dict(base, verdict="suspected_failure" if mismatches else "pass",
                differences=mismatches, observables=observables,
                tolerance_policy={"version": 3, "screening_only": True, **tolerance,
                                  "nonfinite": "unassessed"})
