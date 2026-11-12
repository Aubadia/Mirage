"""Extract API context from the tested installation, never a guessed signature."""

import hashlib
import inspect

from mirage.contracts.apis import validate_api_name


def resolve(api):
    import torch

    validate_api_name(api)
    parts = api.split(".")
    value = torch.Tensor if parts[0] == "Tensor" else torch
    for part in parts[1:]:
        value = getattr(value, part)
    if not callable(value):
        raise ValueError(f"Target is not callable: {api}")
    return value


def describe(api):
    import torch

    target = resolve(api)
    doc = inspect.getdoc(target) or ""
    signature = None
    source = "unavailable"
    try:
        signature = str(inspect.signature(target))
        source = "inspect.signature"
    except (TypeError, ValueError):
        # Builtin operators commonly publish their Python signature in the docstring.
        for line in doc.splitlines()[:12]:
            if "(" in line and ")" in line:
                signature = line.strip()
                source = "installed docstring (unparsed)"
                break
    return {
        "api": api, "torch_version": str(torch.__version__), "signature": signature,
        "signature_source": source, "documentation": doc[:14000],
        "documentation_truncated": len(doc) > 14000,
        "documentation_sha256": hashlib.sha256(doc.encode()).hexdigest(),
        "kind": "tensor_method" if api.startswith("Tensor.") else "callable",
    }
