"""Extract API context from the tested installation, never a guessed signature."""

import hashlib
import inspect
import re

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
    references = referenced_documentation(api, doc)
    return {
        "documentation_references": references,
        "api": api, "torch_version": str(torch.__version__), "signature": signature,
        "signature_source": source, "documentation": doc[:14000],
        "documentation_truncated": len(doc) > 14000,
        "documentation_sha256": hashlib.sha256(doc.encode()).hexdigest(),
        "kind": "tensor_method" if api.startswith("Tensor.") else "callable",
    }


def referenced_documentation(api, documentation, *, limit=3, budget=16000):
    """Resolve a bounded one-hop closure of installed Sphinx API references."""
    references = []
    seen = {api}
    for _, raw in re.findall(r':(meth|class|func):`([^`]+)`', documentation):
        name = raw.lstrip('~')
        candidates = [name] if name.startswith(('torch.', 'Tensor.')) else [
            api.rsplit('.', 1)[0] + '.' + name, 'torch.nn.' + name, 'torch.' + name]
        for target_name in candidates:
            if target_name in seen:
                continue
            try:
                target = resolve(target_name)
            except (AttributeError, ValueError):
                continue
            doc = inspect.getdoc(target) or ''
            if not doc:
                continue
            seen.add(target_name)
            supplied = doc[:budget]
            references.append({'api': target_name, 'source': 'installed docstring reference',
                'referenced_by': api, 'documentation': supplied, 'truncated': len(doc) > len(supplied),
                'sha256': hashlib.sha256(doc.encode()).hexdigest(),
                'supplied_sha256': hashlib.sha256(supplied.encode()).hexdigest()})
            budget -= len(supplied)
            break
        if len(references) >= limit or budget <= 0:
            break
    return references
