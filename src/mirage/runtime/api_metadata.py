"""Extract API context from the tested installation, never a guessed signature."""

import hashlib
import inspect
from importlib import import_module
import re
from types import ModuleType

from mirage.contracts.apis import validate_api_name


def resolve(api):
    import torch

    validate_api_name(api)
    parts = api.split(".")
    value = torch.Tensor if parts[0] == "Tensor" else torch
    for part in parts[1:]:
        try:
            value = getattr(value, part)
        except AttributeError as missing_attribute:
            if not isinstance(value, ModuleType):
                raise
            # A public alias may point at another Torch module. Resolve against
            # its actual name, but never import external or private namespaces.
            requested_module = value.__name__ + "." + part
            try:
                validate_api_name(requested_module)
            except ValueError:
                raise missing_attribute from None
            if not requested_module.startswith("torch."):
                raise missing_attribute from None
            try:
                value = import_module(requested_module)
            except ModuleNotFoundError as missing_module:
                if missing_module.name != requested_module:
                    raise
                # Missing internal dependencies are installation failures, not
                # evidence that the requested public API does not exist.
                raise missing_attribute from None
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


def specification(api, *, probe_device=None):
    """Build a version-bound description; proposals and runtime facts stay separate."""
    import torch

    result = {
        "version": "api-spec-1", "api": api, "framework": "pytorch",
        "framework_version": str(torch.__version__),
        "framework_build": getattr(torch.version, "git_version", None),
        "status": "unresolved", "sources": [], "unknown_fields": [],
        "semantic_guarantees": [], "model_proposals": [], "runtime_observations": [],
    }
    try:
        target = resolve(api)
    except (ValueError, AttributeError) as error:
        result.update(status="unavailable", reason=str(error))
        return result

    metadata = describe(api)
    result.update(status="resolved", metadata=metadata,
                  resolved_identity={"module": getattr(target, "__module__", None),
                                     "qualname": getattr(target, "__qualname__", None)},
                  capabilities={"context_manager_factory": inspect.isclass(target)
                                and hasattr(target, "__enter__") and hasattr(target, "__exit__"),
                                "callable_product": inspect.isclass(target) and any(
                                    "__call__" in cls.__dict__ for cls in target.__mro__),
                                "future_product": inspect.isclass(target) and issubclass(target, torch.futures.Future),
                                "steppable_product": inspect.isclass(target) and issubclass(
                                    target, (torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler)),
                                "optimizer_product": inspect.isclass(target) and issubclass(target, torch.optim.Optimizer),
                                "scheduler_product": inspect.isclass(target) and issubclass(target, torch.optim.lr_scheduler.LRScheduler),
                                "backward": "unknown", "device_support": "unknown"},
                  environment={"cuda_available": torch.cuda.is_available(),
                               "distributed_world_size": torch.distributed.get_world_size()
                               if torch.distributed.is_available() and torch.distributed.is_initialized() else 1})
    result["sources"].append({
        "kind": "installed_documentation", "framework_version": result["framework_version"],
        "identity": api, "sha256": metadata["documentation_sha256"],
        "truncated": metadata["documentation_truncated"],
    })
    if metadata["signature"] is None:
        result["unknown_fields"].append("signature")
    schemas, schema_reason = operator_schemas(target)
    result["operator_schemas"] = schemas
    if schemas:
        result["sources"].append({"kind": "installed_dispatcher_schema",
            "framework_version": result["framework_version"],
            "sha256": hashlib.sha256("\n".join(s["schema"] for s in schemas).encode()).hexdigest()})
    else:
        result["unknown_fields"].append("operator_schemas")
        result["schema_reason"] = schema_reason
    if not metadata["documentation"]:
        result["external_documentation"] = official_documentation(api, result["framework_version"])
    else:
        result["external_documentation"] = {"status": "not_needed",
                                             "reason": "installed documentation available"}
    from mirage.runtime.input_constraints import input_spec, probe_input_domain
    result["constraints"] = input_spec(target, result)
    if probe_device and (probe_device == "cpu" or torch.cuda.is_available()):
        probe_input_domain(result, probe_device)
    result["unknown_fields"].extend(("semantic_guarantees", "backward", "device_support"))
    return result


def operator_schemas(target):
    """Use the installed callable registration; Python-only wrappers may lack it."""
    import torch

    try:
        schema = getattr(target, "_schema", None)
        schemas = [schema] if schema is not None else list(getattr(target, "_schemas", {}).values())
        if not schemas:
            from torch.jit._builtins import _find_builtin
            operator = _find_builtin(target)
            if operator is None:
                return [], "installed callable has no dispatcher registration mapping"
            schemas = torch._C._jit_get_schemas_for_operator(operator)
        return [{"name": schema.name, "overload": schema.overload_name, "schema": str(schema),
                 "arguments": [{"name": arg.name, "type": str(arg.type), "keyword_only": arg.kwarg_only,
                                "write": arg.alias_info.is_write if arg.alias_info is not None else None}
                               for arg in schema.arguments]}
                for schema in schemas], None
    except (AttributeError, TypeError, RuntimeError, ImportError) as error:
        return [], f"schema introspection unavailable: {type(error).__name__}: {error}"


def official_documentation(api, framework_version):
    """Bounded fallback for missing installed docs, restricted to the release line."""
    from html.parser import HTMLParser
    from urllib.error import URLError
    from urllib.request import urlopen

    release = re.match(r"^(\d+\.\d+)(?:\.|$)", framework_version)
    if release is None:
        return {"status": "unavailable", "reason": "release documentation version unknown"}
    public_path = "torch." + api if api.startswith("Tensor.") else api
    url = f"https://docs.pytorch.org/docs/{release[1]}/generated/{public_path}.html"
    class PageText(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    try:
        with urlopen(url, timeout=5) as response:
            raw = response.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024:
            return {"status": "unavailable", "url": url, "reason": "documentation size limit exceeded"}
        page = PageText()
        page.feed(raw.decode("utf-8"))
        text = " ".join(page.parts)
        if re.search(r"PyTorch\s+" + re.escape(release[1]) + r"(?:\D|$)", text) is None:
            return {"status": "unavailable", "url": url, "reason": "documentation release not verified"}
        return {"status": "available", "url": url, "framework_release": release[1],
                "sha256": hashlib.sha256(raw).hexdigest(), "text": text[:16000],
                "truncated": len(text) > 16000, "role": "generation_context_only"}
    except (OSError, URLError, UnicodeError, ValueError) as error:
        return {"status": "unavailable", "url": url,
                "reason": f"documentation fetch failed: {type(error).__name__}: {error}"}
