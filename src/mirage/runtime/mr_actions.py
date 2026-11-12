"""Materialization hooks and conservative batch-separability certificates."""

import ast
from contextlib import nullcontext


def value_operations(names):
    """Conservative MR capability catalog, separate from the open API resolver."""
    known = {
        "matmul", "bmm", "mm", "addmm", "linear", "conv2d", "layer_norm", "gelu",
        "softmax", "log_softmax", "dropout", "gather", "max", "sum", "mean",
        "relu", "tanh", "sigmoid", "silu", "sin", "cos", "exp", "log", "abs",
        "add", "sub", "mul", "div", "pow", "square", "sqrt", "neg", "clamp",
        "float", "double", "to", "type_as", "flatten", "reshape", "view", "clone",
        "transpose", "permute", "contiguous", "squeeze", "unsqueeze", "numel", "size",
        "dim", "__get__", "__getitem__", "register_hook",
    }
    return all(name in known for name in names)


class SavedTensors:
    def __init__(self, enabled):
        self.enabled = enabled
        self.packed = 0
        self.unpacked = 0
        self.valid = True

    def context(self):
        import torch

        if not self.enabled:
            return nullcontext()

        def pack(tensor):
            self.packed += 1
            value = tensor.detach().clone()
            reference = tensor.detach().cpu().clone()
            self.valid &= bool(torch.equal(value.detach().cpu(), reference))
            return (value, tensor.shape, tensor.dtype, tensor.device, reference)

        def unpack(record):
            value, shape, dtype, device, reference = record
            self.unpacked += 1
            self.valid &= value.shape == shape and value.dtype == dtype and value.device == device
            self.valid &= bool(torch.equal(value.detach().cpu(), reference))
            return value

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

    def evidence(self):
        return {"enabled": self.enabled, "packed": self.packed,
                "unpacked": self.unpacked, "roundtrip_valid": bool(self.valid)}


def accumulation_supported(source, ctx, operations):
    """Certify a small, explicit subset; LLM declarations are not certificates.

    Batch input indexing and data-dependent branches are deliberately excluded.
    Unknown operators can still execute as coverage-only cases.
    """
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "sample_loss" not in functions or not ctx["batch_axes"]:
        return False
    safe = {"linear", "matmul", "mm", "addmm", "conv2d", "gelu", "relu", "tanh", "sigmoid",
            "silu", "add", "sub", "mul", "div", "pow", "square", "float", "to", "mean",
            "sum", "flatten", "reshape", "view", "__get__", "register_hook", "size", "dim"}
    if any(op["name"] not in safe for op in operations):
        return False
    # A reduction over the sample axis invalidates per-sample independence.
    for op in operations:
        if op["name"] in ("sum", "mean"):
            dim = op["dim"]
            dims = dim if isinstance(dim, (list, tuple)) else (dim,)
            if dim is None or any(axis == 0 or any(axis == -len(shape) for shape in op["shapes"])
                                  for axis in dims):
                return False
    for name in ("forward", "sample_loss"):
        function = functions[name]
        aliases = {}
        for statement in function.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                aliases[statement.targets[0].id] = statement.value

        def expand(node, seen=None):
            seen = set() if seen is None else seen
            if isinstance(node, ast.Name) and node.id in aliases and node.id not in seen:
                return expand(aliases[node.id], seen | {node.id})
            if isinstance(node, ast.Subscript):
                return ast.Subscript(expand(node.value, seen), node.slice, ast.Load())
            return node

        def shared_parameter(node):
            text = ast.unparse(expand(node))
            return text.startswith("ctx['parameters'][")

        for node in ast.walk(function):
            if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.While, ast.Try, ast.Lambda)):
                return False
            if isinstance(node, ast.Subscript):
                # Only dictionary lookup is allowed; tensor indexing could mix samples.
                shape_extent = (isinstance(node.value, ast.Attribute) and node.value.attr == "shape"
                                and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int
                                and node.slice.value > 0)
                if not shape_extent and (not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, str)):
                    return False
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Attribute) or node.func.attr not in safe:
                    return False
                if node.func.attr in ("reshape", "view"):
                    return False
                if node.func.attr in ("matmul", "mm", "linear", "conv2d"):
                    if len(node.args) < 2 or not shared_parameter(node.args[1]):
                        return False
                if node.func.attr == "addmm":
                    if len(node.args) < 3 or not shared_parameter(node.args[2]):
                        return False
                if node.func.attr == "flatten":
                    start = node.args[0] if node.args else next(
                        (kw.value for kw in node.keywords if kw.arg == "start_dim"), None)
                    if not isinstance(start, ast.Constant) or start.value != 1:
                        return False
    # Matmul is only certified for matrix inputs and shared matrix parameters.
    for op in operations:
        if op["name"] in ("matmul", "mm"):
            if len(op["shapes"]) != 2 or len(op["shapes"][0]) < 2 or len(op["shapes"][1]) != 2:
                return False
        if op["name"] == "addmm" and any(len(s) != 2 for s in op["shapes"][-2:]):
            return False
    return True
