"""Load checked forward programs and collect compiled-graph execution evidence."""

from mirage.generator.validation import validate_source


def load_program(source: str, api: str):
    import torch

    validate_source(source, api)
    namespace = {"torch": torch}
    exec(compile(source, "<mirage-program>", "exec"), namespace)
    return namespace["forward"], namespace["training_loss"]


def compile_forward(forward, config, recorder):
    import torch

    if config.backend == "eager":
        return forward
    from torch._dynamo.backends.registry import lookup_backend

    backend_name = "eager" if config.backend == "compile_eager" else config.backend
    underlying = lookup_backend(backend_name)

    def backend(graph, example_inputs):
        nodes = [str(node.target) for node in graph.graph.nodes if node.op.startswith("call_")]
        recorder.emit("compiled_graph", nodes=nodes, graph=str(graph.graph), backend=backend_name)
        compiled = underlying(graph, example_inputs)

        def execute(*args):
            recorder.emit("compiled_graph_enter", nodes=nodes)
            result = compiled(*args)
            recorder.emit("compiled_graph_exit")
            return result
        return execute

    return torch.compile(forward, backend=backend, fullgraph=True)
