"""Append-only runtime events and semantic checkpoint snapshots."""

import json
from pathlib import Path
import time

from mirage.runtime.representations import metadata, representation_matches
from mirage.runtime.state import cpu_copy


class Recorder:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.events = []
        self.started = time.monotonic()
        self.phase = "setup"
        self.iteration = -1

    def emit(self, event: str, **data) -> None:
        item = {"event": event, "phase": self.phase, "iteration": self.iteration,
                "seconds": time.monotonic() - self.started, **data}
        self.events.append(item)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, allow_nan=False) + "\n")

    def gradient_hook(self, name: str):
        def observe(gradient):
            self.emit("gradient", name=name, shape=list(gradient.shape), dtype=str(gradient.dtype))
            return gradient
        return observe


def checkpoint(name, iteration, tensors, target, *, checkpointing, optimizer_steps, backend, warmup):
    observed = metadata(tensors)
    reached = (
        name == target.checkpoint and iteration == target.iteration
        and representation_matches(observed, target.representation)
        and checkpointing == target.checkpointing and warmup == target.warmup
    )
    return {"checkpoint": name, "iteration": iteration, "metadata": observed,
            "checkpointing": checkpointing, "optimizer_steps": optimizer_steps,
            "backend": backend, "warmup": warmup, "target_reached": reached}


def inputs(x, weight, bias) -> dict:
    return {"x": cpu_copy(x), "weight": cpu_copy(weight), "bias": cpu_copy(bias)}
