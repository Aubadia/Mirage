"""Append-only runtime events and semantic checkpoint snapshots."""

import json
from pathlib import Path
import time



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
