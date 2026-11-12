"""Controlled training lifecycle shared by generated cases and saved replays."""

from contextlib import contextmanager
from pathlib import Path

from mirage.contracts.apis import PROFILES
from mirage.generator.validation import normalized
from mirage.runtime.environment import configure, describe
from mirage.runtime.operations import compile_forward, load_program
from mirage.runtime.probes import Recorder, checkpoint, inputs
from mirage.runtime.representations import construct, metadata
from mirage.runtime.state import cpu_copy, restore_rng, restore_training, rng_state, training_state


def execute(program, config, initial: dict, recorder: Recorder, partial_path: Path | None = None) -> dict:
    import torch
    from torch.utils.checkpoint import checkpoint as activation_checkpoint

    configure()
    environment = describe(config.device)
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if program.actions != config.actions:
        raise ValueError("Program actions do not match execution configuration")
    if PROFILES[config.api].mutation and config.target.checkpointing:
        raise ValueError("Checkpointing a side-effecting forward is outside this harness")

    parameter = torch.nn.Parameter(initial["parameter"].to(config.device).clone())
    bias = initial["bias"].to(config.device).clone()
    if list(parameter.shape) != [config.size, config.size] or list(bias.shape) != [config.size]:
        raise ValueError("Initial tensors do not satisfy the API profile")
    optimizer = torch.optim.Adam([parameter], lr=1e-3, foreach=False)
    scaler = torch.amp.GradScaler(config.device, enabled=config.precision == "fp16")
    if initial.get("optimizer") is not None:
        optimizer.load_state_dict(initial["optimizer"])
    if initial.get("scaler") is not None:
        scaler.load_state_dict(initial["scaler"])
    parameter.grad = None if initial.get("gradient") is None else initial["gradient"].to(config.device).clone()
    restore_rng(initial["rng"], config.device)
    initial_training = training_state(parameter, optimizer, scaler, config.device)
    recorder.emit("action", name="restore_initial_state")

    raw_forward, loss_function = load_program(program.source, config.api)
    forward = compile_forward(raw_forward, config, recorder)
    snapshots = []
    observations = []
    successful_updates = 0
    checkpoint_active = False
    recomputing = False

    def step_done(*_):
        nonlocal successful_updates
        successful_updates += 1
        recorder.emit("optimizer_step", successful_updates=successful_updates)

    optimizer.register_step_post_hook(step_done)

    @contextmanager
    def checkpoint_scope(recompute: bool):
        nonlocal checkpoint_active, recomputing
        checkpoint_active, recomputing = True, recompute
        recorder.emit("checkpoint_enter", recompute=recompute)
        try:
            yield
        finally:
            recorder.emit("checkpoint_exit", recompute=recompute)
            checkpoint_active, recomputing = False, False

    def iteration(index: int, representation: str, warmup: bool = False):
        recorder.iteration = index
        recorder.phase = "warmup" if warmup else "training"
        optimizer.zero_grad(set_to_none=True)
        x, weight, anchor = construct(parameter, representation)
        tensors = {"x": x, "weight": weight, "anchor": anchor, "parameter": parameter}
        recorder.emit("action", name=f"construct:{representation}")
        logical_inputs = inputs(x, weight, bias)
        before_metadata = metadata(tensors)
        before_rng = rng_state(config.device)
        gradient_seen = {"input": False, "output": False}

        def gradient_hook(name):
            def hook(gradient):
                gradient_seen[name] = True
                recorder.gradient_hook(name)(gradient)
                return gradient
            return hook

        # In-place view rebasing can bypass a hook on the old view node. The
        # retained anchor observes the same logical coordinates before the
        # representation transform and survives that graph lifecycle change.
        anchor.register_hook(gradient_hook("input"))

        def invoke(value, other, offset):
            if not warmup and not recomputing:
                snapshots.append(checkpoint(
                    "before_api", index, tensors, config.target,
                    checkpointing=checkpoint_active, optimizer_steps=successful_updates,
                    backend=config.backend, warmup=config.target.warmup,
                ))
            recorder.emit("api_enter", api=config.api, recompute=recomputing,
                          input_dtypes=[str(value.dtype), str(other.dtype), str(offset.dtype)])
            output = forward(value, other, offset)
            recorder.emit("api_exit", api=config.api, recompute=recomputing,
                          output_dtype=str(output.dtype), shape=list(output.shape))
            return output

        dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
        with torch.autocast(config.device, dtype=dtype, enabled=config.precision != "fp32"):
            if config.target.checkpointing:
                output = activation_checkpoint(
                    invoke, x, weight, bias, use_reentrant=False, preserve_rng_state=True,
                    context_fn=lambda: (checkpoint_scope(False), checkpoint_scope(True)),
                )
            else:
                output = invoke(x, weight, bias)
            loss = loss_function(output)

        output.register_hook(gradient_hook("output"))
        output_value = cpu_copy(output)
        mutation = {"x": cpu_copy(x), "weight": cpu_copy(weight)}
        recorder.emit("action", name="forward")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        recorder.emit("action", name="backward")
        gradient = cpu_copy(parameter.grad)
        used_checkpoint = any(e["event"] == "checkpoint_enter" and not e["recompute"]
                              and e["iteration"] == index and e["phase"] == recorder.phase
                              for e in recorder.events)
        if not warmup:
            snapshots.append(checkpoint(
                "after_backward", index, tensors, config.target, checkpointing=used_checkpoint,
                optimizer_steps=successful_updates, backend=config.backend, warmup=config.target.warmup,
            ))
        previous_updates = successful_updates
        scaler.step(optimizer)
        scaler.update()
        recorder.emit("action", name="adam_update", skipped=successful_updates == previous_updates)
        if config.device == "cuda":
            torch.cuda.synchronize()

        if not warmup:
            snapshots.append(checkpoint(
                "after_update", index, tensors, config.target, checkpointing=used_checkpoint,
                optimizer_steps=successful_updates, backend=config.backend, warmup=config.target.warmup,
            ))
            observations.append({
                "iteration": index, "attempted_update": index + 1,
                "successful_updates": successful_updates, "skipped": successful_updates == previous_updates,
                "inputs": logical_inputs, "input_metadata": before_metadata,
                "rng_before": before_rng, "rng_after": rng_state(config.device),
                "loss": cpu_copy(loss), "output": output_value, "gradient": gradient,
                "parameter": cpu_copy(parameter), "optimizer": cpu_copy(optimizer.state_dict()),
                "scaler": cpu_copy(scaler.state_dict()), "mutation": mutation,
                "backward": gradient_seen, "actual_output_dtype": str(output.dtype),
            })
            if partial_path is not None:
                torch.save({"observations": observations, "snapshots": snapshots}, partial_path)

    if config.target.warmup != "none":
        warmup_representation = config.target.representation
        if config.target.warmup == "alternate":
            warmup_representation = "strided" if warmup_representation == "contiguous" else "contiguous"
        for index in range(config.warmup_steps):
            iteration(-index - 1, warmup_representation, warmup=True)
    recorder.phase = "setup"
    recorder.iteration = -1
    recorder.emit("action", name=f"warmup:{config.target.warmup}")
    restore_training(initial_training, parameter, optimizer, scaler, config.device)
    successful_updates = 0
    restored = training_state(parameter, optimizer, scaler, config.device)
    recorder.emit("action", name="restore_training_state")
    recorder.emit("action", name=f"checkpoint:{int(config.target.checkpointing)}")
    for index in range(config.updates):
        iteration(index, config.target.representation)

    return {"version": 1, "environment": environment, "initial": initial_training,
            "restored": restored, "bias": cpu_copy(bias), "observations": observations,
            "snapshots": snapshots, "events": recorder.events, "grammar_verified": True,
            "program_ast": normalized(program.source),
            "resources": {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                          if config.device == "cuda" else None}}
