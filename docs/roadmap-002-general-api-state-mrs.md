# General API generation, state exploration, and conditional MRs

## Approved scope

- Replace the fixed API/input grammar with a version-three program contract.
- Supply versioned API signatures and documentation to the LLM. Resolve additional APIs without handwritten input templates.
- Validate the eight existing APIs and add addmm, bmm, conv2d, sum, mean, gather, max(dim), and log_softmax to the acceptance matrix.
- Reuse canonical programs across target states and alternative action paths. Explore tensor representations, gradients, optimizer history, graph lifecycle, checkpointing, and existing execution modes.
- Retain layout, storage, checkpoint, and mutation rules. Add lossless saved-tensor roundtrip and deterministic FP32 gradient accumulation equivalence.
- Reuse one healthy vLLM service during development iterations, using Slurm steps. Stop the service, release the allocation, and close connections at task completion.
- Preserve new evidence, distinguish generation/runtime/applicability/oracle failures, and reproduce new cases without the LLM. Do not claim new framework defects without corroboration.
- User steering during implementation: legacy versions and legacy replay can be removed. Obsolete result artifacts may be deleted after preserving model-environment configuration, active jobs, and meaningful failure evidence.

## Implementation sequence

1. Add development-session lifecycle and immutable iteration snapshots.
2. Add API metadata extraction and version-three generation contracts.
3. Implement generic initialization, execution, observation, and state-preserving replay.
4. Implement multi-target canonical reuse and verified action trajectories.
5. Adapt existing conditional MRs and implement the two new rules.
6. Validate contracts, negative oracle cases, runtime behavior, new-case reproduction, real-model generation, and session cleanup.
7. Document exact results, limitations, costs, and remaining unsupported capabilities.

## Validation plan

- Lightweight syntax, AST, diff, and shell checks may run on the login node.
- Runtime tests, integration suites, compilation, inference, and replay run through categorized Slurm scripts.
- Test missing API signatures, multi-input/output programs, gradients and aliasing, malformed plans, absent runtime evidence, MR inapplicability, deliberate output violations, and resource cleanup.
- Real-model acceptance covers 16 APIs. Report API-specific outcomes rather than interpreting a successful job as an MR pass.
- Exploration budgets are configurable: up to 3 canonicals/API, 12 targets/canonical, and 2 paths/target; these are attempts, not proven feasible states.
- Save code, environment, API metadata, prompts, initial state, plans, traces, MR version, comparison policy, and costs.
- Preserve existing user changes and the move from doc/MVP.md to docs/roadmap-001-mvp.md.

## Progress

- Approved plan recorded; existing code and execution constraints inspected.
- Development session job 762086 started after shell/preflight validation. Model startup took 192 seconds; all iterations reused that service.
- Initial version-three runtime/oracle suite: 12 tests passed in 14.941 seconds inside Slurm job 762086. Includes uneven microbatch weighting, hooks, history, layout, storage, checkpoint, and negative oracle cases.
- User approved explicit deletion of superseded modules/tests after automatic review requested a more precise scope. Shared HTTP/process/artifact/comparison/sampling infrastructure remains.
- Removed transitional `program_v3` filenames. Current modules use functional names; artifact format versions remain explicit.
- Current-only suites passed 40 tests (9.809 s), then 44 tests (61.818 s), then 47 tests (64.722 s), all inside the same session.
- One GPU test fixture initially overflowed FP16 GradScaler. The oracle returned unassessed, not a suspected defect. The equivalence fixture was bounded; a separate nonfinite-policy test remains.
- `real-02`: 8/16 canonical API paths verified; import aliases and diagnostics exposed generator contract problems.
- `real-03`: 15/16 canonical API paths verified; 47 provider calls, 2 repairs, 255.743 s. The failed addmm program used an invalid matrix rank; its runtime failure was preserved without speculative repair.
- `acceptance-01`: all 16 preserved real-LLM programs validated with one engine revision. Saved-tensor MR passes and accumulation MR passes for conv2d/sum observed; this is not an all-API/all-MR claim.
- `real-04`: 3 canonicals reused for 72 state/path attempts; 34 verified cases, 19 distinct verified states, 5 distinct verified histories, 12 MR passes. 53 plan repair calls exposed insufficient final-state diagnostics; messages now give required and actual update/gradient/graph state.
- Removed 23 obsolete result directories after scanning for suspected/confirmed defect manifests (none found). Kept summaries in `result/archive/legacy-summaries/`, prepared serving configuration, and all current-session evidence.
- `real-05`: four canonicals (two each for conv2d and sum) produced 48 executed state/path cases in 232.735 s. Of these, 40 had verified path/state evidence: 16 MR Pass, 2 Inapplicable, and 22 coverage-only. The other 8 were 4 Invalid and 4 incomplete executions. There were 18 unique verified state configurations, 6 histories, and 17 observed transition sequences; applicable API × MR combinations were 5/12 declared. Generation used 74 calls, including 22 repairs, with 56,029 input and 3,850 output tokens. These bounded candidates do not establish theoretical state-space coverage.
- Final runtime/oracle/GPU suite: 55 tests passed in 66.039 s (`iteration-hNYtXJ`). Includes CUDA FP32/FP16/BF16, compile_eager/aot_eager/inductor, six MR families, incorrect-loss normalization, detached or unused target output, actual target-operand layout, input-gradient restoration, controller reuse, HTTP failures, and session lifecycle.
- `acceptance-final/validation.json`: 16/16 preserved real-model API programs executed their target and backward path under the final runtime. Programs were selected from `real-02` and `real-03`, not regenerated until success within a single batch. Saved-tensor: 12 Pass, 4 Inapplicable. Checkpoint: 13 Pass, 3 Inapplicable. Accumulation: 2 Pass (conv2d/sum), 5 Inapplicable, 9 execution/contract failures retained without a correctness verdict.
- Static validation: 42 Python files parsed; 15 shell/job scripts passed `bash -n`; development-session and full-suite jobs passed `sbatch --test-only`. These are static/preflight results, not extra executed Slurm jobs.
- Final current-code real-model batch (`real-final`): 2 validated canonicals, 12 state/path attempts, 11 executions and 1 exhausted plan repair. Of the 11 executions, 9 had verified path/state evidence: 5 Pass, 2 Inapplicable, and 2 coverage-only. Two remaining coverage-only attempts failed execution. Generation took 23 calls (9 repairs), 18,780 input and 1,761 output tokens; batch wall time was 71.383 s. This run exercised the final status separation; failed plans are excluded from the MR-verdict denominator.
- After adding reproduction resource accounting, the 4 controller tests passed again in 54.085 s (`iteration-ATZj17`). A final real-model storage-MR artifact replayed with Pass, zero provider calls, and no environment changes while the configured provider endpoint was deliberately unavailable (`real-final/replay-93732c47ddb84f2c.json`).
- Cleanup verified: job 762086 completed with exit 0:0; testing terminal and all owned steps exited, vLLM's step was cancelled during requested shutdown, and `squeue` showed no remaining job. The owned testing compilation cache was removed. The model loaded once (192 s), with zero restarts. Allocation elapsed 3,828 s with two GPUs: 7,656 allocated GPU-seconds, not a utilization measure. Per-iteration durations and final status are recorded in `result/validation/762086/completion.json`.
- Approved implementation and validation are complete; limitations below remain explicit.

## Current implementation contract

`build(device, seed)` returns `inputs`, `parameters`, `optimizer`, and `batch_axes`,
with optional registered `modules` and serializable extra state. Named trainable
leaf tensors are accepted, not only Parameter subclasses. Module parameters are
rebound to saved tensors when restoring optimizer/module state. Tensor storage
sharing is preserved by saving the connected initial tree together.

`forward(ctx)` and `loss(output)` may use general PyTorch operands and structured
outputs. Optional `sample_loss(output)` provides one value per sample. Input
generation is not an API-specific template. API context comes from the installed
signature/docstring; missing signatures remain explicit rather than guessed.

The open API resolver and the conservative MR capability catalog are separate.
Unknown operation semantics do not prevent generation/execution, but they cannot
establish purity or batch separability. Effects, representation introspection,
unknown comparison conditions, and unsupported compiled operand correspondence
prevent a correctness verdict.

State histories use `zero:none`, `zero:zero`, `forward`, `backward`,
`backward:retain`, and `step`. The LLM chooses bounded sequences; static validation
checks order and intended final state, while execution verifies gradients,
successful updates, API operands, and backward paths. Sampling uses the existing
inverse-count policy with uniform exploration over declared candidates, records
attempts separately, and retains legal coverage-only trajectories without an MR.

Saved-tensor hooks verify same-device lossless content/metadata restoration and
actual pack/unpack activity. Gradient accumulation requires deterministic FP32,
certified per-sample operations, agreement between scalar and per-sample loss,
and sample-count weighting for uneven microbatches. It compares accumulated
gradients and a single optimizer update, not equal backward counts.

Nonfinite observations are unassessed pending a suitable numerical policy.
Successful update counts are observable outputs, so divergent AMP skip behavior
is not silently excluded by checkpoint alignment. Suspected violations retain
artifacts and trigger bounded saved-engine reproduction; no automatic defect
confirmation or general-purpose minimization is claimed.

## Explicit limits

The 16-API matrix is an acceptance sample, not a whitelist and not a claim that
every PyTorch API already works. The open resolver is additionally exercised with
`torch.sin` and a forward-only `Tensor.size` target. Constructors, specialized
objects, unsupported sparse/quantized layouts, and execution paths the recorder
cannot prove remain unresolved rather than being called supported.

The current adapter supports dense tensor input trees, named trainable leaves,
optimizer state, registered modules, serializable extras, RNG, AMP, and a bounded
history vocabulary. It does not implement distributed state, DataLoader/scheduler
transitions, CPU offload, or arbitrary code minimization. Forward-only explicitly
relaxes target-backward requirements; a training loss still executes backward.

MR capability checks are conservative and separate from API generation. The
accumulation certificate excludes unknown or cross-sample computations. Compiled
backward evidence follows corresponding target-dependent FX output leaves;
unsupported operand correspondence cannot establish a representation MR.
Unknown/nonfinite comparisons remain unassessed. Initial data, gradients, alias
topology, and RNG are restored from saved state rather than inferred from a seed.

No new PyTorch defect is confirmed by these runs. Deliberately corrupted
observations in oracle tests validate detection logic and are not framework bugs.
