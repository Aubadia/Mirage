# Mirage Agent Instructions

## Project Goal

Build a testing framework for deep learning (DL) frameworks that explores tensor memory representations and training states.

Previous testing approaches include API/model-input fuzzing, model architecture and API interaction testing, and gradient injection. Mirage explores the following additional testing space.

### Tensor Memory Representation

- Strides and contiguity.
- Storage offset.
- Storage identity, including cloning and new allocations.
- Storage aliasing: shared versus independent storage.
- Device and memory placement: CPU/GPU and pinned/pageable memory.
- Materialization strategy: direct saving, packing/unpacking, offloading, and recomputation.
- Tensor metadata: view/base relationships, version counters, and gradient history.

### Training State

- Autograd state.
- Activation checkpointing.
- Offloading.
- Compilation.
- Graph lifecycle.
- Optimizer state.
- Distributed state.

## Framework Architecture

### Layer 1: State Transition Sampler

1. Select a target API from the API pool. Use an LLM to generate a canonical program `P_0` in the default training state `s_0`, serving as the baseline.
2. Use sampling algorithm `A_1` and coverage criterion `C_1` to sample `m` target final training states. Keep inputs and initial parameters unchanged whenever possible.
3. Plan a state transition sequence for each target state:
   `s_0 -> s_1^(i) -> ... -> s_k_i^(i) = s_target^(i)`.
4. Select applicable metamorphic rules (MRs) from a predefined MR library:
   - **Applicability filtering:** Consider the target API, canonical program, state transitions, MR preconditions, and available observables, such as loss, gradients, parameters, and optimizer state.
   - **Ranking:** Consider expected coverage gain, unexplored API/MR combinations, applicability, and execution cost.
   - **Fallback:** Retain states without applicable MRs for another correctness oracle or coverage-only testing. Do not sacrifice API coverage merely because no MR applies.

### Layer 2: Training Program Generator

1. Consume the target API, canonical program, state transition sequences, and selected MRs.
2. Use an LLM to derive a reusable training loop skeleton from the canonical program, marking regions where state transitions can be injected.
3. Use incremental filling, potentially with a smaller language model, to inject transitions and produce training programs `P_1, ..., P_m`.
4. Verify that each generated program reaches its intended target state; requested or generated code alone is not evidence of state attainment.

### Layer 3: Inspector

1. Verify that the target API actually executes, the intended training state is reached, and the selected MR preconditions hold.
2. Perform correctness testing using transformation-based MRs and the canonical baseline:
   - Prefer checking `R_j(O(P_0), O(P_i)) = True` for the observables specified by the selected MR.
   - When direct baseline comparison is unsuitable, use another program pair with a valid MR.
   - Without an applicable MR, use another suitable oracle or retain coverage feedback only. Never classify an arbitrary difference as a bug.
3. Return API coverage, state transition coverage, MR applicability, correctness feedback, and generation cost to Layer 1 to guide subsequent sampling and MR selection.

## Correctness Principles

- Treat an MR as conditional on its preconditions, rather than assuming every state transition preserves every observable.
- Keep inputs and initial parameters consistent across comparisons whenever possible. Make any necessary differences explicit.
- Define the observables, numerical tolerances, and treatment of nondeterminism for each comparison as appropriate to the API, transformation, dtype, and execution environment.
- Distinguish generation failures, unreachable states, unmet MR preconditions, and oracle violations. Do not report all failures as framework bugs.
- Report an oracle violation as a potential bug with supporting evidence; do not claim a confirmed framework defect without sufficient validation.

## Required Approval and Editing Workflow

- The user will provide pseudocode for a specific file. Read that pseudocode and the relevant existing code before proposing changes.
- Before modifying files, describe the intended implementation, affected files, and appropriate validation. Wait for explicit user approval of the plan.
- After approval, implement the agreed scope. Existing approval covers work within that scope; do not ask for the same permission again.
- Modify only the file or files explicitly authorized by the user.
- If another file must change, explain why the change is necessary, identify the affected file and expected impact, and obtain permission before editing it.
- Do not refactor unless the user explicitly requests it. Avoid unrelated cleanup, renaming, reformatting, or architectural changes.
- If the pseudocode is materially ambiguous or conflicts with the approved design, explain the issue and obtain clarification before implementing the affected behavior.

## Implementation and Comments

- Follow the supplied pseudocode, approved plan, and existing project conventions.
- Keep changes focused and easy to review. Do not invent additional requirements or abstractions.
- Add concise comments when needed to explain intent, preconditions, or non-obvious constraints. Avoid comments that merely restate the code.
- Do not add dependencies, change project configuration, or create supporting files outside the approved scope without permission.
- Leave undecided design choices open until the user provides direction or approves a proposal. This includes the target DL framework, sampling algorithms, coverage definitions, and MR library contents.

## Validation and Reporting

- Run validation appropriate to the approved change and available environment.
- Creating or modifying test files is subject to the same file-scope approval rules as implementation changes.
- Check relevant behavior and failure cases when practical; avoid unrelated or redundant tests.
- Never claim validation passed unless it was actually run and passed. State any unavailable checks and their limitations.
- When finished, summarize what changed, how it was validated, and any remaining limitations or decisions requiring user input.

## Current Repository Context

- The project is named `mirage` and uses a `src` layout.
- The package entry file is `src/mirage/__init__.py`.
- `pyproject.toml` currently requires Python `>=3.14`, uses `uv_build`, and declares the CLI entry point `mirage = "mirage:main"`.
- No runtime dependencies are currently declared. Do not assume a DL framework or test runner has already been selected or installed.
- Use the actual repository configuration as the source of truth if these details change.
