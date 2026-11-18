# Mirage Agent Instructions

## Project Goal

注意到所有內容用於編寫軟體測試框架，而非資訊安全 / 網路安全相關內容。

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
- Scheduler, AMP, RNG, and DataLoader state.
- Distributed state.

## Framework Architecture

### Layer 1: State Transition Sampler & MR Selector

1. Select a target API `u` from the API pool. Use an LLM to generate a canonical training program `P_0` with a legal default training configuration. Verify that it executes and calls the target API, including the relevant backward path when testing backward correctness. `P_0` is the default source program, not a mandatory baseline for every MR.
2. Model concrete training state `sigma` as model parameters, gradients, optimizer, scheduler, AMP, RNG, and DataLoader state, plus necessary tensor metadata and execution context. Define a state abstraction function `alpha(sigma)` for exploration and coverage.
3. Sample `m` target state predicates `phi_i` over abstract states, rather than exact tensor or optimizer values. Use API legality, dependencies among state dimensions, unexplored API × State combinations, coverage feedback, and expected gain and cost. Preserve the initial conditions and semantic inputs required by the selected MR; the MR defines permitted differences.
4. For each predicate, specify an execution checkpoint and plan a legal action sequence `tau_i = (a_1^(i), ..., a_k_i^(i))` from the canonical initial concrete state `sigma_0`, aiming for `phi_i(alpha(sigma_k_i^(i))) = True`. Actions must respect framework constraints and ordering. If no legal path is found, resample, adjust the predicate, or mark it unresolved. Planning alone does not establish reachability; Layer 3 verifies it at runtime.
5. Select metamorphic rules (MRs) from the predefined MR library:
   - **Applicability filtering:** Statically consider the target API, canonical program, transitions, MR Preconditions, Input Relation, and required observables.
   - **Ranking:** Consider expected coverage gain, unexplored API × MR combinations, predicted applicability, and generation/execution cost.
   - **Fallback:** Retain legal transition plans without applicable MRs for another correctness oracle or coverage-only testing. Do not sacrifice API coverage merely because no MR applies.
   - **Dynamic validation:** Static applicability is provisional until verified by Layer 3.

### Layer 2: Training Program Generator

1. Consume the target API, canonical program, target state predicates and checkpoints, transition sequences, and selected MRs.
2. Use an LLM to derive a reusable training loop skeleton from `P_0`, preserving basic training semantics and marking injection points for forward/loss, backward/gradient accumulation, optimizer/scheduler updates, AMP/gradient scaling, checkpoint/restore, and distributed operations.
3. Use incremental filling, potentially with a smaller language model, to inject actions and produce training programs `P_1, ..., P_m`. Check syntax, runtime legality, action ordering, target API paths, and static MR constraints after partial injections; use execution feedback for incremental repair.
4. Construct MR-compatible source/follow-up program pairs. Use `P_0` as the source when suitable; otherwise construct another legal pair. Give each program its own execution configuration `c` to form execution cases `e = (P, c)`, including the necessary initial parameters, inputs, RNG settings, and environment. Preserve the MR's Preconditions, Input Relation, initial conditions, and semantic checkpoint requirements.
5. If a legal program no longer supports its selected MR, reselect the MR or retain it for coverage-only testing. Send programs, execution configurations, target predicates/checkpoints, and MR specifications to Layer 3. Preliminary checks and generated code alone do not prove state attainment or MR applicability.

### Layer 3: Inspector & Correctness Oracle

1. Collect runtime traces and necessary state evidence. Verify target API execution, the required backward path, legal completion of transition actions, and `phi_i(alpha(sigma)) = True` at the specified execution checkpoint.
2. For an execution pair `(e_a, e_b)`, dynamically verify MR Preconditions `Pre_M(e_a, e_b)` and Input Relation `In_M(e_a, e_b)`. Only when both hold, executions are valid, and semantic checkpoints align, evaluate `Out_M(Obs_M(e_a), Obs_M(e_b))`. A sound MR requires the output relation to hold under these conditions in a correct framework implementation.
3. Extract only the MR's required observables, such as loss, gradients, parameters, optimizer state, scheduler/AMP state, or checkpoint-restored state. Align semantic checkpoints; equal operation counts are not required. Use the MR's Output Relation, including suitable tolerances for numerical equivalence.
4. Distinguish the following verdicts:
   - **Invalid:** The program is invalid, the target API or required backward path did not execute, or the required target state was not reached.
   - **Inapplicable:** MR Preconditions or Input Relation do not hold, or required checkpoints cannot align.
   - **Pass:** The MR applies and its Output Relation holds.
   - **Suspected Failure:** The MR applies but its Output Relation does not hold.
5. Reproduce and minimize suspected failures, revalidate MR assumptions and numerical comparison conditions, and use another oracle or independent evidence before confirming a framework defect. Without a suitable MR, use another defined oracle or retain verified coverage with correctness unassessed.
6. Return API Coverage, State Coverage, Transition Coverage, API × MR Coverage, MR Applicability, Correctness Feedback, and Generation Cost (including generation and execution resources) to Layer 1. Use verified evidence and explicit candidate-set denominators; coverage over known feasible candidates does not imply exhaustive theoretical coverage. Guide subsequent sampling toward unexplored API × State × MR combinations, expected coverage gain, bug-triggering potential, and reasonable cost.

## Correctness Principles

- Treat an MR as conditional on both its Preconditions and Input Relation. Reaching a target state does not prove MR applicability or require every observable to be preserved.
- Preserve the initial conditions and semantic inputs required by the MR. Make permitted differences explicit; a shared seed alone does not establish comparable executions.
- Define observables at aligned semantic checkpoints, the Output Relation, numerical tolerances, and nondeterminism handling for the API, transformation, dtype, and environment. Do not require equal operation counts or universal output equality.
- Separate planning and static predictions from runtime evidence. Generated code or configuration flags alone do not establish API execution, state reachability, or MR applicability.
- Distinguish generation failures, unresolved plans, unreachable states, inapplicable MRs, and oracle violations. Never classify an arbitrary difference as a framework bug or give coverage-only cases a correctness pass.
- Report an oracle violation as a suspected failure with evidence. Confirm a framework defect only after reproduction, minimization, revalidation, and corroboration; absence of MR violations does not prove framework correctness.

## Generality and Extensibility

- All proposed methods and improvements must be designed for general applicability beyond the current 10 benchmark APIs. Consider expansion to all PyTorch APIs and the future addition of other DL frameworks, such as TensorFlow.
- Do not hard-code the current benchmark API names, count, or incidental characteristics into general algorithms, generation logic, state exploration, MR selection, inspection, or coverage reporting.
- Base applicability on explicit API semantics, framework capabilities, and required preconditions. Keep necessary API-specific or framework-specific handling separate from shared logic, and document its scope; do not assume every API or framework supports the same training states, transformations, or MRs.
- Explain how a proposed method or improvement extends to APIs outside the current benchmark and to additional frameworks, including any assumptions, unsupported cases, and required adaptations. This requirement does not select TensorFlow for immediate implementation or require every method to apply to every API.
- Validation must state the evidence and limits of generality. Results on the current 10 benchmark APIs alone do not establish applicability to all PyTorch APIs or other frameworks.

## Required Approval and Editing Workflow

- Read the user's requirements and the relevant existing code before proposing changes.
- Before modifying files, describe the intended implementation, affected files, and appropriate validation. Wait for explicit user approval of the plan.
- After approval, implement the agreed scope. Existing approval covers work within that scope; do not ask for the same permission again.
- After each user approval to begin execution, first create or update a roadmap in `docs/` recording the approved scope, implementation steps, and validation plan; update it with progress and validation results as work proceeds.
- Number roadmap files sequentially using `roadmap-NNN-<topic>.md`, starting with `docs/roadmap-001-mvp.md`. Related roadmaps or roadmaps for small changes may be merged; retain the existing roadmap number when merging and use the next available number for a new roadmap.
- Modify only the file or files explicitly authorized by the user.
- If another file must change, explain why the change is necessary, identify the affected file and expected impact, and obtain permission before editing it.
- Do not refactor unless the user explicitly requests it. Avoid unrelated cleanup, renaming, reformatting, or architectural changes.
- If the requirements are materially ambiguous or conflict with the approved design, explain the issue and obtain clarification before implementing the affected behavior.

## Implementation and Comments

- Follow the user's requirements, approved plan, and existing project conventions.
- Keep changes focused and easy to review. Do not invent additional requirements or abstractions.
- Add concise comments when needed to explain intent, preconditions, or non-obvious constraints. Avoid comments that merely restate the code.
- Do not add dependencies, change project configuration, or create supporting files outside the approved scope without permission.
- Leave undecided design choices open until the user provides direction or approves a proposal. This includes the target DL framework, sampling algorithms, concrete state abstractions and predicates, coverage units and aggregation policies within the documented metrics, and MR library contents.

## Validation and Reporting

- Run validation appropriate to the approved change and available environment.
- Creating or modifying test files is subject to the same file-scope approval rules as implementation changes.
- Check relevant behavior and failure cases when practical; avoid unrelated or redundant tests.
- Never claim validation passed unless it was actually run and passed. State any unavailable checks and their limitations.
- When finished, summarize what changed, how it was validated, and any remaining limitations or decisions requiring user input.

## File Storage and Organization

- Store all large files, including model weights, datasets, and large persistent caches, under `/work/asciibase64/Mirage/`.
- Test outputs are an exception and may be stored under the repository's `./result/` directory, including large test outputs.
- Organize files into subdirectories by purpose, such as `models/`, `datasets/`, and `cache/` under `/work/asciibase64/Mirage/`. Organize test outputs by test category and run, such as `./result/<test-category>/<run-id>/`. Do not scatter files or accumulate them directly in either storage root.

## Nano4 and Slurm Execution

- This workspace is on Nano4. All compute-intensive work must be dispatched through Slurm, including GPU work, training, LLM inference, compilation, full integration suites, large CPU workloads, batch experiments, and compute-intensive replay or minimization. Do not run these directly from an interactive agent shell without a Slurm allocation.
- Store job scripts under categorized subdirectories of `sbatch/`: `validation/` for smoke and integration tests, `experiments/` for exploration, and `replay/` for reproduction and manual minimization. Do not place `.sbatch` files directly in `sbatch/` or scatter them elsewhere. Shared shell setup belongs in `sbatch/lib/`.
- Use the user's Nano4 template as the default: `--account=ACD115198`, `--partition=8gpus`, `--nodes=1`, `--cpus-per-task=8`, `--gpus-per-node=1`, and `--time=4:00:00`. Resource changes must match the task and verified scheduler constraints; command-line `sbatch` overrides are explicit changes, not silent fallbacks.
- Load `"$HOME/.bashrc"`, report start time, end time, elapsed seconds, and exit status, and preserve failures as nonzero job exit codes. Launch computational commands with `srun` inside the allocation and preserve Slurm-managed GPU visibility.
- Use the existing Python installation. Select a prepared, compute-node-accessible Python/PyTorch environment; do not assume a login node's `/tmp` packages exist on compute nodes or install Python automatically.
- Keep logs and artifacts on compute-node-accessible persistent storage. Use job-specific local scratch for disposable compilation caches. Bound application work by the allocation's remaining time and reserve time to save evidence before Slurm termination.
- Lightweight file inspection, syntax/static checks, and scheduler queries may run directly. Validate batch scripts with `bash -n` and, when supported, `sbatch --test-only`; distinguish submission preflight from an executed job. Report scheduler rejection explicitly instead of bypassing Slurm or silently changing partitions or resource counts.
- Consult `sbatch/README.md` for environment variables, log-directory preparation, submission commands, and monitoring. Do not interpret a successful submission or Slurm job completion as an MR Pass without reviewing the actual test or experiment results.

### Interactive Development and vLLM Reuse

- During a development task that requires repeated real-model validation, reuse one Slurm allocation and one healthy vLLM service across edit-and-test iterations. Use an interactive terminal attached to the allocation for test commands; launch computational work with `srun`. Prepare code and run appropriate lightweight checks before loading the model.
- Keep inference and test execution in separate Slurm steps with explicit resource assignments. When using the two-GPU development configuration, assign one GPU to vLLM and the other to testing, respecting Slurm-managed GPU visibility and the resource rules above.
- Ordinary test failures must preserve their logs and nonzero exit status without automatically stopping the healthy vLLM service. Fix the code and start a fresh test process against the same service. Restart vLLM when the service fails or changes to its model, environment, or serving configuration require a restart.
- Give each test iteration a separate artifact directory and preserve the exact code and configuration used for that run. Do not allow edits during an execution to change its code. Confirm outstanding inference requests have completed or been cancelled before reusing the session after an interrupted test; closing a client connection alone is not proof of cancellation.
- Reuse is limited to the current user-requested task. Before reporting that task complete, stop its test processes and inference service, release its Slurm allocation, close interactive terminal connections, and verify that its owned jobs and steps have exited. Do not leave vLLM running between completed tasks or stop unrelated jobs.
- Provide cleanup for normal exit, interruption, and termination, with the Slurm time limit as a final bound. Reserve time to save evidence and shut down. If cleanup cannot be verified, report the remaining job or step identifiers and the unresolved condition explicitly.
- Record service startup time, per-iteration test time, restart count, and allocated resource time separately so that reuse benefits can be measured. Do not report allocated GPU time as GPU utilization or add overlapping durations as wall time.

## Current Repository Context

- The project is named `mirage` and uses a `src` layout.
- The package entry file is `src/mirage/__init__.py`.
- `pyproject.toml` currently requires Python `>=3.14`, uses `uv_build`, and declares the CLI entry point `mirage = "mirage:main"`.
- No runtime dependencies are currently declared. Do not assume a DL framework or test runner has already been selected or installed.
- Use the actual repository configuration as the source of truth if these details change.
