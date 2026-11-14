# Roadmap 005: State-semantic coverage

## Approved scope

This roadmap records three separately approved phases. All three implementation
phases and their bounded validation are complete; performance improvement is not
established. Subsequent approvals, scope extensions and results appear below.

Initial phase 1 scope: evidence-backed target/operand/path coverage, distinguish
output gradients from target backward execution, and add negative controls.

Authorized implementation files:
- `src/mirage/contracts/capabilities.py`
- `src/mirage/runtime/stateful.py`
- `src/mirage/backends/pytorch.py`
- `src/mirage/inspector/stateful.py`
- `tests/test_stateful.py`

The initial phase 1 plan required no dependencies, configuration changes or new
implementation/test files. Later file additions were separately approved below.

## Implementation steps

1. Define versioned coverage evidence with per-call, per-operand attribution.
2. Preserve output-gradient observations separately; certify backward only from
   attributable Node execution. Direct operand gradient contributions require
   unambiguous autograd edges. Unsupported/composite paths remain unknown.
3. Collect normalized relation signatures with checkpoint and environment scope,
   preserving raw witnesses independently of cross-execution deduplication.
4. Store coverage inside inspection so existing worker persistence and replay
   comparisons include it without changing worker files.
5. Add counterexamples and compatibility checks for eager, compiled and captured
   paths. No saved-tensor hooks or claim of backward representation preservation.

The existing `target_backward_count` predicate retains its documented historical
unit (output gradient events); a distinct node execution count is added. The
`backward_executed` flag will require attributable Node execution.

## Validation plan

- Lightweight syntax and diff checks on the login node.
- Runtime tests only through Slurm using the existing prepared `.venv` and
  `sbatch/validation/full_suite.sbatch` (default account/partition/resources).
- Freeze each test iteration's source/tests/scripts under categorized `result/`
  artifacts; preserve logs, source hashes, environment, status and timings.
- Negative controls: unrelated backward, output-only gradient, cross-call and
  cross-operand attribution, valid zero gradients, materialized representation,
  unavailable evidence, duplicate outputs, replay and normalized signatures.
- Inspect actual test results, then verify owned Slurm jobs/steps have exited.

## Progress

- Repository and Slurm conventions inspected. Working tree initially clean.
- Implementation in progress; no runtime validation executed yet.

### Implementation and first validation submission

- Added versioned path evidence and normalized relation signatures under
  `inspection.semantic_coverage`; existing replay compares this field.
- Output-gradient events retain their historical counter. `backward_executed`
  now requires an attributable Node event in a completed backward invocation.
- Direct unambiguous autograd edges provide operand contributions, including
  zero gradients. Shared coordinates, opaque/composite paths and saved-tensor
  representation remain unknown/unsupported as appropriate.
- Recorded target-entry context separately from checkpoint state and bounded
  event joins by checkpoint offsets. Hooks are removed on success and failure.
- Compiled operand metadata is certified only for a direct single-operation
  graph; captured-callable boundary evidence does not certify internal operands.
- Added 17 focused counterexample/compatibility tests in `test_stateful.py`.
- Syntax parsing of all five modified Python files, `git diff --check`, and
  `bash -n` for existing batch/environment scripts passed.
- Default Slurm preflight passed, but jobs 772252 and 772271 stayed pending with
  `MaxGRESPerAccount`; both were cancelled before execution. Iteration 01 was
  superseded by the frozen iteration 02 source snapshot.
- Explicit CPU-only preflight on `8gpus` was rejected: that partition requires
  1–8 GPUs per node. No CPU workload was run outside Slurm.
- Verified `dev` allows the requested account and four-hour allocation, with its
  separate GPU account limit. Explicit `--partition=dev` preflight passed.
  Job 772281 runs the existing full suite from the iteration 02 snapshot with
  the original 1 GPU / 8 CPU / 4-hour resource settings. Runtime results pending.


## Final validation and status

Phase 1 is complete within the approved five-file implementation/test scope.

- Job **772281**: completed, exit status **0:0**, all batch/extern/test steps
  completed. Jobs **772252** and **772271** were cancelled while pending and
  consumed no allocation. No model/inference service was started.
- Full suite: **174 tests, 173 passed, 1 skipped**, including all **17 new
  semantic-coverage tests**, compiled cases, CUDA captured execution, isolated
  workflow and zero-provider replay.
- The skipped existing benchmark test requires a saved stateful fixture artifact
  path (`MIRAGE_STATEFUL_AUDIT_ROOT`), which was not supplied. This is not a skipped
  CUDA or newly added coverage test.
- Environment: prepared Python 3.14.10, PyTorch 2.15.0+cu132, NVIDIA H200.
- Test duration: **83.056 seconds**. Allocation elapsed time: **91 seconds** with
  8 CPUs and 1 GPU; this is allocated time, not GPU utilization. These overlapping
  durations are not added together.
- All five current implementation/test files match the executed frozen source
  hashes. Final `git diff --check` passed.

Artifacts:
- [Test log](../result/validation/state-semantic-coverage/iteration-02/772281/tests.log)
- [Environment](../result/validation/state-semantic-coverage/iteration-02/772281/environment.json)
- [Validation summary](../result/validation/state-semantic-coverage/iteration-02/validation.json)
- [Frozen source hashes](../result/validation/state-semantic-coverage/iteration-02/source-sha256.json)

## Explicit limits

- Direct attributable backward nodes and unambiguous operand edges are supported;
  composite/fused internals without this evidence remain unknown. This may reduce
  accepted backward cases relative to the former output-hook criterion.
- Saved-tensor representation preservation is unsupported, not inferred from
  target-entry representation. Captured-callable boundary execution is not a
  certificate of internal per-operand gradient contributions.
- Coverage is versioned observed relation units with no theoretical denominator.
  It does not establish MR applicability or framework correctness.
- Existing `target_backward_count` remains an output-gradient-event count;
  `target_backward_node_count` is the new node-execution count. Backward validity
  additionally requires a completed invocation.
- Local program mutation, multi-seed corpus and adaptive sampling remain deferred
  to phases 2 and 3 of the proposal.

## Phase 2 approval and plan

The user's continuation authorizes phase 2 of the preceding proposal. Preserve
all phase 1 changes. Adaptive coverage/cost scheduling remains phase 3.

Authorized phase 2 files:
- modify `src/mirage/generator/program.py` and `src/mirage/workflow.py`;
- add `src/mirage/generator/mutation.py` and `src/mirage/artifacts/corpus.py`;
- add `tests/test_mutation.py` and `tests/test_corpus.py`;
- update this existing roadmap (same topic and sequential number).

Implementation:
1. Describe bounded editable AST regions in a selected parent. Keep the existing
   provider's `{source: string}` transport: in a mutation request, source is only
   replacement statements, not a regenerated complete program. Compose locally,
   enforce a size/edit budget, preserve protected target statements and all AST
   outside the editable region, and then run the normal static/runtime checks.
2. Maintain a persistent corpus of independently verified execution cases with
   source/configuration, semantic coverage, MR applicability, costs, immutable
   initialization/replay references and parent/edit lineage. Keep parent seeds;
   quarantine suspected failures and never use them as repair/mutation parents.
3. Use deterministic per-API round-robin parent selection and a bounded periodic
   fresh-generation opportunity. Unsupported local structures/compile goals use
   explicitly recorded fresh generation. Repairs retain their original contract.
4. Re-run each child's own source/follow-up MR pair. Parent/child lineage is never
   an MR certificate. Retain legal coverage-only children with correctness
   unassessed, and reject invalid/unverified admissions.
5. Test out-of-region edits, target deletion, wrong parent hashes, edit budgets,
   multiple seeds, deduplication, lineage, quarantine, corpus reload/integrity,
   unchanged parent replay, and isolated end-to-end child replay.

Validation: syntax/diff checks locally; runtime/full regression through existing
Slurm scripts and prepared Python. Freeze each iteration under
`result/validation/local-mutation-corpus/`. Authored fixture providers verify
protocol/plumbing; no real-model efficiency or discovery claim will be made from
these tests. Real-model evaluation is separate from this implementation gate.

Status: implementation in progress.

### Phase 2 implementation and validation progress

- Local edits replace only a bounded statement region: initialization assignment,
  operand setup insertion, or the tail after direct target calls. The framework
  composes the child and verifies the unchanged AST outside that region.
- Existing provider/HTTP `{source: string}` transport is unchanged; mutation
  requests explicitly ask for replacement statements. Complete build/run
  responses, parent hash mismatches, protected target deletion, setup early exits,
  imports, no-op edits and budget overruns are rejected.
- Parent backward-call obligations are checked against the child's runtime
  evidence, separately from its own MR result. Another target's gradient cannot
  satisfy a lost original path.
- `corpus.json` stores replay-backed seeds, semantic signatures, costs,
  applicability and lineage. Source/initialization/engine hashes are checked
  before selecting a seed. New coverage and applicable relations, bounded
  alternate structures, and cheaper equivalent representatives can be retained;
  superseded entries and their replay files are not deleted.
- Suspected failures are quarantined and excluded from both selection and repair.
  Coverage-only seeds retain unassessed correctness. Valid input refinements are
  new local initialization children rather than full-program repair requests.
- Selection is per-API round-robin, with a fresh-generation slot every four API
  proposals. Missing bounded regions and execution-mode goals explicitly fall
  back to fresh structure. This is a baseline policy, not phase 3 adaptive scoring.
- Added 21 tests across `test_mutation.py` and `test_corpus.py`, including isolated
  authored-provider runs, original/child replay and injected failure controls.
- Syntax parsing of six phase 2 files, `git diff --check`, and batch `bash -n`
  checks passed. Default `8gpus` preflight predicted a delayed start. Rechecked
  `dev` constraints and explicitly selected `--partition=dev`; account, 1 GPU,
  8 CPUs and four-hour limit remain unchanged.
- Full-suite job **772332** uses frozen iteration 01 sources. Runtime validation
  is in progress; early corpus and replay checks have passed.


### Phase 2 final results and limits

Phase 2 is complete in the approved two modified implementation files, two new
modules and two new test files. All phase 1 code/tests were preserved.

- Full regression job **772332** completed with exit **0:0**; all batch, extern
  and test steps exited. No inference service was started or left running.
- **195 tests: 194 passed, 1 skipped**, including all **21 new phase 2 tests**.
  The existing saved-fixture audit was skipped because
  `MIRAGE_STATEFUL_AUDIT_ROOT` was not supplied. CUDA and compiled regressions ran.
- Tests took **187.994 seconds**; the allocation elapsed **196 seconds** with
  1 GPU and 8 CPUs. Allocation time is not GPU utilization, and overlapping test
  and allocation durations are not summed.
- Current source/test file hashes match the frozen executed snapshot. Final
  syntax and diff checks passed.
- Validation uses authored deterministic provider responses, including invalid
  edits and injected failures. It establishes protocol, runtime, admission and
  replay behavior, not real-model generation success or discovery improvements.

Phase 2 artifacts:
- [Test log](../result/validation/local-mutation-corpus/iteration-01/772332/tests.log)
- [Environment](../result/validation/local-mutation-corpus/iteration-01/772332/environment.json)
- [Validation summary](../result/validation/local-mutation-corpus/iteration-01/validation.json)
- [Frozen source hashes](../result/validation/local-mutation-corpus/iteration-01/source-sha256.json)

Limits retained for later work:
- Local regions cover direct probe-call programs; nested/opaque structures and
  execution-mode goals explicitly use fresh generation rather than asserting a
  local preservation certificate.
- Each admitted child carries its own initialization/replay evidence and MR
  assessment. Its initial values and coverage may differ from the retained parent.
- Corpus indexes can be reloaded and integrity-checked, while the workflow still
  creates a new run directory; a cross-run resume/import CLI is not introduced.
- Fixed round-robin selection and periodic fresh generation are baseline policies.
  Adaptive API/state/MR gain-versus-cost scheduling and real-model ablations
  remain phase 3.

## Phase 3 approval and plan

The user's continuation authorizes the proposed adaptive feedback and real-model
comparison phase. Preserve completed phase 1/2 behavior as explicit baselines.

Files: modify `src/mirage/sampler/policy.py`, `src/mirage/workflow.py`,
`src/mirage/artifacts/report.py`, and the previously approved corpus module's
selection entry point; pin phase 2 baseline tests in `tests/test_corpus.py`;
add `tests/test_adaptive.py` and `benchmarks/stateful_search.py`. Update this
roadmap. Reuse existing Slurm/service scripts without changes or dependencies.

Plan:
1. Rank available API/seed/edit candidates by smoothed runtime-valid probability,
   verified coverage/MR novelty and measured generation-plus-execution cost,
   with seeded exploration and periodic fresh generation. Record the complete
   candidate-set denominator, scores/probabilities and observed feedback.
2. Separate transition-based and relation-based selection feedback, and local
   versus complete-program generation, while keeping the same runtime oracle.
   Keep per-case evidence, token availability and failures; no reward from
   generated claims, missing state, or failed preservation obligations.
3. Report unique verified semantic/transition/API/MR units, valid execution rate,
   applicability, per-resource gains, and reproduction outcomes without converting
   coverage-only results into correctness passes or inventing theoretical rates.
4. Add deterministic scheduler/report tests and isolated workflow checks; run the
   full suite inside one reusable two-GPU development allocation (one service GPU,
   one testing GPU). Prepare code/static checks before loading the model.
5. Use the prepared Qwen3-Coder-Next-FP8 model, bounded development pilot, then a
   frozen small ablation: full/local generation × transition/semantic feedback,
   plus local-semantic round-robin baseline. Equal per-run budgets, explicit API
   and seed lists and ordered/reversed treatments. This pilot-sized experiment
   cannot establish broad superiority or defect discovery rates.
6. Reuse the same healthy vLLM service across iterations, preserve snapshots and
   logs, verify no outstanding inference before reuse, and stop service/test
   steps/allocation before completion. Record startup, restart count, per-iteration
   duration and allocation duration separately.

Status: phase 3 implementation and bounded validation completed; final results below.

### Phase 3 implementation/startup progress

- Added `FeedbackSampler`: hierarchical API/edit-family and parent estimates,
  smoothed runtime validity × bounded novelty / measured seconds; epsilon
  exploration, first-API proposals and periodic fresh generation remain explicit.
- Every attempt records selection candidates/probabilities and its observed
  feedback. Invalid/unreached/preservation-failed cases receive no novelty, but
  their measured generation/execution time is charged to the selected action.
- `run_stateful` now defaults to adaptive selection, with explicit round-robin,
  full-generation and transition-feedback baselines. The transition ablation
  changes corpus admission units as well as scheduling feedback; every treatment
  is scored using the same runtime semantic evidence.
- Added common search metrics with explicit observed-unit counts, token-usage
  completeness, applicability and failure stages. No theoretical denominator or
  confirmed-defect rate is inferred.
- Prepared eleven adaptive/report/integration tests and the frozen-protocol
  `benchmarks/stateful_search.py` driver. Seven phase 3 files passed syntax checks;
  `git diff --check` and shell syntax checks passed before loading the model.
- Default `8gpus` preflight predicted a delayed start. Explicit `dev` preflight
  passed with existing two-GPU development resources (8 CPUs, four hours).
- Reusable allocation **772406** started with the existing prepared
  Qwen3-Coder-Next-FP8 revision `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`.
  Service startup is in progress; no model requests have been sent yet.

### Phase 3 startup retry

- Allocation 772406 failed at the existing 900-second service-readiness deadline
  while model shards were still loading (31/40); no real-model request had run.
  Its concurrent regression iteration was interrupted by allocation shutdown,
  so it is **incomplete**, not a passing suite. The eleven new adaptive tests
  had finished successfully before interruption. All old steps and the attached
  test terminal exited; job-owned scratch was removed from the same node.
- Retry allocation **772472** explicitly retains `dev`, two GPUs/eight CPUs and
  four hours, with the existing `MIRAGE_SERVE_START_TIMEOUT=2400` parameter.
  No batch-script or model configuration change. This is one startup retry;
  the healthy-service restart count will be reported separately.

- Retry service became ready after **1383 seconds**. Health/model checks and
  serving/testing GPU isolation passed. Startup included missing-cubin cache
  reload warnings, followed by completed warmup and a healthy endpoint. No
  healthy-service restart has occurred. Full regression now runs from a fresh
  snapshot with CUDA, compilation and end-to-end checks enabled.
- Review also tightened explicit corpus selection when an API has no active seed
  and labels transition-basis admission as `new_transition_coverage`; regression
  assertions cover both.

### Phase 3 full regression

- Allocation 772472, iteration `iteration-juVwKl`: **206 tests, 205 passed,
  1 skipped, zero failures**. CUDA, compilation and end-to-end tests enabled.
  The sole skip remains the existing fixture audit without
  `MIRAGE_STATEFUL_AUDIT_ROOT`. Test-reported duration 1590.376 seconds;
  iteration duration 1619 seconds (includes snapshot/process overhead).
- Current Python source, tests and benchmark driver match the executed snapshot.
- Real-model development pilot starts with `torch.sin`, search seed 0,
  300-second search budget and six-case cap. This larger pilot budget reflects
  measured isolated-worker overhead, to permit reaching local edits.

### Real-model development pilot and prompt refinement

- `pilot-01` (iteration `iteration-e3Xipq`): six attempts, one runtime-valid MR
  pass and successful independent replay. Three attempts failed target backward;
  two local edits incorrectly repeated the protected target and were rejected
  statically. Search 193.511 seconds, metadata 16.556 seconds, iteration 244
  seconds, 18,539 reported tokens. Both failure classes remained excluded from
  verified coverage. This is development evidence, not the formal comparison.
- Within the previously authorized local-generator file
  `src/mirage/generator/program.py`, sharpened the local prompt to identify the
  exact statement slice and distinguish insertion, one-assignment replacement
  and suffix replacement. Each kind explicitly states which existing statements
  remain in place and must not be repeated. Bumped local prompt to `stateful-5-local`;
  full-generation prompt and runtime acceptance rules are unchanged.
- Validate this prompt-only refinement with the existing mutation regression
  suite and a second real-model development pilot before freezing formal runs.

- Prompt refinement validation: iteration `iteration-5M1ave`, all nine mutation
  tests passed (11.113 test seconds, 31 iteration seconds).
- `pilot-02`: six attempts, two runtime-valid coverage-only cases (one local
  child); the first valid saved case replayed successfully. Four further local
  edits were rejected (protected target repetition or no AST change). Search
  93.093 seconds, metadata 16.257 seconds, 18,682 reported tokens. No model
  quality improvement is inferred from these two stochastic development pilots.

### Frozen formal pilot-sized comparison

Use the current prompt and engine without further edits during the comparison:
`torch.sin` and `torch.add`, search seeds 11 and 29, all five driver treatments,
480-second search cap and 24-attempt cap per run, worker timeout 90 seconds.
The cap permits several state/edit decisions despite measured worker overhead.
Initial metadata and one independent saved-case replay are timed separately.
Search seeds control framework exploration; they do not establish deterministic
model outputs or independent statistical replication. Two blocks reverse arm
order, and no significance or general-superiority claim is planned.

### Phase 3 results and limits

The frozen ten-run comparison completed with exit status 0. Every run's first
runtime-valid saved case was replayed successfully: **10/10 replays**, without
provider calls. Current source/tests/driver match the formal snapshot; all formal
engine inventories were verified. The prompt refinement followed the full suite
and was covered by nine mutation regressions plus the second pilot/formal runs.
Final syntax validation covered 66 Python files; `git diff --check` passed.

Each table cell is **seed 11 / seed 29**. Semantic units are per-run observed
units, not fractions of the theoretical space or a union across runs.

| Treatment | Attempts | Runtime-valid cases | Semantic units | Local attempts |
| --- | --- | --- | --- | --- |
| full-transition | 11 / 8 | 5 / 4 | 6 / 5 | 0 / 0 |
| local-transition | 11 / 8 | 3 / 3 | 4 / 5 | 3 / 1 |
| full-semantic | 8 / 7 | 4 / 4 | 6 / 7 | 0 / 0 |
| local-semantic | 11 / 8 | 3 / 4 | 4 / 6 | 3 / 1 |
| local-semantic-round-robin | 13 / 9 | 3 / 3 | 7 / 4 | 6 / 3 |

Across the ten runs there were **94 attempts, 36 runtime-valid cases and 15
applicable MR comparisons**. No suspected failures were reported; this does not
establish framework correctness. All 305,761 reported generation tokens had
complete usage records. Search budgets are caps: conservative dispatch reserves
may stop a run before its 480-second cap; actual elapsed time is reported.

The formal runs produced **0 valid local children out of 17 local attempts**:
nine static-contract rejections, six state-attainment failures and two unresolved
runtime failures. A development pilot with no target-state predicates did produce
a valid local child, so that result must not be presented as success under the
formal runtime-derived state goals. Some candidate edit regions cannot directly
change the required training-history facts; edit-region legality alone does not
establish goal feasibility. Future design should connect state-goal requirements
to edit-region capabilities before candidate selection, and clarify bounded
suffix edits when additional target invocations are needed.

These results **do not support a stable efficiency advantage** for the new local
strategy. Full-semantic observed more units than local-semantic for both seeds
(6 vs 4, 7 vs 6); adaptive versus round-robin results were mixed (4 vs 7, 6 vs 4).
This is a small two-API/two-seed comparison, not a significance test. Model output,
initialization/compile cost, early dispatch stops and short exploration histories
limit interpretation. No further prompt or scheduler tuning was mixed into the
frozen comparison. The approved adaptive/local default remains an experimental
policy; full generation and round-robin remain explicit comparison options.

Timing and cleanup:
- Successful service startup: **1383 seconds**; **zero healthy-service restarts**.
  One earlier startup failed at the 900-second readiness deadline (job 772406);
  its interrupted test iteration remains incomplete, not a pass.
- Full-suite iteration: **1619 seconds**; first pilot: **244 seconds**;
  prompt regression: **31 seconds**; second pilot: **144 seconds**;
  formal comparison iteration: **4531 seconds**.
- Formal search: **3890.232 seconds**; API metadata: **323.567 seconds**;
  saved-case replays: **310.532 seconds**, recorded separately.
- Slurm allocation 772472: **8385 seconds**, two GPUs/eight CPUs, completed 0:0.
  Failed startup allocation 772406: **905 seconds**, two GPUs/eight CPUs.
  Combined allocated time: **18,580 GPU-seconds and 74,320 CPU-seconds**;
  these are allocated resource time, not utilization. Overlapping startup, test
  and service durations are not added as wall time.
- Final service metrics showed zero running and zero waiting requests. Test
  processes exited, job-owned temporary caches were removed, the attached
  terminal closed and both owned allocations/all steps exited. Service step
  cancellation is the requested shutdown; the successful allocation completed.

Artifacts:
- [Validation and resource summary](../result/validation/adaptive-search/validation.json)
- [Frozen formal protocol](../result/validation/adaptive-search/772472/formal-01/protocol.json)
- [Per-run metrics and replay evidence](../result/validation/adaptive-search/772472/formal-01/comparison.json)
- [Full regression log](../result/validation/adaptive-search/772472/iteration-juVwKl/stderr.log)
- [Prompt regression log](../result/validation/adaptive-search/772472/iteration-5M1ave/stderr.log)
- [Formal frozen source](../result/validation/adaptive-search/772472/iteration-0Kzkwm/code/)
- [Slurm accounting](../result/validation/adaptive-search/slurm-accounting.tsv)
- [Final service metrics](../result/validation/adaptive-search/772472/final-service-metrics.txt)

## Approved follow-up: original ten-benchmark revalidation

The user's approval after clarification authorizes two missing checks: rerun the
original ten fixed benchmarks with the current implementation, then measure
whether autonomous generation reproduces those same known issues. The prior
10/10 generated-case replay result is explicitly not this benchmark test.

Scope: update this existing roadmap and write execution evidence under
`result/benchmark-revalidation/`; reuse existing production code, fixture programs,
independent auditors and Slurm helpers without editing them or dependencies.
Preserve existing uncommitted implementation changes.

Plan and frozen settings:
1. Run `evaluate.py validate-stateful` on all ten issue-derived fixed cases using
   the current engine, inspect target execution, MR violation and independently
   checked controls. One fixed-input execution per issue, timeout 180 seconds.
2. Freeze a separate revalidation protocol and exact invocation against the
   measured fixture engine. Do not reuse or forge the older G6 acceptance gate:
   its historical pilot/edge/heldout evidence is not evidence for this version.
3. Call current `run_stateful` for the same ten APIs/devices, seed 0, 300 seconds
   per API, mixed forward/backward schedule, input refinement enabled, adaptive
   selection, local mutation and semantic feedback. Use the existing 120-second
   worker cap and effectively nonbinding 100000-attempt cap. Model and prepared
   environments remain as in phase 3. Persist protocol before generation.
4. Keep issue IDs, fixture constants and expected answers outside all generation
   requests. The benchmark catalog selects API/device only; issue attribution is
   post-hoc. Fixed-case outcomes do not count as generated reproductions.
5. Run the existing independent audit/attribution on saved source/follow-up
   evidence. Preserve automatic results; in a separate reviewed result also
   enforce local preservation obligations. Unknown audit results are not negative
   labels. Replay any independently supported known-issue candidates before
   reporting confirmed reproduction; unrelated anomalies remain separate.
6. Record per-issue attempts, runtime validity, suspected failures, independently
   attributed reproductions/detections, unknowns and generation/execution/audit
   costs. Compare historically with the old 0/10 generated result only with
   explicit limitations, not a causal improvement claim.
7. Reuse one Slurm allocation/service; reserve cleanup time, retain nonzero
   failures, verify no outstanding inference before reuse, and release all owned
   jobs/steps/terminal connections at completion. Use the prepared service's
   2400-second startup allowance based on the measured prior startup.

Status: completed on 2026-11-14. The original fixed fixtures were detected
10/10 with independent controls; autonomous rediscovery was confirmed for 0/10
issues in the bounded run below. Production-code changes remain outside this
execution scope and require a concrete follow-up plan.

Follow-up allocation submitted: **774821**, explicit `dev` partition after
preflight (default `8gpus` predicted a later start), existing two-GPU/eight-CPU
four-hour session, startup allowance 2400 seconds. Fixed fixtures run on the
separate testing GPU while the service initializes; generation starts only after
reviewing fixture evidence and service readiness.

Follow-up fixed benchmark results: allocation 774821, iteration
`iteration-xa1zGX`, **10/10 detected and independently corroborated**, exit 0,
77 iteration seconds. These are the original ten fixed issue-derived programs,
not generic generated-case replays. The service is ready and its GPU separation
checks passed; the first testing step/terminal has exited. Next run the benchmark
regression module with the now-available fixture audit path, then autonomous
generation using the saved, syntax-checked invocation in
`result/benchmark-revalidation/invocation.json`.

Benchmark regression `iteration-1A8wr0`: **9/9 passed, no skips**, including the
previously omitted saved-fixture control audit; 2.176 test seconds, 22 iteration
seconds. Service startup was 187 seconds, with no restart. Inference metrics
showed zero running/waiting requests before autonomous generation. The generated
experiment invocation is frozen before execution. Post-hoc replay is capped at
two candidates per issue, prioritizing independently matched mechanisms and
stopping after one confirmed representative; unreplayed cases are not confirmed.

### Follow-up final results — Slurm 774821

The original ten issue-derived fixtures were detected **10/10**, with independent
controls corroborating all ten. The saved-fixture regression passed **9/9**
without skips. These results establish fixture detection in the current
environment, not autonomous rediscovery. The old full G6 gate is not claimed.

Autonomous search finished all ten targets at seed 0 with 300 seconds per API,
adaptive selection, local mutation, semantic coverage, mixed forward/backward
paths and input refinement. It used the frozen engine already validated by the
fixtures; engine, evaluator and invocation hash checks passed. Issue details and
expected answers were not supplied to the generator.

| Issue | API | Attempts | Runtime-valid | Local attempts / valid | Applicable MR comparisons | Confirmed rediscovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 203078 | `torch.Tensor.scatter_add` | 101 | 20 | 75 / 4 | 17 | 0 |
| 204407 | `torch.linalg.slogdet` | 99 | 22 | 78 / 7 | 12 | 0 |
| 200599 | `torch.linalg.lu_solve` | 104 | 19 | 74 / 3 | 16 | 0 |
| 202706 | `torch.autocast` | 87 | 9 | 57 / 4 | 9 | 0 |
| 181723 | `torch.cuda.make_graphed_callables` | 46 | 0 | 0 / 0 | 0 | 0 |
| 204033 | `torch.compile` | 47 | 0 | 0 / 0 | 0 | 0 |
| 204365 | `torch.nn.functional.batch_norm` | 85 | 1 | 51 / 0 | 1 | 0 |
| 201268 | `torch.add` | 84 | 32 | 68 / 18 | 19 | 0 |
| 197089 | `torch.std` | 96 | 19 | 73 / 5 | 0 | 0 |
| 200613 | `torch.Tensor.copy_` | 92 | 9 | 72 / 4 | 4 | 0 |

Totals: **841 attempts, 131 runtime-valid (15.6%), 78 applicable MR comparisons,
53 coverage-only cases, zero suspected failures and zero confirmed benchmark
rediscoveries/detections**. All 78 applicable comparisons passed. Of 548 local
attempts, 45 were runtime-valid (8.2%); 368 failed static constraints. No
independently positive candidate or suspected failure required autonomous replay.

Observed limits and bottlenecks:

- Local static rejections included 124 edits with no semantic AST change, 94
  edits violating the protected-target boundary, 54 exceeding edit budgets and
  96 indentation errors. This is evidence of an inefficient generation/edit
  interface; it is not evidence that the underlying state space is exhausted.
- CUDA graph and `torch.compile` generation produced no runtime-valid cases.
  Recorded failures include capture invalidation, configured-target mismatch
  and unsupported checkpoint data. These unresolved runtime failures were not
  classified as framework defects.
- Batch normalization had one valid forward case. Some backward attempts ran
  but target-node attribution was unsupported, so they were excluded rather
  than credited with backward coverage.
- All 19 valid `torch.std` cases remained coverage-only because MR obligations
  were not established. Valid execution alone did not earn a correctness pass.
- Independent candidate-level audits remained unknown for 113/131 valid cases.
  Across all 841 attempts, 799 were unknown, including attempts without saved
  execution evidence; 18 known labels were inside the metric domain and 24
  outside it. Member-level checks yielded 161 negatives and 207 unknowns, with
  zero positives. Unknowns are not negatives and the supported-subset confusion
  matrix does not measure whole-benchmark accuracy.
- Confirmed rediscovery **0/10** is a measured lower bound under limited auditors,
  one seed, one model/environment and this budget. Historical autonomous search
  also reported 0/10, but this is not a controlled causal comparison. The current
  result does **not** establish improved bug-finding effectiveness.

Suggested next work, not implemented in this execution-only scope: make bounded
edits reliably satisfy the existing contract, establish legal factory/capture
target calls, and distinguish unsupported backward attribution from missing
execution. Re-evaluate actual rediscovery after those bottlenecks are addressed.

Resource and cleanup evidence:

- Service startup: **187 seconds**; restarts: **0**. Fixture iteration: **77
  seconds**; benchmark regression iteration: **22 seconds**; autonomous iteration:
  **2873 seconds**, all exit status 0.
- Search: **2823.862 seconds**; metadata: **21.262 seconds**; incremental post-hoc
  audits: **21.358 seconds**; final audit/review: **3.458 seconds**. The driver
  completed in **2871.560 seconds**. Known generation usage is **3,146,604 tokens**,
  a lower bound because `torch.compile` has incomplete usage reporting.
- Slurm allocation: **3379 seconds**, two GPUs/eight CPUs, `dev`, COMPLETED 0:0.
  Allocated resource time: **6758 GPU-seconds and 27,032 CPU-seconds**; these are
  not utilization. Fixture execution overlapped service startup, so durations
  are not summed as wall time.
- Final service metrics showed zero running/waiting requests. Test processes
  and both attached terminals exited, job-owned temporary caches were removed,
  and all owned Slurm steps/allocation exited. The serving step was cancelled
  for intentional shutdown; the parent job completed successfully.

Evidence:

- [Validation summary](../result/benchmark-revalidation/774821/validation.json)
- [Fixed original benchmark results](../result/benchmark-revalidation/774821/fixed-01/summary.json)
- [Frozen autonomous protocol](../result/benchmark-revalidation/774821/autonomous-01/protocol.json)
- [Automatic audit](../result/benchmark-revalidation/774821/autonomous-01/evaluation-automatic-v1.json)
- [Reviewed attribution](../result/benchmark-revalidation/774821/autonomous-01/evaluation-reviewed-v1.json)
- [Code integrity checks](../result/benchmark-revalidation/774821/integrity-check.json)
- [Accounting](../result/benchmark-revalidation/774821/slurm-accounting.tsv)
- [Cleanup verification](../result/benchmark-revalidation/774821/cleanup.json)

## Approved follow-up: Qwen3.8-27B model substitution

The user explicitly requested switching vLLM to Qwen3.8-27B and rerunning the
benchmark. Reuse the existing engine, benchmark auditors and Slurm helpers;
update this roadmap and save artifacts under
`result/benchmark-revalidation-qwen38/`. Download the official BF16 snapshot
`Qwen/Qwen3.8-27B` at revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0` under
`/work/asciibase64/Mirage/models/Qwen3.8-27B/`. Reuse the prepared vLLM Python
environment; no dependency installation or production-source edits are planned.

Plan: one Slurm allocation with separate serving/testing GPU steps. Download
and verify all indexed shards inside the allocation, record model/environment,
then verify service readiness and structured JSON generation. Preserve the
model's default chat-template behavior unless compatibility evidence requires
an explicitly documented serving adjustment. Rerun the ten original fixed
fixtures and fixture regression, then the same autonomous protocol as job
774821: seed 0, 300 seconds/API, worker timeout 120 seconds, adaptive selection,
local edits, semantic coverage, mixed paths and refinement. Freeze invocation
and engine before running, keep benchmark answers outside generation, and
independently audit/replay eligible known-issue candidates. Compare runtime
validity, local-edit validity, MR applicability and confirmed rediscovery with
the previous model, acknowledging one-seed/time-budget and auditor limits.
Record startup, download, iterations and allocation costs separately. Stop all
owned processes/steps and verify allocation cleanup before completion.

Preparation: official model/API and architecture verified; 27,781,427,952 BF16
parameters, not yet downloaded. Reference: https://huggingface.co/Qwen/Qwen3.8-27B

Qwen3.8 follow-up allocation **775183**, explicit `dev` after preflight
(`8gpus` predicted 19:41 versus `dev` 15:09), two GPUs/eight CPUs/four hours.
The interactive allocation started immediately on `25a-hgpn024`. Download and
shard validation took 145.930 seconds, 18 shards / 55,563,006,776 bytes. Service
startup took 243.921 seconds, GPU isolation verified, no restart. Fixed fixtures
10/10 independently corroborated (iteration O7QJKU, 76 seconds); benchmark
regression 9/9, no skips (zEKnLU, 2.224 test seconds / 3 iteration seconds).
Disjoint `torch.sin` compatibility smoke: 1 valid case / 1 MR pass (MnXhxv,
10 seconds), excluded from formal search. No chat-template or reasoning-parser
override was needed. The frozen engine and autonomous invocation match job
774821 exactly. The model change also changes FP8 to official BF16; the separate
smoke warmup and model defaults are recorded as comparison limitations.

Interruption of allocation 775183: the interactive owner shell auto-logged out
after 900 seconds idle, its trap stopped serving, and Slurm released the job.
The formal iteration ygcIo1 exited **2** after 519 seconds: scatter_add finished
59 attempts / 13 valid; slogdet stopped early at 232.196 search seconds with
48 attempts / 12 valid and connection refused (cancellation unconfirmed then).
All owned Slurm steps have now exited, eliminating outstanding inference. The
parent allocation's COMPLETED 0:0 is not a successful experiment result. This
is an orchestration failure, not evidence of a model/framework failure. Preserve
this incomplete run separately, switch to existing batch dev_session, and repeat
all ten targets without changing model, engine, prompts or budgets. Prepared
weights are reused; node scratch cleanup is tracked in interruption.json.

Recovery batch allocation **775287** uses the existing dev_session helper, the
same `dev` resources, and node 25a-hgpn024 so previous node scratch could be
removed. Previous owned node/login caches were removed; all old steps exited.
Fresh fixed validation: **10/10** (R18HC9, 67 seconds); benchmark regression:
**9/9**, no skips (XdGeU8, 3 seconds). New service startup: **198 seconds**, GPU
isolation verified. The task has one service restart, caused by orchestration,
not a model error. The fixed-engine manifest still matches baseline 774821.
The formal ten-target invocation is unchanged and starts again from target one;
interrupted counts are excluded from this complete-run comparison.

### Qwen3.8-27B complete rerun results — Slurm 775287

The requested model substitution is complete. Official BF16 Qwen3.8-27B at
revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` ran through vLLM 0.33.0
in the existing prepared environment. No production code, tests, dependencies
or shared serving helpers were changed. The model can be selected for subsequent
runs with `source result/benchmark-revalidation-qwen38/serving.env`. This is a
saved runtime configuration; repository-wide default model values were not edited.

The complete formal rerun used the same engine, benchmark, invocation, seed 0,
300 seconds/API, 120-second worker cap, adaptive/local/semantic search and
mixed forward/backward paths as baseline 774821. All ten runs stopped normally
at their time budgets. The interrupted allocation 775183 remains separate.

| Metric | Qwen3-Coder-Next-FP8, 774821 | Qwen3.8-27B BF16, 775287 |
| --- | ---: | ---: |
| Fixed benchmark detection + independent controls | 10/10 | 10/10 |
| Benchmark regression | 9/9 | 9/9 |
| Autonomous attempts | 841 | 519 |
| Runtime-valid cases | 131 (15.6%) | 87 (16.8%) |
| Local edits / runtime-valid | 548 / 45 (8.2%) | 286 / 28 (9.8%) |
| Applicable MR comparisons (all Pass) | 78 | 52 |
| Coverage-only cases | 53 | 35 |
| Suspected failures | 0 | 0 |
| Confirmed known-issue rediscovery | 0/10 | 0/10 |

Validity per attempt was slightly higher, but the fixed time budget yielded
fewer valid cases and applicable comparisons. There is **no demonstrated
improvement in confirmed benchmark rediscovery**. One seed, different precision
(FP8 versus BF16), model-default behavior and warmup/cache history prevent a
general model ranking or strict causal claim.

| Issue | API | Attempts | Runtime-valid | Local attempts / valid | Applicable MR | Confirmed rediscovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 203078 | `torch.Tensor.scatter_add` | 58 | 13 | 42 / 4 | 9 | 0 |
| 204407 | `torch.linalg.slogdet` | 59 | 14 | 41 / 5 | 7 | 0 |
| 200599 | `torch.linalg.lu_solve` | 61 | 12 | 42 / 2 | 10 | 0 |
| 202706 | `torch.autocast` | 37 | 9 | 0 / 0 | 6 | 0 |
| 181723 | `torch.cuda.make_graphed_callables` | 36 | 0 | 0 / 0 | 0 | 0 |
| 204033 | `torch.compile` | 35 | 0 | 0 / 0 | 0 | 0 |
| 204365 | `torch.nn.functional.batch_norm` | 50 | 2 | 28 / 0 | 1 | 0 |
| 201268 | `torch.add` | 66 | 15 | 48 / 4 | 15 | 0 |
| 197089 | `torch.std` | 53 | 14 | 39 / 6 | 0 | 0 |
| 200613 | `torch.Tensor.copy_` | 64 | 8 | 46 / 7 | 4 | 0 |

Observed bottlenecks and audit limits:

- 164 attempts failed static edit checks: 66 crossed the protected target
  boundary, 63 made no semantic AST edit, 30 exceeded edit budgets and five had
  indentation errors. Of 286 local edits, only 28 were runtime-valid.
- 170 attempts failed runtime validation. The largest recorded category was
  108 failures to register trainable leaf tensors as parameters, followed by
  21 no-grad backward errors and 13 configured-target mismatches. These are
  generated-program/contract failures, not confirmed DL framework defects.
- CUDA graph and `torch.compile` again produced zero runtime-valid cases.
  Some compiled cases reached execution but lacked required backward evidence.
  All 14 valid `torch.std` cases were coverage-only.
- Candidate-level independent audits remained unknown for **73/87 valid cases**
  and **505/519 total attempts**. The all-attempt count includes missing execution
  evidence. Fourteen candidates were independently negative inside the supported
  metric domain; member-level labels were 91 negative / 195 unknown / zero
  positive. Unknown is not negative.
- No suspected failure or independently positive known-mechanism candidate
  required replay; there were zero replay attempts. Confirmed rediscovery 0/10
  is a measured lower bound under the available certificates, not proof that
  every unadjudicated case is correct.

Validation, resource accounting and cleanup:

- Fresh fixed fixture iteration R18HC9: 10/10 detected with independent controls,
  67 seconds, exit 0. Regression XdGeU8: 9/9, zero skips, 2.230 test seconds /
  3 iteration seconds, exit 0. Formal iteration w80p8n: **2843 seconds**, exit 0.
- Formal search: **2800.028 seconds**; metadata: **21.179 seconds**; incremental
  audit: **16.915 seconds**; final review: **2.694 seconds**. Driver wall time:
  **2842.353 seconds**. Generation usage: **1,913,229 tokens**, fully reported.
- Model download/shard validation: 145.930 seconds (Slurm preparation step:
  152 seconds). First service startup: 243.921 seconds; recovered startup:
  198 seconds. One service restart was required because the interactive owner
  auto-logged out; the complete rerun used the stable batch helper.
- Interrupted allocation 775183: **1188 seconds**; complete allocation 775287:
  **3178 seconds**, both two GPUs/eight CPUs. Total allocation time: **4366
  seconds**, **8732 allocated GPU-seconds**, **34,928 allocated CPU-seconds**.
  These are allocations, not utilization; overlapping startup/fixture work
  is not added as wall time. Initial incomplete search is excluded from the
  outcome comparison but included in task cost and preserved in artifacts.
- Final service metrics were zero running/waiting requests. All owned processes,
  Slurm steps/allocations and attached terminals exited. Old and new job-specific
  node/login scratch was removed. Serving-step cancellation was intentional
  shutdown. Prepared weights and reusable model caches remain under `/work`.
- Engine, benchmark and invocation hash checks passed against baseline; final
  working-engine hashes match the frozen execution engine.

Evidence:

- [Model configuration](../result/benchmark-revalidation-qwen38/serving.env)
- [Comparison summary](../result/benchmark-revalidation-qwen38/comparison.json)
- [Full validation and costs](../result/benchmark-revalidation-qwen38/775287/validation.json)
- [Frozen protocol](../result/benchmark-revalidation-qwen38/775287/autonomous-01/protocol.json)
- [Reviewed audit](../result/benchmark-revalidation-qwen38/775287/autonomous-01/evaluation-reviewed-v1.json)
- [Interrupted-run diagnosis](../result/benchmark-revalidation-qwen38/775183/interruption.json)
- [Integrity](../result/benchmark-revalidation-qwen38/775287/integrity-check.json)
- [Accounting](../result/benchmark-revalidation-qwen38/slurm-accounting.tsv)
- [Cleanup](../result/benchmark-revalidation-qwen38/775287/cleanup.json)

## Approved continuation: explicit state plans and serving validation

User approved the proposed repair scope and automatic continuation between phases
on 2026-11-14. First implement and validate explicit LLM state-goal planning,
operand-bound runtime predicates, goal-compatible bounded edits, no-op feedback,
and installed API reference documentation. Then validate Native MTP and retain
CUDA graphs, optimized attention and prefix caching with measured performance.

Authorized production files: generator/state_plan.py (new), contracts/capabilities.py,
contracts/exploration.py, generator/provider.py, generator/program.py,
generator/mutation.py, generator/http_provider.py, sampler/policy.py, workflow.py,
runtime/api_metadata.py, runtime/worker.py, runtime/stateful.py,
inspector/stateful.py, artifacts/corpus.py, artifacts/report.py.
Also authorized: sbatch/lib/serving.sh, sbatch/README.md,
benchmarks/stateful_search.py, tests/test_state_plan.py (new), existing
 test_mutation.py, test_adaptive.py, test_corpus.py, test_stateful.py,
 test_program.py, test_http_provider.py, test_capabilities.py, test_exploration.py,
and this roadmap. Artifacts go under result/.

Sequence and acceptance:
1. Implement explicit plans with up to three candidates, novelty/compatibility
   filtering, bounded multi-region patches and independently verified predicates.
2. Expand installed API doc references with provenance; improve parameter errors
   and stable-prefix prompt layout. Run deterministic regressions through Slurm.
3. Validate real-model generation on disjoint APIs and fixed benchmark controls.
4. Compare frozen baseline, repaired engine MTP off, repaired engine MTP on with
   seed 0 and ten APIs at 300 seconds/API; planning/repair count against budgets.
   Measure MTP request replay first, including JSON validity, tokens and latency.
5. Review actual results, update evidence, stop owned services and allocations.

Planning and generated AST changes are not runtime state evidence. Unknown remains
unknown, coverage-only remains correctness-unassessed. Preserve existing canonical
initial-state execution and MR preconditions. No unrelated refactoring/dependencies.

Status: implementation started; no new runtime validation yet.

Implementation progress: explicit `state_goal` requests, validated goal identifiers,
operand/path predicates, immutable multi-region patch preimages, scoped no-op retry,
goal-specific sampler feedback and planner token accounting are implemented.
Installed API references now expand one hop with source hashes. Parameter errors
identify the exact field/property. MTP supports 0/1/2/3 tokens and records its
configuration. AST parsing, shell syntax and submission preflight passed.
Slurm job 775669 (explicit dev partition, two isolated GPUs) is running the first
frozen regression snapshot and starting the MTP-off service. Further edits and
added regressions will be validated from a separate immutable snapshot.

Validation progress (775669): first regression 212 tests exposed one outdated
initial-predicate/dimension assertion; corrected for operand-bound dimensions.
Second regression: 216 tests, 215 passed, one saved-fixture audit skipped until
fresh benchmark artifacts exist. Targeted follow-ups: 20 tests and then 28 tests passed. First
real-model smoke reached additional backward/target and compiled states on cos,
but insertion prompts caused protected-boundary copies; tightened the prompt.
A tanh generation exceeded the smoke's 60-second worker cap. Server metrics later
verified zero running/waiting requests; recorded drain evidence before reusing
service. Subsequent smoke uses the formal 120-second worker cap.
Second smoke exposed tuple/list normalization at corpus admission for operand
paths. Fixed JSON normalization and added an end-to-end representation corpus
roundtrip regression. Added explicit measurement definitions distinguishing
negative view bits from negative values and forward boundaries from backward
counts. All unsuccessful attempts remain in their immutable iteration artifacts.

Phase 1 acceptance: the third unrestricted disjoint-API smoke produced 7/11 valid
cos cases and 8/11 valid tanh cases, with five attained planned cases per API.
These are smoke observations, not controlled performance claims. A separate
explicit non-contiguous goal with `operand_gradient_contribution` passed on two
planned attempts (representation-smoke/acceptance.json). Coverage-only remained
unassessed. Added a count-bucket filter (one -> multiple, no automatic 2 -> 3
novelty claim) and retained verified operand backward requirements in automatic
representation/data goals. Explicit goals now follow a verified canonical.

Final source/test/script snapshots are under result/state-plan-repair/fixed-code;
original engine snapshot under baseline-code. Eight fixed HTTP requests (two
canonical, two plans, four patches) are saved for identical MTP performance
replays. Formal treatment arguments remain seed 0, ten APIs, 300 seconds/API,
120-second workers, planning/repair included in search budgets. Service remains
MTP off for the original and repaired-engine comparison.

Pre-formal correction: final review identified a mixed-path hidden-obligation
change: an operand-only edit could inherit a forward-only parent but be assigned
requires_backward=True by the attempt rotation. Child obligations now inherit the
parent's required/observed backward path. Added a real two-case workflow regression
with a forward-only parent and a second target call. Preserved the unused first
snapshot as fixed-code-v1, froze the corrected fixed-code, and required the entire
state-plan regression suite in the formal runner before its fixture/search gates.
The original-engine formal search is unaffected; repaired-engine formal search
had not started before this correction.

Formal MTP-off comparison completed (job 775669, same H200 service): original
530 attempts / 88 valid / 71 AST no-ops; repaired 271 attempts / 110 valid /
0 AST no-ops / 54 attained goal cases. Local validity improved from 29/294 to
46/96. Search time was 2803.52s versus 2641.98s, with planning included. Both
fixed-fixture gates detected 10/10; both autonomous searches confirmed 0/10
rediscoveries. Independent audits retained 74 and 80 unknown valid cases,
respectively. Repaired token accounting is incomplete for a truncated response.
Evidence: result/state-plan-repair/comparison-off.json. Two original-driver
preflight failures (bytecode inventory, legacy relative path) are preserved;
formal results use comparison-driver-v2.py and autonomous-final directories.

Eight fixed measured requests with MTP off completed in 25.464s, 1667 output
tokens, 8/8 JSON-shape-valid responses (not semantic correctness). Request hash
and raw responses/metrics are preserved. No requests remained running/waiting
before shutdown. Job 775669 and its test step completed, service step cancelled
by normal cleanup; allocation 02:12:21 at two GPUs, not GPU utilization.
MTP 1/2/3 measurements and the final repaired-engine MTP-on arm remain pending.

MTP fixed-request measurements completed on the same H200 node: off 25.464s,
MTP1 14.061s, MTP2 10.782s, MTP3 9.313s for eight measured requests after
one warmup pass. Every setting returned 8/8 valid JSON shapes. MTP1/2/3 emitted
1592 tokens each versus off 1667; MTP3 throughput ratio is 2.611x, batch latency
ratio 2.734x. Draft-token acceptance including warmup: 93.22%, 87.79%, 83.46%.
These are one-batch observations, not semantic correctness or general speedup.
Counter deltas include warmup; pre-warmup cache history differed. All raw
responses, metrics and startup logs are preserved in performance-comparison.json
and referenced directories. MTP1 job 776358 and MTP2 job 776428 cleaned up;
MTP3 job 776501 retains its healthy service for the final formal arm. Selected
configuration is result/state-plan-repair/selected-serving.env. Native MTP,
CUDA graphs, FlashAttention/FlashInfer and prefix caching are visible in logs.

All three formal arms completed. Repaired engine + MTP3 (776501): 454 attempts,
174 runtime-valid, 93 attained goal cases, 81 applicable comparisons and 93
coverage-only cases, 0 AST no-ops, 1 suspected failure. Search time 2802.10s;
valid cases/s 0.062096 versus repaired/off 0.041636 and original/off 0.031389.
All three fixed fixture/control gates passed 10/10, including benchmark regressions.
The fixed automatic independent audit confirmed 0/10 rediscoveries in each arm;
unknown valid audits were 74 / 80 / 143 and remain unadjudicated.

Additional post-hoc confirmation: generated slogdet case 204407/attempt-0038
(fresh canonical plus negative representation MR) reproduced the known gradient
failure. Immutable replay passed and reproduced saved observations. A minimized
2x2 real-matrix test used exact Python Fraction cofactors as an independent
inverse-transpose derivative. Four negative-bit cases (float32/64, log-only and
sign+log losses) yielded an incorrect transposed gradient, max absolute error 0.5.
Twenty controls passed: direct, materialized, identical-stride non-negative, and
identity transformation derivatives. All logical inputs/forward results agreed.
The original source gradient matches the analytic derivative, follow-up does not.
This is one additional manually corroborated benchmark rediscovery; automatic
metrics are intentionally unchanged. See posthoc-confirmation.json and analytic
review-v3/review.json. Review-v2 had an artifact filename lookup failure; preserved
its nonzero log and reran corrected immutable v3 successfully. No evaluator edit.

Remaining limitations: make_graphed_callables and torch.compile had zero valid
canonical cases in all three arms (parameter/device/capture and configured target
contract failures). Backward saved-tensor representation is still unsupported;
entry representation plus verified backward relation is not an internal-kernel
representation witness. Fixed-off token usage is incomplete for a truncated
response. One seed and one short performance batch do not establish general or
causal gains. No new framework defect is claimed.

Phase implementation, regressions, real-model smokes, MTP selection, formal arms,
and reproduction/corroboration are complete. Final cleanup/accounting verification
is in progress; results: result/state-plan-repair/comparison.json and
performance-comparison.json. Selected config: selected-serving.env.

Final verification complete: jobs 775669, 776358, 776428, 776501 are COMPLETED;
all owned test/service steps exited, PTY sessions closed, and each service had
zero running/waiting requests before shutdown. Disposable test caches removed.
Four service starts / three configuration changes / zero failure restarts.
Allocation total 12023 seconds at two GPUs = 6.6794 allocated GPU-hours, not
utilization. Individual startup/iteration/Slurm accounting is saved separately
in execution-ledger.json; overlapping durations are not added as wall time.
Final git diff --check, serving shell syntax, and byte-for-byte comparison of
89 frozen source/test/benchmark/script files passed. No source changes after
formal freezing. All approved phases are complete; no service remains running.
