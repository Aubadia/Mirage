# Roadmap 007: Connect existing discovery capabilities

## Approval and scope

The user approved the proposed first implementation phase on 2026-11-14:
reduction/matrix MR admission, legal compile/capture generation protocols and
execution attribution, legality-aware exception triage, and a fresh autonomous
search using the existing general numerical oracle. Later relational-coverage,
training-history, saved-tensor, distributed and cross-framework phases are not
part of this implementation. No dependencies or unrelated refactors.

Implementation files within this phase:
- src/mirage/backends/pytorch.py
- src/mirage/generator/program.py
- src/mirage/runtime/stateful.py
- src/mirage/runtime/worker.py
- src/mirage/inspector/stateful.py
- src/mirage/workflow.py
- tests/test_capabilities.py, tests/test_relation_catalog.py
- tests/test_program.py, tests/test_stateful.py
- tests/test_adaptive.py, tests/test_oracle_routing.py
- this roadmap; immutable execution artifacts under result/discovery-bottlenecks/

Existing benchmark and Slurm scripts will be reused without edits. If another
production or test file is necessary, identify it and obtain scope approval first.

## Implementation steps

1. Add conservative reduction numerical admission and distinguish multiplication
   from solve/decomposition conditioning; retain finite/domain/numerical guards.
2. Describe reusable callable-factory and captured-callable execution contracts,
   register module parameters and verify devices, and link factory/callable/graph
   events without claiming unsupported internal operand/backward attribution.
3. Classify established harness-contract violations separately from unresolved
   runtime exceptions. Preserve unresolved cases, avoid semantic repair of them,
   and perform bounded no-provider reruns with explicit uncertainty.
4. Add focused positive and negative regressions, including unsupported effects,
   composed compiled graphs, capture lifecycle, and exception classification.
5. Run a frozen full regression and fixed benchmark controls, then a fresh
   autonomous campaign with the current general numerical oracle. Freeze the
   protocol before formal search and report all stages/unknowns independently.

## Validation plan

Read-only inspection and syntax/static checks may run on the login node. All
Torch, compilation, inference and runtime tests run under Slurm using the prepared
project Python and Qwen3.8 BF16 service with the previously selected MTP setting.
Use one reusable two-GPU development allocation with separate one-GPU/four-CPU
service and testing steps for repeated real-model checks; freeze each iteration.
Try the default 8gpus/account resources first and report scheduler constraints.
Use the existing development helper and persistent result directories, preserve
nonzero failures, reserve cleanup time, and confirm owned steps/jobs exit.

Formal evaluation uses unchanged ten-API benchmark inputs only in independent
controls/evaluation, never production generation. Keep historical results intact.
Prefer the existing seed-0, 300-second/API protocol for historical comparison;
report this as a one-seed evaluation, not a general causal improvement. Additional
held-out lifecycle controls validate the generic protocol beyond issue fixtures.
Record service startup, iteration, inference usage and allocation time separately.

## Progress

- Read relevant source, previous experiment artifacts, repair semantics and Slurm
  instructions. Initial worktree clean. Roadmap created before implementation.
- Implementation and runtime validation pending.

### Implementation progress

- Worker emits typed contract-vs-unknown diagnostics with harness origin; only
  confirmed contract failures authorize semantic repair. Unknown exceptions are
  retained, with at most replay_failures same-invocation reruns per API. Reruns
  do not assert initial-state equivalence or framework defects. Suspects use the
  existing immutable replay path and remain suspects after reproduction.
- Reduction/matrix policy and callable lifecycle changes are in parallel review.
- Default two-GPU 8gpus preflight succeeded but estimated next-day availability.
  dev constraints were checked (account allowed, four-hour limit); explicit
  dev preflight succeeded with two GPUs/eight CPUs. No allocation started yet.
- Focused triage Python AST checks passed; runtime tests remain pending.

### Frozen validation started — Slurm 777522

- One dev allocation, two GPUs/eight CPUs/four hours; serving and testing each
  use one GPU/four CPUs/204800 MiB. Prepared Qwen3.8 BF16 + MTP3 reused.
- Initial fixture snapshot iteration-CLB2iv: 10/10 detected, independent controls
  passed, exit 0, 69.258 seconds driver time. Subsequent source corrections need
  a fresh fixture gate before formal search.
- Full regression iteration-niDWtk is running. Its source/tests match the latest
  working implementation byte-for-byte. New tests distinguish callable boundary
  backward from internal target attribution, unknown exceptions from confirmed
  contract failures, and preserve original numerical-audit replay budgets.
- Serving health became available while regression ran. The serving helper's
  final GPU environment check waits for the testing step, so session-ready wall
  time includes that overlap; do not interpret it all as model-loading time.

### Regression diagnosis and repair

- Full suite iteration-niDWtk: 276 tests / 321.654 seconds; 274 passed, one new
  capture error and one pre-existing shell-test failure. No checks skipped.
- New capture error came from our factory construction probe creating a leaf
  autograd edge before capture, attaching AccumulateGrad to the default stream.
  Removed this construction-time observation; returned-callable backward evidence
  remains. No stream override or source workaround was introduced.
- Fresh targeted iteration-oiyOUR: all 68 stateful tests passed in 17.667 seconds,
  including every new callable test and the previously failing CUDA control.
- Existing shell failure: oracle_audit.sbatch expands missing SLURM_SUBMIT_DIR
  before a no-allocation guard, returning 1 instead of expected 2. Script, shell
  test and environment helper match HEAD and the executed snapshot. This is
  outside the approved implementation files; retained without claiming full-suite
  success. No computation starts outside an allocation.
- Driver artifact frozen separately before fresh fixture/smoke/formal execution.
  Initial detection timestamps remain separate from automatic replay cost.

### Real-model lifecycle smoke and next repair

- Fresh fixture iteration-MNfdKy: 10/10 plus independent controls, exit 0,
  54.461 seconds. No historic artifacts modified.
- lifecycle-smoke-01: compile 6/12 runtime-valid, all six MR comparisons Pass;
  capture 0/12, every case statically rejected a top-level Module class. The
  model service remained healthy and is reused without restart.
- Expand the existing Module-class contract to inert top-level declarations,
  keeping public-base, no-side-effect/class-decorator and existing RNG checks.
  This is the same module lifecycle scope, without issue-specific templates.
- Review also found callable seeds need their own protected backward call sets.
  Workflow now derives them from existing saved witnesses, independently of
  target-node obligations, without changing corpus or coverage vocabulary.
  A new integration control checks loss of the original callable backward while
  another invocation supplies a valid backward. Validation pending.

### Final capability gates

- Inert top-level Module declarations and actual named-parameter registration
  guidance implemented; oracle certificates unchanged. Module declaration
  effects/default execution/decorators and duplicate names remain rejected.
- Callable backward mutation obligations now remain separate from internal-node
  requirements at all local/planned/refinement entry points. Parent witnesses
  supply the requirements without expanding corpus or coverage contracts.
- iteration-0MWhdh: 135 affected integration tests passed, 235.623 seconds.
  Includes both loss/preservation controls for a protected compiled invocation.
- lifecycle-smoke-02 passed: compile 7/12 valid, 5 applicable comparisons (Pass);
  capture 7/12 valid including required backward, 0 applicable comparisons.
  Capture's existing source semantic certificate remains narrower than legal
  generated modules. Driver time 173.806 seconds; no service restart.
- Preparing fixtures-03 from the final engine, then unchanged seed-0 ten-API
  formal protocol. Smoke artifacts are development evidence, excluded from formal
  discovery counts. No benchmark inputs were supplied to production generation.

## Approved extension: numerical-oracle stability false positive

The user explicitly approved adding these two files during the frozen campaign:
- src/mirage/inspector/numerical_gradients.py
- tests/test_numerical_gradients.py

Observed counterexample: formal-01/204407/attempt-0040 uses a real FP32 diagonal
[1e-8,2] matrix. Both source/follow-up VJPs match the elementary local derivative,
but finite-difference displacements cross a singularity. Directional slopes vary
strongly; the existing stability tolerance depends on the large candidate VJP,
incorrectly certifying convergence. Immutable replay repeats the same screening
error, which is not framework-defect confirmation.

Plan: derive finite-difference step and one-sided convergence thresholds from
finite-difference observations/roundoff independently of the VJP under test. Keep
uncertain directions unknown; retain sensitivity to stable injected incorrect
gradients including very large ones. Add the generated counterexample, positive
wrong-gradient controls and existing negative-view consistency regressions.

Validation: finish and preserve the original frozen ten-API campaign. Changes to
working files do not affect its copied engine. Then run numerical tests, routing
and runtime regressions, held-out controls (relabel prior held-out as development
if used for tuning), and a separate fresh-engine re-audit of numerical suspects
and the historical negative-view failure through the same Slurm allocation.
Do not rewrite original counts, verdicts or manifests; report corrected audits
separately. No other file scope extension is authorized.

Frozen new held-out numerical validation set (selected before corrected-oracle
execution): torch.erf, torch.atan (unary); torch.hypot, torch.atan2 (multi-input);
torch.var, torch.linalg.vector_norm (reduction); torch.linalg.det, torch.linalg.inv
(matrix). Use explicit small, distinct FP64 operands and original VJP evidence,
without derivative formulas. Require at least one usable numerical judgment per
family and no violations; unknown stays unknown. Do not tune on these outcomes.
The previous eight-API held-out set is not reused as a new generalization claim.

Formal progress (original engine): scatter 42 attempts/18 valid/13 applicable;
slogdet 47 attempts/31 valid/14 applicable/one numerical suspect. The slogdet
suspect is the newly identified stability-screen false positive; original metrics
remain immutable and corrected review will be separate.

Independent review also found the baseline VJP reconstruction check retained a
fixed 1e-12 scale floor. It could certify mismatched tiny observed/reconstructed
gradients. The approved correction now checks finite values and normalized
relative differences, with exact equality handled explicitly. Negative controls
corrupt only recorded VJPs at small cotangents; positive controls reproduce an
injected wrong backward. Both are development controls, not framework defects.

## Remaining bottlenecks and proposed later phases

These are proposals, not additional implementation scope. Increasing API count
alone does not address the successive losses at legal execution, verified state
attainment, oracle applicability and independent corroboration.

| Gap and current evidence | Proposed capability | Acceptance evidence |
| --- | --- | --- |
| `generator/state_plan.py:goal_catalog` mostly requests counts and individual operand properties; inspector coverage omits relative extents and singleton-stride distinctions. | Relational goals for axis roles, operand extents, alias/overlap and device relationships, with explicit feasible candidate sets. | Frozen feasible pairwise/three-way combinations; runtime witnesses and controls that distinguish superficially similar but different relations. |
| `runtime/stateful.py:Probe.mark` primarily records flags and counts; equal counts can conceal different graph or buffer histories. | Graph/buffer ownership and history predicates at intermediate semantic checkpoints. | Retained versus rebuilt graphs, same versus different buffers and reused versus fresh contexts yield distinct verified units. Invalid ordering/ownership remains invalid. |
| Training state is snapshotted, but `contracts/capabilities.py:STATE_DIMENSIONS` lacks dedicated optimizer/scheduler/AMP/restore dimensions. | Explore optimizer initialization, skipped AMP updates and restored scheduler/RNG/DataLoader progress. | Uninterrupted/resumed pairs align by semantic checkpoint; incomplete restores are rejected or detected. |
| `inspector/stateful.py:semantic_coverage` explicitly leaves backward representation unsupported. Callable boundary evidence does not establish fused/internal operands. | Separate evidence for entry, saved, restored and consumed representations, plus scoped compiled/captured correspondence. | Instrumentation-on/off controls; saved-tensor and recomputation identities; unsupported paths stay unknown. |
| Legal Module/capture execution exceeds the existing semantic/effect certificate. Generic finite differences cover bounded eager dense real FP32/FP64 only. | Composable effect contracts and capability-routed independent oracles for additional legal program classes. | Positive/negative applicability controls, aligned observables and numerical conditions per contract; oracle violations remain suspects until corroborated. |
| `sampler/policy.py:FeedbackSampler` rewards novelty/applicability/cost; novelty inherits current coarse coverage units. | Estimate generation validity, state attainment and assessment probability separately, with exploration reserved for difficult capability families. | Matched-budget multiple-seed ablations reporting verified joint coverage, applicable comparisons, unknown causes and corroborated findings. |

Prioritize relational tensor/history evidence and composable applicability before
distributed or cross-framework expansion. Cross-framework work needs equivalent
backend evidence contracts and backend-specific tolerances, not just API-name
translation. No universal correctness or exhaustive theoretical coverage claim
is justified by passing these bounded tests.

### Why supplied fixtures and autonomous discovery differ

The ten supplied fixtures already encode joint trigger conditions, legal action
order, required observations and independent reference values. Free generation
must discover all of those. The mapping below comes from
`benchmarks/pytorch_correctness/cases.py` and current generic contracts; it is a
capability diagnosis, not a claim that an undetected condition was never generated.
These fixture inputs are not supplied to the production generator.

| Mechanism | Missing or weak generic exploration unit |
| --- | --- |
| Expanded scatter index with a wider destination | Cross-operand extents plus valid index bounds and zero stride; changing representation alone preserves destination size. |
| Negative-view slogdet on a nonsymmetric matrix | Joint representation, matrix structure and attributable backward. Existing numerical support and a historical confirmed case already exist; symmetric data can hide transpose errors. |
| LU solve with a singleton batch axis and unusual stride | Semantic axis role and aligned LU/pivots/RHS, rather than any singleton dimension. |
| Recursive reuse of the same autocast instance | Instance identity and overlapping enter/exit history, rather than restoration count alone. |
| Captured gradient accumulation across different microbatches | Ordered capture, gradient reset and repeated backward, with distinct data and buffer/gradient storage identity. |
| Compiled output alias followed by a write | Output alias graph plus an effect sequence observing the base/partner, not just returned values. |
| MKLDNN batch norm with strided affine operands | Operand-role-specific mixed layout; current goals mainly inspect the first two operand witnesses and do not proactively choose MKLDNN. |
| CSR add into strided shared out-values | Sparse destination topology and collateral-write checks on backing storage; return values alone do not cover untouched regions. |
| Compiled FP32 std on tiny nonuniform data | Joint dtype-relative magnitude, nonuniformity, compilation and internal gradient evidence; eager local finite differences do not substitute for compiled VJP checks. |
| CPU-to-CUDA broadcast copy with lazy conjugation | Cross-operand devices, broadcasting and legal writable topology; representation-only mutations preserve shape/device. |

Measure candidate proposal, legal completion, runtime joint-state attainment,
oracle applicability, violation and independent confirmation separately. Validate
new capabilities with frozen structural variants and controls removing one
required condition, rather than optimizing only supplied-fixture rediscovery.

## Final results — approved phase complete

Completed on 2026-11-15 (Asia/Taipei). Production/test files match the final
executed snapshot byte-for-byte. No additional capability phases were implemented.

| Metric | Historical MTP3, seed 0 | Frozen current campaign, seed 0 |
| --- | ---: | ---: |
| Attempts | 454 | 431 |
| Runtime-valid source cases | 174 | 216 |
| MR-applicable comparisons | 81 | 106 |
| MR-unassessed valid sources | 93 | 110 |
| Search seconds | 2802.104 | 2789.640 |
| Known tokens | 2715397 | 2889399 |

Runtime-valid means verified source API/state/path and protected mutation
obligations, not that the follow-up or MR is valid. The current campaign retained
103 unresolved runtime exceptions, performed ten bounded no-provider exception
reruns (all repeated the exception), and automatically replayed its one suspect.
None of those exception reruns establishes a framework defect. Prompts,
capabilities, evidence and oracle routing changed together; these one-seed
results are descriptive, not a causal or statistical superiority claim.

The callable changes produced 19 valid capture source cases (nine with verified
callable backward) and 16 valid compile source cases (nine with verified callable
backward). Applicable comparisons were one and eleven respectively. Boundary
backward evidence does not certify internal saved tensors or fused operands.

Original frozen automatic benchmark confirmation remains 0/10. Its sole overall
suspect was the numerical stability-screen false positive described above, with
an inapplicable MR. Original artifacts and counts are unchanged. Separate final
engine review changed both numerical members to unknown, preserving exact
initial/operand/observation/VJP evidence and repeat results. The historical
negative-view positive remained a source agreement/follow-up verified violation;
this is an existing discovery, not a new autonomous finding.

Validation:
- Final affected regressions: 121/121, no skips (numerical 24, routing 6,
  stateful 70, adaptive 21). Earlier final generator/corpus/flow coverage also
  passed 135 affected integration tests; overlapping suites are not summed.
- Final engine fixtures-04: 10/10 supplied cases, all independently corroborated.
  This is supplied-trigger validation, not autonomous discovery.
- Frozen new held-out gate: 8/8 source agreements across four API families;
  16/16 source/follow-up agreements, zero violations. No checker tuning followed
  these results. This establishes bounded local consistency, not full semantics.
- Full-suite limitation remains the pre-existing unchanged no-allocation shell
  test mismatch. Do not describe the full suite as all-pass.
- First re-audit driver stopped after successful regressions because its strict
  environment check rejected the intentionally changed engine hash. Corrected
  only the artifact driver: original/new environments may differ solely by their
  verified engine hashes; the formal engine inventory may differ only in
  numerical_gradients.py. New/new replay requires exact environment equality.
  A separate postcheck run passed all re-audit, historical and held-out stages.
- Modified Python files passed AST parsing and git diff whitespace checks.

One Slurm development allocation/service was reused, zero restarts. Session-ready
startup was 571 seconds including the documented overlapping environment-check
wait. Formal driver wall time was 2845.610 seconds; search-only time is shown
above. Per-iteration durations and failures are preserved separately. Allocation
elapsed 5277 seconds with two GPUs = 10554 allocated GPU-seconds, not utilization;
overlapping startup/iteration intervals are not added to wall time.

Cleanup verified: zero running/waiting inference requests before shutdown,
owned temporary caches removed, test terminal exited, all owned steps exited,
and job 777522 completed with batch exit 0. The service step was cancelled/killed
by the existing serving cleanup; this is recorded separately from test results.
Unrelated jobs were not touched.

Artifacts:
- [Comparison and interpretation](../result/discovery-bottlenecks/777522/final-review/comparison.md)
- [Final validation, snapshot checks and resource/cleanup evidence](../result/discovery-bottlenecks/777522/final-validation.json)
- [Final corrected re-audit and held-out results](../result/discovery-bottlenecks/777522/oracle-review-02/summary.json)
- [Executed regression results](../result/discovery-bottlenecks/777522/oracle-review-01/regression/summary.json)
