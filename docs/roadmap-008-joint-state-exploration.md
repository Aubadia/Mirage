# Roadmap 008: Joint tensor-state exploration

## Approved scope

Approved on 2026-11-15. Implement the proposed bounded relational tensor slice:
selected-axis extent ordering, selected-axis zero/singleton stride, shared storage,
and conjunctions with existing representation predicates. Connect planning,
generation, runtime evidence, joint coverage, corpus admission and reporting.

The user reduced final evaluation to one seed per treatment. Both the current
corrected baseline and the new engine will use seed 0, ten APIs and 180 seconds
per API, with isolated corpora and the same model, oracle and execution policy.
This is a descriptive paired comparison, not a statistical superiority claim.

Authorized files:
- src/mirage/contracts/capabilities.py
- src/mirage/generator/state_plan.py
- src/mirage/generator/program.py
- src/mirage/inspector/stateful.py
- src/mirage/artifacts/corpus.py
- src/mirage/artifacts/report.py
- tests/test_capabilities.py
- tests/test_state_plan.py
- tests/test_stateful.py
- tests/test_corpus.py
- tests/test_adaptive.py
- this roadmap
- validation scripts, frozen snapshots and results under result/joint-state-exploration/

Existing Slurm scripts are reused. No changes to runtime hooks, backend adapters,
worker, workflow, sampler formula, MRs or numerical oracle. No dependencies or
unrelated cleanup. The previous phase is already staged by the user; preserve
the index and all existing changes.

## Implementation

1. Freeze the approved pre-change source/tests/benchmark baseline and inventories.
2. Extend typed predicates with optional axis and peer selectors, preserving old
   unary semantics and backward-path meanings. Shared storage does not imply
   occupied-coordinate overlap. Unknown metadata cannot establish independence.
3. Build bounded goals from actual same-call operand witnesses. Retain the
   existing list-of-predicates conjunction and maximum three goals per plan.
   Include selectors in conflict/deduplication checks. Explain measurement and
   legal coordinated edits to both planning and code generation.
4. Verify selected relations from existing call-entry metadata plus checkpoint
   path evidence. Joint signatures include both operands and their relation;
   never join separate calls/checkpoints or claim unobserved backward evidence.
5. Introduce state-semantic-2 coverage, explicitly separate unary/joint counts,
   and reject mixed/legacy active corpus coverage rather than silently migrating.
   Historical artifacts and frozen-engine replay remain intact.
6. Execute controls and real-model smoke; freeze the final engine and protocol
   before one baseline and one treatment campaign. Score both raw-record sets
   with the same frozen v2 observer; never compare raw v1/v2 coverage totals.

## Validation

Read-only/static checks may run on the login node. All Torch, runtime, compilation,
LLM inference and test suites run through Slurm. Reuse one healthy model service
and an attached testing step in one two-GPU/eight-CPU allocation, with explicit
one-GPU/four-CPU steps. Verify scheduler constraints and preflight the default
8gpus partition before any explicit resource/partition change. Snapshot every
iteration, preserve failures, and reserve cleanup time within the four-hour limit.

Controls cover schema validation, old unary semantics, true/false/unknown relations,
wrong axes and peers, cross-call/checkpoint mismatches, repeated operand objects,
storage identity versus overlap, required backward paths, conjunctive attainment,
coverage normalization/deduplication, corpus versions and feedback integration.
Run affected regressions and unchanged fixed ten-fixture independent controls.
Freeze structural held-out controls before their execution; do not tune on them.

Final comparison: seed 0 for each treatment, 180 seconds/API, same prepared
Qwen3.8-27B BF16/MTP3 service and existing numerical oracle. Record joint-condition
attainment, common-scored unary/joint units and rates, MR applicability, unknown
causes, suspected failures, independent evidence and generation/execution costs.
No issue-specific generation templates or fixture inputs enter generation.
Unknown cases remain unknown and replay alone never confirms a defect.

Record service startup, per-iteration elapsed time, restart count and allocation
resource time separately. Confirm no outstanding inference, remove owned scratch,
stop the service, release the allocation and verify all owned jobs/steps exit.

## Progress

- Read the approved plan, AGENTS.md, current contracts/planner/inspector/corpus/
  reports and Slurm instructions. Current staged changes belong to the prior phase.
- Roadmap created before implementation. Baseline freeze and implementation pending.

### Implementation and execution preparation

- Baseline frozen before edits at result/joint-state-exploration/baseline/code;
  all 95 source/test/script/benchmark files have recorded checksums. Index untouched.
- Predicates now support typed axis/peer selectors and independent primary/peer
  path obligations. The observer uses same-call/checkpoint metadata only.
- Automatic pair observation is bounded to four operands in typed-path order,
  first/last axes (at most two per operand), with omitted/ambiguous/unknown counts.
  Unary signatures also include bounded axis stride classes, so single-operand
  goals have observable feedback. Explicit predicates can inspect other axes;
  automatic candidate/coverage bounds do not claim those axes are explored.
- Planner catalog is bounded to 29 goals, with two selected pair candidates and
  four axis slots; parent proposal_index/attempt rotates within the bounded
  vocabulary. Each plan retains at most three conjunctive predicates.
- state-semantic-2 and stateful-corpus-2 are explicit. Old/mixed corpus resume is
  rejected; original artifacts/frozen replay remain untouched. Reporting separates
  unary/joint units and requested relational-goal attainment from correctness.
- Approved production files, tests and driver passed AST/static checks. Runtime
  validation has not yet completed.
- Default 8gpus preflight estimated 05:51 on 2026-11-15. Verified dev account/time
  constraints and explicit two-GPU/eight-CPU dev preflight succeeded. Job 778244
  started 00:50:54 Asia/Taipei on 25a-hgpn031; model service is warming up. Existing
  prepared Qwen3.8-27B BF16, MTP3, context 16384 and max concurrency 2 are retained.

### Validation in progress

- Model startup completed in 205 seconds, with zero restarts. Testing began after
  readiness, in a separate one-GPU/four-CPU Slurm step in allocation 778244.
- Regression iteration TAVK7W passed all 184 tests in the six affected/adjacent
  groups: capabilities 14, state_plan 20, program 32, stateful 81, corpus 14,
  adaptive 23. Driver elapsed time was 301.06 seconds; no failures or skips were
  reported. Frozen evidence: result/joint-state-exploration/778244/regression-01/.
- Subsequent controls and final evaluation use that exact regression snapshot.
  The fixed ten-fixture gates are currently running for both engines.
- Both fixed-fixture gates passed 10/10 (128.12 seconds combined), including
  compilation and CUDA. These supplied triggers are validation, not autonomous
  discoveries. All four pre-frozen structural runtime controls also passed
  (expanded distinct operands, shared-but-disjoint slices, separate clone,
  repeated operand). Missing storage remained Unknown; invalid records gained
  no signatures and repeated-operand gradient attribution was not overclaimed.
- Real-model smoke is running on generic torch.mul and torch.cos, separate from
  the final single-seed campaigns. Production source is unchanged since regression.
- Smoke-01 completed in 147.91 seconds: mul 10 attempts/7 valid and cos 11/10.
  Both generated legal programs, but zero executed attempts selected a new
  relation/axis goal. Review found only one new-family plan among 18 accepted
  plans across six planning calls; it was not selected. Both catalogs contained
  the new candidates, so this is a proposal/selection bottleneck, not evidence
  of unreachable tensor relations.
- Within the approved planner/test scope, refine the planning prompt to identify
  relation/axis goal IDs and request one such plan when a concrete legal sequence
  is available. Keep the three-plan bound, diverse alternatives, single-goal
  option, unsupported-goal escape and runtime proof requirements. No sampler
  formula or validation schema change. This is development-smoke feedback;
  formal single-seed campaigns have not started. Revalidate the planner, new
  fixed-fixture gate and real-model smoke before freezing the final treatment.
- Prompt refinement is frozen in iteration-urBccI/code. Its only differences
  from the 184-test snapshot are state_plan.py and test_state_plan.py; all 21
  planner tests passed in 51.41 test seconds (55 seconds for the iteration).
  The exact new engine again passed the 10/10 fixture gate in fixtures-02
  (57.94 seconds). The unchanged baseline retains its fixtures-01 gate.
  Smoke-02 is in progress. Baseline checksum verification still matches 95 files.
- Smoke-02 passed in 152.37 seconds: mul 11 attempts/8 valid, 4 relation-goal
  attempts/3 attained; cos 11/9 valid, 3 selected-axis attempts/3 attained.
  Runtime attainment is separate from MR applicability/correctness. No production
  changes follow this smoke. Freeze iteration-urBccI/code as the final new engine;
  start paired-01 with baseline fixtures-01 and new fixtures-02, one seed (0)
  and 180 seconds per API per arm. Formal outcomes will not guide implementation.

### Formal-run review preparation

- Formal protocol independently checked: exactly 20 searches, seed 0 only,
  180 seconds/API, five baseline-first and five new-first pairs; identical
  benchmark/model/provider settings, isolated corpora, approved engine differences
  and matching fixture inventories. Production code remains frozen.
- Early scatter_add evidence exposes further bottlenecks: accepted action text
  may describe an unresolved/blocked plan; generated edits can address src when
  the predicate selects index; index bounds/backward shape legality can fail;
  attained relations can still lack an MR certificate. These findings are for
  the next plan, not modifications to this formal treatment.
- Baseline slogdet attempt-0008 was autonomously marked suspected_failure by the
  existing numerical oracle and reproduced automatically. This is a baseline
  candidate, not a discovery attributable to the new relational planner.
- Source review found that the existing evaluator's target_vjp branch reuses
  stored finite-difference violations plus same-call mechanism attribution.
  Its independent_label/confirmed_* names do not by themselves establish a
  second independent oracle. Preserve those raw outputs, explain their scope in
  the comparison report, and add a separate analytical review after the campaign.
- Prepared result/joint-state-exploration/slogdet_independent_review.py (SHA256
  42b26dc6df4f4c79fa707dcc926c9f03af9fe63b125d485f016428a336671313),
  with Slurm guards, original/replay hash protection, actual Node cotangents,
  exact Fraction inverse-transpose and separately implemented signed storage
  pullback, plus fresh recorded-layout/resolve_neg/contiguous controls. Source
  and cross-review passed; runtime review is still pending. This post-hoc work
  supplies no generation input and does not add search seeds or alter arm verdicts.

### Completed results and limitations

- paired-01 completed all 20 single-seed searches, two existing-evaluator reviews
  and common-v2 scoring: all 23 steps exited zero, driver elapsed 3258.04 seconds.
  comparison-01 gates all passed, no JSON or valid source evidence was missing,
  all 20 native/common source-valid populations matched, and original records
  remained unchanged. Both arms used seed 0 only; no second formal campaign.
- Baseline versus new: attempts 236 / 227; verified source cases 132 / 94;
  applicable MR comparisons 69 / 57; common-v2 source unary units 174 / 116;
  common-v2 source joint units 327 / 167; suspected failures 1 / 0. New capability
  is functional, but this one-seed experiment shows lower search yield, not an
  improvement. Native v1/v2 totals were never directly compared.
- New tensor-relation goals: 41 requested, 9 attained, 2 also MR-applicable.
  New axis-or-relation goals: 73 requested, 15 attained, 3 also MR-applicable.
  The 58 unsuccessful new-goal attempts consist of 32 state-attainment failures,
  25 unresolved runtime failures and one static-contract failure. These attempt
  denominators include repair retries. Coverage-only evidence remains useful,
  but reaching a new state is not equivalent to enabling a correctness check.
- Of 133 invalid new-arm source cases, the largest losses were 59 unresolved
  runtime cases, 38 unmet states and 18 runtime-contract failures. Among 37 valid
  but MR-inapplicable cases, overlapping missing obligations included deterministic
  (20), numerical_domain (13) and semantic_execution_contract (9). No conclusions
  about actual nondeterminism follow merely from a missing source certificate.
- Existing-evaluator counts were 1 known issue reproduced/detected in baseline
  and 0 in new; the report explicitly discloses same-oracle reuse in target_vjp
  review. Independent-slogdet-01 then passed (unknown 0, 3.44 seconds): original
  and both saved replay sources agree with exact analytical gradients, while all
  followups violate them. Fresh recorded-layout controls reproduce the VJP;
  resolve_neg and contiguous controls agree with the analytical reference and
  preserve forward outputs. All original hashes and replay observations matched.
- The standalone minimal_repro.py was subsequently executed in iteration-KENQdU
  (exit 0, four seconds), independently confirming ordinary/resolved gradients
  and the negative-view off-diagonal mismatch. The earlier review.json correctly
  retains executed=false as its creation-time snapshot; posthoc-evidence.json
  records the later execution. This is one autonomously generated baseline
  candidate corroborating a known local PyTorch 2.15.0+cu132 CPU mechanism, not
  three bugs or a new upstream defect claim. No automatic minimization claim.
- Baseline token usage is complete (1,601,183); new recorded usage (1,651,996)
  is incomplete due to four provider length failures. New per-token efficiency
  remains unknown. The bounded observer retains operand/axis omission counts;
  no exhaustive state-space or zero-false-positive claims are made.
- Final static scope check: all 95 baseline inventory files intact; only the
  eleven approved production/test files differ; current files match the validated
  final snapshot. Existing staged changes/index were preserved.

### Cleanup and cost accounting

- All ten validation/experiment/review iterations exited zero. Per-iteration
  command, exit status and elapsed time are recorded in cleanup-and-costs.json.
  Regression 184 tests and the later 21-test planner rerun passed; both final
  engine fixture gates were 10/10, four structural controls passed, and smoke-02
  established autonomous new-goal attainment before formal evaluation.
- One model service was reused: startup 205 seconds, zero restarts. Formal
  search time baseline/new was 1614.79 / 1569.30 seconds, with metadata time
  21.68 / 21.93 seconds. Existing-evaluator post-hoc review/replay took
  8.60 / 4.55 seconds; common scoring took 6.25 seconds. Embedded numerical
  audits and automatic replays are already included in search wall time.
- Inference running/waiting metrics were both zero before shutdown. Owned
  service and test scratch were removed, the testing terminal closed, and
  squeue showed no owned jobs or steps. Job/batch/testing step completed 0:0;
  service step 778244.0 ended CANCELLED 0:9 during the requested shutdown.
- Slurm accounting: 2026-11-15 00:50:53 to 02:15:43 Asia/Taipei, 5090 allocated
  wall seconds, two GPUs = 10180 allocated GPU-seconds (2.8278 GPU-hours).
  This is allocation time, not measured GPU utilization; overlapping durations
  are not added as wall time.

### Recommended next work (not implemented in this scope)

1. Represent planned/unresolved/blocked outcomes structurally, require actionable
   transitions, and bind each operand selector to the actual source expression.
   Stop repeated repairs that do not change the selected relation; resample.
2. Add validated API-role/shape/index/backward legality constraints to canonical
   generation and coordinated edits, preserving unknown cases as unknown.
3. Explain certificate rejection per operation and expand evidence-backed pure
   initialization support. Measure attained-and-oracle-applicable cases as a
   primary yield metric while retaining coverage-only exploration. Do not loosen
   oracle soundness or label unsupported domains as Pass to raise counts.

User-facing results: result/joint-state-exploration/778244/completion.md;
full common-score comparison: comparison-01/comparison.md; independent evidence:
independent-slogdet-01/review.json and posthoc-evidence.json. The new corpus/coverage
versions intentionally reject old corpus resume; use a fresh/revalidated corpus.
