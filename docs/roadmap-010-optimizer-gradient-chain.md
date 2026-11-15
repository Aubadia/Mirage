# Roadmap 010 — Gradient representation and attributable optimizer updates

## Authorization and scope

User approved starting the next vertical slice after roadmap009: gradient memory
representation, attributable target backward, real optimizer update and aligned
parameter/slot comparison. Existing autonomous implementation authorization and
one final seed per treatment remain in force. Preserve every pre-existing change
and the user's index; do not commit the root repository. This roadmap is the
first mutation after the new approval.

The first slice verifies target backward provenance before changing the resulting
parameter.grad representation at the native optimizer pre-update boundary. It tests
how the optimizer consumes that representation. It does not claim backward wrote
into the transformed layout; preallocated accumulation buffers need another MR.
No arbitrary gradient-value injection, API-specific bug recipes or oracle threshold
relaxation. Unknown/out-of-domain cases retain execution/coverage-only handling.

## Focused file scope

- runtime/optimizer.py (new): bounded registration/provenance/update evidence and
  harness-owned, exact-value-preserving gradient representation actions.
- runtime/stateful.py; backends/pytorch.py: native parameter/backward linkage,
  pre/post step hooks and a separate optimizer source-effect certificate.
- contracts/capabilities.py; generator/state_plan.py; generator/program.py;
  inspector/stateful.py: new predicates, source-case configuration actions,
  required companion obligations, runtime verification and separate chain metrics.
- rules/catalog.py; runtime/worker.py: conditional optimizer-gradient MR, legal pair
  construction, per-member state obligations and replay evidence.
- artifacts/corpus.py; artifacts/report.py; workflow.py: preserve new evidence in
  the corpus, select/apply source-case representation settings and report the full
  proposal/attainment/MR funnel. Existing sampler formula remains unchanged.
- tests/test_optimizer_runtime.py, tests/test_optimizer_relation.py,
  tests/test_optimizer_inspection.py (new), plus tests/test_capabilities.py,
  tests/test_state_plan.py, tests/test_program.py, tests/test_adaptive.py and
  tests/test_corpus.py as needed for their integration contracts.

Paths above are under src/mirage/ unless prefixed tests/. Supporting research
artifacts/protocols/snapshots go in result/optimizer-gradient-chain/, documentation
in docs/research-010/. Reuse existing categorized Slurm scripts; no dependency or
project-configuration change. No unrelated refactoring.

## Frozen initial design

1. Start with one completed native exact-type SGD update, dense real CPU float64,
   explicit foreach=False/fused=False/differentiable=False/maximize=False,
   finite positive scalar lr, momentum in [0,1), weight_decay=0, dampening=0,
   nesterov=False, no closure, AMP/scheduler/capture/compile or foreign hooks.
   Momentum>0 provides a real cold-start momentum slot. Additional optimizer
   domains/history depths require their own validation, not silent admission.
2. Stable parameter keys/group ordinals replace transient state_dict indices.
   All optimizer parameters must be registered as the same objects. Reject
   duplicate/ambiguous registration and unsupported aliases/effects from proof.
   Do not use parameter-value change as an execution-validity precondition:
   a missing update may itself be a defect. Report nonzero change separately.
3. Prove direct target operand identity -> actual target Node contribution ->
   completed backward -> parameter post-accumulate event -> same gradient
   object/version/values at pre-step. Start from grad=None, and retain explicit
   limitations for extra/ambiguous histories. Source effects require a separate
   restricted certificate; success of a hook alone is not proof.
4. Artifact contract: optimizer_evidence version optimizer-update-1 containing
   registry, bounded steps and limitations; each step contains index,
   pre_checkpoint/post_checkpoint, completed, native-domain evidence,
   pre/post parameter/slot state, gradient provenance and representation action.
   Reserved checkpoint names are optimizer:0/step:0/pre_update and post_update.
   Source probe.mark cannot manufacture these observer checkpoints.
5. New dimensions gradient_representation and optimizer. Scoped predicates use
   parameter_key plus call_index/operand_path and relation target_gradient_update.
   Gradient fields: contiguous, offset_nonzero; optimizer field: completed.
   A global optimizer.target_update_count at final permits planning a real update
   from a valid parent without inventing a parameter key. Its count requires the
   full target-linked chain, not an arbitrary optimizer_step event.
6. A gradient goal carries its same-parameter completed-update companion even
   when already attained by the parent. It is an obligation, not fresh novelty.
   The source execution case explicitly configures a harness-owned action via
   optimizer_gradient_action={step_index,parameter_key,action}; source predicates
   must be reached by the source itself. Actions are canonical/strided/offset
   exact-value copies, with actual metadata/alias checks. Followup receives the
   MR-permitted alternate action and only representation-specific source goals
   are removed from its obligations; target/backward/update obligations remain.
7. New optimizer_gradient_representation MR checks complete valid executions,
   reviewed native optimizer/source effects, target-linked consumed gradients,
   same program/initial state, same pre-step parameters/group options/slots/all
   logical gradients, matching semantic step and sole permitted representation
   differences. Compare post-step parameter and momentum-slot logical values with
   existing dtype_scaled_v1 screening; retain exact equality and update magnitude.
   Never convert missing proof or an unsupported layout into Pass.
8. Separate optimizer-chain-1 coverage/funnel, without rewriting the existing
   state-semantic-2 observer: source-valid linked gradient/update cases and units,
   source vs followup, fresh vs planned, goal conjunctions, MR applicability,
   suspects/unknowns and costs. No borrowed ordinary tensor joint counts.

## Implementation and validation

- Freeze phase009 current code and staged/unstaged state; keep a local protocol
  ledger and durable heartbeat/state. Prepare interfaces before parallel edits.
- Implement runtime proof and action controls, predicates/planning, conditional MR
  routing and corpus/feedback. Cross-review boundaries between components.
- Lightweight static checks first; all Torch/LLM/integration computation in Slurm.
  Reuse one healthy vLLM service/allocation across real-model development checks.
- Positive native controls across unary/binary/matrix target APIs; no-optimizer
  parent bootstrap and source-side gradient-goal attainment; simultaneous layout
  and completed update obligations; strict negative provenance/effect/state/MR
  controls; synthetic output corruption only as oracle sensitivity, never a
  claimed framework defect. Preserve source-owned step/backward sequencing.
- Validate previous numerical-oracle/representation regressions where touched.
- Preregister development targets sin/add and final panel sin/add/mm/linear/tanh/
  bmm before model runs. Formal comparison, if feasible, uses phase009 generator
  with identical new runtime/MR observation versus the new goal/generator path;
  otherwise label it one-method capability validation, never treat absent old
  measurements as zeros. Final only seed0 per treatment, no tuning on final results.
- Save results, limitations, reproducible artifacts and resource accounting; drain
  inference, stop owned jobs/steps/heartbeat and verify cleanup before completion.

## Progress

- Read AGENTS.md, prior architecture-gap/next-step reports and relevant existing
  runtime, inspector, generator, relation and Slurm code.
- Three read-only design reviews established the need for source-side execution
  configuration actions, preserved companion update obligations, and explicit
  provenance beyond a completed-step counter.

- Integration now persists optimizer witnesses/counts separately; versioned chain units participate in corpus admission and feedback while the sampler scoring is unchanged. Reports keep old operand metrics separate and distinguish unmeasured evidence, source/followup and fresh/planned origins.
- Review clarification: a zero direct VJP is valid provenance when real Node, accumulation and unchanged-value evidence all hold. Gradient/update nonzero magnitude is diagnostic only; zeroing/replacement without provenance remains unknown.
- Owned reusable Slurm job779931 submitted after script/static checks, using verified explicit dev partition and400G memory override; first runtime test iteration pending model startup.

- Runtime/predicate/planner/MR/corpus/controller vertical slice implemented. Integration fixes normalize native tuple/JSON-list operand paths, isolate optimizer observer units from old semantic units, retain completed-update companions, and reuse immutable verified source for configuration-only transitions.
- Partitioned Slurm regression currently covers351 distinct tests:350 passed, one exception-rerun assertion under diagnosis. Latest runtime21 passes; both-arm native MR controls13 each pass. Fixed10 gate reference compile/std case hit180s timeout before compiled execution evidence; both-arm600s validation-only retry prepared. Formal one-seed120s/API budgets unchanged.
- Research positioning checked against primary GPU-Fuzz/NablaFuzz sources in docs/research-010/novelty-positioning.md; retain the bounded attributable-update-chain claim, no layout-testing priority or bug-discovery improvement claim.

- Development01 completed before formal experiments: both arms2 source-valid cases; reference0 update chains, candidate1(sin), no planned attempts. Every90s run stopped after one case with remaining43.40–52.72s below whole-round reserve60.37–74.40s. Candidate add source was legal but build indexedwrites/loop fell outside optimizer source certificate; all candidate numerical member audits unsupported. Preserve this negative/partial evidence.
- Within the approved files, next implement a bounded library-known-action candidate from already verified parents (workflow/state_plan/report and corresponding tests), retain exact child obligations, and separate LLM examined goals from selected goals. Add candidate-specific cost admission only for this known configuration branch using its own parent completed-worker measurement with unchanged1.5x+5/floor10 reserve; ordinary model/repair reserve, sampler formula and fresh quota stay intact. Missing or invalid timing retains the ordinary reserve. Actual execution remains bounded by remaining time.
- Generator prompt will explicitly describe existing straight-line construction constraints for optimizer certification (program.py/test_program.py), without widening the source certificate or adding API recipes. Validate metadata rejection/selection accounting and real Slurm configuration-only controller cases; rerun affected regression and exact-engine native/fixed gates before development02/final. Formal seed0, six APIs and120s/arm remain unchanged; this is a bundled system comparison, not isolated component attribution.

### Revision3 validation and final freeze

- The library candidate, selected/examined distinction, own-parent worker reserve,
  and source-fragment prompt changes are implemented. No sampler formula or oracle
  admission/threshold change. Code and tests were frozen at ledger `ecd3ff7`.
- Regression05 passed all172 affected tests in6 modules. Across preserved frozen
  batches,365 distinct related test IDs were observed:360 have ok evidence and5
  CUDA-only tests were skipped (final per-test audit correction). This is not a
  single-snapshot full-suite result. The earlier interrupted/failed runs remain.
- Current exact-engine native controls02 passed13/13 per arm. Fixed fixtures03
  passed10/10 detection and independent checks per arm. Their600s validation-only
  deadline does not change the120s final search budget.
- Development02 completed: reference3 valid/0 chains; candidate4 valid/4 chains,
  2/2 planned layout+update conjunctions attained,2 configuration reuses and4
  optimizer MR Passes. No suspects. Both successful library dispatches would fail
  the old reserve condition at the recorded history/remaining-time values; this
  is not a full-scheduler causal ablation.
- Numerical member records remain separate: reference4 agreement+2 unscheduled
  unknown, candidate4 unsupported+4 unscheduled unknown. Generated momentum was0;
  positive-momentum controls do not imply autonomous nonempty-slot coverage.
- Final launch preregistered at ledger `cc513b5` before the first formal call:
  exactly seed0 per arm, six fixed APIs,120s/API, shared runtime/oracles and common
  scoring of saved evidence. Final search is executing with no implementation
  edits or outcome-dependent reruns. Analysis, resource collection and verified
  cleanup follow completion of all13 driver steps.


### Final results and closure

- Final seed0 completed12 searches+common scoring,13/13 steps exit0;29 report
  integrity gates passed. Reference13 attempts/10 valid/0 optimizer chains;
  candidate18/15/15,17 chain units,10/10 planned gradient+update goals,15 MR Passes.
  All-oracle suspects0. Both arms attained5/6 APIs; linear limitations retained.
- Ordinary semantic units40→17 and old transition units10→5; numerical agreement
  member records8→0. New capability does not establish general bugfinding gains.
  Raw saved-record audit found no nonempty slots among valid candidate members.
- Final test-ID audit corrects earlier runner-total shorthand:365 distinct tests
  observed,360 with an ok result and5 CUDA-only tests skipped. Current affected
  regression05 remains172/172 actual ok. Exact-engine controls13/arm and fixed
  fixtures10/arm passed; original failed/incomplete records remain available.
- Owned requests drained0/0; recorded job-local caches removed. Test PTY exited0,
  job779954 completed, earlier779931 failed startup preserved; all owned steps
  terminal and heartbeat stopped. Resource/report evidence is under
  result/optimizer-gradient-chain/779954/validation-resource-01/.
- Novelty wording narrowed after checking direct PyTorch optimizer test precedents:
  research hypothesis is the integrated exploration/proof framework, not invention
  of the underlying gradient-layout/post-update comparison. Future work is explicit
  in docs/research-010/next-steps.md; no post-final production tuning was performed.

### Authorized follow-up: latest-engine slogdet-only search

User requested one autonomous search with the latest framework targeting only
`torch.linalg.slogdet`. No production/test changes. Before execution, freeze the
current source and protocol under result/slogdet-latest/; preserve the main index.
Use one seed0,180 search seconds (the historical autonomous-hit per-API budget),
CPU eager, backward-required, adaptive/local search, semantic coverage/refinement,
worker120s and existing bounded numerical-audit/replay policy. Fresh corpus; no
historical trigger/source/issue supplied to generation. Same prepared Qwen3.8-27B
MTP3 service as phase010. This is a new single-arm measurement, not a matched
causal comparison to the historical run. No outcome-dependent rerun or tuning.
Reuse existing Slurm service/session scripts; verify scheduler resources and freeze
each execution. Record all outcomes, oracle eligibility, actual elapsed time and
coverage. Reproduce/corroborate suspects separately if any, without relabeling
posthoc findings as autonomous discoveries. Drain/stop service, testing step and
owned allocation; preserve all failures and report final artifacts.

- Follow-up frozen49sourcefiles; submitted owned job780549 with explicit dev/1h/2GPU/8CPU/400G,1800s readiness limit. Default8gpus preflight predicted18:29 vs dev15:24. No source/test changes. Artifacts:result/slogdet-latest/seed0-180s/.

- Job780549 failed before service/model/search:port8000 alreadyinuse,0s. Preserve logs. Replacement780559 explicitly uses port18549 andsame node25a-hgpn048 (also permits cleanup of failedjob localcache),dev/1h/2GPU/8CPU/400G. Search protocol/sourceunchanged.

- Follow-up completed exactly one seed0 latest-engine slogdet search:27 attempts,
  23 valid target-backward cases,16 optimizer-gradient-representation MR Passes,
  7 inapplicable coverage-only cases,0 autonomous suspects. Search160.934s;
  driver163.316s. Numerical members14 unsupported+32 unscheduled unknown,0
  effective audits. All50 saved target operand entries were contiguous with
  negative=False; all23 valid sources lacked the general source certificate
  (19 optimizer.step receiver,2 grad writes,2 list construction blockers).
- No production/test change or rerun. All49 current source hashes match the frozen
  snapshot and the main index hash is unchanged. Job780559 completed0:0; requests
  drained0/0,owned scratch removed,PTY closed,and all780549/780559 steps terminal.
  Startup192s; allocation603s/1206 allocated GPU-seconds,not utilization. Report:
  result/slogdet-latest/seed0-180s/report.md. Proposed generic oracle-admission and
  exploration-budget remedies are analysis only,not implemented in this follow-up.
