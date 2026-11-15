# Roadmap 009: Evidence-guided general state exploration

## Authorization and scope

The user explicitly authorized autonomous overnight analysis, implementation and
experiments on 2026-11-15, including routine decisions without further approval.
This extends the previous eleven-file phase-008 scope for this research task.
Preserve Mirage's tensor-memory-representation x training-state exploration,
runtime state/path evidence, conditional MRs, and independent defect adjudication.
Do not substitute issue-specific generation recipes or weaker oracles for gains.

Expected production scope: generator/state_plan.py, generator/program.py,
workflow.py, sampler/policy.py, backends/pytorch.py, artifacts/report.py and directly
related contracts/runtime evidence only if code review establishes necessity.
Tests may cover those modules. Record concrete file changes and rationales here
before expansion. No unrelated refactoring, dependency changes or fixture tuning.
Documentation lives in docs/research-009/ and this roadmap; scripts, immutable
snapshots, protocols, heartbeat logs and results in result/general-state-research/.
Any new Slurm script goes in sbatch/experiments/ or sbatch/validation/.

Keep the user's existing index and all prior changes intact. Use an isolated
research-protocol Git ledger under result/ for preregistration; do not commit the
user's staged changes. Freeze the current phase-008 engine before production edits.

## Continuity, evaluation and implementation steps

1. Set up durable state/findings/logs and a 20-minute lightweight heartbeat before
   research. No session cron is exposed in this environment: a local heartbeat
   records liveness/state/scheduler status while this active task continues; it
   cannot wake an ended chat. Stop it on completion and preserve its log.
2. Audit the phase-008 funnel by parent/goal/repair/certificate, distinguishing
   proposal supply, scheduling allocation, runtime attainment and MR applicability.
3. Review primary literature to bound the novelty claim and generic mechanisms.
4. Preregister hypotheses and held-out API/protocol choices before new executions.
   Optimize verified useful states and applicable comparisons per wall time, while
   reporting all attempts, unknowns, failures, tokens and exploration diversity.
5. Implement focused general improvements with an ablation-friendly protocol;
   no API-name-specific bug generators, fixed benchmark inputs, or oracle relaxation.
6. Validate negative controls and relevant regressions through Slurm. Reuse one
   healthy vLLM service across development iterations; freeze each execution.
7. Run developmental mechanistic comparisons, reflect, then freeze final choices.
   Final campaigns retain the user's one-seed-per-treatment constraint (seed 0),
   comparable time/model/oracle budgets and a common coverage observer. Treat
   historical known-issue APIs as seen development evidence, not unseen tests.
8. Produce findings plus a readable HTML/Markdown report, disclose negative
   outcomes and limits, then drain inference, remove owned scratch, stop all owned
   jobs/services/heartbeat and verify cleanup before reporting completion.

## Initial observations (existing evidence, not new experiments)

Phase 008: baseline/new source validity 132/94, MR comparisons 69/57, common joint
units 327/167. New goals: 73 requested -> 15 attained -> 3 MR-applicable. Explicit
new-goal prompting changed proposal supply but may distort search allocation.
Invalid source cases include 59 unresolved runtime, 38 unmet state and 18 runtime
contract failures. Missing certificates and repeat ineffective repairs are distinct
losses. The baseline slogdet candidate was independently corroborated as a known
local issue; it is not evidence that the new planner improves discovery.

## Progress

- Read the invoked autoresearch skill, continuity reference, repository rules and
  relevant workflow/certificate code. No session automation tool is available.
- This roadmap is the first mutation after the new authorization.

- Baseline frozen (97 files), user index archived read-only, separate protocol ledger initialized.
- Implementing scoped workflow.py/sampler/policy.py/test_adaptive.py plus generator state_plan.py/program.py and corresponding tests; backend pytorch.py and certificate tests in test_stateful.py. Shared corrected certificate will be used in both experiment arms.
- Seven primary-source literature notes and scoped novelty synthesis recorded.

- Regression01 executed10modules:262passes/1failure in263tests. The failure is the existing HTTP schema assertion expecting only `plans`; v2 intentionally requires `examined_goal_ids` too. Extend test_http_provider.py within protocol integration validation to assert the new exact schema; production HTTP adapter needs no change.

- Static API-resolution inspection found getattr-only lookup cannot resolve public lazy submodules (e.g. torch.utils.checkpoint) before they are imported. Extend src/mirage/runtime/api_metadata.py and test_program.py to import only a missing public submodule from a module receiver, preserve nested import failures, and validate generic aliases/invalid paths. Share this infrastructure fix across both final arms alongside certificate; update experimental overlay/hash gates. This is API-availability preflight before heldout search, not heldout outcome tuning.

- Development01 did not improve throughput (valid22→18,MR17→12). Deterministic protocol findings justify extending generator/http_provider.py and its tests: request-specific exact goal-ID enums and mutually exclusive planned/unresolved shapes; scope unknown operand-binding guidance to operand-selected goals. Root will stop identical known-rejected repairs before worker execution in workflow.py/test_adaptive.py. A fixed-request paired encoding-hint experiment will precede any generic source-quoting prompt change. No numeric threshold relaxation or per-API generation recipe.

- Encoding hint met preregistered fixed-request adoption criterion (3/4recoveries,0/3control regressions; syntax/static only). Latest114protocol/generator/workflow tests pass; fixtures03both10/10. Running development02 from immutable fixture03 engines. Final public resolver is a common overlay, and final campaigns remain one seed each.

## Postformal oracle-eligibility correction (separate from frozen comparison)

Formal32searches/35steps completed, original outcomes frozen (0suspects both). Saved candidate slogdet8 reached nonsymmetric negative-view target andbackward but whole-source certificate rejected run requires_grad_, gating both MR and local numerical audit. Bounded saved-evidence extraction confirmed differing VJPs. This is manual posthoc diagnosis, not autonomous discovery.

Under the existing autonomous general-improvement authorization, extend focused scope to src/mirage/runtime/stateful.py, src/mirage/inspector/oracles.py and tests/test_oracle_routing.py, tests/test_numerical_gradients.py alongside already-scoped backend pytorch.py and tests/test_stateful.py. Add a distinct local-numerical source-effect certificate reusing strict binding/effect checks and admitting only reviewed native Tensor.requires_grad_(literal True) in run. Keep representation-MR program_certificate and all numerical tolerances unchanged. Reject global changes, hooks, unknown receivers and dynamic flags. No unqualified removal of source-effect guards.

Validation: strict positive/negative source controls; real correct multi-API gradient-setup audits; existing routing/numerical/representation tests; exact saved-case replay as separate posthoc evidence, plus independent Fraction/minimal reproduction. No new search seed or rewriting/rescoring of formal artifacts. Report formal pre-correction version and postformal current version separately. If sound admission cannot be established, leave the correction unimplemented and retain the diagnosed gap.

## Completion

Formal32searches/35steps complete,15gatespass: valid110→105,MR70→78,qualifiedjoint17→54allfresh; plannedvalid29→17,memorygoalvalid19→2,explicitmemory×training0,autonomoussuspects0both. This is mixed, not evidenceofimprovedcoreplannedinteractionexploration.

Postformal independentFraction/minimal controls corroborate knownlocal slogdet negativeviewmiss. SeparatelocalVJPcertificate correction validated130tests+unchanged10/10fixedfixtures; exactsavedsource/configreplay nowflags suspected_failure whileMRremainsinapplicable. Formalartifactsunchanged, noadditionalsearchseed. Firstpostformalvalidation129/130 withstaleedgepromptassertion preserved; correctedtest checks exactedgeinsertionandunchangedbaseprompt, productionunchanged.

Current17source/testfiles differfrominitialbaseline; postformalpatch6files. Userindex/stagedpatchand97baselinefilespreserved. Currentcode matchesexecutedpostformal snapshot. Ownedjob778971/testingstepsclosed, modeldrained/stopped, scratchremoved, heartbeatstopped. Startup191sec,restarts0,allocated9922sec×2GPU=5.512GPUhours (notutilization). Reports andartifactlinks savedunderdocs/research-009.
