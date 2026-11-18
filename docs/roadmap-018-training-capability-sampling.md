# Roadmap 018: capability-based training candidates

## Approved scope

Implement runtime-evidence-based candidate admission and bounded legal training
paths in the existing `training` strategy. Keep the source program contract v3;
version the candidate contract separately. Shared candidate logic must not depend
on PyTorch or benchmark API names. Framework adapters supply observed facts,
while the runtime and MR inspector independently validate attainment and oracle
preconditions. Unknown semantic properties remain unknown, and legal cases with
no applicable MR remain coverage-only.

The user approved implementation and a subsequent run of the existing ten
benchmark APIs, replacing the proposal's four held-out campaign. Use 500 seconds
of framework wall-time per API and the prepared Qwen3.8 environment. Start the
campaign and hand off its job/artifact locations without real-time monitoring.
The unattended job must save evidence and clean up its service and allocation.

## Authorized files

New: this roadmap; `contracts/candidates.py`, `sampler/candidates.py` under
`src/mirage/`; `tests/test_training_candidates.py`;
`benchmarks/training_candidates.py`.

Existing: `src/mirage/backends/{base,pytorch}.py`,
`src/mirage/contracts/exploration.py`,
`src/mirage/generator/{program,http_provider}.py`,
`src/mirage/runtime/{program,mr_actions,worker}.py`,
`src/mirage/workflow.py`, `src/mirage/rules/conditional.py`,
`src/mirage/artifacts/report.py`, and
`tests/test_{exploration,program,http_provider,artifacts}.py`.

## Implementation and validation

- [x] Capture nested input paths, tensor/alias/gradient facts, optimizer binding,
      backward and slicing capabilities with provenance from canonical execution.
- [x] Retain the full candidate registry and rejected/unresolved reasons; sample
      only admitted candidates and return unmet requirements to later sources.
- [x] Offer finite legal complete action paths, with LLM choice constrained to
      those paths; revalidate prerequisites at execution checkpoints.
- [x] Preserve independent MR evidence and coverage-only outcomes. Record
      admitted, sampled, executed, attained, applicable and verdict counts.
- [x] Run meaningful contract/negative-control regressions under Slurm, including
      missing optimizer/shared partner, retained graph endings, nested aliases,
      gradient coordinates, optimizer bindings and a framework-neutral fake backend.
- [x] Freeze code/configuration and submit the ten-API batch job under Slurm.
- [ ] Confirm actual benchmark API execution; job submission/startup alone does
      not complete this step.

## Evidence and limits

Implementation and regression validation complete. Prior slogdet evidence is in
`result/training-slogdet/20261117-500s/`: missing optimizer/alias partner caused
18 execution exceptions and retained-graph endings exhausted eight repairs.
Admission is checkpoint-specific feasibility evidence, never proof of state
attainment, MR applicability, exhaustive coverage, or framework correctness.
Ten APIs cannot establish support for all PyTorch APIs or other frameworks;
additional backends must implement the capability adapter and runtime evidence.

## Progress

- 2026-11-17: approval recorded. Repository HEAD is now `c139acd`; preserve the
  user's existing `AGENTS.md` changes. No campaign has been launched yet.
- The new sampler is framework-neutral and uses a separately versioned
  `training-candidates-1` contract. PyTorch supplies dense-tensor and optimizer
  facts. Aliased/repeated input coordinates are described explicitly; transforms
  requiring preservation of an existing alias topology are conservatively
  rejected by this adapter, with their requirements retained. Shared transforms
  require a concrete equal-valued nontrainable partner. Nested independent input
  transforms preserve input gradient hooks and use typed, unambiguous paths.
- Canonical target-result autograd ancestry and executed backward hooks provide
  target-to-registered-parameter evidence. Optimizer binding alone does not
  establish this relation. Mechanics are rechecked at the injection checkpoint;
  attainment and MR preconditions still require the existing runtime inspector.
- Source generation receives unmet capability requirements from earlier canonical
  evidence. Primitive path prompts omit the full source and expose only the
  bounded legal catalogue. No benchmark API identities occur in the shared core.
- Validation job `789358` waited on `8gpus` with `MaxGRESPerAccount` and was
  cancelled before execution. Explicitly switched validation to the verified
  `dev` partition (8 CPUs, 1 GPU, 96 GiB, 30 minutes).
- Job `789364`: 93 tests, one integration failure exposed enum objects in saved
  capability records. Changed payloads to primitive data and added a
  `weights_only` round-trip regression. Job `789369`: 94 tests passed.
- Final substantive regression job `789374`: **95 tests passed**, including
  saved-campaign controls excluding all 12 optimizer and 6 partner failures and
  all 8 invalid complete histories. Logs and frozen code are retained under
  `result/validation/<job>/`. Tests cover existing MR positive/negative controls,
  provider schemas, controller/replay, fake backend, nested gradients and aliases.
- Candidate identifiers are excluded from state coverage and sampling feedback
  keys, so provenance identifiers cannot inflate state counts or defeat feedback
  across canonical seeds. A final focused controller check follows this change.
- The unattended campaign performs a separate real-service path-schema preflight
  before benchmark API budgets. Each API uses its roster device and a separate
  500-second training batch; source contract remains v3. The runner contains only
  benchmark IDs/API names/devices, without issue triggers or oracle fixture hints.
- Follow-up controller job `789380`: four tests passed after removing candidate
  provenance from coverage/feedback keys. Final runtime job `789384`: **96 tests
  passed**, including typed saved gradient coordinates and repeated-object input
  hooks (each coordinate recorded once, without doubling gradient values).
  The frozen campaign's source/tests match this job's validation snapshot byte
  for byte. AST checks and `git diff --check` passed.
- Campaign prepared under `result/training-candidates/20261117-10api-500s/`.
  `protocol.json` records the roster, exact source hashes and per-API budgets;
  `submission.json` records the complete unattended Slurm command using frozen
  existing setup/serving helpers. `bash -n` and `sbatch --test-only` succeeded.
  Explicit campaign resources: `dev`, 8 CPUs, 2 GPUs, 192 GiB, four hours.
- Submitted campaign job **789395**, running on `25a-hgpn009`. Model startup is
  separate from per-API budgets. Session logs/artifacts are in
  `result/training-candidates/20261117-10api-500s/sessions/789395/`;
  submission stdout/stderr are under the campaign's `logs/`. The user explicitly
  requested an unattended campaign and later result review; leave this owned
  job running after handoff, with automatic service/cache cleanup at its end.
- Startup attempt `789395` exceeded the existing 900-second readiness limit
  during cold model initialization/weight loading. **No API batches started**.
  It exited nonzero after 905 session seconds; `cleanup.json` confirms cache
  removal and Slurm confirms the allocation and serving step are terminal.
- Submitted replacement job **789437** with readiness limit 1800 seconds on the
  warmed `25a-hgpn009` host. Allocation size/time, model/inference settings,
  frozen code and all per-API budgets are identical. Both attempts and the one
  service restart are recorded in `launch-attempts.json`; exact second submission
  and successful preflight are preserved separately. `job-id.txt` points to the
  current job. This retries startup, not benchmark API execution.
- Handoff: job `789437` is RUNNING and model initialization is ongoing. The
  unattended job owns readiness validation, real-service path-schema preflight,
  sequential API execution, evidence persistence and cleanup. API execution and
  benchmark verdicts have not been confirmed at handoff. In accordance with the
  user's request, stop active monitoring here; later result review is deferred
  to the user's next request. The running allocation is intentionally retained.

## Result review and launch correction

- On the user's result-review request, inspected scheduler accounting and actual
  campaign evidence. Job `789437` failed (exit 1) after 23:27 allocation time.
  Service startup succeeded after 1398 seconds, but campaign execution failed
  after 0.395 seconds, with `runs: []`: **zero of ten APIs started**.
- Root cause is the new benchmark launcher's missing creation of the
  `path-schema-preflight/` evidence directory. `HTTPProvider.generate()` saves
  request evidence before HTTP dispatch and raised `FileNotFoundError`.
  This is a launcher defect, not an MR violation or a framework defect.
  Prior 96-test validation did not exercise this real launch path.
- Cleanup is confirmed: `cleanup.json` records cache removal and nonzero exit;
  Slurm confirms all owned job/steps terminal. Both startup attempts are retained.
- Correct within the existing approved file scope: create the evidence directory
  in `benchmarks/training_candidates.py`; add a campaign-preflight regression in
  `tests/test_training_candidates.py`, using a local HTTP fixture and no model.
  Validate under Slurm, freeze a fresh run, and relaunch the originally authorized
  ten APIs at 500 seconds each. Do not modify the original frozen evidence.
- Correction implemented: create the preflight evidence directory before
  constructing its HTTP provider. Added a fresh-campaign local HTTP regression
  asserting evidence exists before dispatch and records HTTP 200 afterward.
  AST/static checks passed; the new regression has **not executed yet**.
- Scheduler conditions changed since the previous run. `dev` now has a one-hour
  limit and its QoS reports minimum GPU count 16 versus maximum per-account 8;
  validation job `791250` was pending `QOSMinGRES` and cancelled before execution.
  Default `8gpus` submission was accepted but is pending `QOSGrpGRES`. Initial
  validation job `791255` was replaced before execution to validate frozen code.
- Fresh immutable run: `result/training-candidates/20261117-10api-500s-r2/`.
  Validation job **791262** uses this frozen code, pending `QOSGrpGRES`.
  Benchmark job **791268** depends on `afterok:791262` with invalid-dependency
  cancellation enabled. It is pending `Dependency`; no new benchmark API has
  started. Resources: validation 1 GPU/8 CPUs/96 GiB/30 minutes; campaign
  2 GPUs/8 CPUs/192 GiB/four hours, both in `8gpus`. The campaign keeps its
  1800-second service readiness bound and 500-second wall budget per API.
  Both submission preflights and shell syntax checks passed; these are not
  executed validation results. Current result-review evidence is saved as
  `result/training-candidates/20261117-10api-500s/result-review.json`.
