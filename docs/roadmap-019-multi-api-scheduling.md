# Roadmap 019: shared multi-API scheduling

## Approved scope

On 2026-11-17 the user approved the proposed independent per-API sessions and
shared, dynamically allocated generation/execution capacity for both training
and stateful (including alternating generation and durable traversal drain).
Keep API feedback, seeds, runtime evidence, MR semantics and budget types local.
Use bounded fair, work-conserving scheduling with event-driven slot borrowing.
Do not alter existing queued jobs or original frozen experiment evidence.
Do not add dependencies, change prompts/oracles, or claim speedup without data.

## Authorized files

New: this roadmap; src/mirage/contracts/scheduling.py;
src/mirage/runtime/{scheduler,multi_api}.py;
tests/test_multi_api_scheduling.py;
sbatch/validation/multi_api_scheduling.sbatch.
Existing: src/mirage/{workflow,cli}.py; contracts/programs.py;
runtime/{pipeline,process}.py; generator/provider.py; artifacts/report.py;
benchmarks/{training_candidates,e2e_latency}.py;
tests/test_{exploration,pipeline,cli,artifacts}.py; README.md; sbatch/README.md.
All src paths above are under src/mirage. Preserve pre-existing working changes.

## Implementation steps

- [x] Typed session/resource options and framework-neutral fair capacity broker.
- [x] Independent sessions, stable seeds, bounded active API admission and reports.
- [x] Route training and stateful execution and generation through shared limits;
      controller waits do not occupy subprocess slots; no nested-pool deadlocks.
- [x] Preserve training wall deadlines and stateful model-budget/drain semantics,
      alternating feedback causality, local failures and shared-service uncertainty.
- [x] Integrate CLI and the existing training/stateful campaign launchers.
- [x] Add adversarial scheduler and strategy regression coverage and documentation.
- [x] Freeze and execute Slurm validation; review actual results and cleanup.
- [x] Perform bounded fixed-work/live checks if scheduler permits; report any
      unavailable measurements without asserting speedup.

## Validation plan

Run lightweight AST, diff and shell checks directly. All execution tests, model
requests and GPU work run in Slurm using existing prepared environments. Validate
fairness/borrowing, global and per-session caps, deadlines, independent feedback,
local versus shared cancellation, CPU/CUDA serialization, alternating mode,
drain/resume, MR evidence and reports. Compare fixed semantic inputs rather than
requiring identical adaptive exploration trajectories. Measure request/worker
queue times and campaign makespan separately; allocated GPU time is not utilization.
Default allocation uses ACD115198/8gpus, 8 CPUs and the appropriate 1 test GPU or
2 serving/testing GPUs. Preflight is not an executed test. Keep each iteration's
code immutable and preserve failures; clean up owned work/services at completion.

## Generality and limits

The broker knows session identifiers, dependencies and resource requests, not
benchmark API names, PyTorch tensors, samplers or MRs. Current adapters support
training/stateful in the existing PyTorch runtime. Other frameworks need an
execution/isolation/resource adapter; unknown GPU requirements serialize
conservatively. Initial work shares one service and one test GPU; automatic
Slurm resizing, multi-replica serving and training resume are outside this scope.

## Progress

- Approval recorded; inspected both execution paths and existing pipeline resume.
  No implementation tests have run yet. Existing uncommitted capability-sampling
  work and pending jobs 791262/791268 are preserved.

- Continuation: recovered the approved scope and reviewed partial contracts,
  broker, subprocess integration and CLI. No executed scheduler validation exists
  yet. Existing jobs 791262/791268 remain pending and unchanged. Next work is
  adversarial review, campaign integration, focused regression coverage and a
  frozen Slurm validation attempt; preserve capability-sampling changes.

- First implementation includes a shared round-robin broker, independent APIJob
  sessions, training and stateful adapters, CLI controls and both campaign runners.
  Unknown additional device use/distributed tasks require a resource adapter; the
  initial CUDA policy is one subprocess on the allocated test GPU at a time.
- AST parsing of 84 Python files, shell syntax and git diff --check passed before
  submission. Frozen iteration 001 is under result/multi-api-scheduling/
  20261117-iteration-001/. Validation job **792424** is pending QOSGrpGRES;
  its preflight estimated 2026-11-18 09:55 (scheduler local time). No execution
  test or performance claim has been made. Subsequent edits need a fresh snapshot.

- Final implementation preserves distinct job-level caps (the shared scheduler
  does not silently raise them), native replay cancellation scopes and independent
  stateful model budgets. Alternating capacity waits occur before acquiring the
  shared generation lease. Root report CLI supports the multi-API summary.
- Newly prepared benchmark protocols are version 2 and explicitly select the
  shared coordinator even for a one-active-API baseline. Previously frozen v1
  protocols keep their original execution path. Both strategy adapters preserve
  roster devices and explicit search treatments.
- Added 24 adversarial/shared-session/strategy test methods plus focused CLI,
  report and public entry regressions. These include CPU fixed-work semantic
  comparison, two CUDA sessions, alternating feedback isolation and stateful
  no-generation resume. They have not executed yet.
- Superseded validation jobs 792424, 792465 and 792508 were cancelled while
  still PENDING; no test run was replaced or hidden. Their frozen snapshots
  and submission evidence are retained. Final snapshot is
  result/multi-api-scheduling/20261117-iteration-004/.
- Final validation job **792562** submitted to 8gpus: 8 CPUs, 1 GPU,
  96 GiB, 30 minutes. Matching shell syntax and sbatch test-only preflight
  passed. Actual regression execution and live-model/performance validation
  remain pending scheduler resources; no speedup or MR Pass is claimed.


## Additional approved live measurement (2026-11-17)

The user requested one additional 8gpus job measuring the current multi-API
parallel implementation over all ten benchmark APIs. The proposed dev migration,
cancellation and segmented campaign are abandoned. Preserve jobs 791262, 791268
and 792562 and all existing frozen artifacts. No application, benchmark or test
source changes are authorized or needed for this measurement.

Steps: freeze the current source/tests/launch helpers and existing E2E driver in
result/multi-api-scheduling/20261117-live-e2e-001/; prepare the existing stateful
E2E protocol with ten active APIs, eight shared workers, two shared generation
slots and the existing default 180 provider-request seconds per API; create only
run-specific batch/measurement artifacts; preflight and submit one 8gpus job
using 16 CPUs, two GPUs, 192 GiB and four hours. Use one vLLM service, separate
serving/testing srun steps, and afterok:792562 as the regression gate. Verify the
measurement source matches that validation snapshot. Record campaign and
per-session timing, admission/resource queues, execution slots, pipeline/search
metrics and sampled vLLM latency/token/cache metrics. Service startup, cleanup
and allocated resource time are separate from campaign wall time; this single
arm does not establish a speedup.

Validation: lightweight syntax/hash/protocol checks and bash -n; matching
sbatch --test-only before submission. Actual regressions and measurements remain
unexecuted until their Slurm jobs run. Each API must retain its own verdicts and
evidence. Preserve incomplete/failed sessions. The unattended job must terminate
its owned test/service steps, remove disposable cache and retain nonzero errors,
startup/test/cleanup timing and final service-idleness evidence. No interactive
service is started or left running by the submitting agent.

Progress: scope recorded; preparing frozen artifacts and scheduler preflight.

- Additional measurement job **792609** submitted successfully to **8gpus**;
  dependency **afterok:792562**, with kill-on-invalid-dependency enabled.
  Original jobs 791262/791268/792562 were not updated or cancelled.
- Frozen run: `result/multi-api-scheduling/20261117-live-e2e-001/`.
  All 90 source/test/benchmark Python files match validation iteration 004.
  AST parsing of 148 frozen/wrapper Python files, batch bash -n, protocol/hash
  checks and matching sbatch --test-only passed. These are preparation checks,
  not executed regressions, measurements or an MR verdict.
- One stateful parallel arm: ten active APIs, eight global execution workers,
  two global generation slots, 180 cumulative provider-request seconds per API,
  original devices and independent representation drain. Allocation: one node,
  16 CPUs (four serving/twelve testing), two GPUs (one per concurrent step),
  192 GiB and four hours. The unattended batch preserves failures and stops its
  owned processes/service on exit; no interactive allocation was started.
- `measurement/metrics-summary.json` will retain campaign makespan, per-session
  E2E (controller entry to durable report), admission/submission-to-completion
  timing, request mean/p50/p95 split by success/failure, pipeline/search metrics,
  scheduler events and final idle evidence. Shared-process session E2E is not
  labeled as the legacy separate-process launch-to-exit measure. Raw vLLM
  before/after metrics and two-second samples preserve service-level latency,
  TTFT, token and cache observations across overlapping APIs. Job-specific
  `sessions/792609/cleanup.json` records startup/test/cleanup durations, resource
  allocation estimates and exit status. Actual execution/results remain pending.


## Additional stateful E2E submission (2026-11-17)

The user explicitly requested one new ten-benchmark-API stateful E2E measurement
job. Preserve all existing jobs and snapshots. Reuse the current benchmark and
run-only metrics wrapper with a new immutable source snapshot and independent
output root under result/stateful-e2e/20261117-001/. No production source, test,
configuration or dependency changes. Record the selected active-API count in the
new protocol before submission; the existing 792609 arm already uses ten active
APIs, so clarify whether this new run is a one-active-API comparison or another
parallel repetition. Keep eight workers, two generation slots, 180 cumulative
provider-request seconds per API, existing stateful search/drain treatment and
Qwen3.8-27B/MTP3 serving configuration. Use 8gpus, 16 CPUs, two GPUs, 192 GiB,
four hours and a fresh service port; retain the existing afterok:792562 gate.

Validation: frozen source comparison, Python AST/protocol/hash checks, bash -n
and matching sbatch --test-only. Submit exactly one new job and record its ID,
queue state and output paths. No execution or MR Pass claim follows from
submission. The unattended job owns cleanup and preserves failed/incomplete
sessions and separate startup/test/cleanup/resource accounting. Application
execution remains exclusively inside Slurm srun steps.

Progress: requested scope recorded; preparing mode-independent frozen artifacts.


## Confirmed training parallel E2E submission (2026-11-17)

The user clarified that the additional job must measure the CURRENT training
parallel E2E over all ten benchmark APIs. This replaces the tentative additional
stateful submission above. No extra stateful job was submitted; its preparation
artifacts remain preserved under result/stateful-e2e/20261117-001/.

Approved scope: record this clarification first, prepare a fresh immutable run
under result/training-e2e/20261117-parallel-001/ using the existing training
campaign version 2, add only run-specific metrics and batch artifacts, preflight
and submit exactly one new 8gpus job. Preserve all existing jobs and code.
Use ten active API sessions, eight shared execution workers, two generation
slots, 500 seconds of framework wall-time per admitted API (shared slot waits
consume this budget), the existing training candidates/MR configuration, seed
and device roster, and prepared Qwen3.8-27B/MTP3. Allocation: one node, 16 CPUs,
two GPUs, 192 GiB and four hours, separate serving/testing srun steps and one
service. Retain afterok:792562 as the same-source regression gate.

Record campaign/per-API controller E2E and admission wait separately, scheduler
execution/generation slot timing, successful/failed request latency mean/p50/p95,
raw service latency/TTFT/token/cache metrics, training canonical/case statuses,
verified evidence and existing MR verdict summaries. Generation failures and
coverage-only cases must remain explicit; neither a completed controller nor
an exit status zero implies an MR Pass or all-API support. Startup/test/cleanup
and allocated resource estimates remain separate from campaign wall time.

Validate AST, shell syntax, ten-target protocol and immutable hashes; compare
source with the queued regression snapshot; run matching sbatch --test-only.
Actual computation remains in Slurm. Preserve nonzero errors and evidence, stop
owned services/steps and remove disposable cache on job exit. Report the new
job ID and queue state without waiting for full execution.

Progress: clarified approval recorded; preparing training-specific run artifacts.

- Confirmed training-parallel job **792664** submitted successfully to **8gpus**,
  with **afterok:792562** and kill-on-invalid-dependency enabled. Exactly one
  new training job was submitted; existing stateful job 792609 and training
  job 791268 remain untouched. No additional stateful job was submitted.
- Frozen run: `result/training-e2e/20261117-parallel-001/`.
  All 90 Python source/test/benchmark files match validation iteration 004;
  AST checks passed for 91 files, categorized batch bash -n and protocol/hash
  checks passed, and exact-resource sbatch --test-only passed. The JSONL sample
  delimiter was checked through its AST. These are preparation checks only.
- Protocol: current training campaign v2, ten active API sessions, eight shared
  execution workers, two generation slots, 500 framework wall seconds per API,
  including shared-capacity waits, original device roster and candidate/MR
  settings. Prepared Qwen3.8-27B/MTP3 service on a fresh port. Allocation: 16
  CPUs, two GPUs, 192 GiB and four hours; serving/testing use separate steps.
- Results will be written to `metrics-summary.json`, `parallel/multi-api.json`,
  per-API reports and service raw/sample metrics under the frozen run. Job logs,
  service startup evidence and cleanup/resource timing live in `logs/` and
  `sessions/792664/`. Actual regression/measurement execution is still pending;
  no E2E result, speedup, universal support or MR verdict is claimed.


## Approved repair and resubmission (2026-11-17)

The user requested fixing the failed validation and resubmitting the measurement
jobs. Continue within this roadmap's already approved implementation/test/launcher
scope; preserve existing uncommitted work and all frozen runs. First reproduce
StrategyRuntimeTests.test_training_fixed_semantics_match_serial_and_parallel
alone in Slurm so unittest emits its complete failure before any later timeout.
Record native failure evidence. Correct only the demonstrated issue in the
previously authorized scheduler/coordinator/process/pipeline/workflow or strategy
tests, with focused regression coverage; do not weaken MR evidence or merely
increase a timeout to disguise a failure. The existing validation launcher and
strategy test file are authorized. If another file outside the existing approved
scope is necessary, obtain file-scope approval before editing it.

Use four-hour validation submissions rather than the earlier explicit thirty-
minute override, with early/durable failure output and bounded cleanup. Run
focused and complete regressions in Slurm using frozen per-iteration snapshots.
No model is needed for deterministic regression diagnosis. Before any new live
campaigns, verify the regression result and cleanup. Prepare fresh stateful and
training parallel measurement runs with their original respective 180 model-
request-second and 500 framework-wall-second API budgets, ten active APIs, eight
workers, two generation slots and prepared Qwen3.8-27B/MTP3; preserve failures
and submit new job IDs. Never requeue or overwrite cancelled run evidence.

Progress: inspected the failure log, test fixtures, training transitions and
scheduler/process paths. The earlier unknown canonical results include explicit
"Target API/backward execution not verified" diagnostics; they are not arbitrary
exceptions or proof of framework defects. First collect the actual equality-test
assertion and preserve generation/runtime evidence. No repair test has run yet.

- Focused original-code reproduction **792681** failed in 59.121 seconds,
  exit 1, with the complete recursive comparison traceback. The mismatching
  values are integers in the Python RNG state, which rng_state includes while
  the fixture previously seeded only torch. Reproduction cleanup is verified.
- Repair is limited to `tests/test_multi_api_scheduling.py` and
  `sbatch/validation/multi_api_scheduling.sbatch`: seed Python RNG in the
  deterministic fixture, compare actual saved initial tensors/optimizer/RNG in
  addition to all final observations, require nonempty result records and label
  comparison paths; enable unittest --failfast for immediate complete failure
  output. No production scheduler, oracle, runtime or generation behavior changed.
- Test/embedded-source AST, shell syntax and git diff --check passed. Frozen
  repaired snapshot: `result/multi-api-repair/20261117-fixed-001/`.
  Focused verification **792693** submitted; complete original regression suite
  **792695** depends on it. Both have explicit four-hour limits and matching
  successful scheduler preflight. Tests have not been claimed passed yet.

- Focused repaired verification **792693 PASSED**: one test in 64.971 seconds,
  exit 0; actual initial-state and full final-observation equality succeeded;
  cache_removed=true and all focused/reproduction steps are terminal.
- Complete original regression suite **792695** is running with its four-hour
  bound and fail-fast diagnostics; do not claim it passed until results are read.
- Fresh replacement jobs submitted: stateful parallel **792705** under
  `result/multi-api-scheduling/20261117-live-e2e-002/`; training parallel
  **792706** under `result/training-e2e/20261117-parallel-002/`. Both are gated
  by afterok:792695 and retain their original model/framework budgets, limits,
  prepared service settings and self-cleanup. Exact matching preflight passed;
  all 90 Python source/test/benchmark files match the repaired validation snapshot.
- Submission handoff retains the earlier instruction to start campaigns and
  provide job/artifact locations without real-time campaign monitoring. Full
  regression and live metrics remain pending; no MR or speedup conclusion.
- Final source-scope check confirmed that this repair changes only the approved
  strategy test file and validation launcher, plus this roadmap and run artifacts.
  No production source, MR semantics or numerical tolerances were changed.


## Approved stateful persistence repair (2026-11-17)

The user requested fixing the failed stateful measurement. Continue within this
roadmap's authorized files: src/mirage/workflow.py and tests/test_artifacts.py
for transactional report publication, plus this roadmap and new run artifacts.
Preserve every earlier result and do not modify training generation, MR semantics,
observables, tolerances, or sampling. Existing runtime/pipeline/coordinator and
benchmark/test files remain available within the previously approved scope only
if reproduction demonstrates another necessary issue.

Evidence: complete repaired regression 792695 passed all 94 tests in 342.245s;
stateful 792705 ran and exited 1 without a final measurement summary. Its large
scatter_add summary contains a hole of zero bytes beginning at offset 8388608,
and another pipeline pending file and campaign stderr are empty. /home is now
100% full with about 655 MiB free, while /work has about 571 GiB available. The
old failure did not preserve explicit ENOSPC, so storage exhaustion is strongly
suggested rather than proven. Current write_json publishes by truncating and
writing the destination, so any partial I/O can damage the previous valid report.

Implementation: serialize first, write a unique temporary file in the same
directory, close successfully, then replace the destination; clean temporary
files on exceptions and propagate failures. Preserve exact JSON bytes and all
existing serialization rejection behavior. Add meaningful regressions for a
partial write, publication failure, and a reader during a delayed write. No
numerical or oracle conditions are relaxed.

Execution: put NEW large artifacts under
/work/asciibase64/Mirage/results/stateful-repair/20261117-001/, with a fresh
result/stateful-repair/20261117-001 symlink for repository navigation. Do not
move or delete previous outputs. Record filesystem capacity, errors and result
paths. Validate static checks and targeted/full regressions via Slurm. Use one
owned two-GPU Slurm development allocation with one prepared Qwen3.8/MTP3 vLLM
service for repeated real-model iterations, separate serving/testing srun steps,
immutable iteration snapshots, actual traceback/metrics and cancellation checks.
Preserve ordinary failures without restarting a healthy service. On task exit,
stop owned test/service processes, remove local caches, release the allocation
and verify terminal steps. If a separate new measurement is submitted, keep its
artifacts/cleanup self-contained and label pending results honestly.

Progress: storage and failed artifacts inspected; preparing transactional writer
regressions and compute-node-accessible output storage before loading a model.

- User steering: subsequent NEW test/measurement artifacts may now be written
  to repository `./results/` (plural). Preserve the already submitted /work
  validation/session artifacts and existing result/ navigation link. Place new
  live iterations under results/stateful-repair/20261117-001/ and record capacity
  before execution. No earlier output is moved or deleted.

- Corrected user steering: use the ORIGINAL `./result/` (singular) and existing
  category paths. New stateful live measurement is a real directory at
  `result/multi-api-scheduling/20261117-live-e2e-003/`, not results/ and not a
  /work symlink. Existing submitted /work validation/development records remain
  in place to avoid changing running jobs; no additional output relocation.
  /home now has about 7.7 GiB free. New measurement capacity is recorded before
  execution; preserve all earlier evidence.

- Implemented atomic write_json publication and three injected-failure/reader
  regressions. Static AST and diff checks passed; all three new tests passed
  within Slurm validation **792807**. The full regression is still in progress.
- User corrected storage twice and confirmed original ./result/ category paths.
  Pending development allocation **792811** was cancelled before model startup
  (no service existed), and replacement session **792821** was submitted with
  service logs and artifacts physically under
  `result/multi-api-scheduling/20261117-live-e2e-003/`. Both matching submission
  preflights passed. Only one actual model service is planned/owned for this task.
- The new ten-API protocol preserves 180 provider-request seconds per API, ten
  active APIs, eight execution workers, two generation slots and all prior
  stateful treatments. All 90 source/test/benchmark Python files match 792807's
  frozen snapshot; AST checks passed for 148 snapshot/wrapper files. Run-only
  periodic thread-stack and durable exception diagnostics were added. Previous
  failed stateful output occupies about 792 MiB; current /home has about 7.7 GiB
  available. No earlier artifacts were removed or relocated by the agent.

- Storage repair regression **792807 PASSED**: 97 tests in 348.101 seconds;
  exit 0 and cache_removed=true. A copy of the actual logs/submission/hash
  evidence is available at
  `result/multi-api-repair/20261117-stateful-storage-001/` for the original
  repository navigation convention. Existing /work evidence remains unchanged.
- Original-location session **792821** is allocated and starting the prepared
  service. No actual model-service restart has occurred. Live ten-API results
  remain unverified; periodic traceback diagnostics will distinguish a further
  runtime failure from missing or corrupt persistence.

- First repaired live iteration in job **792821** produced complete final
  evidence: **18,549 JSON files parsed successfully**, all ten sessions recorded
  and terminal, final campaign/metrics summaries saved, service independently
  idle. It retained exit 1 and partial pipeline states rather than inventing Pass.
  Evidence audit is in the original live-e2e-003 diagnostics directory.
- The preserved provider ledger demonstrates a second issue: torch.compile
  launched a canonical request with about 1.17 seconds remaining, less than
  previously observed same-contract request/process cost. HTTP timed out after
  dispatch; cancellation uncertainty correctly aborted the shared endpoint and
  interrupted other APIs. Do NOT weaken that protection or relabel it success.
- Continue the already approved stateful scope in runtime/pipeline.py,
  workflow.py and tests/test_pipeline.py: before dispatch, compare remaining
  per-API model budget with the maximum observed completed same-stage/contract
  request cost including observed provider-process overhead. A budget is a
  maximum, not an obligation to start a request with insufficient known cost.
  Decline the request before dispatch, record estimated cost and unspent budget,
  stop generation normally and fully drain already generated work. No seed,
  prompt, MR, case result or budget type changes; no API-name special cases.
  Predictions cannot guarantee a future request duration; genuine dispatched
  timeouts still retain cancellation uncertainty and shared abort semantics.
- Add tests for non-dispatch/no shared cancellation, contract-specific evidence,
  resume/default compatibility and actual controller drain on budget admission
  rejection. Validate in the existing isolated testing step with fresh immutable
  snapshots before a second real-model iteration. Keep the current healthy
  service and verify quiescence independently before reuse. Do not modify the
  first iteration's executed code, protocol or evidence.

- First live provider-tail diagnosis is saved in live-e2e-003 diagnostics. It
  confirms actual dispatched-request timeout at about one second remaining,
  not an invented framework defect or a failed JSON writer.
- Added same-stage/contract observed-cost admission evidence in pipeline state
  and timing (optional on resume for backward compatibility). The stateful
  controller now records a not-dispatched budget attempt, stops generation with
  its budget reason and drains existing work. True ProviderFailure uncertainty
  handling remains unchanged. Added three focused budget/isolation/drain tests.
- Auto-review rejected deletion of the historical cancellation marker because
  original failure evidence must be preserved. No deletion was performed.
  Safer resolution: retain that marker untouched and save separate independent
  server-gauge quiescence evidence (three samples spaced two seconds apart,
  all running/waiting gauges zero) before service reuse. Saved under
  result/multi-api-repair/20261117-stateful-budget-001/service-quiescence.json.
- Complete budget-regression suite is running in existing testing step 792821.3
  against a fresh frozen snapshot. The same healthy vLLM service remains at
  zero restarts. New outputs stay in original ./result/ category paths.

- Budget-tail regression suite PASSED: **100 tests in 345.379 seconds**, exit 0,
  in the existing testing step (no model restart). Includes independent
  cancellation isolation, typed per-contract admission evidence, legacy/resume
  compatibility and actual controller full drain after declining generation.
- Original result filesystem now has only about 4.6 GiB free. Asked the user
  asynchronously about freeing more space; no response yet. Proceed with a
  separately labeled ten-API **60-second diagnostic** at
  `result/multi-api-scheduling/20261117-stateful-tail-smoke-001/` using the same
  healthy service. This is NOT the original 180-second benchmark and must not
  be used as an equal-budget performance comparison. Full 180-second rerun
  remains subject to enough output space. Its unused model budget/admission
  decisions and genuine failures remain explicit.

- User answered the storage question: run the full benchmark under
  /work/asciibase64. This supersedes the earlier original-result-only preference
  for the NEW full run; existing outputs remain unchanged. Prepared
  `/work/asciibase64/Mirage/results/stateful-repair/20261117-full-001/` with all
  ten APIs, maximum 180 provider-request seconds, eight shared workers and two
  generation slots. Use the existing healthy service after short-diagnostic
  audit/quiescence, retaining its warm-cache state and zero service restarts.
  No new model is loaded and no earlier cancelled/failure record is overwritten.

- Interruption recovery: reattached existing testing step 792821.3 with sattach;
  the new terminal tool handle is 46674. No duplicate allocation/service was
  started. Original 20636 tool handle was no longer available after interruption.
- Short diagnostic audit PASSED evidence integrity: **14,098 JSON files valid**,
  all ten sessions terminal, **10/10 pipeline_complete**, all stop reasons
  llm_time_limit, **scheduler cancelled=false**, service idle, no controller
  exception. It retains exit 1 and four failed session classifications due to
  generation/service errors (six completed); this is not an all-API/MR Pass.
  The maximum model budget was 60 seconds, with unspent budget explicitly
  recorded by admission decisions. Actual diagnostic campaign wall was
  910.913619 seconds; this is not a 180-second benchmark comparison.
- User authorized the full rerun on /work/asciibase64. Full 180-second run is
  starting under `/work/asciibase64/Mirage/results/stateful-repair/20261117-full-001/`
  after independent quiescence verification, in the SAME allocation/testing step
  and healthy vLLM service (restart count remains zero). All existing result/
  evidence is preserved. Before task completion, audit full results and verify
  termination of testing shell, model service, allocation and owned local caches.

- User asked for a directly submit-able sbatch. Added run-specific
  `/work/asciibase64/Mirage/results/stateful-repair/20261117-full-001/sbatch/experiments/stateful_e2e.sbatch`.
  It derives its own frozen source root (ignoring stale inherited session roots),
  creates fresh runs/JOB_ID output, starts one service, runs the ten-API 180-second
  campaign and read-only JSON audit in separate Slurm steps, preserves nonzero
  statuses and cleans owned scratch/service. Explicit resources remain
  8gpus/16 CPUs/2 GPUs/192 GiB/4h. Shell syntax and test-only submission passed;
  no duplicate actual job was submitted. Current full iteration continues in
  existing allocation 792821; the new standalone script has not executed.

- Standalone preflight review found that Slurm executes a spooled copy, so
  BASH_SOURCE cannot locate the frozen run directory at execution time. Within
  the approved run-script scope, use the explicit immutable run root and record
  the actual iteration status/timing before audit. Repeat syntax/test-only
  checks; do not edit the running full campaign or its frozen inputs.

- Corrected standalone script root for actual Slurm spooling (explicit
  persistent frozen root) and added native iteration status/timing before audit.
  Embedded Python AST, bash -n and matching sbatch --test-only passed again;
  standalone remains unsubmitted. Verified 179 frozen input hashes unchanged
  and all four current repair source/test files match the 100-test snapshot.
  Full 180-second campaign is still running in 792821.3 with normal per-API
  generation stops and no reported pipeline exception.

- Accounting review found a planning pre-dispatch flag set before provider
  admission: rejected state-goal calls can retain planning_dispatched=true.
  Within the existing approved workflow.py/tests/test_pipeline.py scope, clear
  that flag only for a GenerationBudgetExhausted in state_plan and validate
  both planning and program admission rejection through the real controller.
  Do not change the running frozen 180-second campaign, prompts or measured
  admission behavior. Run the updated regression snapshot in the same testing
  step after the full campaign, and identify the reporting-only snapshot
  difference in the final evidence instead of rewriting completed measurements.

- Full 180-second maximum per-API rerun finished in existing job/step
  792821.3: wrapper elapsed 3689 seconds, actual campaign wall
  3582.213197 seconds, native exit 1 retained. Evidence audit checked
  **46,813 JSON files with zero parse errors**, all ten sessions terminal,
  **10/10 pipeline_complete**, no controller exception, service idle.
  Session classifications remain **3 completed / 7 failed**; integrity is not
  an MR Pass. All ten API ledgers record normal generation-budget stops.
  Detailed failure/latency accounting and reporting-only regression are now
  running sequentially in the same testing step with the same healthy service.
  Full original evidence and its executed code stay unchanged.

- Reporting regression iteration accounting-001 correctly failed after two
  tests (66.375s): the new no-dispatch flag passed, but planning_calls still
  counted the preallocated planning_usage key. Retain this failed snapshot/log.
  Continue within originally authorized artifacts/report.py and
  tests/test_artifacts.py scope: count explicitly dispatched planning rows,
  exclude never-dispatched usage from token completeness, preserve legacy
  fallback for old reports and retain attempted family-selection evidence.
  Validate rejected calls, actual failed dispatches and legacy rows, then
  rerun the full selected suite in a NEW frozen iteration. No model restart.

- Final reporting-only regression snapshot accounting-regression-002 PASSED:
  **101 tests in 362.001 seconds**, iteration elapsed 363 seconds, exit 0.
  Includes actual planning/program budget rejection with full drain and explicit
  rejected/failed/legacy dispatch accounting. All 120 frozen submission input
  hashes match; five current repair source/test files match the tested snapshot;
  git diff --check passed. Earlier failed accounting-001 remains preserved.
- Final full measurement details: campaign wall **3582.213197 seconds**, wrapper
  **3689 seconds**, **581 successful / 746 failed model requests**. Failures:
  **719 HTTP 400 context-length rejections**, **27 completions truncated at
  output length**. Keep these generation failures; no prompt/context/oracle
  treatment was silently changed. Session states remain 3 completed / 7 failed,
  native measurement exit 1, scheduler cancelled=false, execution/generation
  peaks 8/2, no unfinished requests/workers, all ten pipeline drains complete.
- Full reports include **13 suspected failures for slogdet**, all 13 with valid
  automatic reproduction evidence. They are not confirmed framework defects;
  minimization, assumption/numerical revalidation and corroboration remain
  separate triage. Current benchmark evidence does not generalize to all APIs.
- Original full-001 snapshot/results remain unchanged. Five original planning
  calls were marked dispatched before budget admission; corrected derived
  counts are in full-001/diagnostics/detailed-audit-final.json. The only later
  code differences are dispatch/usage reporting and corresponding tests; exact
  patch and source-scope evidence are in accounting-regression-002/diagnostics/.
  Full per-API E2E and successful/failed request latency metrics are in
  full-001/measurement/metrics-summary.json and per-api-metrics.csv.
- Latest directly submit-able script uses the final tested reporting repair:
  `/work/asciibase64/Mirage/results/stateful-repair/20261117-accounting-regression-002/sbatch/experiments/stateful_e2e.sbatch`.
  Bash syntax and matching Slurm test-only preflight passed. It has NOT been
  submitted; the measured full run reused 792821. The earlier full-001 script
  remains available for exact reproduction of its original snapshot.
- Cleanup VERIFIED: testing terminal/step 792821.3 exited 0; owned vLLM service
  step 792821.0 was stopped, all job steps exited, and allocation 792821 is
  COMPLETED (02:26:41). No owned local srun/sattach connection remains.
  cleanup.json confirms both main and testing scratch caches removed. Service
  startup was 186 seconds, actual restarts zero. Cleanup session accounting is
  8799 seconds / 17598 estimated allocated GPU-seconds, separate from scheduler
  allocation duration; neither is utilization or a sum of overlapping API times.
- Historical cancellation marker remains untouched after auto-review rejected
  deletion. Independent three-sample idle evidence before final cleanup remains
  in accounting-regression-002/diagnostics/service-quiescence-before-cleanup.json.
  Final evidence, risks and lifecycle/resource accounting are consolidated in
  accounting-regression-002/repair-summary.json; original failed evidence is
  retained, and no task-owned service/allocation is left running.
