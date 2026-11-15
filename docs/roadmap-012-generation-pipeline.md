# Roadmap 012: Generation budget, pipelined execution and complete representation traversal

## Approved scope

User approved the implementation/file scope proposed in this conversation, and expanded acceptance from torch.linalg.slogdet to three additional existing benchmark APIs. Selected panel: torch.linalg.slogdet, torch.Tensor.scatter_add, torch.linalg.lu_solve, torch.add. These exercise matrix backward, indexed operands, heterogeneous solver operands, and multiple elementwise operands. Only API names enter generation; benchmark trigger programs do not.

Each API receives 180 seconds of cumulative model-request time (planning, generation, repairs, including failed calls). Framework execution and queue waiting do not consume that budget. Every valid generated source/configuration must drain its finite observed representation inventory. Unsupported candidates retain reasons; runtime failure or unavailable evidence is not coverage or Pass. A run with pending work is incomplete and resumable.

## Implementation

1. Add bounded overlapping generation/execution with one in-flight model request and one buffered generated program. Consume only completed verified feedback, label request feedback snapshots, and preserve immutable source/configuration and original MR obligations.
2. Separate model request timing, framework worker timing, controller/queue time, overlap and total wall time. Account failures/cancellation and preserve existing provider timeout contracts. Per-API model budgets stop generation only; worker timeouts and Slurm allocation limits remain independent.
3. Drain every eligible call/path/action for each accepted source/configuration, including sources with a suspected failure. Record all exclusions and outcomes. Preserve completed evidence and resumable pending work across interruption; do not recursively register representation children.
4. Integrate CLI/configuration/reporting and focused tests for budget accounting, overlap, feedback causality, backpressure, complete traversal, failure handling and resume.
5. Freeze code/protocol, run seed0 acceptance for the four selected APIs, 180 model seconds each, CPU eager and required backward. Record generation and framework times, candidate completeness, oracle outcomes, and replay/corroborate suspects (especially slogdet). No injected known trigger sources and no claim that fixed controls are autonomous discovery.
6. Drain requests and stop owned test/service/allocation resources; record evidence and remaining limitations.

## Authorized files

- src/mirage/workflow.py
- src/mirage/sampler/representations.py
- src/mirage/runtime/pipeline.py (new)
- src/mirage/generator/provider.py
- src/mirage/generator/http_provider.py
- src/mirage/runtime/worker.py
- src/mirage/contracts/programs.py
- src/mirage/cli.py
- src/mirage/artifacts/report.py
- README.md
- tests/test_pipeline.py (new)
- tests/test_adaptive.py
- tests/test_http_provider.py
- tests/test_cli.py
- tests/test_artifacts.py
- this roadmap
- supporting validation scripts, frozen snapshots, logs and results under result/representation-pipeline/

Existing categorized Slurm scripts are reused. No dependency changes or unrelated refactoring. Preserve existing worktree/index. All computational validation uses Slurm and the prepared Python/model environments.

## Validation plan

Run static syntax checks before loading the model. Run focused deterministic mock tests and required affected regressions in Slurm, then real-model acceptance on the frozen four-API panel. Check actual time intervals for overlap, per-request feedback versions, all candidate terminal statuses, target and backward evidence, and separate MR/numerical verdicts. Exhausted generation budget is not an execution deadline. An interrupted drain must be resumed or reported incomplete. Suspects require saved-evidence replay and independent corroboration before any defect claim.

## Progress

- Read workflow/provider/worker/traversal/CLI/reporting and Slurm paths; confirmed current generation and execution share one wall deadline and are serial.
- Created roadmap as first change after approval. Implementation and validation pending.

- Implemented separate provider request timers and a background representation consumer sharing one serialized execution slot with source/replay work. Model requests use completed-feedback snapshots; source backlog applies backpressure independently of the model budget.
- Added source staging, durable candidate inventory, explicit drain resume, SIGTERM cleanup, CLI opt-out for legacy wall budgets and distinct dispatch/attainment completion fields. Static syntax checks passed; runtime validation pending.
- Slurm preflight: default 8gpus estimated next-day 04:51; explicit dev/400G preflight estimated 21:32. Submitted task-owned job782013 with dev, two GPUs/eight CPUs/400G/four hours, prepared Qwen3.8-27B MTP3, port18712 and 1800s service readiness bound. Actual start21:28 Asia/Taipei. This is an explicit resource override, not a fallback hidden from results.

- Validation iteration6oJV5Q:37 tests passed (pipeline unit controls, existing provider/CLI/traversal regressions). Initial iterationue7KLe failed solely because the unittest module path omitted the frozen tests directory; retained logs, corrected the runner.
- Runtime iterationoCjCcy exposed the fixture provider subprocess's missing test import path; fixed the test environment. Iteration1yAARL then passed the real PyTorch slogdet source-limit/full-drain test, including actual negative representation attainment. These are authored controls, not autonomous discovery.
- Added idempotent resume validation and an acceptance driver that records independent generation/framework/overlap times and confirms owned-service quiescence before reusing it after a deadline-truncated request. Final affected validation pending.

- Final affected snapshot iteration68l9tk passed53 tests in35.20s, including real slogdet full traversal and LLM-free idempotent resume. git diff --check passed. Froze that source and the four-API driver under result/representation-pipeline/final-engine and final-lock.json before acceptance.

- Frozen acceptance is running on job782013. The first slogdet source autonomously produced a negative-view local-gradient suspected failure, with a matching valid automatic replay. Continue the full budget/drain; separate analytical review and standalone controls are queued.
- During read-only controller review, found a multi-API reporting edge case: cancellation after API1 returned its complete single-API report as if it represented the requested panel. Corrected the single-report shortcut to require a single requested API, and added a focused regression. The four independently invoked single-API acceptance searches continue using their original frozen engine; this aggregation-only guard does not change any accepted single-API path. Follow-up regression is queued in the same Slurm testing step.

- slogdet search completed its full drain:180.0077 model seconds,1335.3871 framework-worker seconds,169.8398 overlapping seconds,1369.2056 wall seconds. Candidate states:163 reached,7 exact prior runtime evidence,90 executed/not verified,156 ineligible,0 pending/dispatched.21 suspected cases had21 valid matching replays.
- Independent review iterationcGigq5 and standalone control iterationvbKrUp both exited0:exact 2x2 inverse-transpose confirms original/replay negative-view gradient violations; ordinary and resolve_neg controls agree. This corroborates one known local mechanism, not21 distinct defects.
- slogdet outer driver iterationirbEaf exited1 at its final whole-directory inventory equality check. Read-only diagnosis found exactly12 new __pycache__/*.pyc files, no changed/deleted original files and no mismatch among any locked Python sources. Search/drain and service-quiescence evidence are complete. Preserve the nonzero original; independently audit locked sources and permit only incidental bytecode additions, without rerunning or changing the frozen search.

- User requested generation pause and a full metric report. Stopped scatter_add at22.756975 model seconds/8 requests; lu_solve and add were never started. Original interrupted driver status137 is retained. A fresh step in the same allocation resumed only saved work, with IsolatedProvider.generate patched to raise if called:0 new model requests, full candidate drain, exit0 (iterationqa1sKF).
- Scatter_add partial-generation snapshot:386.1405 framework-worker seconds,16.3709 overlap seconds,400.8022 active wall seconds and585.9015 elapsed including the interruption gap.77 reached+2 exact prior evidence+5 executed/not verified+60 ineligible;0 pending/dispatched. Resume did not persist prior backpressure time:raw0 is unavailable, not a measured zero. Recovered source controller bookkeeping is incomplete for library_configuration_outcomes.
- Additional pipeline regression iterationI3eFre passed9/9 in0.121s;54 distinct tests have passed across affected validation. The only production difference from the frozen acceptance engine is the separately tested multi-API aggregation guard.
- Saved all existing nested search metrics, cohort legality, explicit processing/attainment denominators, request outcomes/stages/tokens and measured CPU/RSS resources in result/representation-pipeline/observation-metrics.json and all-observation-metrics.md. Counts for both APIs include representation reuse; LLM-only cohort metrics are separate.
- Cleanup verified:service running/waiting0/0, both owned test scratch directories removed, test PTYs closed, job782013 and all steps terminal. Allocation21:28:16–22:13:06 Asia/Taipei,2690wall seconds/5380allocated GPU-seconds (not utilization); service startup194s/restarts0. Further generation and the remaining two API acceptances are paused at the user's request.

## Authorized follow-up: isolated representation concurrency measurement

User approved measuring2,4,8 workers before integrating parallelism. Keep LLM generation paused and do not start the remaining API acceptances. This follow-up changes no production scheduler:use a standalone harness under result/representation-parallel/ and a categorized sbatch/experiments/representation_parallel.sbatch launcher; update this roadmap first and preserve all prior worktree changes.

Freeze the253 completed representation-child requests from the slogdet acceptance, their engine, source/initialization hashes and expected runtime/coverage/oracle signatures. All arms execute the exact same ordered cohort in fresh artifact directories. Each pool thread supervises one isolated existing worker process; each worker runs CPU eager with one compute thread and a120s timeout. Retain the original per-pair numerical audit budget and initialization reference. Source/follow-up execution within a case remains unchanged. No provider or model service is started.

Use one Slurm allocation with8 CPUs,1 GPU(default template; target computation remains CPU),16GiB host memory and4h limit. Validate the script and scheduler preflight. Record any explicit partition/resource override. Use the prepared project Python. Bound the harness by remaining allocation time and reserve300s for evidence/cleanup.

Run two predeclared rounds:2,4,8 then8,4,2. Warm up the same8 fixed cohort cases before each arm, retaining warmup logs separately from measured results. Measure complete arm wall time, cumulative worker time, worker-time interval union, throughput, latency distribution, queue wait, CPU time and sampled aggregate process-tree RSS. Aggregate RSS is a sum that double-counts shared pages, not physical memory use. Report2-worker-relative speedups, both rounds/ranges and observed peak concurrency; do not claim speedup versus an unmatched historical1-worker run.

Check all253 outcomes against the saved candidate-specific process, legality, state attainment, semantic coverage, MR and numerical-oracle signatures. Keep not_verified/inapplicable cases in the denominator. Verify frozen hashes before/after, prove zero model calls by the execute-only harness, preserve failures with nonzero status, and confirm owned jobs/steps/scratch terminate. Scheduler completion alone is not correctness acceptance. Integration into the production pipeline remains outside this standalone measurement.

- Frozen protocol-01 contains253 cases,43 initialization records and96 pinned files; protocol SHA2561fd96b7b87fa60c8205f8255987e5910e36dcaf30500ec246e8628a3d41a5467. Syntax/bash checks passed. Default8gpus preflight predicted2026-11-16 06:13; dev predicted2026-11-15 22:49. Submitted owned job782311 with explicit --partition=dev,8CPUs/1GPU/16GiB/4h; actual start22:38:45 on25a-hgpn014. No model service or production edits.

- Same-allocation measurement in progress:actual affinity is8 physical cores across2 sockets; OMP/MKL/OpenBLAS each1. First round2/4/8 workers completed253/253 matching cases in540.85/277.67/148.04s. Reverse-round8/4 completed in148.06/279.78s, again253/253 matches. Last2-worker arm remains in progress. All warmups so far match;0 model calls and no production scheduler changes.

- Standalone experiment complete:all6 arms and6 warmups passed their saved-signature checks.1518 measured cases+48 warmups,1566 actual execute_stateful_pair requests,0 provider/model requests. All96 frozen files verified; observed peak concurrency exactly2/4/8. Every measured arm retained20 suspected_failure+233 inapplicable and163 attained+90 not_verified; these counts are not all correctness Passes.
- Two-round mean wall times:2workers542.720s,4workers278.726s,8workers148.046s. Speedup versus matched2-worker baseline:1.947x/3.666x. p50 case latency4.280/4.354/4.620s; maximum sampled process-tree RSS sum2.778/5.515/7.505GiB(shared pages counted repeatedly). Full observations in result/representation-parallel/782311/report.md and analysis.json.
- Job782311 and every step COMPLETED0:0; allocation22:38:45–23:12:12 Asia/Taipei,2007seconds,8CPUs/1GPU/16GiB,SlurmTotalCPU4915seconds. Submitted EXIT trap completed cleanup of its owned scratch.0LLM services started. Production source verification found no changes from the pre-experiment state. LLM generation and remaining API acceptances remain paused; production parallel-pipeline integration was not part of this measurement.
