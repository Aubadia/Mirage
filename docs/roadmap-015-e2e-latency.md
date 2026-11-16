# Roadmap 015: Ten-API E2E latency and equivalent optimization

Status: completed on 2026-11-16. JSON conversion optimization accepted; 32 workers
remain experimental and the product default remains eight. The complete live
candidate did **not** pass every acceptance gate: `batch_norm` recorded 179.851486
request seconds, below the original 179.98-second lower tolerance. See the final
review below; earlier progress entries describe their respective intermediate state.

## Approved scope

The user explicitly approved this plan after reviewing the file scope and equivalence requirements. This roadmap is the first edit after approval.

Measure the ten selected benchmark APIs (scatter_add, slogdet, lu_solve, autocast, make_graphed_callables, compile, batch_norm, add, std, copy_) with their existing CPU/CUDA and forward/backward policies. Use the prepared Qwen3.8-27B snapshot, MTP3, seed 0 and 180 cumulative model-request wall seconds per API. Framework work is outside that budget. Run one complete baseline, bounded fixed-work diagnostics, then one complete candidate campaign.

Authorized additions: this roadmap, benchmarks/e2e_latency.py, sbatch/experiments/e2e_latency.sbatch, and result/e2e-latency/. Necessary edits are limited to src/mirage/runtime/{pipeline,process,worker,environment,stateful}.py, workflow.py, generator/{provider,http_provider}.py, artifacts/report.py, cli.py, contracts/programs.py; sbatch/lib/{serving,environment,development}.sh; relevant existing tests; README.md and sbatch/README.md. Preserve pre-existing working-tree changes. No new dependencies, prompt edits, MR changes or coverage-definition changes.

## Method

1. Freeze the current engine, experiment driver, model configuration, API/device roster, policies and resource allocation. Keep issue-derived examples outside all generation requests. Use the existing benchmark mixed forward/backward policy, model-request budgeting and full representation drain. Record individual API E2E and entire-campaign makespan separately.
2. Reuse one Slurm allocation and healthy model service across iterations. Isolate serving and testing GPU steps, respect scheduler CPU/GPU limits, use explicit resources and preserve job-specific caches. Record service startup/restarts and allocated resource time separately. Reserve cleanup time and keep partial evidence if the allocation limit is reached.
3. Measure model request latency and usage, serving queue/prefill/decode/TTFT metrics where supported, worker time and concurrency, framework/model overlap, controller/report costs, queue/backpressure, final drain, errors and all existing quality metrics. Model request time includes service/network waiting and is not pure inference time.
4. Select optimizations from actual bottlenecks. Test worker overhead/concurrency or serving/cross-API scheduling using fixed requests and saved cases before integration. Keep isolation, normal process exit, cancellation, recovery, per-API feedback causality and complete candidate processing. Preserve existing externally defined fields and semantics.
5. Require fixed-case equality of process outcomes, validity, state/path attainment, coverage, oracle status/reasons and replay evidence. Compare identical serving requests' outputs before accepting serving changes. Keep unknown/nonfinite evidence unknown. A shorter run caused by less generation, less coverage, incomplete drain or more unknowns is not an accepted speedup.
6. Run the complete candidate E2E campaign with the same per-API model budget and immutable model/prompt/sampling settings. Report workload and quality alongside latency: faster generation may produce more verification work, so equal-budget live E2E and equal-work speedup are distinct claims. Do not claim identical adaptive LLM text trajectories or statistical equivalence from one campaign per arm. Reject or explicitly retain as experimental any optimization that fails the approved checks.
7. Run appropriate regressions and benchmark controls; compare final code with validated snapshots. Stop owned services/test processes, release the allocation, close terminals, verify exited steps and preserve all evidence before completion.

## Initial observations

- Current stateful multi-API workflow runs APIs sequentially. Each API has one causally dependent model request at a time; increasing serving batch capacity alone does not create independent work.
- The prepared service is vLLM 0.33.0 with Qwen3.8-27B BF16, MTP3, prefix caching, context 16384 and output allowance 4096. The previous service allowed two concurrent sequences.
- Roadmap 014 measured worker throughput at 8/16/32 slots on 32 CPUs and retained equivalent report-conversion/persistence optimizations. Those improvements are part of this new baseline. Individual worker latency rose with concurrency; CPU-only results do not establish GPU-worker or live E2E gains.
- The site caps CPUs at 12 per allocated GPU. Resource overrides must be preflighted and explicit. No running model service from the previous task is to be reused.

## Progress

- Approval received; roadmap created before implementation. The baseline engine and driver are frozen under `result/e2e-latency/baseline/`, with per-file hashes and the exact protocol.
- The 8gpus submission (785047) remained pending at the account GPU cap. The attempted partition update was rejected; the unstarted job was cancelled. An explicit dev/4-hour resubmission retains 36 CPUs, 3 GPUs and 192 GiB. Serving uses 4 CPUs/one GPU; testing uses 32 CPUs/one GPU. The third allocated GPU is scheduler overhead because of the site CPU/GPU ratio.
- Pre-model checks in job 785060 exposed an existing no-allocation exit-status mismatch. The shared environment guard now consistently returns 2. That job exited before model loading. Job 785068 passed all 24 preflight tests and loaded the unchanged service in 193 seconds, with zero restarts so far.
- Baseline campaign is running sequentially in job 785068 using a separate testing step. Service metrics are sampled every two seconds. Ordinary per-API failures retain nonzero status and do not prevent later APIs when server idleness is independently confirmed.
- Initial service/kernel warmup and later cache reuse are recorded separately. Live campaigns may see different prefix-cache warmth; their latency comparison is descriptive. Causal speedup claims require fixed-work comparisons, with workload and quality checks.

### Prepared acceptance checks

- Fixed-work screen: select up to 32 saved execution pairs per API by observed source/representation/replay, process, mutation, verdict and audit classes; add complete reference-parent closure. Run the identical set at 8 and 32 workers, keeping APIs sequential and rebuilding each arm's strict references. Compare every process/verdict/path/state/coverage/audit/replay signature, then exact saved initial/training evidence with nonfinite results still unassessed. This is a bounded screen, not an exhaustive equivalence proof.
- Reject worker configurations with any unexplained fixed-case mismatch, newly missing evidence, timeout/crash or changed unknown classification. Live 180-second-budget runs must separately report all attempted/generated/validated cases, complete representation drainage, usage, failures, coverage and detection. Different adaptive trajectories cannot establish statistical equality from one campaign.
- A small serialization trial exists only under `result/e2e-latency/json-trial/`: avoid recursive calls for builtin scalar leaves and copy containers only when special values need conversion. Re-walk each call, retain strict key/type/nonfinite behavior, and do not cache mutable input. Adoption requires byte-identical report outputs and a measured gain, plus existing serialization/recovery regressions. Production files have not received this trial.

### Baseline observations (campaign still in progress)

The first two API runs completed their 180-second model budgets and all pending
representation work. Their last budget-truncated provider requests retain
`provider_cancellation_unconfirmed` and child exit status 1; the campaign's
external serving monitor observed empty running/waiting queues before continuing.
These statuses are preserved, not relabeled as successful generation.

| API | Parent E2E s | Model request s | Worker busy union s | Worker sum s | Model/worker overlap s | Controller/unattributed s | Runtime-valid / attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scatter_add | 2155.336 | 180.007 | 1529.386 | 8131.933 | 93.298 | 528.425 | 1644 / 1755 |
| slogdet | 534.597 | 180.007 | 430.773 | 1600.842 | 94.758 | 17.451 | 331 / 356 |

These validity denominators include framework representation reuse; they are not
LLM-only generation success rates. Slogdet recorded 19 suspected cases and 19
successful automatic replays. This is not a distinct-defect count. Post-hoc
benchmark attribution and the ten fixed benchmark controls are scheduled outside
live E2E intervals; VJP attribution reuses recorded numerical evidence and must
not be described as an additional independent execution.

Scatter_add also reproduced the existing HTTP 400 context overflow. Prompts,
context/output limits and budget charging remain unchanged. In its partial
serving sample, internal inference summed to about 116 s (prefill 36 s, decode
80 s), while request queue time was only 0.113 s. This points toward worker and
controller costs rather than serving queue congestion; final per-API counter
deltas will be retained in the observation report.

### Resumed execution

- User requested continuation of the approved scope. Recovered job 785068 with seven of ten baseline API attempts recorded; the existing model service remains healthy according to the completed per-API idle checks. The prior local terminal handle is unavailable, so recover monitoring through the owned allocation without duplicating its running work. Complete baseline and bounded diagnostics, adopt only accepted changes, then attempt the full candidate within verified allocation time; preserve partial evidence if time is insufficient.


The optimized live arm will run from its frozen experimental engine. Production
adoption follows the fixed-work and live-quality review, rather than preceding
it. The session launcher accepts an explicit `MIRAGE_PREFLIGHT_ENGINE` so a new
allocation can run pre-model regressions on that exact candidate. The current
running baseline uses Slurm's submitted script snapshot and its immutable engine;
these launcher edits do not change it. Fixed evidence checks additionally compare
finite values and NaN/Inf positions/signs, and reject explicit differences even
when the existing numerical comparator retains an unknown verdict.

- Reattached the original testing step 785068.4 via `sattach`; no model restart or duplicate workload. Eight baseline API attempts are complete, each with full representation drain. `torch.add` took 1444.225 s parent E2E, 180.029 s model requests, 1333.378 s worker busy union, and 72.911 s controller/unattributed time; runtime validity is 1363/1444 including representation reuse. `torch.std` is running. Current production/driver hashes still match the baseline snapshot. Prepared an additional ten-API integrity audit from the prior accepted checks; regression and benchmark-control snapshots match current files.

### Fixed-work acceptance and candidate freeze

- Baseline completed all ten API attempts in 8106.480 seconds campaign makespan. All ten retained the budget-cutoff cancellation status; external idle checks and corrected integrity checks passed. No candidate remains pending or dispatched. A zero-byte HTTP log is retained as missing evidence. The initial diagnostic mistakenly required positive overlap for source-only APIs and omitted two files from a copied test fixture; original failures are preserved, corrected diagnostics pass.
- Stratified fixed cohort: 359 pairs from 5143 available executions, including reference-parent closure, run at 8 and 32 workers. Each raw comparison matches 356/359; the remaining three are identical torch.compile exceptions except for built-in-method process-local addresses. Separate narrow review explains these addresses and finds zero unresolved semantic differences. Both arms have 622 exactly equal saved members and 96 absent/nonfinite unassessed members, with no explicit tensor mismatch.
- Fixed-work wall times: 250.566 s at 8 workers and 145.940 s at 32 workers (1.7169x observed throughput). Worker p50/p95 increased from 4.570/6.628 s to 6.143/15.823 s; peak summed descendant RSS increased from 9.143 to 34.752 GB. Shared pages may be counted repeatedly. This bounded ordered screen is not exhaustive equivalence or a live E2E speedup claim.
- Report ABBA trial: baseline 9.003/7.991 s, candidate 3.197/3.203 s for the same 406523172-byte report; every SHA-256 matches. Adopt the tested conditional-container-copy/scalar conversion in src/mirage/workflow.py, retaining all serialization validation and writes. 18 artifact regressions and 13 pipeline regressions pass; all ten fixed benchmark controls were detected and independently corroborated.
- Freeze the candidate at 32 workers with the unchanged model, prompts, 180-second budgets and API/device policy. Keep the product default at eight workers. The dev partition hard limit is four hours; the remaining time is insufficient to safely budget a complete candidate plus analysis/cleanup from observed durations. Use a follow-on allocation with the same explicit dev/4-hour, 36-CPU, 3-GPU, 192-GiB resources, dependent on the old allocation ending. Both scheduler preflights passed; unrestricted H200 node selection is earlier than forcing the old node. Record the new service instance and cold prefix cache separately. No overlapping owned serving allocation is intended.

- Baseline allocation 785068 completed with batch/test exit 0; serving was intentionally terminated after two empty-queue checks. Elapsed allocation 9568 s; startup 193 s; no in-session restart. EXIT cleanup completed; direct post-release scratch inspection was denied by the cluster because no active job remained on that node.
- Follow-on job 785575 on 25a-hgpn009 failed the unchanged 900-second serving readiness bound before any candidate API or model request. Its preflight 24 tests passed, but multi-modal warmup took 110.931 s (baseline 12.091 s) and model loading 186.883 s (baseline 30.587 s). Preserved the failed startup and terminal steps. Retry uses the same dev/36-CPU/3-GPU/192-GiB/4-hour allocation, explicitly excludes this slow node, and increases readiness waiting only to 1800 s; model/prompt/inference settings remain unchanged. Added cache-path and verified cache-removed artifacts to the authorized batch script so future cleanup is directly auditable.

- Retry job 785647 on 25a-hgpn041 passed 24 preflight tests and became ready in 225 s with unchanged model configuration. Readiness includes GPU-isolation verification and brief resource waiting from an early empty testing attachment, which was closed before verification completed. Multi-modal warmup was 13.300 s. A saved CPU source control then matched its exact semantic signature in 5.773 s, with zero LLM requests. The complete frozen 32-worker candidate campaign is now running in separate step 785647.5; model service remains in step .1.

## Final review and disposition

- Both campaigns attempted all ten APIs and drained every pending/dispatched
  candidate. Baseline makespan was 8106.480 s; candidate was 7653.745 s (5.58%
  shorter observed wall time). All twenty per-API children preserve exit 1 and
  `provider_cancellation_unconfirmed`; separate service observations confirmed
  idle queues at each boundary. Campaign completion is not generation success
  or an MR Pass.
- Candidate integrity passes every check except `batch_norm`'s original budget
  tolerance. Its last provider process consumed 0.869819 s, of which 0.719933 s
  was measured request time; the 0.148514-second budget shortfall is consistent
  with process overhead consuming the enclosing deadline. Preserve the failure,
  the original threshold, and all evidence in `budget-review.json`. Do not resume
  or rerun generation to fill this shortfall.
- Workload changed: baseline/candidate recorded 5687/6236 total attempts,
  4931/5561 runtime-valid attempts, 546/531 generated programs, and 357/367
  runtime-valid generated programs. These are different denominators. `std`
  semantic units fell 20 to 17; representation units fell for slogdet, lu_solve,
  std and copy_; numerical unknown fractions rose for scatter_add, slogdet and
  add. Hence no equal-quality live acceleration or statistical equivalence claim.
- The initial matching completed request payload for each API produced identical
  response bytes in both arms. Later adaptive trajectories differ. Different H200
  nodes, cold service instances and cache history further limit live comparisons.
- Baseline/candidate have 19/12 suspected slogdet cases, all with valid automatic
  no-LLM replay. Post-hoc attribution reports 19/12 mechanism matches, but reuses
  production finite-difference evidence and is not a new independent execution or
  distinct-defect confirmation. The ten supplied benchmark controls were detected
  and independently corroborated separately from blind generation. No new
  confirmed framework-defect claim is made.
- Keep the validated conditional JSON-container copying/scalar handling in
  `src/mirage/workflow.py`. Its four ordered ABBA report outputs are byte-identical
  and the mean baseline/candidate timing ratio is about 2.66x for that report.
  Keep 32 workers as an experimental option despite the fixed-cohort throughput
  gain; do not change the default of eight. No serving, prompt, MR or coverage
  semantics changed.
- Validation completed: 18 artifact and 13 pipeline regressions; 24 candidate
  startup preflight tests; ten supplied controls; exact candidate/production hash
  equality across 51 Python files; Python AST, batch syntax and diff checks.
  Initial diagnostic failures, pointer-only fixed-signature differences and the
  final candidate budget failure remain recorded. Unknown/nonfinite evidence is
  not promoted to passing numerical evidence.
- [Full report](../result/e2e-latency/report.md),
  [decision](../result/e2e-latency/decision.json),
  [comparison](../result/e2e-latency/comparison.json), and
  [resource accounting](../result/e2e-latency/resource-accounting.json) contain
  the detailed evidence and limitations.

## Final cleanup and resource cost

All task-owned jobs (785047, 785060, 785068, 785575 and 785647) and their steps
are terminal. Candidate testing step 785647.5 exited 0:0; batch exited 0:0;
serving step .1 was deliberately cancelled after two empty request-queue samples.
The testing terminal is closed and the batch verified the exact scratch directory
was absent before writing `785647/cache-removed`.

Allocated job seconds were 0, 7, 9568, 921 and 8354 respectively: 18850 seconds
total, 15.7083 allocated GPU-hours and 188.5 allocated CPU-hours. These include
startup, tests, diagnostics, review waiting and cleanup; they are not utilization.
Healthy startup times were 193 and 225 seconds, with zero in-session restarts.
There were three service-start attempts: two healthy services and one failed
900-second readiness attempt (921 allocated seconds). No owned serving
allocations overlapped. Per-iteration times remain separate in campaign, fixed-work,
controller-trial and diagnostic status artifacts; overlapping steps and request/
worker intervals are not summed as wall time. Baseline scratch removal is supported
by its successful EXIT trap, but independent post-release probing was unavailable
because cluster access required an active allocation; current cleanup has a direct
absence marker.
