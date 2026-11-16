# Roadmap 014: Worker overhead and equivalent execution optimization

## Approved scope

The user approved profiling saved slogdet and scatter_add cases under Slurm at 8, 16 and 32 workers, followed by optimizations that preserve test semantics, candidate processing, oracle decisions, replay and recovery guarantees. Context overflow is proposal-only in this task: do not change prompts or generation policy. Do not issue new LLM requests.

Authorized files: this roadmap; sbatch/experiments/worker_overhead.sbatch; artifacts under result/worker-overhead/; necessary changes to src/mirage/runtime/{pipeline,process,worker}.py, src/mirage/workflow.py, src/mirage/artifacts/report.py, related existing tests and README.md. Preserve previous worktree changes. No unrelated refactor, dependencies, weakened checks, reduced replay or candidate skipping. This roadmap is the first edit after approval.

## Method and equivalence boundary

1. Freeze the current production source and preserved inputs from result/parallel-pipeline/acceptance-01. Keep source, configuration, initialization and audit limits fixed. Begin with a deterministic stratified cohort containing both APIs, representation actions, valid/invalid/inapplicable/suspected outcomes and audit statuses; use the full retained cohort for final comparison when practical.
2. Measure 8/16/32 workers with explicit Slurm CPU/memory allocation. Separate startup, dispatch/test, oracle, evidence I/O, interpreter exit, pool queue, persistence and controller costs. Instrumented diagnostics and uninstrumented throughput measurements are distinct. Do not infer wall-time savings from sums of overlapping durations.
3. Profile report computation/serialization and pipeline persistence on preserved data. Prioritize redundant copies and repeated calculations without changing reporting semantics or durability. Keep process isolation, normal interpreter cleanup, oracle conditions, audit budgets and replay counts.
4. Adopt only measured, focused improvements. For fixed work, compare verdicts, state/path attainment, coverage, mutation and replay evidence. Timing, process-local identities and output paths are not semantic equality targets. Live adaptive search uses observed costs and feedback; its exact generated trajectory is not promised identical across performance changes.
5. Run relevant regressions under Slurm, including cancellation, multiple in-flight recovery, ledger integrity, shared worker limits and report equivalence. Recheck current source against the measured snapshot. Record any mismatches and limitations rather than loosening comparisons.
6. Release all owned Slurm jobs/steps and scratch before completion. Report measured gains, single-case latency and end-to-end costs separately, and recommend a worker count from actual evidence.

## Existing evidence and context proposal

Roadmap 013 scatter_add has a 373,684,678-byte final summary rewritten after each source round. Pair subprocess median is about 4.606 seconds; cumulative worker-event time outside run_worker is 859.711 seconds and controller/unattributed wall time is 578.011 seconds. These are candidate bottlenecks, not yet causal profiling results.

Context proposal (no implementation here): count final rendered messages with the actual tokenizer/chat template and reserve output tokens; losslessly factor repeated facts and remove execution-only bookkeeping from the prompt copy; select optional history by tokens while preserving required MR/state/operand/backward evidence; partition oversized planning goals with dependency closure and explicit examined/omitted denominators. Validate against all 86 saved context-limit failures. Increasing context alone is not a bounded solution to growing feedback.

## Progress

- Approval received; roadmap created before any other edit. Baseline engine frozen with 51 source-file hashes before production edits.
- Slurm rejected 32 CPUs with one GPU: the site imposes a maximum of 12 CPUs per GPU. An explicit `--partition=dev --gpus-per-node=3` override passed preflight and started job 783115 with 32 CPUs and 64 GiB host memory. All retained cases are CPU cases; the three allocated GPUs are scheduler cost, not utilized compute. One test step uses 32 CPUs and one visible GPU; no visibility override or LLM service.
- Fixed 128-case stratified screen includes 83 slogdet and 45 scatter_add cases, with exceptions and applicability/audit classes. Order: 8, 16, 32, 32, 16, 8 after a separately reported warmup. All six arms matched every expected semantic/process signature.
- Final implementation reuses one traversal snapshot for both existing durable ledger writes and fast-paths builtin JSON data before the existing special-type converter. File format, strict key/nonfinite rejection, write cadence and normal worker process cleanup remain unchanged.
- A per-report coverage cache was tried, but the uninstrumented measurements showed no meaningful gain (scatter_add final metrics: 1.071 s baseline, 1.080 s trial). It was reverted. `artifacts/report.py` is byte-identical to baseline; final production changes in this task are only `runtime/pipeline.py` and `workflow.py`.
- All six uninstrumented scaling arms matched 128/128 expected signatures. Times at 8 workers: 75.777/76.315 s; 16 workers: 42.391/42.204 s; 32 workers: 30.268/30.500 s. Means: 76.046/42.297/30.384 s, or 1.00/1.80/2.50× throughput. Single-worker latency rises with concurrency; this is not a 2.50× reduction in per-worker computation.
- The diagnostic engine clone changed the engine hash, so one strict-reference suspect was correctly rejected by the environment guard. Its diagnostic result is excluded from equivalence acceptance. No guard was disabled. Candidate full validation reconstructs the retained source→representation→replay dependency graph and points each reference to its newly executed candidate parent; source/configuration/initialization and all oracle/replay checks remain the same. Initial conditions and replay evidence are compared against original artifacts in a separate check.
- Full candidate run completed in **359.400 s**, **1,887/1,887** matched, zero pending or signature/process mismatches: 183 source pairs, 1,686 representation pairs, 18 replay pairs. Levels: 183 cases/42.620 s, 1,687 cases/310.995 s, 17 cases/5.652 s. No new LLM calls. Detailed evidence comparison and regression results follow below.
- Full run used the frozen `candidate` snapshot, before reverting the ineffective reporter cache. The final snapshot is `final`; worker/process/oracle execution code is unchanged. Final verification repeated report/metric/ledger equivalence, 38 focused regressions, and a strict source→representation→replay chain (3/3 signatures and 6/6 saved evidence members equal). Final source hashes match the working tree.

## Measured results

All times below are seconds. Scaling rows are means of two uninstrumented runs on the same 32-CPU/64-GiB allocation. RSS is the sampled sum across controller/descendants, including repeated shared pages; it is not unique physical memory or GPU memory.

| Workers | 128-case wall | Speedup vs 8 | Worker p50 | Worker p95 | Worker time sum | Peak summed RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 76.046 | 1.00× | 4.445 | 4.945 | 586.742 | 7.01 |
| 16 | 42.297 | 1.80× | 4.795 | 5.529 | 624.461 | 13.51 |
| 32 | 30.384 | 2.50× | 5.909 | 6.451 | 741.329 | 23.66 |

The 32-worker configuration gives the best measured throughput, while each worker takes longer and cumulative worker time increases. Use `--workers 32` with sufficient allocated CPU/memory; keep the default of eight for the ordinary resource configuration. The full cohort reached 36.30 GiB summed RSS, above the stratified screen's peak.

| Controller operation | Baseline mean | Optimized mean | Reduction |
| --- | ---: | ---: | ---: |
| slogdet final report conversion/serialization/write | 0.495 | 0.379 | 23.4% |
| scatter_add final report conversion/serialization/write | 8.449 | 7.083 | 16.2% |
| slogdet 30 durable persist calls | 0.578 | 0.510 | 11.9% |
| scatter_add 30 durable persist calls | 1.895 | 1.679 | 11.4% |

Controller order was baseline, candidate, candidate, baseline. Both APIs' complete reports and half-row fixtures matched byte-for-byte; all computed metrics matched as objects. Both pipeline ledger hashes matched with a fixed clock. The half-row fixture retains other final-report metadata; it is a serialization/aggregation fixture, not a reconstructed historical wall-time sample. After reverting the reporter cache, final controller verification again matched all bytes/metrics/ledgers; final full-report writes were 0.384/7.001 s, consistent with the retained implementation. No intermediate write was omitted.

The full fixed execution cohort took 359.400 s; cumulative subprocess wall time was 10,958.554 s. Recorded worker CPU sums were 3,665.905 user seconds and 2,006.215 system seconds, recorded before final interpreter cleanup. Numerical audit time summed to 36.198 s and is already included in worker time. Exact evidence checking was a separate 93.861 s validation. LLM calls and LLM generation time were both zero. These overlapping/cumulative durations are not added together as elapsed time. This fixed-work driver does not reproduce the entire adaptive controller/LLM loop, so no new live-search end-to-end acceleration is claimed.

### Worker cost diagnosis

An external wrapper around the unchanged engine measured one pass and one strict-reference suspect, both matching original signatures:

| Phase | Pass | Suspect |
| --- | ---: | ---: |
| Python process startup to worker main | 0.107 | 0.101 |
| Main, including PyTorch initialization/execution/oracle/evidence | 3.464 | 3.564 |
| Main return to final registered atexit callback | 0.007 | 0.008 |
| Callback to parent wait return | 0.542 | 0.599 |
| Total | 4.120 | 4.272 |

The last interval includes Python teardown and parent wait observation; it is not pure cleanup CPU time. Separate cProfile diagnostics on the pass case attributed 4.196 s of a 4.736 s main to cumulative imports. Direct torch import took 2.240 s; stateful execution took 2.335 s **including** 1.816 s in deterministic setup, which triggered additional compiler-module imports. Audit and torch-save wrappers measured 0.017/0.035 s. These instrumented timings overlap and are not production latencies. The scatter_add report profile made 9,076,812 converter calls, explaining why avoiding repeated scalar type tests helped. Normal worker exit, process isolation and all evidence saves are retained.

### Quality and completeness

The following counts include the 18 automatic replay executions in slogdet. Thus its 36 suspected outcomes include 18 original suspects plus their replays, not 36 distinct defects. Exceptions are retained as exceptions rather than classified as proven API misuse or framework defects.

| Metric | slogdet | scatter_add |
| --- | ---: | ---: |
| Execution pairs / matching original classifications | 400 / 400 | 1487 / 1487 |
| Completed / runtime exception | 399 / 1 | 1462 / 25 |
| Verified source validity / denominator of all pairs | 395 / 400 | 1462 / 1487 |
| Pass / Inapplicable / Suspected / Invalid verdict | 40 / 320 / 36 / 3 | 94 / 1368 / 0 / 0 |
| Verified target-state attainment | 395 | 1462 |
| Verified backward execution | 398 | 1462 |
| Numerical audit agreement / violation / unknown (members) | 491 / 36 / 268 | 2514 / 0 / 410 |
| Unknown: bounded schedule did not select | 73 | 120 |
| Unknown: original execution/state/path not verified | 155 | 290 |
| Unknown: unresolved numerical/stability evidence | 40 | 0 |
| Representation children processed | 321 | 1365 |
| Representation children attained / not verified | 166 / 155 | 1075 / 290 |
| Strict replay source and followup evidence valid | 18 / 18 | No retained replay cases |

All these classifications, target paths, coverage signatures, mutation obligations and audit status/reasons match original saved cases. No timeout/crash/OOM outcome was introduced. The denominator is the retained execution cohort, not all generated programs or all theoretically possible representations. Untested/ineligible candidates and generation failures were not silently counted as new executions.

Saved tensor/training/initial-state evidence comparisons: **3655 exactly equal, 0 unequal, 64 unassessed due to NaN/Inf, 55 absent members**. Nonfinite evidence remains unknown under the existing comparator; missing members have their separately matched process/classification evidence. The new work does not upgrade unknowns to correctness passes or reconfirm distinct framework defects.

### Validation and cleanup

- Broader frozen-candidate suite: **96 tests passed**, 278.994 s (pipeline, artifacts, CLI, adaptive, exploration).
- After reporter-cache removal: **38 focused tests passed**, 11.445 s; final source/representation/replay smoke passed 3/3 classifications and 6/6 exact saved-member comparisons. Existing cancellation, multiple-in-flight recovery, shared worker limit and feedback/budget tests remain covered.
- JSON regression covers special dataclass/Enum values, shared references followed by mutation, exact bytes, invalid keys, nonfinite numbers, unsupported objects and cyclic input; failed conversion leaves existing output intact. Recovery asserts both saved traversal ledgers agree.
- `git diff --check`, Python AST checks and `bash -n` passed. Scheduler preflight and actual execution are recorded separately.
- Job **783115** and steps **.0–.3**, batch and extern are **COMPLETED, exit 0:0**. The attached terminal closed. The batch EXIT trap removed job scratch; the 150,553,374-byte default PyTorch compilation cache was separately inventoried and removed only after proving its directory birth and every entry belonged to this task/UID and postdated setup. Older shared Triton cache was retained. The batch session environment now carries the cache variables for future attachments.
- Slurm elapsed allocation: **1847 s**, 32 CPUs, 64 GiB, 3 GPUs. This is **59,104 allocated CPU-seconds** and **5,541 allocated GPU-seconds** (1.539 GPU-hours), including analysis/idle time between steps. It is not CPU/GPU utilization. No model service was started; startup time/restart count are zero. Unrelated jobs were untouched.

Artifacts: `result/worker-overhead/comparison.json`, `protocol.json`, `cohort.json`, baseline/candidate/final engine inventories, and `result/worker-overhead/783115/` with raw screens, profiles, full execution/evidence comparisons, final acceptance and regression logs. No persistent service or owned allocation remains.

## Context overflow proposal (not implemented)

The saved scatter_add acceptance recorded 153 provider requests: 66 completed, 86 HTTP 400 context-limit failures and one budget timeout. The first failing planning request had approximately 58,282 message characters; the server reported at least 16,385 input tokens. Characters are diagnostic sizes, not token estimates. Actual prompt assembly is in `src/mirage/generator/http_provider.py::_payload`, including API documentation, serialized context and optional assistant/repair messages. There is currently no input-token admission check.

1. Count the **final messages rendered with the service's actual tokenizer and chat template**, including repair turns and generation markers. The installed vLLM service supports chat `/tokenize`; a matching local tokenizer is also possible after checking parity against the service. Reserve the requested output allowance and a recorded safety margin before sending inference. With the present 16,384 context and 4,096 output limits, a proposed 256-token margin leaves 12,032 input tokens. Measure tokenizer/admission overhead separately from generation.
2. First apply lossless compaction: compact JSON; factor identical API/backend/checkpoint/context/environment/state fields shared by coverage entries; reference stable evidence IDs within a self-contained prompt. In the first failed request, factoring common fields across seven known-coverage dictionaries would remove 5,322 characters before tokenization. These fields are dictionaries, not double-encoded JSON strings. Remove execution-only bookkeeping such as `oracle_seen_units` from the **prompt copy** while keeping complete execution configurations and artifacts unchanged.
3. Give optional history a token budget. Deduplicate the overlap between completed feedback, representation feedback and known coverage. Preserve the canonical source, API legality constraints, MR Preconditions/Input Relation, required initial conditions, selected predicates/checkpoints, actual operand and peer paths, backward obligations and optimizer companion facts. Do not truncate source code or drop a required witness. Keep the complete evidence in artifacts, with snapshot version, selected IDs and omitted counts recorded for each request.
4. If mandatory material still exceeds the budget, partition the goal catalog using dependency closure: each group retains required companions, cross-family conjunction options and matching operand/optimizer witnesses. Rotate groups fairly, recording offered/examined/omitted goal IDs and denominators. A goal absent from a prompt is neither examined nor unreachable. This changes planning presentation and requires separate implementation approval; it is not part of the current runtime optimization.
5. If one mandatory group cannot fit, record a local context-budget failure with field-level token costs instead of repeatedly submitting an identical oversized request. A larger context may be an explicitly measured fallback, but does not bound growing feedback by itself.

Future validation: replay all 86 saved failed prompt constructions through token admission without inference; verify exact token boundaries including repair messages and output allowance; reconstruct losslessly factored facts; assert required goal IDs/witnesses/companions survive; then assess generation legality, runtime state attainment, MR applicability and explicit coverage denominators on an approved model run. Token fit alone is not test quality. Report before/after prompt tokens, compaction and admission time, retained/omitted facts, overflow/HTTP failure counts and successful-generation time. No prompts or model policies changed in this task.
