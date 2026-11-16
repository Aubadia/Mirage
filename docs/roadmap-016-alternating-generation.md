# Roadmap 016: Alternating generation and framework execution

Status: completed. One complete alternating campaign was measured after a
preserved ENOSPC-interrupted attempt; the historical pipeline arm was not rerun.
The experiment shows mixed quality outcomes and longer E2E, so defaults remain
unchanged. Context-overflow remediation remains explanation-only.

## Approved scope

The user confirmed that the alternating mode must be measured, while the existing
pipeline results must be reused without rerunning that arm. The approved plan
keeps framework worker parallelism and changes only model/framework overlap:
finish generated-source testing, representation traversal and required replay,
publish completed feedback, then construct the next generation request.

Use the completed roadmap-015 candidate as the reference: ten APIs, 32 workers,
seed 0, 180 cumulative model-request seconds per API, unchanged prepared
Qwen3.8-27B BF16/MTP3, CPU/CUDA roster, mixed backward policy, prompts, numerical
oracles and finite full traversal. The product default remains eight workers.
Preserve the reference batch_norm budget-gate failure and all failed requests;
neither rerun the reference nor widen its acceptance threshold.

Authorized source files: src/mirage/runtime/pipeline.py,
src/mirage/workflow.py, src/mirage/contracts/programs.py, src/mirage/cli.py,
benchmarks/e2e_latency.py, tests/test_pipeline.py, tests/test_cli.py,
tests/test_artifacts.py, README.md, and this roadmap. Experiment scripts,
snapshots, analyses and outputs belong under result/pipeline-alternating/.
Reuse the existing categorized Slurm launcher without editing it; its ordinary
job-specific session/preflight/serving records remain under result/e2e-latency/.
Preserve all pre-existing worktree changes. This roadmap is the first edit after
execution approval. Context-overflow remediation is explanation-only here.

## Implementation

1. Add an explicit generation-overlap option, enabled by default. Disabling it
   retains model-only budgets and complete traversal; the existing legacy
   no-generation-pipeline switch retains its current distinct semantics.
2. Await all pending/dispatched framework work before selecting the next proposal
   and building its prompt. Publish completed evidence before selection. Guard
   provider admission as well; preserve framework concurrency, worker isolation,
   failures, cancellation and durable recovery. Persist the scheduling mode;
   old ledgers default to their original overlapping behavior.
3. Record the scheduling mode in options, protocol, timing and durable state.
   Extend the frozen benchmark driver without rewriting historical protocols.
4. Validate ordering, feedback causality, no overlap, budgets, shared worker cap,
   cancellation/resume, CLI and protocol compatibility. Freeze the final engine,
   driver, tests and protocol before any real generation.
5. Execute one fresh alternating campaign using the same 32-worker resource
   configuration as the reference. Use Slurm, one healthy service and separate
   serving/testing steps; record startup, iterations, allocation cost and cleanup.
6. Compare LLM-generated source coverage separately from representation children,
   LLM validity and all-case validity with explicit denominators, parent E2E,
   coverage sets, suspected outcomes and replay evidence. Reproduce/corroborate
   new mechanisms if needed; no distinct-defect claim from case counts alone.

## Validation and interpretation

- Lightweight AST, shell syntax and diff checks precede service startup.
- Focused tests and real-framework controls run under Slurm on frozen snapshots.
- Actual model/framework intervals must not overlap in the alternating arm;
  no pending/dispatched candidates may remain at successful drain completion.
- Same seed and model do not establish identical adaptive trajectories. A fresh
  service/node/cache and the preserved reference budget exception limit causal
  and statistical claims; report them explicitly.
- Read-only reference extraction found 795 verified LLM-source semantic units,
  49 representation projections, 83 training-state projections, 367 valid / 531
  generated programs, 5561 valid / 6236 total attempts, and 12 slogdet suspects
  with 12 valid automatic replays. Reference makespan is 7653.744682 seconds.
- Fixed-work timing arithmetic (not an experiment) adds 580.943372 seconds of
  observed model/framework overlap, giving 8234.688054 seconds. Actual workload
  and generation quality may change with feedback timing.
- Stop task-owned test/service processes and release the allocation before
  completion; verify terminal steps and scratch cleanup. Do not stop unrelated jobs.

## Progress

- Approval received; existing scheduler, provider, feedback construction,
  benchmark driver and reusable Slurm session inspected. No new model execution
  has begun. Existing no-pipeline mode was confirmed to change budget semantics,
  so it cannot serve as the isolated scheduling ablation.
- Implemented `--no-generation-overlap` with model-only budgets and the existing
  framework worker limit. The capacity barrier runs before proposal selection and
  again at provider admission, waits for traversal/replay completion, and retains
  cancellation and failure propagation. The mode is saved and restored; absent
  legacy fields default to overlapping behavior. Added focused ordering, feedback,
  cancellation, recovery, CLI and frozen-protocol regressions.
- Eight changed Python files passed AST parsing; diff whitespace and unchanged
  launcher shell syntax checks passed. Frozen development snapshot is
  `result/pipeline-alternating/development-01/`.
- Both scheduler preflights passed. Owned job **786574** uses explicit dev/4h,
  36 CPUs/3 GPUs/192 GiB, matching the reference resource configuration. Excluded
  25a-hgpn009 because its prior service startup failed; actual node is 25a-hgpn004.
  Session outputs use the launcher's existing `MIRAGE_OUTPUT_ROOT` override under
  `result/pipeline-alternating/sessions/786574/`. Pre-model validation passed
  **25 tests in 4.855 seconds**; serving startup and full focused validation follow.
- Service startup completed in **189 seconds**, with zero restarts. Separate
  testing step uses 32 CPUs/one GPU/98304 MiB and task-local compilation caches.
  Frozen pipeline, CLI, artifacts and adaptive suites passed **100 tests in
  246.290 seconds** (247 seconds wrapper), including two real PyTorch rounds with
  completed representation feedback before request two and exactly zero overlap.
- Froze `alternating-01/protocol.json`, engine, driver and analysis helpers in
  `final-lock.json`. All reference model, roster, budget, path, oracle and worker
  controls match. Exactly four production Python files differ from the reference:
  CLI, options, workflow wiring and scheduler; prompts/oracle/report aggregation
  are unchanged. The single new alternating campaign is ready to start in the
  existing healthy allocation; the reference arm will not be rerun.
- The first alternating attempt stopped during scatter_add because `/home`
  returned **ENOSPC**, including when the shell tried to save the exit status.
  The campaign process exit was **120**; its ledger retains **94.001903 model
  seconds / 69 requests**, 100 source proposals, and no pending/dispatched
  representation work. `summary.json` is truncated at 33554432 bytes; the campaign
  metadata still says running because final persistence failed. This is an
  incomplete infrastructure failure, not a completed comparison or oracle failure.
- Confirmed the service running/waiting gauges were both zero twice, closed the
  testing terminal, stopped the owned service, and verified job **786574** and all
  steps terminal. Allocation elapsed **1973 seconds**; startup **189 seconds**;
  no in-session service restart. The batch's cache-removed marker exists. Batch
  completion does not override the preserved campaign failure.
- Relocated only this task's artifact tree to
  `/work/asciibase64/Mirage/experiments/pipeline-alternating-786574/artifacts`,
  retaining `result/pipeline-alternating` as a symlink. **14297 files / 596718414
  bytes** were verified before removing the duplicate home copy. Original failed
  artifacts, frozen engine and prior pipeline reference remain intact.
- A complete alternating attempt must use the writable `/work` output location.
  Preserve the partial attempt and charge its resource cost; a fresh complete
  retry addresses storage failure, not an unfavorable test outcome. The reference
  arm is still not rerun. Storage placement now differs from the historical
  reference, so E2E differences cannot be attributed solely to scheduling. Keep
  this limitation explicit alongside node/cache and budget-gate differences.
- Retry job **786736** uses the same explicit dev/4h, 36 CPUs/3 GPUs/192 GiB
  allocation on 25a-hgpn008. Its preflight passed **25 tests in 3.725 seconds**;
  service startup took **187 seconds**, with zero in-session restarts. The frozen
  retry engine and driver hashes equal the previously validated versions.
- Reconstructed the interrupted run's report in `partial-recovery-01` from saved
  outcomes and the ledger inside the Slurm testing step. Recovery exited zero;
  model and worker execution were prohibited by guards. The original truncated
  report remains unchanged. This recovered metadata is partial evidence only.
- Started the fresh `alternating-02` campaign against the healthy retry service.
  Separate testing and inference steps retain explicit CPU/GPU assignments. All
  outputs now use the verified `/work` artifact directory.
- The retry has completed the first seven API children with E2E times of
  1427.706, 726.109, 1875.270, 567.084, 361.059, 555.582 and 518.583 seconds,
  respectively. Each preserved exit status 1, and independent boundary checks
  confirmed service idle before proceeding. Full quality and integrity analysis
  remains pending until the campaign ends, avoiding competing analysis workload
  during live timing. The healthy service has not been restarted.
- Overflow remains observable under alternating execution: scatter_add's first
  HTTP 400 was provider-018, reporting at least 16385 input tokens against the
  16384 limit. Later lu_solve requests mix fast HTTP 400 failures with completed
  generation and substantial framework work. This illustrates why model-request
  time alone does not bound E2E when failures return quickly; remediation remains
  proposal-only in this comparison.

## Final measurement and review

- All ten API children completed their bounded work; campaign E2E was
  **9499.971081 seconds (158.332851 minutes)**, versus the reused pipeline's
  **7653.744682 seconds (127.562411 minutes)**: **+24.12%**. The complete campaign
  retains `complete_with_failures`, child exits 1, and
  `provider_cancellation_unconfirmed`; independent service gauges were idle at
  every API boundary. These statuses are not converted into generation success.
- LLM source semantic/representation/training-state units were **795/49/83 →
  803/52/68**. Semantic sets share 473 units, with 330 gained and 322 lost;
  training-state sets lose 16 and gain one. Counts do not establish exhaustive
  coverage or a uniform quality improvement.
- LLM generated/runtime-valid counts were **531/367 → 559/390**; total
  attempts/verified-valid were **6236/5561 → 6806/6145**. The generation funnel
  and framework-derived cases retain separate denominators. LLM valid per
  campaign minute falls from 2.877 to 2.463.
- All alternating integrity checks pass, including the original per-API model
  budget tolerance, complete candidate drain, feedback causality, immutable
  engine, worker limits, source-child ordering, and **zero** inner/outer model
  interval overlap with framework work. Reference overlap was 580.943372 seconds;
  its batch_norm budget failure remains unchanged. All ten initial request
  payloads and response contents match; later adaptive trajectories differ.
- Suspected slogdet cases and valid no-LLM automatic replays are **12/12 →
  16/16**. Posthoc same-call attribution matches the known negative-view backward
  mechanism, but reuses production VJP numerical evidence and is not another
  independent confirmation or a distinct-defect count.
- The posthoc evaluator additionally labeled 27 batch_norm cases as FN. A bounded
  Slurm diagnostic of all 54 members found one source-program hash and the same
  near-zero flagged element: absolute error **3.413788e-9** at output about
  **-2.5e-6**. Decimal80 scalar references and ordinary float32/float64 controls
  support cancellation amplified by a pure output-relative threshold. The
  ordinary source control reproduces the saved output; 47/54 full member outputs
  match that control bitwise, while the other seven retain small layout-dependent
  rounding differences. Maximum saved float32 term-scaled error is 0.941182
  epsilon units. Preserve all original FN labels, but do not claim 27 established
  misses or derive recall from them. No production oracle/tolerance changed.
- Context/token-limit failures remain **342 → 310**. The explanation proposes
  exact final-message token admission, lossless factoring, bounded optional
  history, dependency-preserving goal partitions, and bounded failure handling.
  No prompt, model context setting, or generation policy was changed here.
- Validation: 100 focused tests passed in 246.290 seconds; startup preflights
  passed 25 tests in each allocation; analysis, integrity, comparison, attribution
  and numeric review all exited zero. All 51 production Python files match the
  validated frozen engine; frozen helper/protocol hashes and diff checks pass.
- This is one historical arm and one complete new arm, with different nodes,
  service/cache histories, adaptive workloads and storage locations. E2E cannot
  be attributed solely to scheduling. The ENOSPC partial attempt remains outside
  the paired comparison and inside resource accounting.

## Cleanup and artifacts

Both owned jobs **786574** and **786736**, and all steps, are terminal. Testing
terminals are closed; both inference services were intentionally stopped after
idle confirmation. Both exact scratch paths have `cache-removed` evidence.
Allocated job seconds are **1973 + 11238 = 13211**, or **11.009167 allocated
GPU-hours / 132.11 CPU-hours**. This includes failed-attempt, validation, startup,
idle/review, analysis and cleanup costs; it is not GPU utilization. Healthy
startup times were 189/187 seconds, with zero in-session restarts.

- [Full report](../result/pipeline-alternating/report.md)
- [Machine-readable comparison](../result/pipeline-alternating/comparison.json)
- [Integrity checks](../result/pipeline-alternating/alternating-02/integrity.json)
- [Posthoc findings and numerical review](../result/pipeline-alternating/findings.md)
- [Context-overflow proposal](../result/pipeline-alternating/context-overflow.md)
- [Resource accounting](../result/pipeline-alternating/resource-accounting.json)
