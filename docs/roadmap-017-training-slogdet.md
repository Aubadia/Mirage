# Roadmap 017: Training v3 slogdet run

Status: completed. One frozen training batch finished; owned resources cleaned up.

## Approved scope

The user requested one run of the latest identified training delivery for
`torch.linalg.slogdet` (the typed `alogdet` is interpreted from the preceding
discussion). After clarification, the user changed the budget to 500 seconds
of batch wall time. This supersedes the proposed 180-second model-only budget;
no model-budget implementation or production-code changes are needed.

Use the unmodified training v3 delivery at Git commit `c954727`, previously
verified against all 34 Python files in the final training experiment snapshot.
Freeze source and configuration under `result/training-slogdet/`. Reuse the
existing categorized Slurm development-session launcher and prepared Python
environments. Do not edit production code or unrelated working-tree changes.

## Execution plan

1. Freeze commit c954727 and record file hashes, exact command and protocol.
2. Use Qwen3.8-27B revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0,
   native MTP3, context 16384, output limit 4096, temperature 0 and top-p 1.
3. Run one training batch on CUDA/FP32, seed 0, required target backward,
   500 seconds batch wall-time budget, two workers and two concurrent model
   requests. Retain training defaults of three canonicals, twelve targets per
   canonical, two paths per target and at most 100 case attempts. Early exhaustion
   of these finite candidates is reported rather than replaced with another run.
4. Keep model serving and testing in separate Slurm steps and GPUs. Use the
   existing two-GPU development allocation, eight CPUs, account ACD115198,
   default partition 8gpus and four-hour allocation bound as the starting
   resource template. Preserve nonzero test exits.
5. Inspect the actual report, failed attempts, state/path evidence and MR verdicts.
   Measure wrapper/batch time, provider usage and resource/startup costs separately.
6. Confirm no outstanding inference requests, stop owned processes/service,
   release the allocation, remove owned scratch and verify terminal job states.

## Validation and evidence

- Lightweight source-hash, AST and shell checks before model loading.
- `sbatch --test-only` checks submission feasibility; it is not an executed test.
- This single live run is the requested validation. Do not repair code, change
  prompts/oracles, or rerun a failed campaign to replace its result.
- Report canonical acceptance, case attempts/executions/verification, MR
  applicability and verdicts, states/paths/transitions, token usage, actual time
  and incomplete evidence. A Pass does not establish universal correctness.
- Model startup, queue time and allocation time are outside the 500-second
  training batch budget. Preserve any deadline/cancellation uncertainty.

## Progress

- Read the training CLI, controller, provider and existing Slurm helpers.
- Confirmed training v3 uses a shared batch wall deadline, not model-only time.
- No owned Slurm jobs were active at the initial scheduler check.
- Created `result/training-slogdet/20261117-500s/`; froze unmodified commit
  c9547274b5d9f533047c5ced88e7c2a597d3b929 and recorded source/launcher hashes.
  All 42 source/test Python files parsed; existing launch/helper shell checks pass.
- The default 8gpus/four-hour submission preflight accepted the request but
  estimated a substantially later start. Read the dev partition constraints
  (maximum four hours), and explicitly selected dev, 30 minutes, 192 GiB,
  eight CPUs and two GPUs. The 30-minute bound fits the existing 900-second
  service-start limit, 500-second batch and cleanup reserve. Matching dev
  `sbatch --test-only` passed. These are preflights, not executed jobs.
- Submitted job 788909 with those explicit overrides; it started on
  25a-hgpn035. Qwen3.8-27B/MTP3 readiness took 191 seconds. Separate serving
  and testing GPUs were verified before attaching testing step 788909.3.
- Created a measurement-only `run-once.py` in the run artifact directory.
  It verifies the frozen files, launches the original training CLI exactly once,
  preserves its stdout/stderr and exit status, and records service queue gauges.
  It changes neither the training engine nor the 500-second wall budget.
- Pre-run service gauges independently showed zero running/waiting requests.
  The exact testing step is recorded in `execution-start.json`; job-local test
  scratch is recorded for cleanup. No model restart or prior task corpus reuse.
- The only campaign completed with CLI exit 0 and `stop_reason=completed`.
  Batch wall time was 293.988132 seconds (294.484622 seconds launch-to-exit),
  below the 500-second cap because all three canonical branches and 72 finite
  state/path attempts were exhausted. No replacement/extension batch was run.
- All three canonical target/backward paths verified. Of 72 case attempts,
  64 were dispatched and eight exhausted plan repair. There were 46 verified
  cases: 40 coverage-only and six MR-inapplicable; 18 case executions raised
  exceptions. No applicable MR, Pass or suspected failure was recorded.
  Verified coverage includes nine state configurations, four histories and
  twelve transition sequences, with the original bounded denominators retained.
- Provider calls/repairs: 106/31; all 106 HTTP responses were 200. Input/output
  tokens were 61,335/3,898, with complete usage. Measured HTTP-provider request
  time summed to 31.522366 seconds, including service/network/local handling
  across concurrent calls; it is neither pure inference nor additive wall time.
- Evidence inspection in the same testing Slurm step found all three canonical
  optimizers absent. Twelve update histories consequently failed; six shared
  storage cases lacked a compatible equal-valued partner. Frozen runtime
  operations are `linalg_slogdet` and `mean`; the capability catalog omits the
  former, so purity is uncertified and most selected MRs fall back to coverage.
  The six accumulation comparisons lack purity/batch-separability certificates.
  These limitations do not establish a framework defect or correctness.
- Before shutdown two independent service samples showed zero running/waiting
  requests. Testing terminal 788909.3 closed with exit 0; owned test and batch
  scratch were removed and checked on 25a-hgpn035. Job 788909 and all steps are
  terminal: batch/test exit 0:0; serving step deliberately cancelled (0:9).
  Allocation: 771 seconds, eight CPUs/two GPUs/192 GiB; 0.428333 allocated
  GPU-hours and 1.713333 allocated CPU-hours, not utilization.
- All frozen source/test hashes and the measurement runner hash remain intact.
  Final [report](../result/training-slogdet/20261117-500s/report.md),
  [metrics](../result/training-slogdet/20261117-500s/metrics.json),
  [canonical facts](../result/training-slogdet/20261117-500s/canonical-runtime-facts.json)
  and [cleanup](../result/training-slogdet/20261117-500s/cleanup.json) are saved.
