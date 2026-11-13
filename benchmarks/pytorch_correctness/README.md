# PyTorch correctness regression fixtures

These issue-derived fixtures are benchmark data, never generator prompts or general sampling templates. See candidates.json for upstream attribution and docs/pytorch-correctness-reproduction.md for reviewed outcomes.

Submit from the repository root after creating result/pytorch-correctness/logs:

    bash -n sbatch/replay/pytorch_correctness.sbatch
    sbatch --test-only sbatch/replay/pytorch_correctness.sbatch
    sbatch --parsable sbatch/replay/pytorch_correctness.sbatch

Optional runner arguments: --cases ISSUE ... --repetitions 1 --variants 0 --timeout 240.
The default is one fixed-input execution per selected issue, per user request. No selected case requires RNG or timing races. Every cell runs in a fresh process. Variant 0 is the minimized issue-derived fixed-input example; optional variant 1 changes values/shape/depth. No repeated runs or variants are required for admission. Results retain conditions, independent expected values, control, numerical policy, raw tensors, logs, environment and immutable code hashes. A control mismatch or nonfinite result is unresolved, not a bug. Exceptions and timeouts never count as reproduction.

Exit 0: no observed violation; 3: observed oracle violation; 2: unresolved execution/oracle condition. A nonzero Slurm job is expected when the benchmark detects violations; inspect summary.json and each result.json. Reproduced requires every requested cell to violate; the default has exactly one cell per issue. This automated status does not establish distinct root causes or confirm a framework defect without report review.

Full tensor evidence is in evidence.pt; JSON retains at most 64 values per tensor. Load trusted local evidence only. Computational inspection and reruns belong inside Slurm. `run.py` does not call an LLM; the separate `evaluate.py` experiment and its actual results are documented below.

## Framework integration and blind generation

`evaluate.py validate` routes the ten `INTEGRATION_CASES` through the production
property worker, recording actual target execution, applicability, controls and
oracle violations. `evaluate.py experiment` requires a successful ten-case
integration report, then passes only target API/device to production generation.
Post-hoc issue matching remains in this benchmark directory and never enters the
LLM prompt or general sampler.

Inside the allocated testing step (see `sbatch/README.md`):

```bash
mirage_iteration "$MIRAGE_PROJECT_ROOT/benchmarks/pytorch_correctness/evaluate.py" \
  validate --output "$MIRAGE_SESSION_DIR/phase2"
export MIRAGE_ITERATION_TIMEOUT=3600
mirage_iteration "$MIRAGE_PROJECT_ROOT/benchmarks/pytorch_correctness/evaluate.py" \
  experiment --phase2 "$MIRAGE_SESSION_DIR/phase2" \
  --output "$MIRAGE_SESSION_DIR/phase3" --budget 300
```

Use fresh output directories. Inspect per-case results, not only process exit codes.
Structural trigger matching is provisional; `evaluation.json` leaves independently
unadjudicated sensitivity/FPR fields null. The evaluation report documents actual
independent review and distinguishes generated trigger coverage from oracle ability.
Development smoke tests and benchmark replay are excluded from Phase 3 metrics.

Completed run: job **763298**, Phase 2 **10/10**, Phase 3 reviewed known issues
**3/10** from **1,154** candidates. See
[the evaluation report](../../docs/pytorch-correctness-evaluation.md) for exact
denominators, the post-hoc copy-matcher correction, the independent audit subset,
non-benchmark std candidates, and verified cleanup.
