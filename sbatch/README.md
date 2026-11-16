# Nano4 jobs

All computational work runs through Slurm. Default resources are account
`ACD115198`, partition `8gpus`, one node, eight CPUs and four hours. Validation
without a model uses one GPU. Model-serving development and parallel experiments
explicitly request two GPUs: one for vLLM and one for testing, with four CPUs and
half the allocated host memory per step. Do not override Slurm GPU visibility.

## Prepared environments

Use the existing project `.venv/bin/python` for PyTorch tests. Select a different
prepared interpreter with `MIRAGE_PYTHON`; no job installs Python automatically.
The current prepared model configuration is:

```bash
source result/serving/761235/serving.env
```

It records `MIRAGE_SERVE_PYTHON`, `MIRAGE_MODEL_PATH`, `MIRAGE_MODEL_REVISION`, and
`MIRAGE_VLLM_MODEL`. The environment and model snapshot live under
`/work/asciibase64/Mirage/`. For a new installation, review the explicit
`serving/prepare.sbatch` job; preparation is separate from experimentation.

Serving defaults: localhost port 8000, context length 16384, maximum output tokens
4096, and generation concurrency 2. Override with `MIRAGE_VLLM_PORT`,
`MIRAGE_MAX_MODEL_LEN`, `MIRAGE_MAX_TOKENS`, and `MIRAGE_GENERATION_CONCURRENCY`.
`MIRAGE_SERVE_START_TIMEOUT` bounds readiness waiting. An external prepared endpoint
can be used with `MIRAGE_PROVIDER=vllm` and `MIRAGE_VLLM_BASE_URL` via
`experiments/run.sbatch`.

## Reusable interactive development

Prepare directories and validate submission before starting a session:

```bash
mkdir -p result/validation/logs
bash -n sbatch/validation/dev_session.sbatch sbatch/lib/development.sh
sbatch --test-only sbatch/validation/dev_session.sbatch
sbatch --parsable sbatch/validation/dev_session.sbatch
```

Wait for `result/validation/JOB_ID/ready`, then attach a testing terminal. Replace
JOB_ID and use the per-step memory value recorded in `session.env`:

```bash
srun --jobid=JOB_ID --exclusive --exact --ntasks=1 --cpus-per-task=4 \
  --gpus-per-node=1 --gpus-per-task=1 --mem=204800M --pty bash
source result/validation/JOB_ID/session.env
source sbatch/lib/development.sh
mirage_iteration -m unittest discover -s "$MIRAGE_PROJECT_ROOT/tests" -v
mirage_iteration -m mirage run --provider vllm --device cuda \
  --canonicals-per-api 1 --targets-per-canonical 2 --paths-per-target 1 \
  --cases 32 --workers 2 --output "$MIRAGE_SESSION_DIR/cases"
```

`mirage_iteration` creates a fresh source/test/script snapshot and separate logs,
runs a new Python process, preserves the exit status, and bounds its runtime by
allocation time remaining. A normal failing test does not stop the service. After
editing the helper itself, source it again. An unconfirmed inference cancellation
blocks reuse until the service is restarted; a closed HTTP connection is not proof
that inference stopped.

When the user-requested task is finished, ensure all test processes have ended,
remove the job-specific testing cache recorded in `test-cache-path.txt` if present,
then stop the owning job and close the attached terminal:

```bash
touch "$MIRAGE_SESSION_DIR/stop"
exit
```

Confirm cleanup using `squeue -j JOB_ID` and `sacct -j JOB_ID`. Closing the terminal
alone does not stop vLLM. Use `scancel JOB_ID` for an interrupted owned session;
do not cancel unrelated jobs. The batch time limit and termination trap bound the
service lifetime. Preserve logs if cleanup cannot be verified.

## Other jobs

- `validation/full_suite.sbatch`: current complete test suite.
- `validation/smoke.sbatch`: runtime and oracle smoke tests, without an LLM.
- `validation/generation_planning.sbatch`: program and exploration controller tests.
- `validation/provider_scheduler.sbatch`: HTTP, CLI, artifact and shell tests.
- `validation/vllm_smoke.sbatch`: small real-model generation batch.
- `validation/llm_graph_plans.sbatch`: 16-API real-model canonical acceptance batch.
- `experiments/parallel.sbatch`: model service plus a bounded exploration batch.
- `experiments/run.sbatch`: exploration against a supplied provider.
- `replay/replay.sbatch`: reproduce current-format artifacts without an LLM.

Create the corresponding `result/<category>/logs` directory before submission.
Scripts source `"$HOME/.bashrc"`, record start/end time and exit status, and retain
failures as nonzero statuses. `sbatch --test-only` is only preflight; neither
submission nor job completion establishes an MR Pass. Inspect actual artifacts.

## PyTorch correctness Phase 2–3 session

`experiments/pytorch_correctness.sbatch` explicitly allocates two GPUs and reuses
one prepared vLLM service across validation and the ten-API evaluation. It creates
`result/pytorch-correctness/JOB_ID/session.env` and `ready`. Prepare logs, run
`bash -n`, then `sbatch --test-only` before submission. Attach a testing step as
above and source that session's environment plus `sbatch/lib/development.sh`.

```bash
mkdir -p result/pytorch-correctness/logs
bash -n sbatch/experiments/pytorch_correctness.sbatch
sbatch --test-only sbatch/experiments/pytorch_correctness.sbatch
sbatch --parsable sbatch/experiments/pytorch_correctness.sbatch
```

Inside the testing step, enable full validation with `MIRAGE_TEST_COMPILE=1`,
`MIRAGE_TEST_CUDA=1`, and `MIRAGE_TEST_E2E=1`. Run the integration/evaluation commands
in `benchmarks/pytorch_correctness/README.md`. Set `MIRAGE_ITERATION_TIMEOUT=3600`
for ten sequential 300-second API budgets; the helper still respects allocation
time remaining and reserves cleanup time. It snapshots benchmark scripts as well
as production code, tests and batch helpers. Leave the service healthy after
ordinary test failures; stop it and verify all owned steps exit at task completion.

## Qwen3.8 Native MTP validation

Use `source result/benchmark-revalidation-qwen38/serving.env` for the prepared
Qwen3.8-27B BF16 snapshot. Set `MIRAGE_MTP_TOKENS=0` (default/off), `1`, `2` or `3`
before submitting a service session. Nonzero enables vLLM native `mtp` using the
model's own prediction head; it requires a compatible model and installed vLLM.
The serving environment records the setting. Prefix caching is explicitly enabled;
CUDA graph and attention choices remain vLLM's hardware/model-specific automatic
selection and must be verified from startup logs. Do not claim an acceleration
from flags alone. Compare identical structured-output requests, record speculative
acceptance, latency, tokens and JSON validity, then compare valid cases/second.
Changing MTP requires restarting the service; preserve per-configuration artifacts.

## Ten-API E2E latency session

`experiments/e2e_latency.sbatch` starts the prepared Qwen3.8-27B BF16/MTP3 service
and waits for an attached testing step. It runs frozen lightweight regressions
before loading the model. The explicit allocation is 36 CPUs, three GPUs and
192 GiB: four CPUs/one GPU for serving and 32 CPUs/one GPU for tests. The extra
allocated GPU satisfies Nano4's CPU/GPU ratio; allocation is not utilization.
Prepare `result/e2e-latency/logs`, check `bash -n`, and use `sbatch --test-only`
with the same partition/time overrides as the actual submission. Defaults are
8gpus and four hours; a dev submission is an explicit resource choice, subject
to its scheduler limits and account quota.

Set `MIRAGE_PREFLIGHT_ENGINE` to a prepared arm's absolute `engine` directory to
validate experimental code before model loading; otherwise preflight uses `src`.
An experimental candidate can remain outside production until live quality review.

Prepare a new frozen arm before execution (this only copies code and metadata):

```bash
python benchmarks/e2e_latency.py prepare --engine src \
  --output result/e2e-latency/MY_BASELINE --label baseline --workers 8
```

After the job's `ready` file appears, attach a separate testing step with the
recorded memory limit (98304 MiB for this allocation), then source its environment:

```bash
srun --jobid=JOB_ID --exclusive --exact --ntasks=1 --cpus-per-task=32 \
  --gpus-per-node=1 --gpus-per-task=1 --mem=98304M --pty bash --noprofile --norc
source result/e2e-latency/JOB_ID/session.env
"$MIRAGE_PYTHON" result/e2e-latency/MY_BASELINE/driver.py campaign \
  --protocol "$MIRAGE_PROJECT_ROOT/result/e2e-latency/MY_BASELINE/protocol.json"
```

The driver verifies frozen source hashes, records each API launch-to-exit E2E,
keeps failed requests and incomplete attempts explicit, and confirms empty serving
running/waiting gauges before proceeding between APIs. A model-budget timeout
can retain `provider_cancellation_unconfirmed` and a nonzero child status even
when the external monitor subsequently observes an idle service. This does not
turn the failed request into successful generation. Inspect each API's
`metrics.json`, process logs, final traversal and raw serving metrics; a complete
campaign must account for all ten API attempts. The current driver refuses an
automatic campaign rerun; interrupted generation budgets remain partial.

Use the same healthy service for fixed-case diagnostics and a separately frozen
candidate arm. Keep heavy diagnostics outside live latency intervals. Record
prefix-cache warmth when comparing real generation, and compare saved identical
work before claiming an optimization speedup. At task completion, finish testing,
exit the attached terminal, touch the session `stop` file, and verify the owned
job/steps have exited with `squeue` and `sacct`.

The launcher records `cache-path.txt` and writes `cache-removed` only after
verifying removal of that job's scratch directory. Preserve final idle samples,
terminal-step status and `sacct` allocation accounting. A successful batch exit
does not imply that the experiment's acceptance gates passed.

Roadmap 015 completed baseline job 785068 and candidate job 785647, with zero
in-session service restarts. Intermediate job 785575 failed service readiness
before candidate work; its cost remains included. All owned jobs and steps are
terminal. The [results](../result/e2e-latency/report.md) retain the candidate
`batch_norm` budget-gate failure and keep 32 workers experimental; eight remains
the product default. Resource accounting includes all startup/diagnostic/idle
time and must not be interpreted as GPU utilization.
