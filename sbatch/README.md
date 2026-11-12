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
