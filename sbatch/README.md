# Nano4 Slurm jobs

Run computational work through these categorized scripts. They reject execution
without a Slurm allocation, load `"$HOME/.bashrc"`, use `srun`, preserve nonzero
failures, and print start/end timestamps, elapsed seconds and exit status.

| Script | Purpose | GPUs |
| --- | --- | --- |
| `serving/prepare.sbatch` | Prepare a separate vLLM environment, kernels and a fixed model snapshot | 1 |
| `validation/provider_scheduler.sbatch` | HTTP stub, scheduler, process cancellation and CLI tests; no model required | 1 |
| `validation/full_suite.sbatch` | Full suite with CUDA, compilation and end-to-end tests | 1 |
| `validation/smoke.sbatch` | CUDA reference-provider case and LLM-free replay | 1 |
| `validation/vllm_smoke.sbatch` | Real-model single-worker case and replay after serving stops | 2 |
| `validation/slurm_steps.sbatch` | Concurrent step/GPU isolation with PyTorch and no model weights | 2 |
| `experiments/run.sbatch` | Exploration with an explicit provider, including an existing HTTP endpoint | 1 |
| `experiments/parallel.sbatch` | Qwen3-Coder-Next-FP8 serving and parallel Mirage cases in one allocation | 2 |
| `replay/replay.sbatch` | Saved-case replay or manual minimization | 1 |

All jobs retain the supplied account `ACD115198`, partition `8gpus`, one node,
eight allocated CPUs and four-hour limit. The integrated serving jobs explicitly
request two GPUs; each concurrent step gets four CPUs, one GPU and half the
allocated host memory. Both per-node GPU count and step memory must override
inherited job defaults, or the other step can remain pending. No partition
or resource fallback occurs automatically. Four hours is an upper bound; jobs
finish when their work ends.

## Submission environment

Submit from the repository root. Slurm opens log files before the script runs,
so create the parent directories first:

```bash
cd /home/asciibase64/Aubadia/Mirage
mkdir -p result/validation/logs result/experiments/logs result/replay/logs
export MIRAGE_PYTHON="$PWD/.venv/bin/python"
```

`MIRAGE_PYTHON` must be a compute-node-accessible Python >=3.14 environment with
the PyTorch version under test. The project declares PyTorch as a dependency;
`uv sync` does not install vLLM or download model weights. Experiment and validation
jobs do not install dependencies; `serving/prepare.sbatch` explicitly prepares them.
No job installs Python. For explicitly prepared external dependencies/provider
modules, set `MIRAGE_EXTRA_PYTHONPATH`. An incidental login-node `/tmp` package
installation is not a shared environment. Settings must survive `.bashrc`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MIRAGE_PROJECT_ROOT` | `SLURM_SUBMIT_DIR` | Repository root |
| `MIRAGE_PYTHON` | `<root>/.venv/bin/python` | Testing interpreter, inherited by case subprocesses |
| `MIRAGE_EXTRA_PYTHONPATH` | Empty | Explicit shared dependency/provider directories |
| `MIRAGE_OUTPUT_ROOT` | `<root>/result/<category>` | Output parent; each job uses `<parent>/<job-id>/` |
| `MIRAGE_PROVIDER` | Required by `run.sbatch` | `vllm`, `module:factory`, or explicit `reference` fixture |
| `MIRAGE_TIME_LIMIT` | `13800` | Requested batch seconds, capped by allocation time remaining |
| `MIRAGE_WORKER_TIMEOUT` | `600` | Per-process limit, also bounded by the batch deadline |

Exploration retains 300 seconds for saving evidence and cleanup. Replay divides
its remaining budget between two sequential execution cases. Source manifests
must be writable because replay creates immutable child artifacts beside them.

## vLLM serving and parallel testing

To prepare the environment and weights using the existing Python:

```bash
mkdir -p result/serving/logs
sbatch sbatch/serving/prepare.sbatch
```

The preparation job installs into
`/work/asciibase64/Mirage/environments/vllm/qwen3-coder-next/`, prepares the
matching FlashInfer kernels, and resolves a model commit before downloading.
An optional `MIRAGE_MODEL_REVISION` pins a specific revision on preparation or retry.
It records `serving-requirements.txt`, `preparation.json`, and, after successful
download and shard checks, `serving.env` in `result/serving/<job-id>/`.

```bash
source result/serving/PREPARE_JOB_ID/serving.env
export MIRAGE_PYTHON="$PWD/.venv/bin/python"
sbatch sbatch/validation/vllm_smoke.sbatch
```

The preparation job needs network access to package indexes and Hugging Face.
It permits downloads explicitly; the experiment jobs below remain offline for
model loading. Installation failures retain nonzero exit codes and logs.

Prepare a separate serving environment so vLLM's dependencies do not replace
the PyTorch being tested. Select an existing compatible Python; no script installs
Python. The serving environment must have a `vllm` executable beside its Python.

```bash
export MIRAGE_SERVE_PYTHON=/work/asciibase64/Mirage/environments/vllm/ENV/bin/python
export MIRAGE_MODEL_PATH=/work/asciibase64/Mirage/models/Qwen3-Coder-Next-FP8/REVISION
export MIRAGE_MODEL_REVISION=REVISION

sbatch --test-only sbatch/experiments/parallel.sbatch
sbatch sbatch/validation/vllm_smoke.sbatch
sbatch sbatch/experiments/parallel.sbatch --apis torch.matmul --cases 10
```

Paths above are placeholders for prepared installations. The job requires a
local model directory under `/work/asciibase64/Mirage/models/` with `config.json`.
`MIRAGE_MODEL_REVISION` records the declared snapshot revision; it does not prove
weight provenance. Serving metadata additionally records the config hash and
package versions. Hugging Face/vLLM persistent caches are under
`/work/asciibase64/Mirage/cache/`. Offline mode prevents implicit downloads.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MIRAGE_SERVE_PYTHON` | Required | Separate serving interpreter |
| `MIRAGE_MODEL_PATH` | Required | Prepared local FP8 model snapshot |
| `MIRAGE_MODEL_REVISION` | Required | Declared snapshot revision |
| `MIRAGE_VLLM_MODEL` | `Qwen/Qwen3-Coder-Next-FP8` | Served/API model name |
| `MIRAGE_VLLM_PORT` | `8000` | Loopback port; must be unused |
| `MIRAGE_MAX_MODEL_LEN` | `16384` | Serving context limit |
| `MIRAGE_WORKERS` | `2` | Concurrent experiments sharing the testing GPU; smoke fixes this to 1 |
| `MIRAGE_GENERATION_CONCURRENCY` | `2` | Client generation slots and server maximum active sequences |
| `MIRAGE_SERVE_START_TIMEOUT` | `900` | Readiness seconds, capped by remaining allocation time |
| `MIRAGE_MAX_TOKENS` | `2048` | Maximum generated tokens per request |
| `MIRAGE_TEMPERATURE` | `0` | Sampling temperature |
| `MIRAGE_TOP_P` | `1` | Nucleus sampling setting |
| `MIRAGE_HTTP_TIMEOUT` | `120` | HTTP timeout, also bounded by process/batch deadlines |

The service binds loopback on the allocated node. It loads once, passes health
and model-ID checks, then serves the testing step. Slurm owns GPU visibility;
the scripts never select physical GPU indices. Actual GPU UUIDs must differ.
The serving process is stopped on completion, cancellation or failure. A model
that cannot fit or load is reported as a failure; resources are not silently
increased. There is no A/B model switching.

An existing compatible endpoint can instead be used with `run.sbatch`:

```bash
export MIRAGE_PROVIDER=vllm
export MIRAGE_VLLM_BASE_URL=http://HOST:PORT/v1
sbatch sbatch/experiments/run.sbatch --workers 2 --generation-concurrency 2 --cases 10
```

In that mode, the external endpoint's lifecycle belongs to its owner. Mirage
stops new work when cancellation is unconfirmed, but does not stop an externally
managed service. Each request records `max_tokens`, sampling seed and schema;
input plus output must fit the configured context. Context errors are retained
rather than silently truncating the testing contract.

## Validation and replay

```bash
bash -n sbatch/lib/serving.sh
sbatch --test-only sbatch/validation/full_suite.sbatch
sbatch sbatch/validation/provider_scheduler.sbatch
sbatch sbatch/validation/slurm_steps.sbatch
sbatch sbatch/validation/full_suite.sbatch
```

The control suite uses a deterministic HTTP stub, not an LLM. The full suite
enables `MIRAGE_TEST_COMPILE`, `MIRAGE_TEST_CUDA` and `MIRAGE_TEST_E2E`, including
real PyTorch execution through the HTTP stub with one and two workers. Missing
requirements are reported as failures or explicit skips, never as passed checks.

```bash
export MIRAGE_PROVIDER=reference
sbatch sbatch/experiments/run.sbatch --cases 1
sbatch sbatch/replay/replay.sbatch result/experiments/JOB_ID/cases/EXPERIMENT.json
sbatch sbatch/replay/replay.sbatch result/experiments/JOB_ID/cases/EXPERIMENT.json \
  --size 2 --updates 1 --iteration 0
```

`reference` is an explicit fixture, not a fallback model. To validate a saved
case with nightly, select a prepared nightly interpreter and pass
`--allow-environment-change` to replay. Replay reports zero provider calls and
preserves the original evidence.

## Logs and monitoring

Application outputs are `result/<category>/<job-id>/`; scheduler logs are
`result/<category>/logs/<job-name>-<job-id>.out` and `.err`. Integrated jobs keep
service stdout/stderr and metadata in their run's `serving/` directory. Disposable
compilation caches use job-specific local scratch. Large persistent caches and
weights belong under `/work/asciibase64/Mirage/`, not the repository or home cache.

`MIRAGE_OUTPUT_ROOT` changes application output only. To move scheduler logs,
pass explicit `sbatch --output`/`--error` paths with existing parent directories.
Managed CLI options cannot override the integrated job's provider, device,
output or time budget. Use documented environment variables instead.

```bash
squeue --user="$USER"
sacct --jobs=JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

Syntax/preflight success is not an executed job. Submission success is not test
success. Even a completed experiment job is not an MR Pass: inspect its saved
verdicts, applicability and evidence. Service/cancellation failures return nonzero
status; suspected MR violations remain findings requiring reproduction.
