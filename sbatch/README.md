# Nano4 Slurm jobs

Run computational work through the categorized batch scripts. Do not execute
training, GPU/compile tests, full integration suites or large experiments directly
on the login/interactive host. The scripts reject execution without a Slurm job ID.

| Script | Purpose |
| --- | --- |
| `validation/smoke.sbatch` | One CUDA reference-provider case and LLM-free replay; both must pass |
| `validation/full_suite.sbatch` | All unittest tests with compile, CUDA and end-to-end checks enabled |
| `experiments/run.sbatch` | Coverage-guided exploration using an explicitly selected provider |
| `replay/replay.sbatch` | Replay, manual minimization or validation in a prepared nightly environment |

All scripts follow the supplied Nano4 template: account `ACD115198`, partition
`8gpus`, one node, eight CPUs per task, one GPU per node, and a four-hour limit.
They source `"$HOME/.bashrc"`, use `srun`, and print start/end timestamps, elapsed
seconds and exit status. Shared setup is in `lib/environment.sh`. A job finishes
when its work ends; the four-hour request is an upper bound.

## Prepare the submission environment

Submit from the repository root. Slurm opens its output files **before** running
the script, so create the log directory first:

```bash
cd /home/asciibase64/Aubadia/Mirage
mkdir -p artifacts/slurm/logs

# Select an existing Python >= 3.14 environment containing the PyTorch to test.
export MIRAGE_PYTHON=/shared/path/to/prepared/environment/bin/python
```

`MIRAGE_PYTHON` defaults to the repository's `.venv/bin/python`. `uv sync` alone
does not install PyTorch. Neither Python nor dependencies are installed by these
jobs. The interpreter, dependencies, provider module, repository and replay
bundles must be accessible from the allocated compute node.

If dependencies or a provider module are in a separate shared directory:

```bash
export MIRAGE_EXTRA_PYTHONPATH=/shared/path/to/dependencies:/shared/path/to/provider
```

The scripts build `PYTHONPATH` from the repository's `src/` and this explicit
setting. They do not inherit an incidental `PYTHONPATH` from prior interactive
experiments. In particular, `/tmp/mirage-validation-deps` on the current host
must not be assumed to exist on an allocated node.

Settings supplied before submission must also survive the user's `.bashrc`.
The preflight runs under `srun`, checks Python/PyTorch and exactly one visible
CUDA GPU, and saves environment details. It does not replace Slurm GPU visibility
with a physical GPU index.

## Validate and submit

Syntax and scheduler preflight do not run the workload:

```bash
bash -n sbatch/lib/environment.sh
bash -n sbatch/validation/full_suite.sbatch
sbatch --test-only sbatch/validation/full_suite.sbatch
```

Submit validation jobs explicitly:

```bash
sbatch sbatch/validation/smoke.sbatch
sbatch sbatch/validation/full_suite.sbatch
```

The full suite enables `MIRAGE_TEST_COMPILE`, `MIRAGE_TEST_CUDA` and
`MIRAGE_TEST_E2E`. Its complete output is retained in the job artifact directory,
including when a test fails. An outer timeout reserves time before Slurm's hard
deadline; its nonzero exit status is preserved.

Start exploration with a Python provider factory:

```bash
export MIRAGE_PROVIDER=my_provider:create_provider
sbatch sbatch/experiments/run.sbatch \
  --apis torch.matmul torch.nn.functional.linear \
  --backend inductor --precision fp16 --cases 100 --seed 0
```

For a deterministic integration fixture, explicitly set
`MIRAGE_PROVIDER=reference`. That provider is not an LLM. The exploration job's
exit status reflects CLI completion; inspect the experiment verdicts in the
saved report to determine whether MRs passed or failed.

Replay a saved manifest on shared storage:

```bash
sbatch sbatch/replay/replay.sbatch artifacts/slurm/JOB_ID/cases/EXPERIMENT.json

sbatch sbatch/replay/replay.sbatch artifacts/slurm/JOB_ID/cases/EXPERIMENT.json \
  --size 2 --updates 1 --iteration 0
```

To validate with nightly, select an already prepared environment and explicitly
allow the environment difference:

```bash
export MIRAGE_PYTHON=/shared/path/to/nightly/bin/python
sbatch sbatch/replay/replay.sbatch artifacts/slurm/JOB_ID/cases/EXPERIMENT.json \
  --allow-environment-change
```

Replay writes a child artifact next to the source manifest, as defined by the
Mirage CLI. Its source directory must be writable. The batch job's `replay.json`
contains the child path, verdict and environment changes.

## Configuration and budgets

| Variable | Default | Meaning |
| --- | --- | --- |
| `MIRAGE_PROJECT_ROOT` | `SLURM_SUBMIT_DIR` | Repository root |
| `MIRAGE_PYTHON` | `<root>/.venv/bin/python` | Existing interpreter, also used by child workers |
| `MIRAGE_EXTRA_PYTHONPATH` | Empty | Explicit shared dependency/provider paths |
| `MIRAGE_OUTPUT_ROOT` | `<root>/artifacts/slurm` | Persistent job output root; each job gets its own ID directory |
| `MIRAGE_PROVIDER` | Required for exploration | `module:factory` or explicit `reference` fixture |
| `MIRAGE_TIME_LIMIT` | `13800` | Requested exploration budget in seconds |
| `MIRAGE_WORKER_TIMEOUT` | `600` | Requested exploration/replay worker timeout in seconds |

Exploration and replay retain at least 300 seconds of allocation headroom.
The common helper reads Slurm's remaining duration and caps the requested
application budget accordingly. Replay divides the available budget between
its two sequential workers. Exploration's own batch deadline bounds its workers.
The smoke test requests smaller run/replay budgets and also retains headroom.

Use the variables above for managed options; positional CLI options cannot
override the provider, CUDA device, output root or time budgets. Other supported
CLI options must be spelled out in full. Slurm resources can be explicitly
overridden before the script path, for example:

```bash
sbatch --time=00:30:00 sbatch/validation/smoke.sbatch
```

Resource overrides still need to satisfy the selected partition/account's rules.
A scheduler rejection must be resolved explicitly; the scripts never silently
switch partition, reserve extra GPUs or fall back to unscheduled execution.

## Logs and monitoring

Default scheduler logs are `artifacts/slurm/logs/<job-name>-<job-id>.out` and
`.err`, relative to the submission working directory. Changing
`MIRAGE_OUTPUT_ROOT` does not change these `#SBATCH` paths; use `sbatch --output`
and `--error` with existing parent directories to move scheduler logs.

Application output and `environment.json` go under
`$MIRAGE_OUTPUT_ROOT/<job-id>/`. Disposable Inductor/Triton caches use a unique
directory under `SLURM_TMPDIR`, `TMPDIR` or the allocated node's `/tmp`; they
are not required to replay saved cases.

```bash
squeue --user="$USER"
sacct --jobs=JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

A successful `sbatch --test-only` is submission validation, not an executed test.
A queued job is not a completed job, and a completed exploration job is not
automatically an MR Pass. Inspect the saved test logs or experiment reports.
