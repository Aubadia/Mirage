# Mirage

Mirage tests PyTorch tensor memory representations and training states using
conditional metamorphic relations (MRs). It preserves the generated programs,
actual initial state, runtime traces and comparison evidence needed to replay
an experiment without calling an LLM.

The MVP includes eight API profiles, representation and checkpoint transitions,
Adam training, FP32/AMP, eager and compiled execution, bounded generation repair,
coverage feedback, immutable artifacts, replay and manual minimization support.
See [the MVP specification](doc/MVP.md) for the research goals and exclusions.

## Environment

Use Python **3.14 or later**. The execution interpreter must have the PyTorch
version being tested and its required dependencies available. Mirage deliberately
does not install or upgrade PyTorch: stable and nightly belong to separately
prepared environments. `uv sync` installs Mirage, but does not install PyTorch.

Workers use `sys.executable`, inherit `PYTHONPATH` and CUDA visibility, and run
sequentially on CPU or one visible GPU. On Nano4, Slurm assigns GPU visibility;
do not override it with physical GPU indices. FP16 uses autocast with FP32 parameters and GradScaler;
BF16 uses autocast with FP32 parameters. The actual output dtype is recorded.

This workspace is on **Nano4**. Submit all computational work through the
categorized scripts in [`sbatch/`](sbatch/README.md), including full tests,
training, compilation, experiments and replay. Direct shell work is limited to
lightweight inspection and static checks. The scripts use the existing Python
installation; a login host's `/tmp` dependencies are not a shared environment.

## Running a batch

```bash
mkdir -p artifacts/slurm/logs
export MIRAGE_PYTHON=/shared/path/to/prepared/environment/bin/python
export MIRAGE_PROVIDER=my_provider:create_provider
sbatch sbatch/experiments/run.sbatch \
  --apis torch.matmul torch.nn.functional.linear \
  --backend inductor --precision fp16 --cases 100
```

The default API pool also includes `layer_norm`, `gelu`, `softmax`, `dropout`,
`Tensor.add_` and `Tensor.mul_`. API arguments use their full identifiers shown
by `src/mirage/contracts/apis.py`. Backends are `eager`, `compile_eager`,
`aot_eager` and `inductor`; precision modes are `fp32`, `fp16` and `bf16`.

Other CLI controls include `--seed`, `--updates` (default 3), `--size` (square input
width, default 4), `--warmup-steps` (default 1), `--epsilon` (default 0.10) and
`--replay-failures` (default 1). All generation, initialization, warm-up,
compilation and execution consume the batch budget. Workers and providers have
externally enforced deadlines; timeout kills their subprocess group.
The Slurm wrapper caps the application budget against the remaining allocation
time and reserves time for saving results. See the batch documentation for
`MIRAGE_TIME_LIMIT`, `MIRAGE_WORKER_TIMEOUT` and persistent output paths.

For an integration smoke test, use the explicit deterministic fixture:

```bash
sbatch sbatch/validation/smoke.sbatch
```

`reference` is **not an LLM**. It runs real PyTorch cases but supplies fixed
programs, reports zero generated tokens, and is labelled in every artifact.
Mirage never silently substitutes it for an unavailable model.

## Python LLM provider

`--provider module:factory` imports a zero-argument factory inside an isolated
provider process. The returned object implements:

```python
from mirage.generator.provider import Request, Response, Usage


class LocalProvider:
    def generate(self, request: Request, *, deadline: float | None) -> Response:
        # Call your local model here and honor the monotonic deadline.
        text, input_tokens, output_tokens = call_your_model(request, deadline)
        return Response(text, Usage(input_tokens, output_tokens), model="your-model")


def create_provider():
    return LocalProvider()
```

Each request contains the generation stage, prompt, API profile, supported
program grammar, configuration, ordered actions, previous program and MR.
Return JSON with exactly `source` (Python text) and `actions` (an array of
strings). Unknown usage is `None`, not zero. Each invocation gets a fresh
process; persist any model-service state outside that short-lived provider or
implement model loading in the factory. No HTTP adapter is bundled.

The four stages are canonical, skeleton, transition and follow-up. Each accepted
stage executes in a worker and must provide runtime evidence. Syntax and missing
paths can be repaired at most twice. Unsupported grammar and unexplained
exceptions are retained without speculative repair; suspected failures never
enter the repair loop.

## Supported generation and state space

The harness owns training lifecycle and instrumentation. Generated code supplies
the profiled single target operation and the loss, with no arbitrary Python
side effects. Canonical generation can choose supported literal options for
dropout probability, softmax dimension, GELU approximation and layer-norm epsilon.
Skeleton and transition stages preserve that computation. A legal follow-up
that changes it cannot receive an MR pass because the input relation includes
the normalized program. This controlled grammar bounds the MVP's exploration;
it is not a general-purpose arbitrary-model fuzzer.

The trainable FP32 parameter is a square tensor. Every iteration constructs a
nonleaf API operand from it. Binary API weights are a detached value copy or a
detached alias of that operand; bias is a fixed saved tensor. Gradients are
compared in the trainable parameter's logical coordinates, including when
in-place view rebasing changes the autograd graph.

Representations include contiguous, transposed-stride, contiguous nonzero-offset,
an additional clone, and shared operand/weight storage. Storage groups are
relationships within an execution; cross-process pointers are never compared.
The alias MR is restricted to pure operations. Mutation pairs keep the same
alias topology and compare eager against a selected compiled backend.

Targets combine representation, checkpointing, first/final iteration,
`before_api`/`after_backward`/`after_update`, precision, backend and warm-up
history. Warm-up can be absent, use the tested representation, or an alternate
representation. It runs inside the same worker and compiled callable as training;
parameters, gradients, Adam, scaler and RNG are then restored and verified.

Dropout layout changes are retained for coverage without an equality oracle.
Dropout checkpoint equivalence requires the same backend and representation,
saved RNG and the controlled side-effect-free computation. RNG state after
each iteration is included in its output comparison.

## Verdicts and coverage

Execution outcomes (`completed`, `exception`, `crash`, `oom`, `timeout`) are
separate from MR verdicts:

| Verdict | Meaning |
| --- | --- |
| Invalid | Required API/backward/action/state evidence disproves validity |
| Inapplicable | Valid executions fail MR conditions or checkpoint alignment |
| Pass | Applicable MR satisfies its output relation |
| Suspected Failure | Applicable MR violates its output relation |
| Unassessed | Coverage-only, incomplete execution, or insufficient evidence |

Inputs, initial training/RNG state, permitted configuration differences and
normalized computations are checked before comparing outputs. Successful Adam
update counts must align. Shape, dtype, discrete state and nonfinite values have
explicit checks; NaN is not automatically equal to NaN.

Tolerance policy version 1 is in `rules/tolerances.py` and is saved in each
experiment. FP32/FP16/BF16, API reduction behavior, backend and observable select
different thresholds. Optimizer steps and scaler state compare exactly. These
are screening thresholds for bounded workloads; suspected failures require
numerical revalidation and independent evidence. Replay rejects an edited or
unsupported tolerance policy.

Coverage counts only verified evidence and lists the complete declared candidate
universe. API coverage may survive a later failure; state and combination coverage
require their own evidence. Applicability includes unknown cases in its explicit
denominator. Generation failures and unresolved plans remain separate statuses.
Candidate fractions are not estimates of exhaustive theoretical coverage.

## Artifacts, replay and minimization

Each experiment has an immutable JSON manifest and a sibling bundle containing:

- Generated sources, prompts, responses, repair diagnostics and token usage.
- Shared initial tensor/RNG state, reconstruction actions and pair configurations.
- A snapshot of the Mirage Python engine used to execute the case.
- Worker requests, append-only traces, partial observations, stdout/stderr,
  complete observations, environment and process resource measurements.
- MR definition, tolerance policy, inspection and content hashes for every file.

Batch reports preserve candidate sets, selection counts, outcomes, coverage,
generation/execution costs and stop reasons. Immutable progress reports are
published after each attempted experiment. Missing token usage and resource
measurements remain unknown. Generation timing includes stage validation;
provider time and execution time are also reported separately, so do not sum
overlapping timing categories.

```bash
python -m mirage inspect artifacts/EXPERIMENT.json
python -m mirage report artifacts
sbatch sbatch/replay/replay.sbatch artifacts/EXPERIMENT.json
```

Replay verifies bundle hashes, uses saved programs, initial state, engine and
policy, and makes **zero provider calls**. A new child manifest links to the
original; original evidence is never overwritten. A version/environment mismatch
remains unassessed unless explicitly requested as cross-environment validation:

```bash
export MIRAGE_PYTHON=/shared/path/to/nightly/bin/python
sbatch sbatch/replay/replay.sbatch artifacts/EXPERIMENT.json \
  --allow-environment-change
```

Use another prepared interpreter with Mirage/PyTorch available; the flag does
not install nightly. Both old and new environments are retained.

Manual minimization supports smaller saved tensors and fewer updates. Explicitly
move the target iteration if removing its original iteration:

```bash
sbatch sbatch/replay/replay.sbatch artifacts/EXPERIMENT.json \
  --size 2 --updates 1 --iteration 0
```

`--source-program` and `--followup-program` accept separately edited source files.
They must still satisfy the supported grammar; altered computations invalidate
the MR input relation. Keep original bundle files intact and supply modifications
through these options. Automatic minimization and automatic issue submission are
outside the MVP. A repeated violation alone is not a confirmed framework defect.

## Validation

On Nano4, submit the full suite through Slurm after selecting a prepared
Python/PyTorch environment and creating the scheduler log directory:

```bash
sbatch sbatch/validation/full_suite.sbatch
```

The job enables compilation, CUDA and end-to-end tests. Lightweight syntax and
scheduler preflight can run directly:

```bash
bash -n sbatch/validation/full_suite.sbatch
sbatch --test-only sbatch/validation/full_suite.sbatch
```

These check all eight APIs, MR positives and injected negatives, independent
scalar matmul gradients, mutation backward evidence, checkpoint RNG, warm-up,
precision/backend paths, deadlines, bundle integrity, LLM-free replay and
minimization. GPU tests use the available GPU; they do not substitute for the
specified Nano4/H200 acceptance run. Finding and confirming two previously
unreported framework bugs remains a separate research milestone.
