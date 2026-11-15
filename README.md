# Mirage

Mirage explores PyTorch tensor representations and training states with LLM-generated
programs and conditional metamorphic relations. API names are resolved in the tested
installation; the 16-API acceptance list is not a generation whitelist.

The LLM receives the installed API signature and documentation, then produces
`build(device, seed)`, `forward(ctx)`, `loss(output)`, and optionally
`sample_loss(output)`. Initialization registers inputs, trainable parameters,
optimizer, sample axes, and optional modules. The runner owns state transitions,
actual API/backward observation, conditional comparisons, and evidence storage.

See [the implementation roadmap](docs/roadmap-002-general-api-state-mrs.md) for the
approved scope, exact contracts, validation results, and limitations. The original
[MVP roadmap](docs/roadmap-001-mvp.md) is historical; its old execution contracts and
legacy replay are no longer supported.

## Execution

This workspace is on Nano4. Run inference, runtime tests, compilation, exploration,
and reproduction through Slurm. Use the existing Python/PyTorch environment.
vLLM and model weights are separate, prepared resources; experiment jobs do not
install them. Large persistent resources belong under `/work/asciibase64/Mirage/`.

See [Slurm instructions](sbatch/README.md) for prepared environment selection,
interactive development, resource assignments, and cleanup.

Inside an allocated testing step, a small exploration is:

```bash
python -m mirage run --provider vllm --device cuda \
  --apis torch.matmul torch.nn.functional.conv2d \
  --canonicals-per-api 1 --targets-per-canonical 12 --paths-per-target 2 \
  --cases 48 --workers 2 --generation-concurrency 2 \
  --time-limit 1800 --output result/experiments/example
```

`--cases` caps planned state/path attempts. Canonical generation has its own
`--canonicals-per-api` budget. `--size` is an input-size hint, not a fixed tensor
shape. Workers share the testing GPU. `--epsilon` controls the uniform component
of inverse-attempt-count target sampling. All targets are provisional until their
runtime predicates are verified.

`--forward-only` explicitly removes the requirement for a backward path through
the target API, allowing nondifferentiable targets such as tensor shape queries
inside a training program. The training loss still needs a valid backward path;
the LLM cannot silently disable target-backward validation on ordinary runs.

## Metamorphic rules

The six families are layout, storage, checkpoint, mutation, saved-tensor lossless
roundtrip, and gradient accumulation. Each requires valid executions, its own
preconditions and input relation, and aligned semantic checkpoints before outputs
are compared. Unknown operator capabilities retain coverage-only execution rather
than being assumed compatible with an MR.

The accumulation rule initially certifies deterministic FP32 per-sample computation
with explicit sample axes and correctly normalized losses. Saved-tensor roundtrip
uses same-device lossless copies. Within the training-program strategy, CPU offload, distributed execution, sparse
and quantized tensor adapters, and automatic arbitrary-program minimization remain
outside the implementation. The property strategy below adds selected CSR/MKLDNN
adapters with narrower contracts. Compiled-path evidence is conservative; unsupported graph
correspondence or operand capture is reported instead of claiming correctness.

Outputs distinguish invalid executions, inapplicable relations, Pass, suspected
failure, and correctness unassessed. Nonfinite observations require a more specific
numerical policy and are not automatically reported as bugs. Numerical thresholds
are screening policies, not proof of a defect.

## Evidence and reproduction

Each generated program retains API context, prompts, repairs, initial tensor storage
relationships, plans, runtime traces, environment, code snapshot, comparison results,
and measured resource usage. A canonical is reused across multiple targets and paths.

```bash
python -m mirage inspect result/experiments/example/CASE.json
python -m mirage report result/experiments/example
python -m mirage replay result/experiments/example/CASE.json
```

Reproduction runs through Slurm, uses the saved engine and initial data, and makes
no provider calls. Environment changes are reported and suppress a same-environment
verdict unless `--allow-environment-change` is explicit. Suspected failures trigger
bounded reproduction controlled by `--replay-failures`; no result is automatically
promoted to a confirmed framework defect.

## Deterministic correctness benchmark and property strategy

The [reproduction report](docs/pytorch-correctness-reproduction.md) records ten
fixed-input upstream correctness issues. The [Phase 2–3 report](docs/pytorch-correctness-evaluation.md)
describes integration, property preconditions, the blind generation protocol and
measured results: 10/10 admitted fixtures were detected, and the frozen LLM run
reproduced 3/10 known issues from 1,154 candidates. Issue-derived fixtures live under
`benchmarks/pytorch_correctness/`; they are not imported by production generation.

For reviewed API semantics, `--strategy properties` asks the LLM for a deterministic
input builder (and a small function/module where required). The runtime owns
materialization, analytic-gradient, context-restoration, captured-accumulation and
compiled-alias probes. This supplements the training strategy with explicit
state/representation oracles; it does not certify arbitrary APIs or programs.

```bash
# Inside a Slurm testing step with a healthy prepared provider:
python -m mirage run --strategy properties --provider vllm --device cpu \
  --apis torch.linalg.slogdet --time-limit 300 --cases 100000 \
  --output result/experiments/properties-example
python -m mirage report result/experiments/properties-example
python -m mirage replay result/experiments/properties-example/attempt-0000/execution/request.json
```

Property results preserve invalid, inapplicable, unassessed, pass and suspected
failure outcomes. An exception is not automatically a PyTorch bug. A no-provider
replay verifies saved code/evidence and initial representation. Unsupported snapshot
restoration is a failed replay, not an inferred pass. Property exploration is
currently serial; training-strategy concurrency and canonical/path budgets do not
control it. The property report does not invent ground-truth detection or false
positive rates from oracle outputs alone.


## Stateful generation pipeline

For `--strategy stateful`, the CLI enables `--generation-pipeline`, representation
traversal and scoped numerical checking by default. `--time-limit` then means
cumulative **model-request seconds per API**, including state planning, program
generation and repair requests. Model service startup, framework execution and
backpressure waiting do not consume that budget. `--cases` limits source proposals;
it does not truncate their representation children. Direct Python callers enable
`RunOptions(generation_pipeline=True)` and pass `representation_traversal=True`.
`--no-generation-pipeline` explicitly retains the legacy shared wall-time budget.

One model request can overlap one testing worker. A bounded backlog admits at most
one additional generated source while prior representations drain. Requests record
a versioned snapshot of completed feedback; later results affect subsequent
requests. No pending result establishes coverage, applicability or a defect.

After generation stops, every observed eligible call/operand/action is executed or
linked to exact prior runtime evidence. Ineligible actions retain their reasons;
`not_verified` means execution did not establish the required state. The report
separates `dispatch_complete` from `all_eligible_attained`; neither means all MRs
passed or all theoretical representations were explored.

`pipeline_timing` reports `llm_request_seconds`, `framework_worker_seconds`,
`overlap_seconds`, `wall_seconds`, provider-process overhead and backpressure.
Request time includes server/network waiting and is not pure GPU inference time.
Worker time includes process initialization, execution, oracle and artifact work.
Overlapping durations must not be added and called wall time.

Interrupted execution retains the source and candidate inventory. Inside a Slurm
allocation, `mirage resume PATH_TO_SINGLE_API_RUN` drains saved work without more
LLM calls. It verifies the frozen engine and source/initialization checksums and
retains interrupted attempts separately. Run it with the saved engine on
`PYTHONPATH` for the same controller version. Model generation is not restarted;
a partial generation budget remains explicitly partial. Slurm/worker timeouts
remain independent resource bounds, and unresolved work is never labeled complete.
