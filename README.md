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

For alternating model/framework rounds, use `--no-generation-overlap` while
keeping the generation pipeline enabled. Each round completes source testing,
representation traversal and required replay, and publishes their feedback before
the next proposal is selected and its prompt constructed. Framework workers remain
parallel, and the per-API model budget and full traversal rules are unchanged.
Planning or failed generation need not produce an executable framework case.

One model request can overlap up to eight testing workers by default. `--workers`
sets the shared execution limit for sources, representations, replay and triage;
other strategies retain their existing default of one. Independent worker
processes isolate tensor/RNG state. A bounded backlog admits at most
one additional generated source while prior representations drain. Requests record
a versioned snapshot of completed feedback; later results affect subsequent
requests. No pending result establishes coverage, applicability or a defect.

On the measured 32-CPU allocation, `--workers 32` completed the fixed 128-case CPU
screen in 30.38 seconds versus 76.05 seconds at eight workers (2.50× throughput;
two measurements per setting). Sixteen workers took 42.30 seconds. Higher
concurrency increased individual worker latency and memory demand; the default
remains eight, and a larger limit requires matching Slurm CPU/memory resources.
The complete 1,887-case candidate validation matched saved classifications,
including representation and replay obligations. See
[roadmap 014](docs/roadmap-014-worker-overhead.md) for scope, evidence and limits.

Report conversion avoids repeated special-type checks for builtin JSON data;
both durable pipeline ledgers share one traversal snapshot. Reporting frequency, JSON format, worker
isolation, normal process exit and recovery checks remain unchanged.

After generation stops, every observed eligible call/operand/action is executed or
linked to exact prior runtime evidence. Ineligible actions retain their reasons;
`not_verified` means execution did not establish the required state. The report
separates `dispatch_complete` from `all_eligible_attained`; neither means all MRs
passed or all theoretical representations were explored.
Finding a defect does not blacklist that API/representation: later inputs, states
and paths can expose different failures. Suspected cases are not distinct bugs.

`pipeline_timing` reports `llm_request_seconds`, `framework_worker_seconds`,
`framework_busy_seconds` (the union of worker intervals), `overlap_seconds`,
`wall_seconds`, peak/mean active workers, execution queue time, provider-process
overhead and backpressure. Worker seconds are cumulative across concurrent slots;
overlap is the intersection of model and framework interval unions.
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
Resume restores the saved worker limit and backpressure measurement. Older ledgers
default to their original one-worker limit and explicitly mark missing historical
backpressure as unknown.
The saved generation-overlap setting is also restored; ledgers without that field
retain the original overlapping behavior.

The ten-API latency driver, `benchmarks/e2e_latency.py`, freezes the selected
benchmark API/device roster and engine, measures 180 model-request seconds per
API, and records process E2E alongside framework/model overlap and vLLM metrics.
Its `prepare` command creates a new immutable protocol directory; `campaign`
runs inside a Slurm testing step against the prepared service. Issue fixtures do
not enter generation. Equal-budget live campaigns and fixed-case throughput are
reported separately; a changed adaptive trajectory is not byte-for-byte LLM
equivalence. See [roadmap 015](docs/roadmap-015-e2e-latency.md) and the
[Slurm instructions](sbatch/README.md) for the protocol and executed results.

The completed ten-API study retained the byte-identical JSON report conversion
optimization (about 2.66x throughput for the measured 406 MB report). Its bounded
359-pair screen took 250.566 s at eight workers and 145.940 s at 32 workers,
with higher per-worker latency and memory at 32. The default remains eight.
Live campaign makespans were 8106.480 s and 7653.745 s, but workloads and quality
differed; candidate `batch_norm` also failed the original request-budget tolerance
(179.851 s versus nominal 180 s). These observations do not establish equal-quality
E2E acceleration. See the [full results](result/e2e-latency/report.md), including
the preserved failures, attribution limits, and completed resource cleanup.

Independent API sessions can share generation and testing capacity in either
`training` or `stateful` mode:

```bash
mirage run --strategy training --provider vllm --apis torch.sin torch.cos \
  --active-apis 2 --workers 4 --workers-per-api 3 \
  --generation-concurrency 2 --generation-per-api 2 --time-limit 500 \
  --output result/experiments/multi-training
mirage run --strategy stateful --provider vllm --apis torch.sin torch.cos \
  --active-apis 2 --workers 4 --generation-concurrency 2 \
  --no-generation-overlap --time-limit 180 --output result/experiments/multi-stateful
```

Run these computational commands with `srun` in an appropriate Slurm allocation.
`--workers` and `--generation-concurrency` are campaign-wide capacities in shared
mode. Each API has independent seeds, queues, feedback and reports. The broker
rotates among ready sessions, lends unused slots, and serializes CUDA test
subprocesses on the assigned test GPU. Waiting coordinators do not consume test
slots. CPU and provider subprocess overhead still needs allocated host resources;
worker slots are not CPU cores or GPU utilization. Generated code using extra
undeclared devices and distributed/multi-GPU tests need a resource adapter and
are outside this initial single-test-GPU policy.

The default `--active-apis 1` retains the existing entry behavior. Explicit shared
sessions use fresh output directories. Training gets a separate wall deadline and
case cap per admitted API; resource waits after admission count toward that wall
budget. Stateful pipeline mode retains cumulative per-API provider request seconds,
excluding local dispatch/backpressure waits, followed by full admitted traversal
drain. Service-side queue time remains part of request time. Alternating mode
waits only for its own API's feedback. Disabling the stateful pipeline keeps its
wall-budget behavior. Admission queue time and per-session durations are reported
separately from actual campaign makespan; their sum can exceed makespan.

Each API's native evidence and stateful `pipeline-state.json` stay in its own
`api-NNN` directory. `multi-api.json` records session outcomes and resource events;
it does not turn controller completion into a correctness Pass or combine distinct
coverage denominators. Resume an interrupted stateful drain with
`mirage resume result/experiments/multi-stateful/api-000`; this issues no new model
requests. Training resume and automatic multi-replica serving are not implemented.
Unconfirmed cancellation on the dedicated shared model endpoint stops new work
across its sessions; normal case/controller failures remain local.

Library callers can pass `APIJob` records to `mirage.runtime.multi_api.run_sessions`
with `SchedulingOptions` to retain per-API devices, budgets and explicit stateful
search treatments. Shared scheduling knows no benchmark API names or framework
tensors. New frameworks require their own legal execution/isolation and resource
adapters; success on a bounded PyTorch roster does not establish universal support.
