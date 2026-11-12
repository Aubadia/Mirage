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
uses same-device lossless copies. CPU offload, distributed execution, sparse and
quantized tensor adapters, and automatic arbitrary-program minimization are outside
this implementation. Compiled-path evidence is conservative; unsupported graph
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
