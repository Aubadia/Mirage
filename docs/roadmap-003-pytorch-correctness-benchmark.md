# PyTorch correctness benchmark: Phases 1–3

The opening sections preserve the original plan and chronological execution record. Detailed technical accounts follow for [Phase 1](#phase-1-details), [Phase 2](#phase-2-details), and [Phase 3, including the actual prompts](#phase-3-details). The [evaluation report](pytorch-correctness-evaluation.md) explains reproduction and non-reproduction for each issue.

## Approved scope

The user approved Phase 1 on 2026-11-13: collect ten distinct, reproducible correctness issues involving tensor representation or training state in the prepared PyTorch installation. The scope covered isolated reproducers, independent oracles, controls, one deterministic fresh-process execution per case, and reports supported by execution evidence. It also covered assessing existing Mirage capabilities without changing the implementation. At that stage, Phase 2 changes and Phase 3 experiments required a subsequent plan naming the affected files.

Authorized files: this roadmap; docs/pytorch-correctness-reproduction.md; benchmarks/pytorch_correctness/{README.md,candidates.json,cases.py,run.py}; tests/test_pytorch_benchmark.py; sbatch/replay/pytorch_correctness.sbatch. Execution artifacts: result/pytorch-correctness/<run-id>/.

## Steps

1. Inspect public upstream issues and current implementation; record candidates and exclusions.
2. Implement minimal public-API cases, controls, oracle evidence and isolated runner.
3. Validate static syntax and submission preflight; run all computation through Slurm with the prepared environment.
4. Reproduce qualifying violations once with explicit fixed inputs; investigate alternate inputs and independent semantic evidence; group shared root causes.
5. Record actual outcomes and existing-framework capability gaps. Propose Phase 2 files based on the final benchmark.

## Validation

- Python AST checks, bash -n and sbatch --test-only are static/preflight only.
- Runner tests cover failed applicability, oracle disagreement, unexpected exceptions, process timeout and result aggregation where practical.
- Use account ACD115198, partition 8gpus, one node, eight CPUs, one GPU, four hours; computational commands use srun.
- Snapshot code/configuration before each run, save environment and raw logs; preserve failure statuses and bound work with cleanup time.
- A candidate is not a confirmed bug. An admitted violation needs legal inputs, verified trigger, passing controls and corroborating oracle evidence. Root-cause explanations retain their evidence level.
- Report API/runtime limitations separately from non-reproduction. No generated test or LLM success is inferred from benchmark replay.

## Progress (chronological)

- Read repository contracts, generator, conditional oracle, representation adapter, Slurm helpers and prepared environment metadata.
- Installed version file reports 2.15.0+cu132; compute-node verification pending.
- Existing framework does not sample or record conjugate/negative bits, only supports dense strided registered inputs, and conservatively gates effectful/compiled MRs.
- Roadmap recorded before implementation. No execution results yet.

- Screening job 763186 executed all 14 candidates once: 13 oracle violations with passing controls, one non-reproduction (204347). Three runner tests passed in 1.971 s. Runtime environment: PyTorch 2.15.0+cu132, Python 3.14.10, CUDA 13.2, NVIDIA H200. These are provisional findings, not final admission.
- Initial proposal, later superseded: selected ten mechanism groups for three repeats and two input variants each: 203078, 204407, 200599, 202706, 181723, 198118, 204365, 201268, 197089, 200613. Compiled conjugate clone/alias candidates remain a conservatively grouped reserve; grid-sample reduced precision is a reserve.

## User steering: deterministic single execution

The user explicitly requested no issues requiring randomness and one execution per issue. This supersedes the earlier three-repeat/two-variant validation plan. Job 763201 was cancelled while pending (MaxGRESPerAccount), before allocation or execution. Replace seeded random pinv inputs with an explicit constant matrix, retain independent mathematical oracles and controls within each invocation, and run ten fixed-input cases once. No timing-race cases are selected; autocast uses deterministic recursion. Optional repeat/variant flags remain diagnostic only.

- Zero-GPU preflight was explicitly rejected by Slurm: --gpus-per-node accepts only 1..8. No CPU-only work ran outside Slurm and no partition was changed. Job 763226 was cancelled while pending; no allocation used.
- Final selection reuses ten deterministic cells already executed in job 763186, replacing seeded-input pinv with fixed-input compiled aliasing (204033). Current selected function ASTs and shared oracle/metadata helpers match the executed snapshot exactly. final-review.json records result hashes, function hashes, verified conditions and controls. No rerun is needed under the user's single-execution policy.

## Final Phase 1 status

- Completed ten fixed-input reproductions on PyTorch 2.15.0+cu132: 203078, 204407, 200599, 202706, 181723, 204033, 204365, 201268, 197089, 200613. Each uses one already executed process, verified predicates, an independent oracle and a passing control. The reproduction report distinguishes native-source evidence from root-cause hypotheses.
- Preserved raw upstream issues/comments, version-pinned native source excerpts, installed autocast/CUDA-graph source excerpts, immutable execution code, full tensor evidence and review hashes under result/pytorch-correctness/.
- Final numerical fixture functions are identical to the executed snapshot. Three runner tests passed in Slurm; the updated single-cell aggregation check passed separately in 0.010 s without importing PyTorch.
- Job 763186 ended with the expected oracle-violation exit code 3 after 87 allocated seconds (one GPU). Jobs 763201 and 763226 were cancelled before allocation. Final scheduler inspection found no owned jobs. No vLLM service was started.
- Existing src/mirage implementation, dependencies, configuration and unrelated files were not modified. Phase 2 integration and Phase 3 LLM evaluation remain unexecuted pending their file-specific approval; no discovery metrics are claimed.

- Added a concrete, unapproved Phase 2–3 implementation/file scope to the reproduction report: general lifecycle/representation/oracle capabilities, independent benchmark integration, blind 300-second/API evaluation, Slurm validation and cleanup. No Phase 2 files were changed.

## Approved Phase 2–3 execution (2026-11-13)

The user approved the implementation and exact file scope in the reproduction report, and explicitly requested Phase 3 immediately after Phase 2 correctness validation. No additional phase-transition approval is required.

Implementation sequence: (1) add a reviewed property-program contract and reusable representation/lifecycle/reference oracles alongside the existing training contract; (2) route the ten fixtures through the same runtime worker and validator used for generated property programs; (3) validate negative/applicability cases and the existing suite in Slurm; (4) freeze code and prompts; (5) run blind generation for 300 seconds per target API using the prepared provider, then independently attribute results and report all denominators. General logic must contain no issue IDs or benchmark constants. Single deterministic execution remains the benchmark policy.

Use only files listed in the report's approved Phase 2–3 scope. Update this roadmap with implementation decisions, real validation outcomes, experiment results, resource accounting and verified session cleanup. Phase 1 evidence remains immutable.

### Phase 2 implementation and development evidence

Added a property-program strategy reusing existing provider/worker/evidence infrastructure, five reviewed property families, deterministic representation actions, opaque/sparse metadata, benchmark-to-worker integration and LLM-free checked replay. Production code contains no selected issue identifiers or reproducer constants. The general API registry encodes reviewed semantics, not failing input recipes.

Slurm development job 763298 uses one prepared vLLM service (startup 192 s, zero restarts) and a separate testing GPU. Initial integration found an MKLDNN snapshot serialization failure (9/10); the explicit opaque adapter fixed it (10/10). An intermediate full suite passed 67 tests. A new replay regression then exposed the serialization version-counter increment; the version-insensitive fresh-graph snapshot signature and CLI property-result handling were corrected. All failed iterations remain saved. Final validation is still in progress; no Phase 3 results are claimed yet.

### Phase 2 verified; Phase 3 started

Final full suite: iteration-hz5F7t, 70 tests passed in 63.277 s (65 s iteration). Final integration: phase2-frozen / iteration-7KgQnc, 10/10 detected with actual target evidence and agreeing controls. Both checked replay probes (204407 negative view, 204365 MKLDNN plus affine views) completed with the same suspected-failure verdict, unchanged environment and zero provider calls: iterations aWRPx2 and 5353x3. Earlier rejected replay evidence remains preserved; view/base reconstruction and nonzero failed-replay CLI status are now covered. Real-model torch.sin smoke produced two valid executable passes in property-smoke-02.

Phase 3 is authorized and begins directly. The driver requires production engine hashes to match phase2-frozen, excludes development smoke results, uses 300-second/API budgets with a 45-second dispatch reserve, and retains raw prompts/responses/programs/executions. No tuning to interim benchmark outcomes is allowed during the frozen run.

### Completed Phase 3 and final validation

- Frozen iteration-pwpdkQ exited 0 after 2,597 s. All ten nominal 300-second/API budgets completed; active loops totaled 2,567.91 s, metadata discovery 21.29 s. The predeclared 45-second dispatch reserve explains actual per-API durations of 255.01–263.64 s.
- 1,154 generated candidates, 310 static-contract-valid, 130 completed target executions, 125 applicable comparisons, 28 suspected failures. Static validity 26.86%; executable/valid 41.94%. There were 661 repair responses and 748 API-scoped source fingerprints.
- Independent post-hoc audit (iteration-3ChIy1) labeled 19 positive and 50 negative observations, with TP=19, FN=0, FP=0, TN=50. Its FPR 0/50 and detection 19/19 apply only to that subset; nine suspects and the remaining unsupported/unexecuted candidates are not assumed correct.
- Reviewed reproduction: 204407 slogdet (first 8.78 s), 202706 autocast (10.73 s), 200613 copy_ (33.97 s): 3/10 known issues. The original matcher reported 2/10 because it omitted negative-view copy variants explicitly included upstream. Original matcher output is preserved; post-hoc attribution was corrected without altering generation/engine.
- Std attempts 38/52 independently exhibit wrong gradient signs for negative-bit inputs under Inductor. They are not the tiny-input 197089 benchmark and remain one related, unconfirmed-root/novelty candidate group. They do not increase issue reproduction rate.
- Post-run negative-control validation: iteration-RgKjt3 passed 12 property tests in 7.317 s, including ordinary compiled aliases and identical graph microbatches; iteration-lnZdUc passed four benchmark/audit tests in 2.031 s. Post-hoc attribution report and selected tensor-observation export both completed in Slurm. No production engine changes were made during the frozen experiment.
- Generation limitations are recorded explicitly: strict RNG-call rejection, unresolved nn.Module aliases, missing Tensor-method receivers, function objects returned as snapshot data, and insufficient MKLDNN/CSR/extreme-scale trigger coverage. These are not all PyTorch-invalid programs and are not automatically oracle misses.
- Job 763298 completed with exit 0:0 after 4,208 allocated seconds, two GPUs (8,416 allocated GPU-seconds). One service startup, 192 s, zero restarts. All 23 iteration timings and failed development logs are retained. Testing scratch was removed; terminal closed; squeue was empty; all steps exited, including the service step deliberately cancelled at shutdown. No owned service/allocation remains.
- Final results, MR conditions, reproducible commands, metric denominators and artifact links are in docs/pytorch-correctness-evaluation.md. No new dependencies or project configuration changes were introduced.

## Documentation expansion requested after completion

After completion, the user requested detailed accounts of the work in Phases 1, 2, and 3: the added capabilities, their implementation, the LLM prompts, and the reasons each issue was or was not reproduced. That documentation update was limited to this roadmap and `docs/pytorch-correctness-evaluation.md`.

The update proceeded by comparing the current code with the frozen snapshot; inspecting saved requests, responses, generated programs, worker summaries, and independent audit records; expanding the phase descriptions and per-issue explanations; and checking numerical results, verbatim prompts, links, and edit scope. It changed no implementation, launched no Slurm or LLM processes, and reran no benchmarks. All added descriptions were based on existing execution evidence.

Validation used the Python standard library to compare all 10 execution-funnel rows with `final-analysis.json` and all 10 audit rows with `independent-audit.json`; every row agreed. The system message and complete slogdet user message matched the saved HTTP request verbatim. All five family suffixes matched the implementation strings, as did the common prefix, except that its displayed block omitted the trailing joining space. Local links and explicit anchors resolved in both documents; Markdown fences were paired and no lines had trailing whitespace. Only those two documents changed. Earlier test results retained their original chronology; no new experiments or repeated test passes were claimed.

<a id="phase-1-details"></a>
## Phase 1: Establishing the ten-case benchmark

### 1.1 Separating upstream reports from local reproduction

Phase 1 examined public issues, minimal reproducible examples (MREs), and discussions in the official PyTorch repository against the prepared local environment. The focus was correctness under changes in representation with equal logical values, or under changes in training and context lifecycles. An issue's open status was not treated as proof of a defect, nor was an arbitrary exception treated as an incorrect result.

The retained evidence includes 14 `issue-<ID>.json` files, [version-pinned native-source excerpts](../result/pytorch-correctness/research/upstream-source-excerpts.json), and [source evidence from the installed package](../result/pytorch-correctness/research/installed-source-evidence.json). The [candidate catalog](../benchmarks/pytorch_correctness/candidates.json) records each API, device, trigger, oracle, root-cause group, and inclusion or exclusion rationale. These artifacts support candidate review; they neither exhaust the GitHub issue space nor preserve a complete search-query and ranking history. They therefore do not establish an exhaustive search experiment.

### 1.2 Constructing fixed inputs, independent expectations, and controls

The fixtures in [Phase 1 cases.py](../benchmarks/pytorch_correctness/cases.py) are regression data derived from the issues, **not programs generated by the LLM in Phase 3**. Each returns `actual`, `expected`, `control`, `conditions`, tolerances, and additional metadata. The `check()` function compares full tensors using the existing `mirage.inspector.comparison.differences` implementation.

The validator checks conditions, controls, and numerical comparability before evaluating whether `actual` violates `expected`. A disagreement between the control and expectation, nonfinite values, unmet conditions, exceptions, or timeouts cannot establish confirmed reproduction. JSON previews contain at most 64 tensor elements; full comparisons and `evidence.pt` retain the complete data.

| Issue / fixture | Fixed trigger construction | Correctness evidence and control |
| --- | --- | --- |
| 203078 / `scatter` | Expand index `[[0],[2]]` to `(2,16)` with stride `(1,0)`; use a `(3,32)` destination. | Scalar Python scatter semantics and a target control with the same index made contiguous. Expected row sums are `[16,0,16]`; observed sums are `[16,16,0]`. |
| 204407 / `slogdet` | Apply `conj().imag` to a fixed asymmetric 2×2 complex128 leaf to obtain a real matrix with the negative bit set. | Analytic 2×2 inverse-transpose derivative, finite differences of a scalar determinant implementation, and a `resolve_neg()` control. Phase 1 compares the imaginary gradient of the complex leaf; Phase 2 differentiates directly with respect to the real operand. |
| 200599 / `lu_solve` | For two fixed 2×2 systems, `LU[:, None]` introduces a size-one batch dimension with stride `(4,2,1,2)`. | Closed-form 2×2 solution, contiguous-LU control, and retained residuals. The second solution should be `[1/6,1/3]`; the observed result is `[5,-1]`. |
| 202706 / `autocast` | Recurse through the same decorated function without threads or races. | AMP state must be restored on exit. Recursion with a fresh context at each level supplies the control; observables include enabled state, dtype, and cache state. |
| 181723 / `cuda_graph` | Use a fixed elementwise scaling model and two different microbatches; set parameter `.grad=None` after capture. | The analytic gradient is the sum of the two batch inputs, corroborated by eager autograd. The observed accumulation is twice the second batch gradient. |
| 204033 / `compiled_alias` | Inductor returns a complex base and its `conj().imag` view; mutate the view after return. | The same eager program and the directly calculable change in imaginary values. Correct initial values alone do not establish correct alias semantics. |
| 204365 / `batch_norm` | Use an MKLDNN inference input with stride-two weight and bias views. | Explicit affine-normalization formula and a contiguous control with the same logical affine values. |
| 201268 / `sparse_add` | Construct a valid CSR `out` whose values use stride-two storage, with sentinels in the intervening positions. | Elementwise addition, a contiguous-output control, and checks of values and neighboring storage writes. The last check is additional Phase 1 evidence; the general Phase 2 oracle does not fully implement collateral-storage checks. |
| 197089 / `std_gradient` | Use FP32 `[-2**-100, 2**-100]`, CPU Inductor, and `correction=0`. | Expected std is `2**-100` and gradient is `[-.5,.5]`, with eager control. Scale-appropriate comparison prevents a generic absolute tolerance from masking the discrepancy. |
| 200613 / `broadcast_copy` | Copy a fixed conjugate source synchronously from CPU to CUDA, broadcasting one element to two. | Logical source broadcasting and a resolved-conjugate control. Expected values are `[1-2j,1-2j]`; the observed imaginary parts have the opposite sign. |

The [Phase 1 reproduction report](pytorch-correctness-reproduction.md) provides the full MREs, observed outputs, and evidence strength for each root-cause explanation. The table describes observed symptoms and input constructions; it does not imply that every kernel-level cause was verified with an upstream patch or native debugger.

### 1.3 Execution, deduplication, and the single-execution policy

For each case, [run.py](../benchmarks/pytorch_correctness/run.py) creates a fresh subprocess and process group, separate stdout and stderr, a timeout, an exit status, an environment record, and complete tensor evidence. It runs only inside a Slurm allocation and step. Numerical violations use exit code 3; uncertain results and execution failures use exit code 2. A Slurm `FAILED` status therefore does not itself establish an invalid test, and exit code 0 does not replace an oracle verdict.

Screening job **763186** ran 14 candidates, yielding 13 provisional violations and 1 non-reproduction. The user subsequently required non-random cases and one execution per fixed case, superseding the planned repeats and variants. Jobs 763201 and 763226 were cancelled while pending and contributed no validation evidence. The final ten cases reused deterministic cells that had already executed. [final-review.json](../result/pytorch-correctness/research/final-review.json) checked result hashes, fixture AST hashes, conditions, and controls; no statistical repetitions were performed to claim stability.

| Excluded candidate | Rationale |
| --- | --- |
| 145093 compiled clone | Conservatively grouped with the compiled conjugate/negative alias issue 204033 rather than counting two APIs as independent root causes. |
| 198118 pinv | Screening used an earlier seeded-input fixture. The later fixed-matrix version had not executed and could not inherit the earlier result. |
| 204347 repeated cat | The original screening did not reproduce the issue. The later fixed-input reserve had not executed and was excluded. |
| 204496 BF16 grid gradient | Screening reproduced a discrepancy, retained as a reserve. The final ten cases already covered the selected mechanisms; this reserve did not increase the success count. |

The top-level `selection` field in `candidates.json` still contains the early repeated-execution proposal. The final policy is established by the user's later instruction, the selected rows' `reproduction_policy`, `final-review.json`, and the updated roadmap. The initial screening was not uniformly the final RNG-free corpus; its 14 candidates must not be presented as having the same evidential status as the final ten cases.

### 1.4 Deliverables and scope

Phase 1 added benchmark fixtures, a catalog, an isolated runner, runner tests, a Slurm reproduction script, the reproduction report, and this roadmap. It neither changed the production generator nor conducted an LLM discovery experiment. Inspection identified the need for later capabilities: the existing dense-tensor and training-loop path could not fully represent lazy neg/conj bits, MKLDNN and CSR layouts, context recursion, CUDA graph capture, or alias mutation after return.

<a id="phase-2-details"></a>
## Phase 2: Five property families in a shared worker

### 2.1 Implemented architecture

Phase 2 added `--strategy properties` while retaining the default `--strategy training`. The LLM supplies builders and small functions or modules; the runner controls transformations, state protocols, API invocation, references, and verdicts. Exact constants from the ten issue reproducers remain in benchmark fixtures rather than the generator.

The experiment used the following execution path:

```text
API + installed signature/docstring
    → API-semantic family registry + deterministic configuration cycle
    → LLM build(device, seed) [and forward(ctx) where required]
    → static contract check / limited repair
    → isolated execute_property worker
    → representation injection + snapshot + actual target/graph/backward evidence
    → reviewed property handler + applicability/control gates
    → pass / suspected failure / inapplicable / unassessed / execution failure
    → benchmark-side structural matching and independent post-hoc audit
```

This path implements a bounded addition to the three-layer architecture in AGENTS.md. Layer 1 uses a fixed cycle and predefined family mapping; it adds **neither learned coverage ranking nor a complete `alpha(sigma)` transition planner**. Layer 2 generates builders rather than filling every injection point of an arbitrary training loop. Layer 3 reuses isolated workers and evidence infrastructure while adding property handlers. These capabilities do not constitute a complete implementation of the intended architecture.

### 2.2 Added capabilities and affected files

| Component / location | Implementation |
| --- | --- |
| [contracts/programs.py](../src/mirage/contracts/programs.py), [cli.py](../src/mirage/cli.py) | Add `RunOptions.strategy`, CLI selection, and property run/report/replay dispatch; correct result-format handling and failed-replay exit status. |
| [contracts/exploration.py](../src/mirage/contracts/exploration.py), `property_configuration()` | Cycle deterministically through representations and operand subjects; alternate eager and Inductor for the gradient family. |
| [generator/program.py](../src/mirage/generator/program.py) | Compose prompts in `property_request()`; validate source, AST, and builder contract and produce an AST fingerprint in `analyze_property()`. |
| [runtime/representations.py](../src/mirage/runtime/representations.py) | Construct views preserving logical values in `equivalent_view()`; record conj/neg/layout metadata and CSR values metadata; avoid direct reads of opaque MKLDNN storage. |
| [runtime/program.py](../src/mirage/runtime/program.py), `run_property()` | Execute builders, inject representations, save initial state, invoke handlers, trace API entry and exit, and attach configuration, environment, and metadata. |
| [rules/properties.py](../src/mirage/rules/properties.py) | Implement the API-semantic registry, five handlers, materialization and observation, finite-value and control gates, and snapshot encoding/decoding. No issue-ID branches or benchmark tensor constants are included. |
| [runtime/worker.py](../src/mirage/runtime/worker.py) | Add `execute_property` mode and save `result.pt`, JSON summaries, traces, and process status. |
| [workflow.py](../src/mirage/workflow.py) | Manage deadlines, generation, repair, execution, and artifacts in `run_properties()`; validate checksums and replay saved initial data without an LLM in `replay_property()`. |
| [artifacts/report.py](../src/mirage/artifacts/report.py) | Separate generated, static-valid, target-executed, and suspected counts in `summarize_properties()`; do not infer FPR ground truth from oracle verdicts. |
| [benchmark evaluate.py](../benchmarks/pytorch_correctness/evaluate.py) | Connect ten handwritten cases to production workers with `validate`; run frozen LLM evaluation with `experiment`; match issues with `report`; independently recompute references from saved inputs with the later `audit` command. |
| [sbatch/lib/development.sh](../sbatch/lib/development.sh), [experiment session script](../sbatch/experiments/pytorch_correctness.sbatch) | Include benchmark code in iteration snapshots; separate serving and testing across two GPUs; reuse one healthy vLLM service and retain failures and timings. |

Existing components supplied API metadata resolution, the HTTP provider, process isolation, numerical comparison, and engine snapshots and checksums. For example, the source-only JSON schema in the HTTP provider was reused. Files such as `conditional.py`, `mr_actions.py`, and `artifacts/bundle.py` appeared in the approved scope but were not all rewritten.

### 2.3 Sampler rules and coverage limits

For a new configuration with attempt index `j`:

```python
representations = (
    'original', 'strided', 'offset', 'negative',
    'conjugate', 'singleton_stride', 'expanded',
)
representation = representations[j % 7]  # materialization / gradient_reference only
subject = j // 7
backend = 'inductor' if j % 2 == 0 else 'eager'  # gradient_reference only
seed = j
```

The worker selects from **direct positional Tensor arguments** in `args`, using `subject % tensor_argument_count`. It does not recurse into keyword arguments, sparse values, or module parameters. The same `seed` identifies the LLM request and attempt; the prohibition on RNG calls does not require tests to call `manual_seed(seed)`.

| Action | Implementation | Preconditions and limits |
| --- | --- | --- |
| original | Retain the builder output. | Does not automatically make tensors contiguous. |
| strided | Allocate storage with a final dimension of size 2, select `[...,0]`, and copy logical values. | Produces a stride-two view rather than searching arbitrary strides. |
| offset | Allocate `numel+1` storage, select `[1:]`, and reshape. | Sets storage offset to 1 rather than searching a range. |
| negative | `torch.complex(zeros_like(x), -x.detach()).conj().imag` | Supports FP32/FP64. Logical values remain x while `is_neg()` becomes True, typically with stride-two, offset-one storage. |
| conjugate | `x.detach().conj_physical().conj()` | Requires an already complex operand; does not convert a real dtype to complex. |
| singleton_stride | Find the first existing size-one axis, set its stride to 1, and apply `as_strided`. | Does not add a batch dimension or guarantee that the axis is the LU matrix-batch axis. |
| expanded | Select the first slice of the last axis with length greater than 1, then expand to the original shape. | Legal only when values repeat along that axis. Otherwise reject the action; input values cannot change merely to trigger a bug. |

Each action checks logical equality with `torch.equal(result, tensor)` before saving metadata. A failed precondition can produce `inapplicable` before target invocation. Context, graph, and compiled-alias configurations retain the original representation; their handlers construct the protocols. In particular, the `compiled_alias` handler explicitly uses Inductor even when the generic configuration field `backend` says eager.

**Coverage gaps.** The cycle includes no dense-to-MKLDNN or dense-to-CSR conversion, CSR `out.values()` stride injection, extreme-scale search, or insertion of a singleton batch dimension. The runtime can process supported structures if the LLM constructs them, but the presence of an adapter does not establish that the sampler explored the corresponding state. These gaps directly affect Phase 3 non-reproduction.

### 2.4 Builder contract and execution boundaries

`analyze_property()` requires nonempty source of at most 40,000 characters, permitted top-level torch/math imports and undecorated functions, and a unique `build(device, seed)`. It rejects assertions, exec/eval/I/O, listed RNG and uninitialized-data calls, private attributes, global/nonlocal statements, and most external attribute assignments. Local class bases are checked against the AST spelling `torch.nn.Module`; **`import torch.nn as nn` aliases are not resolved**.

This validator enforces a testing contract, not general Python legality or complete determinism. A `manual_seed` call alone is rejected, whereas the call-name blacklist cannot establish that implicit parameter initialization in `nn.Linear` avoids RNG use. Some legal assignments, such as `x.requires_grad = True`, can also fail the attribute-store rule. The Phase 3 static-valid metric must be interpreted within these limits.

The worker requires the builder to return a dictionary, with tuple/list `args` and dictionary `kwargs`. It limits the total logical `numel` of traversable tensors to 2,000,000; this is not a complete bound on backing-storage bytes or module parameters. Contract violations, snapshot failures, and failure to execute the target do not receive a correctness Pass.

### 2.5 Property semantics and oracle implementation

**A. `materialization`.** Copy the same args and kwargs into materialized controls. Dense tensors use detach, conj/neg resolution, contiguous conversion, and clone; CSR tensors rebuild indices and contiguous values; MKLDNN tensors pass through a dense logical payload before conversion back to MKLDNN. The handler applies only to reviewed APIs in the registry. Writable `out=` must not share observed storage with positional inputs, and BatchNorm must use inference statistics.

The handler supplements representation comparisons with API-specific mathematical references: scalar scatter loops with at most 8192 index elements; LU reconstruction via `lu_unpack`, solution via `linalg.solve`, and a residual gate with `atol=1e-8, rtol=1e-7`; BatchNorm via `(x-mean)/sqrt(var+eps)*weight+bias`; CSR addition via dense arithmetic; and copying via dtype/device conversion, expansion, and cloning of the materialized logical source. These branches encode API semantics, not issue-ID rules. For other pure operators, the materialized control may itself serve as the reference, providing differential evidence rather than a third independent oracle.

**B. `gradient_reference`.** The first operand must be a real tensor requiring gradients, with at most two dimensions and 4096 elements. Detaching it establishes an independent differentiation coordinate; the test evaluates the derivative with respect to that operand and does not preserve its construction history. The handler computes a scalar result and `torch.autograd.grad`, using logabsdet for slogdet. std/var require an all-element scalar reduction, positive degrees of freedom, and nonzero variance. det/slogdet require an unbatched square matrix with `cond<1e6` and `abs(det)>1e-12`. References use float64 formulas and are converted back to the input dtype:

```text
slogdet: d log|det(A)| / dA = A^(-T)
det:     d det(A) / dA = det(A) A^(-T)
std:     s = sqrt(sum((x-mean)^2)/(N-correction))
         ds/dx = (x-mean)/((N-correction)*s)
var:     dvar/dx = 2*(x-mean)/(N-correction)
```

The handler compares the actual result, materialized eager control, and mathematical reference with `atol=0, rtol=1e-4`. Inductor mode wraps lowered-graph invocation to record executed FX call nodes; runtime graph coverage therefore rests on execution evidence rather than a configuration flag. Backend lookup and `_base`/`_version` metadata use version-dependent internal instrumentation, while generated trigger operations use public APIs.

**C. `context_restore`.** The LLM supplies context-constructor args, kwargs, and a depth from 0..4. The handler records thread-local grad, inference, and autocast enabled/dtype/cache state, runs recursion with fresh contexts as a control, and then recurses through a function decorated with the same context object. State must be restored exactly at the exit checkpoint. Recursion is a runtime protocol; the LLM does not independently construct the full decorator reproducer.

**D. `graph_accumulation`.** The LLM supplies a module and 2..4 CUDA batches with fixed signatures. FX checks restrict operations and modules to a certified deterministic subset and reject mutable buffers; a deep copy of the module supplies the eager reference. Both executions start with `.grad=None`, sum real outputs into a loss for each batch, perform backward per batch, and compare final parameter gradients. `make_graphed_callables(module, first_batch)` constructs the target. Here, `reference` and `control` are the same eager result; the analytic sum-of-inputs calculation in Phase 1 provides separate mathematical corroboration for that fixture.

**E. `compiled_alias`.** The LLM supplies `inputs`, `output_path`, `delta`, and a top-level `forward(ctx)`. The handler builds separate materialized contexts, runs eager and Inductor forward, and first checks agreement of initial outputs. It selects a tensor via output_path and verifies that more than one tensor in the eager return collection shares its storage. It then applies the same `add_(delta)` to both executions and compares all returned values. Merely using a view internally while returning one independent tensor does not satisfy this MR. The handler checks observable effects among returned aliases, not complete preservation of input-alias topology.

The shared `validate()` function checks predicates, finite observations, control/reference agreement, and finally the actual/reference relation, in that order. Default tolerances are `atol=1e-5, rtol=1e-4`; context comparisons are exact, and gradients use the tolerances above. Nonfinite values and control disagreement yield `verdict=None, correctness=unassessed`. Automated results retain `confirmed_defect=False`; source review and independent audit are separate steps. These tolerances are conservative screening policies, not error bounds proved for every dtype or kernel.

### 2.6 Runtime evidence, snapshots, and replay

The target wrapper in `run_property()` records `api_enter` before invocation and adds `target_calls` and `api_exit` only after normal return. An exception trace can therefore show target entry without contributing to the completed-result `api_executed` metric. `execution_failed` does not universally mean that the target was never called. Completed results retain shape, stride, offset, storage group, dtype/device, requires_grad, leaf/base/version/grad_fn, neg/conj/layout, graph evidence, verdict, configuration, and environment.

| Artifact | Contents and purpose |
| --- | --- |
| `request.json` / `response.json` / `program.py` | Generation request, provider response with HTTP payload and token usage, and extracted source. |
| `execution/request.json` | Worker mode, source, configuration, and engine path; distinct from the LLM request. |
| `initial.pt` | Initial context and metadata. MKLDNN uses logical payload and layout tags; tagged views preserve shape/stride/offset on restoration; modules save state_dict. |
| `result.pt` / `summary.json` / trace / process logs | Full tensor observations, compact JSON metadata and verdicts, execution events, exceptions, and resource evidence. |
| `execution.sha256.json` / `engine-sha256.json` | Evidence and code checksums for replay verification, preventing a changed engine from silently standing in for the recorded experiment. |

Replay loads the saved engine and source, reconstructs required modules, restores initial data and weights, checks representation metadata, and reruns the property without calling the provider. An environment change suppresses a same-environment verdict unless explicitly allowed. Serialization increments version counters, so fresh-graph replay signatures ignore absolute version numbers; other metadata mismatches reject replay. This mechanism does not restore arbitrary saved autograd graphs or Python objects.

### 2.7 Integration validation and development fixes

`evaluate.py validate` obtains ten issue-specific builders from `INTEGRATION_CASES` and sends them through the same `execute_property` worker used for generated programs. `representation=original` means that the fixture already constructs the required conditions; detection does not depend on sampling them. The std configuration selects Inductor, and the compiled-alias handler also executes Inductor. Admission through the integration gate requires a completed process, target evidence, `suspected_failure`, and empty `control_differences`, rather than exit status alone.

Development exposed several Mirage integration problems. MKLDNN `torch.save` initially failed, limiting detection to 9/10; layout-tagged snapshots brought it to 10/10. A replay test exposed version-counter differences, followed by a fix for view/base reconstruction. The CLI was corrected to read property `records` instead of training `experiments`, and failed replay was changed to return a failing CLI exit status. These fixes addressed Mirage integration, not additional PyTorch bugs.

The final Phase 2 gate in `phase2-frozen/summary.json` detected 10/10 cases, and the full suite passed 70 tests. Negative-view and MKLDNN/affine-view replays succeeded with provider_calls=0. The later 12 property tests and 4 benchmark tests included ordinary compiled views, identical graph microbatches, finite/control/unknown gates, and known-answer checks of independent references. Their timing remains explicit: those later tests were not all run before Phase 3.

<a id="phase-3-details"></a>
## Phase 3: LLM generation and execution under a frozen protocol

### 3.1 Code freeze, budget, and model

`evaluate.py experiment` requires Phase 2 `detected=total=10`, copies the engine to frozen-engine, and verifies that Python source hashes match the validated version. Production `run_properties()` receives only the API and device for each target; issue IDs remain in outer benchmark routing and result directories, outside the prompt context. The local model is Qwen/Qwen3-Coder-Next-FP8, revision `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`, with temperature 0, top-p 1, max_tokens 4096, and context length 16384.

For each API, a worker first retrieves the installed signature and docstring, with metadata time measured separately. Generation and execution then receive up to 300 seconds; dispatch stops when remaining time is less than 45 seconds. Worker timeout is `min(120, remaining)`. The candidate cap is 100000, but every run ended at its time limit. Property generation and execution are serial; the second GPU serves vLLM rather than running another target API concurrently.

### 3.2 Prompts sent to the LLM

The following text comes from saved HTTP requests and request.json files and was checked against `property_request()` and `description()`. It records the prompts used in the experiment, not a retrospectively improved version. Each user message concatenates **the common prefix + family suffix + `\nContext:\n` + JSON context**.

**System message, verbatim:**

```text
Follow the supplied testing contract. Return only the requested JSON object.
```

**Common user prefix, verbatim text:** The display omits one trailing joining space; the actual concatenation retains that space before the suffix.

```text
Generate one small deterministic PyTorch property-test program from the installed API specification. Return JSON with exactly source. Use torch/math imports and define build(device, seed). build must return a dictionary whose named keys are specified below, never a tuple. Use explicitly initialized tensors (tensor, zeros, ones, arange, linspace), bounded dimensions and finite values. No RNG, external I/O, private APIs, assertions, or expected answers. The runner owns API invocation, state evidence, references, gradients and validation. Choose shapes, values, legal options and public representation operations yourself; vary them across attempt indices. Inputs may use ordinary views, strides, complex values, storage offsets or sparse layouts when the API supports them. Keep total storage below 2 million elements. Do not supply unrelated trainable parameters.
```

**Five family suffixes, verbatim:**

`materialization`:

```text
build returns args (tuple), kwargs (dict). They are the exact positional/keyword operands for the target API, including self for Tensor methods. The runner calls the API and compares legal logical materialization. Sparse CSR and MKLDNN values are supported. Avoid aliased writable operands and uninitialized data.
```

`gradient_reference`:

```text
build returns args and kwargs for the target. The first argument must be a real floating tensor requiring gradients. For std/var use a scalar reduction with dim=None; for determinant APIs use a square unbatched nonsingular matrix. The runner differentiates the scalar result (logabsdet for slogdet). Backend is controlled by configuration.
```

`context_restore`:

```text
build returns args and kwargs for constructing the target context manager, plus depth (integer 0..4). The runner checks nested decorator invocation against fresh contexts and observes entry/exit state. No threads are used.
```

`graph_accumulation`:

```text
build returns module (a deterministic torch.nn.Module on device), batches (2..4 tuples of fixed-shape input tensors). Use a small differentiable module without random layers, mutable buffers or control flow. A local class inside build is allowed. The runner sums real tensor outputs, accumulates gradients, and compares captured versus eager execution. Do not capture graphs yourself.
```

`compiled_alias`:

```text
build returns inputs (dict), output_path (list of tuple/list indices or dictionary keys), delta (finite scalar). Also define forward(ctx) returning tensors or a nested tensor structure. Choose a legal program whose output may contain views. The runner compiles forward, checks initial outputs and tests an add_ on the selected returned tensor against eager execution. Do not call compile, assert, or inspect storage/strides inside forward.
```

**Complete recorded user message: slogdet / attempt-0000.** The message includes the installed API documentation and configuration. Its `torch.randn` example comes from the public docstring, not an added instruction to generate random inputs. The common prompt forbids RNG use while the supplied documentation retains random examples. Both were present in the model input; a cleaned docstring cannot retrospectively represent this experiment.

```text
Generate one small deterministic PyTorch property-test program from the installed API specification. Return JSON with exactly source. Use torch/math imports and define build(device, seed). build must return a dictionary whose named keys are specified below, never a tuple. Use explicitly initialized tensors (tensor, zeros, ones, arange, linspace), bounded dimensions and finite values. No RNG, external I/O, private APIs, assertions, or expected answers. The runner owns API invocation, state evidence, references, gradients and validation. Choose shapes, values, legal options and public representation operations yourself; vary them across attempt indices. Inputs may use ordinary views, strides, complex values, storage offsets or sparse layouts when the API supports them. Keep total storage below 2 million elements. Do not supply unrelated trainable parameters. build returns args and kwargs for the target. The first argument must be a real floating tensor requiring gradients. For std/var use a scalar reduction with dim=None; for determinant APIs use a square unbatched nonsingular matrix. The runner differentiates the scalar result (logabsdet for slogdet). Backend is controlled by configuration.
Context:
{"contract": "training-program", "program_kind": "property-1", "api_context": {"api": "torch.linalg.slogdet", "torch_version": "2.15.0+cu132", "signature": "linalg.slogdet(A, *, out=None) -> (Tensor, Tensor)", "signature_source": "installed docstring (unparsed)", "documentation": "linalg.slogdet(A, *, out=None) -> (Tensor, Tensor)\n\nComputes the sign and natural logarithm of the absolute value of the determinant of a square matrix.\n\nFor complex :attr:`A`, it returns the sign and the natural logarithm of the modulus of the\ndeterminant, that is, a logarithmic polar decomposition of the determinant.\n\nThe determinant can be recovered as `sign * exp(logabsdet)`.\nWhen a matrix has a determinant of zero, it returns `(0, -inf)`.\n\nSupports input of float, double, cfloat and cdouble dtypes.\nAlso supports batches of matrices, and if :attr:`A` is a batch of matrices then\nthe output has the same batch dimensions.\n\n.. seealso::\n\n        :func:`torch.linalg.det` computes the determinant of square matrices.\n\nArgs:\n    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.\n\nKeyword args:\n    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.\n\nReturns:\n    A named tuple `(sign, logabsdet)`.\n\n    `sign` will have the same dtype as :attr:`A`.\n\n    `logabsdet` will always be real-valued, even when :attr:`A` is complex.\n\nExamples::\n\n    >>> A = torch.randn(3, 3)\n    >>> A\n    tensor([[ 0.0032, -0.2239, -1.1219],\n            [-0.6690,  0.1161,  0.4053],\n            [-1.6218, -0.9273, -0.0082]])\n    >>> torch.linalg.det(A)\n    tensor(-0.7576)\n    >>> torch.logdet(A)\n    tensor(nan)\n    >>> torch.linalg.slogdet(A)\n    torch.return_types.linalg_slogdet(sign=tensor(-1.), logabsdet=tensor(-0.2776))", "documentation_truncated": false, "documentation_sha256": "6898d9f3ddd28d5031ee08773f52353307b8bf7564996cc75244e0b078176a42", "kind": "callable"}, "configuration": {"api": "torch.linalg.slogdet", "device": "cpu", "seed": 0, "attempt": 0, "family": "gradient_reference", "representation": "original", "subject": 0, "backend": "inductor"}}
```

The original [request.json](../result/pytorch-correctness/763298/phase3/204407/attempt-0000/request.json) and [complete HTTP payload in response.json](../result/pytorch-correctness/763298/phase3/204407/attempt-0000/response.json) provide the source records. Requests for the other families are available for [materialization](../result/pytorch-correctness/763298/phase3/203078/attempt-0000/request.json), [context](../result/pytorch-correctness/763298/phase3/202706/attempt-0000/request.json), [graph](../result/pytorch-correctness/763298/phase3/181723/attempt-0000/request.json), and [compiled alias](../result/pytorch-correctness/763298/phase3/204033/attempt-0000/request.json).

The HTTP `response_format` uses a strict JSON schema:

```json
{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}
```

The schema constrains the JSON envelope, not the Python inside its source string. Strict JSON can still contain a Python SyntaxError and therefore does not establish executability. The context field `contract="training-program"` reuses the HTTP provider's source-only schema selector; `program_kind="property-1"` identifies the new contract. These builders are not full training programs.

### 3.3 Repair messages, stopping rules, and a successful generation chain

The controller permits at most two repairs for `invalid_program`. It appends the previous assistant content and the following diagnostic to the message list:

```text
Repair this diagnosed problem only:
ContractError: Unsupported or nondeterministic builder call: manual_seed
```

This diagnostic occurred in the first successful chain for each of the three reproduced issues. For [slogdet attempt-0004](../result/pytorch-correctness/763298/phase3/204407/attempt-0004/request.json), the previous response contained `manual_seed`; repair removed that call and retained the explicit matrix `[[2,1],[3,4]]`. The request kept configuration attempt/seed=3, negative, subject=0, and eager. The **runtime** then injected the negative-bit representation, computed the target gradient, and reported a violation after approximately 8.78 seconds.

Likewise, [autocast attempt-0003](../result/pytorch-correctness/763298/phase3/202706/attempt-0003/request.json) retained configuration attempt=2. [Copy attempt-0011](../result/pytorch-correctness/763298/phase3/200613/attempt-0011/request.json) retained attempt=10 and subject=1, causing the transformation to select the second positional Tensor, the CPU source. Directory attempt numbers differ from configuration attempts because repair consumes another loop iteration while retaining the original configuration; the discrepancy is not a recording error.

**Repair limits.** Runtime TypeError, snapshot/pickle errors, control disagreement, Pass, and suspected failure do not enter this static-repair branch. Issue-matcher conditions are not fed back to the LLM. Missing Tensor-method receivers and returned local functions therefore did not receive automatic runtime-feedback repair. If provider cancellation cannot be confirmed, the controller writes a marker and stops further service reuse rather than treating a closed client connection as proof of cancellation.

### 3.4 Deriving results from execution records

The run produced 1,154 parsed source candidates, including 661 repair responses. Of these, 310 passed the static contract, 226 workers completed, 130 completed with retained target-execution evidence, and 125 received applicable comparisons: 97 Pass and 28 suspected failures. Generation failure, MR inapplicability, control disagreement, and target violations retain distinct denominators rather than being collapsed into a single non-discovery category.

The original frozen matcher's 2/10 result was preserved. Independent audit and issue/source review subsequently supported three known issues. The original copy matcher checked only the conjugate flag and omitted the upstream negative/sliced variant; the correction changed benchmark attribution without changing prompts, the engine, or generated inputs. The two negative-bit/Inductor std gradient cases were independently corroborated mathematically but did not meet the tiny-input trigger for 197089. They do not increase reproduction of the ten known cases.

The independent audit recomputes references from saved initial inputs using Python Fraction inversion, math.fsum/hypot std, and scalar normalization and copying. It neither reruns targets nor uses production expectations as answers. The final 19 TP, 50 TN, 0 FP, and 0 FN apply only to the 69 audited cases. The remaining 1085 candidates, including 9 suspects, remain unknown. No ground truth is imputed for graph, LU, or compiled-alias cases without corresponding audit support, or for inputs outside the audit's supported domain.

### 3.5 Reproducibility, cost, and remaining capabilities

Job 763298 started the model service once, in 192 seconds, with zero restarts. Each iteration copied src/tests/sbatch/benchmarks before execution. Frozen Phase 3 iteration `iteration-pwpdkQ` took 2597 seconds; API loops totaled 2567.91 seconds and metadata discovery took 21.29 seconds. The allocation lasted 4208 seconds with 2 GPUs, or 8416 GPU-seconds. These figures measure allocated resources, not GPU utilization; overlapping service and test durations cannot be added as wall time. Cleanup removed test scratch, stopped the service, and closed the terminal; scheduler and accounting records confirmed that all owned steps exited.

The evidence supports a bounded conclusion: five controlled property protocols validated ten known positive fixtures, and blind generation combined with the sampler found three of them. It does not establish that the LLM autonomously generated ten complete MREs, that all rejected Python programs were illegal, that MR applicability generalizes to arbitrary models, or that the overall FPR is zero. Alias-aware AST validation, runtime-contract repair, nested/keyword operand sampling, and layout/scale exploration are directions suggested by the results, **not capabilities implemented or experimentally validated in this run**.

## Approved bilingual editorial revision

The user authorized language and organizational edits to this roadmap and `docs/pytorch-correctness-evaluation.md`, with matching Traditional Chinese reports under `reports/` using the same filenames. The revision applies the `ml-paper-writing` skill while preserving the experimental procedures, conclusions, numerical results, limitations, and verbatim prompts. The implementation sequence is to polish the English documents, produce corresponding Chinese translations, and verify section coverage, numerical and technical invariants, prompt fidelity, and links. This is a documentation-only revision; no implementation or experimental artifact changes or experiment reruns are authorized by this edit.

Editorial validation completed: the two English documents contain no Chinese text; each report preserves the corresponding section and paragraph coverage. All original link targets remain represented and local links and explicit anchors resolve. Numerical table rows agree with the pre-edit documents, and both evaluation editions match all 10 execution-funnel rows and 10 independent-audit rows in the saved artifacts. Python examples preserve their executable ASTs, JSON examples are unchanged, and the system message, complete user-message example, common prefix, five family suffixes, and repair message retain their recorded text. Manual review checked wording and qualifications across languages. Only the four authorized documents were edited; no implementation tests or experiments were rerun.
