# PyTorch correctness regression benchmark — Phase 1

Ten publicly reported correctness issues were reproduced with **explicit fixed inputs, one independent process per selected case**, on the prepared installation. Every selected case satisfied its trigger predicates, disagreed with its independent expected result beyond the declared tolerance, and passed its control comparison. No selected fixture calls an RNG or requires a race or random schedule. This follows the user's single-execution instruction; repeated-run statistical stability was not measured or required.

These are confirmed local reproductions of published issues, not newly discovered defects. Exact implementation root causes are distinguished below from supported hypotheses. Ten issue IDs do not prove ten independently patched upstream defects; grouping uses the available mechanism evidence, and related compiled clone/alias reports are counted only once.

## Environment and evidence

- Execution: Slurm job **763186**, step 2, host `25a-hgpn055`, account `ACD115198`, partition `8gpus`.
- PyTorch **2.15.0+cu132**, git `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; Python **3.14.10**.
- NVIDIA **H200**, CUDA **13.2**, cuDNN **92600**; MKL and MKLDNN available; eight CPU threads.
- [Reviewed selection, result hashes, function hashes and conditions](../result/pytorch-correctness/research/final-review.json).
- [Immutable code snapshot](../result/pytorch-correctness/763186/snapshot/) and [SHA-256 manifest](../result/pytorch-correctness/763186/snapshot-sha256.txt).
- [Raw run summary](../result/pytorch-correctness/763186/run/summary.json), [runner tests](../result/pytorch-correctness/763186/runner-tests.txt), and [job log](../result/pytorch-correctness/logs/mirage-correctness-763186.out).

The raw screening summary says `provisional_violation`, reflecting the original three-repeat policy. It remains immutable. The user subsequently replaced that policy with one deterministic execution; `final-review.json` records admission under the updated policy. Current selected function ASTs and the shared `check`/`meta` helpers exactly match their executed snapshot. Changes to unselected fixtures do not inherit screening evidence.

Every case directory contains `result.json`, full tensor evidence in `evidence.pt`, stdout/stderr and subprocess status. JSON previews limit tensors to 64 values; the comparison used full tensors. `candidates.json` is the machine-readable catalog, including rejected/reserve candidates. The installed version above applies to **every admitted row** below.

## Admitted cases

| Issue / affected API | Representation or state | Expected | Observed | Result |
|---|---|---|---|---|
| [203078](https://github.com/pytorch/pytorch/issues/203078), `Tensor.scatter_add` | CPU expanded index, stride `(1,0)`, destination wider than index | Row sums `[16,0,16]` | `[16,16,0]` | Reproduced |
| [204407](https://github.com/pytorch/pytorch/issues/204407), `linalg.slogdet` | `complex128` leaf → `conj().imag`, negative bit set, backward | Imaginary gradient `[[-.3,.2],[.1,-.4]]` | `[[-.3,.1],[.2,-.4]]` | Reproduced |
| [200599](https://github.com/pytorch/pytorch/issues/200599), `linalg.lu_solve` | CPU singleton batch dimension, LU stride `(4,2,1,2)` | Second solution `[1/6,1/3]` | `[5,-1]` | Reproduced |
| [202706](https://github.com/pytorch/pytorch/issues/202706), `autocast` | Deterministic recursive invocation of one decorated function | Entry disabled → exit disabled | Exit enabled | Reproduced |
| [181723](https://github.com/pytorch/pytorch/issues/181723), `cuda.make_graphed_callables` | Two distinct fixed microbatches, initial parameter `.grad=None` | Accumulated gradient `[14,27,40,53]` | `[22,44,66,88]` = twice second gradient | Reproduced |
| [204033](https://github.com/pytorch/pytorch/issues/204033), `compile` with conjugate imaginary view | Inductor output aliasing, post-return view mutation | Base imaginary values `[-101,-100,-99,-98]` | `[-1,0,1,2]`; storage no longer shared | Reproduced |
| [204365](https://github.com/pytorch/pytorch/issues/204365), `nn.functional.batch_norm` | MKLDNN inference, stride-two weight/bias | Explicit affine normalization | Maximum absolute error **69.599757** | Reproduced |
| [201268](https://github.com/pytorch/pytorch/issues/201268), `add(out=...)` | CPU CSR output with stride-two values | Values `[11,22]`, interleaved storage unchanged | Values `[11,-99]`; interleaved slot overwritten | Reproduced |
| [197089](https://github.com/pytorch/pytorch/issues/197089), `std` | Inductor CPU FP32, fixed `[-2^-100,2^-100]`, backward | Std `2^-100`; gradient `[-.5,.5]` | Std zero; gradient zero | Reproduced |
| [200613](https://github.com/pytorch/pytorch/issues/200613), `Tensor.copy_` | Blocking CPU→CUDA broadcast of conjugate source | `[1-2j,1-2j]` | `[1+2j,1+2j]` | Reproduced on NVIDIA CUDA |

The matrix covers representation correctness as well as training state. The slogdet, CUDA graph and std cases execute backward. BatchNorm here explicitly uses inference statistics (`training=False`); the other forward/side-effect cases do not claim backward validation. Their inclusion follows the requested tensor-representation scope.

## Minimal reproductions, oracles and root evidence

All minimal operations below are implemented with their controls in [cases.py](../benchmarks/pytorch_correctness/cases.py). To execute one through the prepared runner, from the repository root:

```bash
mkdir -p result/pytorch-correctness/logs
sbatch sbatch/replay/pytorch_correctness.sbatch --cases 203078
```

Default execution selects these ten cases once each. It does not call an LLM. No model, environment or dependency was installed.

### 203078 — expanded-index destination addressing

```python
base = torch.zeros(3, 32)
index = torch.tensor([[0], [2]]).expand(2, 16)
actual = base.scatter_add(0, index, torch.ones(2, 16))
```

`scatter(0)` compares all output elements against the scalar scatter definition; a materialized index is the control. Integer-valued FP32 additions here are exactly representable, so tolerance is zero. Runtime confirms legal indices, zero stride and the destination/index width mismatch.

Root evidence: the v2.15.0 [expanded-index CPU kernel](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cpu/ScatterGatherKernel.cpp#L910) defines `K` using index width and addresses destination rows with `row*K`. This directly explains the observed wrong row when destination width is 32 instead of 16. The upstream discussion independently describes the same addressing error. Related scatter APIs are not separate benchmark entries.

[Actual evidence](../result/pytorch-correctness/763186/run/203078/repeat-0-variant-0/result.json).

### 204407 — negative-view slogdet gradient

```python
z = torch.tensor([[-4j, -1j], [-2j, -3j]], dtype=torch.complex128, requires_grad=True)
loss = torch.linalg.slogdet(z.conj().imag).logabsdet
loss.backward()
```

`slogdet(0)` verifies the negative bit and nonsingularity. The independent derivative is `-A^{-T}` for `A=-Im(z)`, computed by the elementary 2×2 inverse formula. Central finite differences of Python scalar `log(abs(ad-bc))` corroborate it; a `resolve_neg()` input is the control. Tolerance: `atol=1e-10`, `rtol=1e-9`; observed error is `0.1`, while finite differences agree within `1e-8`.

Root status: local evidence isolates a negative-view-dependent transposition of off-diagonal backward values. The exact responsible native implementation statement is **not confirmed** by a patched build or source-level trace. It differs from 200599's singleton-batch indexing trigger (this case is unbatched), but shared lower-level dependencies are not ruled out merely by different issue IDs.

[Actual evidence](../result/pytorch-correctness/763186/run/204407/repeat-0-variant-0/result.json).

### 200599 — singleton batch stride in LU solve

```python
A = torch.tensor([[[2.,1.],[1.,3.]], [[4.,1.],[2.,5.]]], dtype=torch.float64)
B = torch.tensor([[[1.],[2.]]], dtype=torch.float64)
LU, piv = torch.linalg.lu_factor(A)
actual = torch.linalg.lu_solve(LU[:, None], piv[:, None], B)
```

`lu_solve(0)` uses the elementary 2×2 solution formula and a contiguous-LU control (`atol=rtol=1e-12`). The incorrect second lane has equation residual `[18,3]`. Runtime confirms the singleton dimension and stride `(4,2,1,2)`.

Root evidence: [v2.15.0 apply_lu_solve](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/BatchLinearAlgebraKernel.cpp#L1065) reads `LU.stride(-3)` then advances by a flattened matrix index. Here that stride is 2 although matrices are 4 elements apart. The source and contiguous control support the upstream analysis.

[Actual evidence](../result/pytorch-correctness/763186/run/200599/repeat-0-variant-0/result.json).

### 202706 — reentrant autocast state restoration

```python
@torch.autocast('cpu', dtype=torch.bfloat16)
def f(n):
    if n:
        f(n-1)
# Starting with autocast disabled:
f(1)
# torch.is_autocast_enabled('cpu') is now True.
```

`autocast(0)` wraps entry in an explicitly disabled context and records enabled state, dtype and cache flag before/after. A recursive function that constructs a fresh context per invocation is the control. Comparison is exact. No threads, random schedules or timing barriers are involved.

Root evidence: the installed `torch/amp/autocast_mode.py` reuses `autocast_instance` in the decorator and stores prior thread-local state in mutable `self.prev*` fields. An inner call replaces the outer call's saved state. [Installed source excerpts and hashes](../result/pytorch-correctness/research/installed-source-evidence.json) corroborate the issue's explanation.

[Actual evidence](../result/pytorch-correctness/763186/run/202706/repeat-0-variant-0/result.json).

### 181723 — CUDA graph parameter-gradient accumulation

The minimal module is `forward(x) = x * weight`, with `weight=[1,2,3,4]`. Capture it using `torch.cuda.make_graphed_callables`, set `weight.grad=None`, then execute `sum().backward()` for these fixed inputs:

```python
x1 = [[1.,2.,3.,4.], [2.,3.,4.,5.]]
x2 = [[10.,20.,30.,40.], [1.,2.,3.,4.]]
```

`cuda_graph(0)` checks the first gradient `[3,5,7,9]` before the second backward. The exact independent derivative is `sum(x1,dim=0)+sum(x2,dim=0)`; ordinary eager autograd is the control. All arithmetic is exactly representable, so comparison is exact.

Root evidence: installed `torch/cuda/graphs.py` returns detached aliases of `static_grad_inputs` after backward replay. The observed total equals twice the second gradient, matching the upstream buffer-overwrite mechanism. This benchmark does not instrument an internal pointer hook, so the buffer-stealing step is supported by source/upstream analysis rather than a new local internal trace. It is distinct from compiled output-view alias preservation.

[Actual evidence](../result/pytorch-correctness/763186/run/181723/repeat-0-variant-0/result.json).

### 204033 — compiled output view loses aliasing

```python
def fn(x):
    base = torch.complex(x, x+1)
    view = base.conj().imag
    view.add_(3)
    return base, view
base, view = torch.compile(fn, backend='inductor', fullgraph=True)(torch.arange(1.,5.))
view.add_(100)
```

`compiled_alias(0)` verifies both initial returned values before testing the subsequent mutation. The negative imaginary view should subtract 100 from the base imaginary part; eager execution is the control. Comparison is exact. The compiled output storages are observed to be different.

Root status: failure to preserve the returned view/base alias relationship is directly observed; the specific compiler pass responsible remains a hypothesis. [145093](https://github.com/pytorch/pytorch/issues/145093), a related compiled negative-view clone report, remains a reserve and is **not counted separately**.

[Actual evidence](../result/pytorch-correctness/763186/run/204033/repeat-0-variant-0/result.json).

### 204365 — MKLDNN BatchNorm affine strides

```python
x = torch.arange(32, dtype=torch.float32).reshape(2,4,2,2)/10
w = torch.tensor([1.,10.,2.,20.,3.,30.,4.,40.])[::2]
b = torch.tensor([0.,10.,0.,20.,0.,30.,0.,40.])[::2]
actual = torch.nn.functional.batch_norm(
    x.to_mkldnn(), torch.zeros(4), torch.ones(4), w, b, training=False).to_dense()
```

`batch_norm(0)` independently evaluates `x*w/sqrt(1+eps)+b`; contiguous affine parameters form the control. Tolerance is `atol=2e-6`, `rtol=1e-6`, allowing ordinary FP32 evaluation-order error. The observed maximum error is about 69.6, not a tolerance-boundary discrepancy.

Root evidence: [v2.15.0 MKLDNN normalization](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mkldnn/Normalization.cpp#L151) wraps weight/bias with `itensor_from_tensor` before oneDNN computation. This release uses a different helper spelling from the issue's later-source excerpt. The stride-two failure and materialization control corroborate a stride-handling problem; the precise responsibility inside the wrapper/oneDNN boundary remains unpatched and not conclusively isolated here.

[Actual evidence](../result/pytorch-correctness/763186/run/204365/repeat-0-variant-0/result.json).

### 201268 — CSR output values are written contiguously

Construct diagonal 2×2 CSR inputs with values `[1,2]` and `[10,20]`. Construct `out` with the same indices and values view `backing[::2]`, where `backing=[-99,-88,-99,-77]`, using `check_invariants=True`, then call `torch.add(lhs,rhs,out=out)`.

`sparse_add(0)` expects logical values `[11,22]` and records whether the interleaved storage is preserved. A contiguous CSR output is the control; comparison is exact. Both wrong logical values and changed interleaved storage were observed. The fixture uses a valid, bounded allocation.

Root evidence: [v2.15.0 mkl_result_copy_](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/mkl/SparseBlasImpl.cpp#L141) copies `nnz*sizeof(scalar_t)` bytes directly to the output values pointer. That does not honor a stride-two destination. Other APIs using the same copy path are not counted as additional issues.

[Actual evidence](../result/pytorch-correctness/763186/run/201268/repeat-0-variant-0/result.json).

### 197089 — compiled std loses representable output and gradient

```python
x = torch.tensor([-2.**-100, 2.**-100], requires_grad=True)
y = torch.compile(lambda t: torch.std(t, correction=0), fullgraph=True)(x)
grad = torch.autograd.grad(y, x)[0]
```

`std_gradient(0)` uses the elementary identities `std([-a,a], correction=0)=a` and gradient `[-1/2,1/2]`. Eager is the control. Both expected output and gradients are representable in FP32. Tolerance is `atol=0`, `rtol=1e-6` for this specifically justified tiny exact-value case. A global `1e-5` absolute tolerance would conceal the output failure, but the gradient error of `0.5` independently establishes a substantive violation.

Root status: intermediate FP32 variance underflow in Inductor is the upstream explanation and is consistent with the fixed-value result. The benchmark does not claim a newly validated compiler patch or inspect generated machine code. The fixture is minimized to two values, following the upstream discussion; no random inputs remain.

[Actual evidence](../result/pytorch-correctness/763186/run/197089/repeat-0-variant-0/result.json).

### 200613 — conjugate bit reapplied during broadcast transfer

```python
src = torch.tensor([1+2j], dtype=torch.complex64).conj()
dst = torch.zeros(2, dtype=src.dtype, device='cuda')
dst.copy_(src)  # blocking, broadcasts one logical value
```

`broadcast_copy(0)` synchronizes before reading the destination and compares against explicit `[1-2j,1-2j]`. The `resolve_conj()` source is the control. Comparison is exact; no numerical reduction, dtype conversion or asynchronous ordering is needed.

The original report tested ROCm only. **This run directly establishes the same symptom on NVIDIA H200 / CUDA 13.2**, rather than inferring CUDA behavior. [v2.15.0 Copy.cu](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/Copy.cu#L557) materializes a contiguous broadcast temporary and restores the original conjugate/negative flags onto it before recursive copying. Source and control support double application of the logical transformation.

[Actual evidence](../result/pytorch-correctness/763186/run/200613/repeat-0-variant-0/result.json).

## Screening, validation and resource accounting

Fourteen candidates were screened once: thirteen violations with passing controls and one non-reproduction. Final admission contains ten deterministic cases above. Remaining candidates:

- 145093: related compiled negative-view clone behavior; grouped conservatively with 204033, not counted twice.
- 198118: screening used seeded complex inputs. A constant-matrix reserve fixture is now present, but that revised fixture has **not** run and is not admitted.
- 204347: repeated-cat example did not reproduce in screening. A later fixed-input reserve fixture is unexecuted; neither version is admitted.
- 204496: deterministic BF16 grid-gradient screening produced 128 instead of 2048. Retained as a reserve; the selected ten already cover distinct mechanisms. No extra admission is inferred.

Validation actually performed:

- Python AST checks and `bash -n` passed; default one-GPU `sbatch --test-only` passed. These are static/preflight checks only.
- Three runner tests passed inside job 763186 in **1.971 s**: applicability/control/nonfinite/tolerance gates, completeness of requested cells, and timeout/nonzero-exit preservation.
- After the user's policy update, the pure-Python single-cell aggregation test passed in **0.010 s**. This lightweight check imported no PyTorch and ran no training or compilation.
- Ten selected case functions and shared helpers were verified against the executed snapshot by AST equality, with hashes retained in `final-review.json`. Numerical comparisons reuse the existing `mirage.inspector.comparison.differences` implementation.
- No true-negative population was sampled; ten passing controls do **not** establish a general false-positive rate.

Job 763186 ran for **87 allocated seconds**, one GPU: **87 allocated GPU-seconds**, not GPU utilization. It ended `FAILED`, exit **3:0**, because the runner deliberately preserves oracle violations as nonzero exits; it was not an execution failure. The job log preserves start/end/elapsed/status. Jobs 763201 and 763226 were cancelled while pending, each with zero allocation time. Zero-GPU preflight was rejected (`--gpus-per-node` accepts only 1..8); no CPU-only Slurm job was launched. No partition fallback or direct interactive computation was used. Final `squeue` inspection showed no owned jobs; no vLLM service was started.

## Framework assessment recorded at the end of Phase 1

This section records the **Phase 1** assessment. Subsequent approved implementation and experiment results are recorded in [the Phase 2–3 evaluation report](pytorch-correctness-evaluation.md). At the time of this assessment, integration and LLM discovery were not yet run. The harness reuses Mirage's numerical comparator, while its semantic predicates/references are reviewed benchmark fixtures. It does not feed issue examples or constants into the generator. Current framework gaps are based on code inspection, not invented generated-test outcomes:

| Capability | Current implementation | General extension needed |
|---|---|---|
| Representation generation/evidence | Strided/offset/clone/shared inputs; no conj/neg flags; only top-level input transformations | Verified zero-stride, singleton-stride and conjugate/negative representations; registered operand/parameter targeting |
| Sparse and MKLDNN adapters | Registered inputs must be dense strided; general metadata assumes storage/stride access | Layout-aware metadata, snapshots and legal adapters, with explicit unsupported cases |
| Training lifecycle | Fixed training hooks and bounded history primitives | CUDA graph capture/replay and context-state restoration with aligned observations |
| Aliasing/mutation | Conservative effect rejection; output snapshots copy values | Preserve alias evidence and support post-return mutation observations under a stated contract |
| Compilation | Compiled target evidence exists; pure eager/compiled equivalence is not a general rule | Conditional execution-mode relation with verified inputs, graph execution and required observables |
| Numerical/reference oracles | Finite comparison and global precision tolerances; restricted capability catalog | Independently justified reference/property oracles, gradient checks and scale-aware tolerances |
| Evaluation attribution | General exploration reports, no issue-level blind evaluation | Separate API execution, trigger attainment, oracle applicability, violation detection and issue matching |

Phase 2 should prove detection using the preserved fixtures through the improved framework, then freeze implementation/configuration before Phase 3. Benchmark-specific inputs remain solely in fixture/catalog data. General transformations and oracle contracts must not branch on issue IDs or embed these constant tensors.

At the end of Phase 1, Phase 3 was **not run**. Valid Test Rate, Executable Test Rate, Issue Reproduction Rate, Correctness Violation Detection Rate, False Positive Rate, Time to First Detection and Number of Generated Test Cases are therefore **not measured**, not zero. Planned blind evaluation uses installed API specifications and generic allowed configurations, with 300 seconds per target API for generation plus execution; service startup and allocation cost are reported separately. Trigger not generated, unsupported execution, inapplicable oracle and applicable-but-missed violation must remain separate outcomes. False-positive denominators require independent adjudication; unknown results remain unknown.

## Phase 2–3 implementation scope (subsequently approved)

The user subsequently approved the following concrete scope and authorized Phase 3 immediately after successful Phase 2 validation. The evaluation report records implementation decisions and actual execution.

1. Extend the existing lifecycle contract with bounded execution setup and post-forward observation hooks. This permits context restoration, graph capture/replay, and returned-view mutation checks while preserving target-call and backward evidence. Keep pure/value MRs separate from effect/alias contracts; do not broadly disable existing applicability checks.
2. Add reusable representation actions and metadata for zero strides, singleton batch strides, conjugate/negative views, registered affine operands, sparse values and device transfers. Verify logical values, shape/dtype, storage relationships and reached predicates at runtime. Representation changes that alter autograd or alias semantics require their own input relation.
3. Add reviewed oracle families: logical materialization equivalence; bounded alias-mutation propagation; context entry/exit restoration; eager/compiled equivalence; graph/eager gradient accumulation; and mathematical linear-algebra/reduction properties. Each has explicit preconditions, required observations, numerical policy and a documented independent basis. API-specific mathematics may be used, but issue IDs and fixture constants must not appear in general generation/sampling/oracle code. Unknown semantics remain unassessed.
4. Route preserved benchmark fixtures through framework execution/validation to measure detection independently of LLM generation. Add positive/negative tests for missing predicates, unrelated target calls, invalid inputs, legal numerical differences and unsupported layouts. Re-run the current framework suite in Slurm after changes; freeze the validated engine before evaluation.
5. Run blind generation with the prepared Qwen3-Coder-Next-FP8 provider: installed API specifications plus general configuration only, 300 seconds per target API including generation and execution. Do not give the model issue URLs, fixture source, exact benchmark values or oracle answers. Record prompts, responses, repairs, inputs, code/environment hashes, all outcomes and resource costs. Post-hoc issue attribution reads evidence separately from generator inputs.
6. Compute the requested metrics with explicit denominators. Record generation failure, invalid execution, trigger not reached, oracle inapplicable and triggered-but-undetected outcomes separately. Report FPR only for independently adjudicated negatives; preserve unknown labels. Keep time-to-first-detection censored when the budget expires without detection. Provider pretraining contamination cannot be ruled out by withholding prompts and is an experimental limitation, not something this protocol can prove absent.

Approved exact editable file scope:

| Purpose | Files |
|---|---|
| Contracts and generation | `src/mirage/contracts/programs.py`, `src/mirage/contracts/exploration.py`, `src/mirage/generator/program.py` |
| Runtime and evidence | `src/mirage/runtime/program.py`, `src/mirage/runtime/representations.py`, `src/mirage/runtime/mr_actions.py`, `src/mirage/runtime/worker.py` |
| Conditional/reference oracles | `src/mirage/rules/conditional.py`, new `src/mirage/rules/properties.py`, `src/mirage/inspector/comparison.py` |
| Controller, CLI and reporting | `src/mirage/workflow.py`, `src/mirage/cli.py`, `src/mirage/artifacts/report.py`, `src/mirage/artifacts/bundle.py` |
| Tests | `tests/test_program.py`, `tests/test_exploration.py`, `tests/test_cli.py`, `tests/test_artifacts.py`, `tests/test_pytorch_benchmark.py`, new `tests/test_properties.py` |
| Benchmark/evaluation | `benchmarks/pytorch_correctness/README.md`, `benchmarks/pytorch_correctness/candidates.json`, `benchmarks/pytorch_correctness/cases.py`, `benchmarks/pytorch_correctness/run.py`, new `benchmarks/pytorch_correctness/evaluate.py` |
| Slurm | `sbatch/lib/development.sh` (include benchmark code in immutable iteration snapshots), new `sbatch/experiments/pytorch_correctness.sbatch`, `sbatch/README.md` |
| Documentation | `docs/roadmap-003-pytorch-correctness-benchmark.md`, this report, new `docs/pytorch-correctness-evaluation.md`, `README.md` |

Files in this list would change only when required for the approved capabilities. No unrelated refactor, dependency installation, project-configuration change, upstream issue posting or unrelated job operation is proposed. Artifacts remain under `result/pytorch-correctness/<run-id>/`; prepared model resources remain under `/work/asciibase64/Mirage/`.

Use the existing two-GPU development-session arrangement for real-model iterations: one GPU for vLLM, one for tests, eight allocated CPUs split between steps, same account/partition and four-hour bound. Verify scheduler constraints/preflight before submission. Reuse one healthy service within the task; preserve failed iteration evidence; stop processes/service and release the allocation before completion. Service startup, generation/execution time and allocation time are separate measurements. Scheduler rejection or queue delay does not justify bypassing Slurm.
