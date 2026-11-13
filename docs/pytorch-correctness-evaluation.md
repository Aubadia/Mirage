# PyTorch correctness: Phase 2–3 evaluation

Completed on 2026-11-13: Phase 2 detected **10/10** deterministic benchmarks; the
frozen LLM experiment reproduced **3/10** known issues from **1,154 generated
candidates**. Independent audit rates below apply only to its explicitly reviewed
subset. All owned Slurm processes and the model service have exited.

For detailed explanations, see the [per-issue reproduction analysis](#per-issue-analysis). The [roadmap](roadmap-003-pytorch-correctness-benchmark.md#phase-3-details) documents the implementation workflow and complete prompts.

## Scope and evidence

The ten fixed-input upstream benchmarks and their corroborating Phase 1 evidence are documented in [the reproduction report](pytorch-correctness-reproduction.md). Phase 2 tests whether Mirage can recognize those failures when handed the triggers. Phase 3 tests whether a real model generates triggers without receiving issue descriptions, reproducer code, expected answers or issue identifiers. These are different measurements.

This evaluation adds `--strategy properties` alongside the existing training-program strategy. It reuses installed API discovery, the isolated provider, runtime workers, traces, numerical comparisons and engine snapshots. The LLM constructs inputs and small modules/functions; reviewed runtime strategies construct the source/follow-up executions and own the oracles. This is guided property-program generation, not unconstrained generation of an entire training loop. The existing training strategy remains available.

## Reusable properties

| Family | Preconditions and input relation | Observables and output relation |
| --- | --- | --- |
| Logical materialization | Certified API; supported tensor layout; logical values and options preserved; writable `out` must not alias input storage. Batch normalization requires inference with running statistics. | Target output versus materialized execution. Additional semantic references for scatter addition, LU solve, batch normalization, sparse addition and copying must agree with the control. LU reference also satisfies the defining linear equation. |
| Analytic gradient | Small real differentiable operands; scalar all-element variance/std reduction with positive degrees of freedom and nonzero variance, or a well-conditioned unbatched nonsingular square determinant input. Target backward must execute; compiled mode requires recorded graph execution. | Scalar output and derivative with respect to the first operand, compared with a float64 formula and materialized eager control. The operand is a fresh independent differentiation coordinate; prior construction history is not tested. |
| Context restoration | A supported context/decorator; bounded recursion; same arguments and initial thread-local state; no threads or races. | Grad mode, inference mode, autocast enabled state, dtype and cache setting after a reused decorated invocation must match entry state and a fresh-context control. Exact equality. |
| Captured gradient accumulation | Small deterministic certified FX graph; no mutable buffers; 2–4 CUDA microbatches with equal signatures; identical initial parameters; gradients cleared to None before accumulation. | Captured and eager parameter gradients at the end of the same microbatch sum. Eager is a differential reference, not an independent mathematical proof. |
| Compiled returned alias | Same initial logical inputs; compiled graph executes; initial returned values agree; selected eager returned tensor shares storage with another returned tensor; legal finite `add_` mutation. | Returned values after the same mutation must agree between eager and compiled execution. This checks observable alias behavior, not equality of raw addresses across executions. |

Default screening tolerance is `atol=1e-5, rtol=1e-4`; analytic gradients use `atol=0, rtol=1e-4` to avoid erasing tiny-scale violations. Context state is exact. Comparisons promote floating/complex values for error calculation and check shape/dtype. Nonfinite values or disagreement between the semantic reference and control yield **correctness unassessed**, not a failure. Tolerances are screening policies, not universal error bounds or proof of a defect.

The deterministic representation cycle covers original, stride-two, nonzero offset, negative-bit, conjugate-bit, singleton-stride and expanded representations. An action is used only if it preserves logical values and is legal for that operand; otherwise it is inapplicable. Target calls record actual metadata. Unsupported API/property combinations receive no correctness pass. Sparse CSR values and opaque MKLDNN tensors have explicit adapters; this is not general support for every sparse/quantized layout.

Snapshot replay saves inputs, code and checksums and uses no LLM. MKLDNN uses a logical payload plus layout tag. Tensor serialization increments version counters, so replay compares all recorded metadata except absolute version numbers; these properties construct fresh graphs and do not test saved-graph version-counter behavior. Other metadata changes reject replay. This is deliberately narrower than arbitrary autograd-history restoration.

## Experiment protocol

- One local Qwen/Qwen3-Coder-Next-FP8 service, temperature 0, top-p 1, maximum 4096 generated tokens, prepared model revision recorded in serving artifacts.
- One API per sequential run, maximum 300 seconds for generation plus execution, after separately measured installed-API metadata discovery. The controller stops dispatching new requests with less than 45 seconds remaining; this reserve is reported, not counted as generated work.
- Generic API-family configuration and deterministic representation cycling; attempt index changes the generation seed/configuration. No benchmark fixture imports from production code. Installed public docstrings are retained verbatim in request artifacts, including their own examples.
- At most two repairs for static builder-contract errors. Runtime errors, oracle outcomes and benchmark trigger checks do not supply issue-specific repair hints.
- Each candidate executes once in an isolated worker. Timing failures, unsupported programs and invalid inputs are separate from oracle violations. No repeated trials are used to estimate a probabilistic issue-reproduction rate.
- Freeze implementation and prompts after Phase 2 validation. Preserve all initial smoke/validation failures as development evidence; exclude them from the frozen Phase 3 population.
- One Slurm allocation, two GPUs: one serving step and one testing step. Source snapshots are immutable per iteration. Compilation caches are disposable job-local storage.

A deterministic benchmark does not imply deterministic model generation across hardware/software environments. The recorded run is a single measured experiment, without confidence intervals over independent generation runs.

## Metric definitions

`generated` counts parsed responses containing a source program. `contract_valid / generated` is the static Valid Test Rate; it does not establish runtime legality. `completed target executions / contract_valid` is the Executable Test Rate. Raw statuses also distinguish completed inapplicable programs from actual target executions.

An issue is reproduced only after a generated candidate is independently reviewed against the benchmark mechanism and oracle evidence. Structural trigger matching alone is reported separately as `benchmark_consistent_detection`. Phase 2's detection rate uses ten independently admitted positive benchmark fixtures; Phase 3 sensitivity needs independently adjudicated positive generated cases. False Positive Rate requires independently adjudicated negatives and uses FP/(FP+TN); unadjudicated cases are excluded, never presumed negative. An observed false-discovery fraction would instead be FP/(TP+FP) and must not be mislabeled FPR.

Time to First Detection includes generation, repair and execution from the start of that API budget. These are automated oracle detections later independently attributed to an issue; subsequent review latency is excluded. Targets without detections are censored at their actual elapsed time, not assigned a fabricated detection time. Generated-case count includes duplicates; unique source fingerprints are available for separate inspection.

## Completed execution results

Slurm job **763298** tested PyTorch **2.15.0+cu132** (Python 3.14.10, NVIDIA H200).
The separate model-serving environment used vLLM 0.33.0 / PyTorch 2.14.0+cu132;
that serving version is **not** the tested framework version. The model snapshot
was `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`.

Phase 2's final gate detected **10/10** admitted fixtures through the production
property worker. The full suite passed **70 tests in 63.277 s**. Two checked
saved-input replays (negative view and MKLDNN with strided affine values) completed
with unchanged environment and zero provider calls. After the frozen experiment,
**12 property tests** (including ordinary compiled-view and identical-microbatch
CUDA-graph negative controls) and **4 benchmark/audit tests** also passed. This
validates the tested corpus and guards; it is not a proof of all possible programs.

The frozen experiment produced **1,154 candidates** (including **661 repair
responses**) and **748 API-scoped source fingerprints**; parseable programs use
AST fingerprints, otherwise raw source hashes. All ten API budgets completed.

| Issue / API | Generated | Static valid | Target executed | Applicable | Suspected | First benchmark detection |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [203078](https://github.com/pytorch/pytorch/issues/203078) `torch.Tensor.scatter_add` | 106 | 32 | 1 | 1 | 0 | Not detected (censored) |
| [204407](https://github.com/pytorch/pytorch/issues/204407) `torch.linalg.slogdet` | 122 | 29 | 10 | 9 | 3 | 8.78 s |
| [200599](https://github.com/pytorch/pytorch/issues/200599) `torch.linalg.lu_solve` | 132 | 38 | 20 | 18 | 0 | Not detected (censored) |
| [202706](https://github.com/pytorch/pytorch/issues/202706) `torch.autocast` | 97 | 41 | 34 | 34 | 22 | 10.73 s |
| [181723](https://github.com/pytorch/pytorch/issues/181723) `torch.cuda.make_graphed_callables` | 161 | 0 | 0 | 0 | 0 | Not detected (censored) |
| [204033](https://github.com/pytorch/pytorch/issues/204033) `torch.compile` | 98 | 35 | 2 | 0 | 0 | Not detected (censored) |
| [204365](https://github.com/pytorch/pytorch/issues/204365) `torch.nn.functional.batch_norm` | 108 | 27 | 19 | 19 | 0 | Not detected (censored) |
| [201268](https://github.com/pytorch/pytorch/issues/201268) `torch.add` | 163 | 32 | 17 | 17 | 0 | Not detected (censored) |
| [197089](https://github.com/pytorch/pytorch/issues/197089) `torch.std` | 62 | 34 | 22 | 22 | 2 | Not detected (censored) |
| [200613](https://github.com/pytorch/pytorch/issues/200613) `torch.Tensor.copy_` | 105 | 42 | 5 | 5 | 1 | 33.97 s |

“Applicable” means a Pass or Suspected Failure was actually issued. “Target
executed” requires completed target execution, and can include a subsequently
inapplicable/unassessed oracle. There were 226 completed workers, 130 completed
target executions, 125 applicable comparisons, 97 Passes and 28 suspected failures.
Generated counts include contract-invalid programs and duplicate attempts.

- **Static Valid Test Rate:** 310/1,154 = **26.86%**.
- **Executable Test Rate:** 130/310 = **41.94%**, using completed target executions
  per statically valid program. Target execution per generated candidate is 11.27%.
- **Reviewed known-issue reproduction:** **3/10 = 30%**: 204407, 202706, 200613.
- **Phase 2 positive-benchmark detection:** **10/10 = 100%**.
- **Independent generated-case audit:** TP=19, FN=0, FP=0, TN=50. Sensitivity is
  **19/19 = 100%** and FPR is **0/50 = 0% within this audited subset only**.
  Only 69/1,154 candidates were independently adjudicated; the remaining 1,085
  include generation failures, unsupported cases and unadjudicated observations.
  Nine of the 28 suspects remain unadjudicated by this conservative audit.

The independent positive subset contains 17 known-benchmark-associated cases and
two additional std anomalies. The latter do not increase the known-issue count.
The uncertainty guard leaves near-threshold numerical discrepancies unadjudicated;
equal-dtype copying and context state use exact semantic comparisons. No general
zero-FPR or no-false-negative claim follows from this subset.

### Attribution and additional observations

The original frozen matcher counted **2/10 issues**. Its copy rule recognized
conjugate sources only. Independent review found that [issue 200613](https://github.com/pytorch/pytorch/issues/200613)
explicitly includes negative views and sliced-layout temporary-copy cases. The
generated CPU negative strided source therefore belongs to that issue: logical
values 11..22 became -11..-22 on CUDA, maximum absolute error 44. The corrected
post-hoc matcher and independent copy oracle support **3/10**. The original
`frozen-matcher-evaluation.json` is preserved; only attribution changed. Generation,
execution, source inputs and production code were unchanged.

The first slogdet case uses `[[2,1],[3,4]]`: the expected derivative is
`[[0.8,-0.6],[-0.2,0.4]]`, whereas the observed off-diagonal entries are swapped,
with maximum error approximately 0.4. The first autocast case has depth 2 and
returns with autocast enabled and cache disabled, instead of restoring disabled
and enabled, respectively. Both have passing controls and independent references.

Std attempts 38 and 52 use values 1..12, a negative-bit operand and Inductor.
The standard deviation is correct, but the gradient sign is reversed; maximum
error is approximately 0.27735 for FP64 and FP32. Python scalar mathematics
independently corroborates both discrepancies. They **do not reproduce issue
197089's tiny-input underflow condition**. They are retained as one related
candidate group with unconfirmed root cause/novelty, not two new confirmed bugs
and not extra successes in the ten-issue benchmark.

<a id="per-issue-analysis"></a>
### Per-issue analysis: Reproduction and non-reproduction

The analysis distinguishes observed results, obstacles supported by code inspection, and improvements that have not been tested. All ten issues reproduced in the fixed Phase 1 fixtures and the Phase 2 production-worker integration. Non-reproduction below refers specifically to **Phase 3 automatic generation**; it does not indicate that the bug was fixed or that the oracle cannot detect it.

The [Phase 3 roadmap](roadmap-003-pytorch-correctness-benchmark.md#phase-3-details) records the actual prompts, five family suffixes, complete request example, repair message, and sampler rules. The [Phase 2 account](roadmap-003-pytorch-correctness-benchmark.md#phase-2-details) defines implementation boundaries. The model did not receive issues and rewrite their MREs. Reviewed property protocols guided builder generation, and the sampler injected representations.

#### A. Denominators across execution stages

| Issue | Generated | Static rejection | Execution failure | Worker completed | Target completed | Pass | Suspected | Inapplicable | Unassessed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 203078 | 106 | 74 | 18 | 14 | 1 | 1 | 0 | 13 | 0 |
| 204407 | 122 | 93 | 0 | 29 | 10 | 6 | 3 | 19 | 1 |
| 200599 | 132 | 94 | 2 | 36 | 20 | 18 | 0 | 18 | 0 |
| 202706 | 97 | 56 | 7 | 34 | 34 | 12 | 22 | 0 | 0 |
| 181723 | 161 | 161 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 204033 | 98 | 63 | 33 | 2 | 2 | 0 | 0 | 2 | 0 |
| 204365 | 108 | 81 | 0 | 27 | 19 | 19 | 0 | 8 | 0 |
| 201268 | 163 | 131 | 2 | 30 | 17 | 17 | 0 | 13 | 0 |
| 197089 | 62 | 28 | 0 | 34 | 22 | 20 | 2 | 12 | 0 |
| 200613 | 105 | 63 | 22 | 20 | 5 | 4 | 1 | 15 | 0 |

Inapplicable cases include both transformations whose preconditions failed before target invocation and MRs that failed applicability checks after the target completed. The entire column therefore cannot be interpreted as valid executions missed by the oracle. The full execution funnel is:

```text
1154 generated
 ├─ 844 static contract / syntax rejections
 └─ 310 static-valid
     ├─ 84 execution failures
     └─ 226 completed workers
         ├─ 96 transformation-inapplicable before recorded target completion
         └─ 130 completed target executions
             ├─ 4 MR-inapplicable (2 LU residual gates, 2 compiled-alias gates)
             ├─ 1 correctness unassessed (slogdet reference/control disagreement)
             └─ 125 applicable comparisons
                 ├─ 97 Pass
                 └─ 28 Suspected Failure
```

`execution_failed` describes failure of the worker as a whole. Errors can occur in builders, snapshots, or reference computation, so the label alone does not establish a PyTorch target defect. Similarly, `api_enter` and completed `target_calls` are distinct evidence. Error counts below record the **first reported reason per candidate**, not every potential defect. For example, a source rejected for an `nn.Module` alias can also contain `manual_seed`; counting both would double-count that candidate.

#### B1. 203078 — scatter_add: Expanded-index kernel conditions were absent

**Required trigger.** Phase 1 uses dim=0 scatter, index stride `(1,0)`, and a destination wider than the index to enter the specialized expanded-index path. An arbitrary non-contiguous tensor does not establish this combination. Expected fixture row sums are `[16,0,16]`; observed sums are `[16,16,0]`.

**Observed generation.** Of 106 candidates, 74 were rejected statically: 61 for `manual_seed`, 11 for unsupported top-level structure, and 2 for SyntaxError. Of the 32 entering workers, 18 placed the integer dim first in args and omitted the Tensor-method receiver, causing descriptor TypeError. The other 14 workers completed, but 13 were inapplicable at the transformation gate. Only attempt-0003 completed the target and received Pass.

The [attempt-0003 source](../result/pytorch-correctness/763298/phase3/203078/attempt-0003/program.py) uses dim=1 and destination/index/source shapes `(2,3)`. The sampler gives the destination stride `(6,2)`, while the index remains contiguous with stride `(3,1)`. The destination is not wider than the index. Its [runtime summary](../result/pytorch-correctness/763298/phase3/203078/attempt-0003/execution/summary.json) corroborates these conditions.

**Why reproduction failed.** The sole completed target tested another legal strided configuration without reaching the benchmark's index state. The 13 pre-target inapplicable cases comprise 4 expansions that would change logical values, 4 conjugate actions on real tensors, 3 missing singleton axes, and 2 negative actions on integer operands. The cycle neither redesigns index values to permit expansion nor guarantees selection of the index operand.

**Failure classification.** Receiver-contract and generation errors combined with a trigger-coverage gap. There is no evidence of a positive execution missed by the oracle. Operand-role mapping, generation of values compatible with expansion, and runtime-contract repair are possible general improvements that remain unimplemented and untested in this run.

#### B2. 204407 — slogdet: Asymmetric values, negative-bit injection, and an analytic oracle

**First successful chain.** Attempt-0003 was rejected for `manual_seed`. The [attempt-0004 repair request](../result/pytorch-correctness/763298/phase3/204407/attempt-0004/request.json) retained seed/attempt=3, negative, subject=0, and eager, and requested only correction of that contract violation. The LLM supplied the fixed real matrix `[[2,1],[3,4]]` with requires_grad=True; its source did not construct `conj().imag`. The worker applied the general negative-bit action while preserving logical values. Recorded metadata show shape `(2,2)`, stride `(4,2)`, offset 1, and `neg=True`. The handler then differentiated logabsdet with respect to a fresh leaf coordinate.

```text
A = [[2, 1], [3, 4]], det(A)=5
Correct d(log|det(A)|)/dA = A^(-T)
expected = [[ 0.8, -0.6], [-0.2, 0.4]]
observed = [[ 0.8, -0.2], [-0.6, 0.4]]   (rounded for display)
max absolute error ≈ 0.40000002
```

The error exceeds the `atol=0, rtol=1e-4` comparison threshold by a wide margin, while the materialized eager control agrees with the reference. Independent Python Fraction elimination reconstructs the inverse-transpose from the saved matrix, corroborating the discrepancy without reusing the production verdict as ground truth. The [source](../result/pytorch-correctness/763298/phase3/204407/attempt-0004/program.py), [target and gradient evidence](../result/pytorch-correctness/763298/phase3/204407/attempt-0004/execution/summary.json), and [readable observations](../result/pytorch-correctness/763298/phase3/reviewed-observations.json) document this chain. First detection occurred at 8.78 seconds.

**Full API budget.** The 122 candidates included 93 static rejections: 41 manual_seed calls, 50 SyntaxErrors, and 2 attribute-store restrictions. All 29 remaining workers completed; 19 failed transformation preconditions. The 10 completed targets yielded 3 suspected failures, 6 Passes, and 1 unassessed result. Suspects were attempts 4, 24, and 67. The latter two executed under Inductor with runtime graph evidence and maximum gradient errors of approximately 2.63415.

**Why one negative-bit case passed.** [Attempt-0046](../result/pytorch-correctness/763298/phase3/204407/attempt-0046/program.py) used the symmetric matrix `[[2,1,.5],[1,3,1],[.5,1,4]]` and received Pass despite having the negative bit set. The inverse of an invertible symmetric matrix is symmetric, so inverse and inverse-transpose coincide. This input cannot expose the observed transpose-type discrepancy. It demonstrates that reaching a target state does not guarantee an observable violation; the Pass is not evidence of an oracle miss.

[Attempt-0121](../result/pytorch-correctness/763298/phase3/204407/attempt-0121/execution/summary.json) remained unassessed because a control/reference gradient difference of approximately `1.043e-7` failed the strict zero-atol comparison. Its operand was strided with `neg=False`. This conservative gate reduces assessed coverage but avoids treating a comparison with an inconsistent reference as a confirmed target violation.

**Attribution.** The LLM selected discriminating asymmetric values, the sampler injected the representation, runtime evidence established backward execution, and the mathematical oracle detected the discrepancy. The result cannot be attributed solely to autonomous LLM generation of a complete issue MRE.

#### B3. 200599 — lu_solve: Ordinary systems were covered, singleton-batch strides were not

**Required trigger.** The issue concerns LU matrix indexing with a size-one batch axis. Phase 1 uses `LU[:, None]`, shape `(2,1,2,2)`, and stride `(4,2,1,2)`. Non-contiguity alone does not identify this trigger: ordinary column-major LU also need not satisfy row-major contiguity.

The 132 candidates produced 94 static rejections—56 manual_seed, 12 randn, and 26 syntax errors—and 2 execution failures, from unsupported complex arange and invalid pivots. Of 36 completed workers, 20 completed the target: 18 received Pass and 2 were inapplicable because `reference_residual=False`. The other 16 failed transformation applicability before target execution.

[Attempt-0001](../result/pytorch-correctness/763298/phase3/200599/attempt-0001/program.py) explicitly constructs an invertible 4×4 matrix, calls `lu_factor`, and uses rhs shape `(4,3)`. Its [runtime metadata](../result/pytorch-correctness/763298/phase3/200599/attempt-0001/execution/summary.json) show LU shape `(4,4)` and stride `(1,4)`, with no batch axis. Both the reference solution and reconstructed-system residual passed. This valid result does not exercise singleton-batch addressing.

The sampler changes only an **existing** size-one axis; it does not insert `[:,None]`. Four completed-worker rejections reported `No singleton dimension`, and the final structural matcher found no reproduction. Attempts 38 and 89 completed the target but failed the alternative solution's residual gate. Without further evidence of a target defect, those inapplicable results cannot be counted as missed bugs.

**Failure classification.** Shape/state coverage was insufficient, with additional budget consumed by generation and reference-gate failures. The evidence does not show a confirmed triggering execution incorrectly assigned Pass.

#### B4. 202706 — autocast: Reentrant execution exposed the exit-state invariant

Of 97 candidates, 56 were rejected statically—29 manual_seed and 27 syntax errors—and 7 failed during execution. Execution failures included an incorrect RNG-state setting, undefined variables, out-of-bounds views, and the logical-tensor budget. The 34 completed target and oracle executions yielded 22 suspected failures and 12 Passes.

After manual_seed repair, [attempt-0003](../result/pytorch-correctness/763298/phase3/202706/attempt-0003/program.py) specified CPU, BF16, enabled=True, cache_enabled=False, and depth=2. The builder also created tensors, but this oracle observes thread-local context state rather than their numerical values. `context_restore()` constructs the recursive decorated invocation and fresh-context control.

```text
entry / fresh-context control: grad=True, inference=False, amp=False,
                              dtype=bfloat16, cache=True
reused-decorator exit:         grad=True, inference=False, amp=True,
                              dtype=bfloat16, cache=False
```

The [summary](../result/pytorch-correctness/763298/phase3/202706/attempt-0003/execution/summary.json) records exact mismatches in `amp` and `cache`, with an agreeing control. First detection occurred at 10.73 seconds. These are discrete-state violations rather than small floating-point differences amplified by a tolerance choice.

All 12 Passes used depth=0. The 22 suspects comprised 2 at depth 1, 15 at depth 2, 4 at depth 3, and 1 at depth 4. The executed samples distinguish non-reentrant from reentrant contexts, but they do not prove failure for every autocast parameter combination with depth>0.

**Attribution.** The LLM supplied legal constructor options and depth; the framework provided the reusable recursion/restoration protocol. The same protocol supports other registered grad and inference contexts. Its design was informed by the benchmark, so the result is not prior-free discovery by the LLM alone.

**Audit limits.** The conservative independent audit adjudicated only 13 of the 22 suspects; 9 remain unknown. Its builder allowlist rejects unlisted calls such as `min(4, seed % 5)` in [attempt-0008](../result/pytorch-correctness/763298/phase3/202706/attempt-0008/program.py). That rejection means the auditor cannot certify the builder, **not that ambient-state corruption has been demonstrated**. Eight Passes also remained unaudited. Independent ground truth for this API is therefore 13 TP and 4 TN, not the production counts of 22 and 12.

#### B5. 181723 — CUDA graph accumulation: Frontend rejection prevented oracle evaluation

The benchmark requires backward execution on at least two different microbatches of the same captured module, accumulating from `.grad=None`. The observable distinguishes the correct `g1+g2` from the incorrect `2*g2`. Generating an `nn.Module` class alone does not establish that state.

All 161 candidates were rejected statically: 131 first encountered `Only local torch.nn.Module classes are supported`, and 30 had SyntaxError. No worker executed the target or captured backward, and none received Pass or Suspected Failure. Oracle false-negative rates cannot be estimated from these unexecuted programs.

[Attempt-0000](../result/pytorch-correctness/763298/phase3/181723/attempt-0000/program.py) used:

```python
import torch.nn as nn
# local class inside build:
class SmallNet(nn.Module):
    ...
```

The alias is legal PyTorch syntax, but the validator requires the base's `ast.unparse(...)` to equal **exactly** `torch.nn.Module`, without import-alias resolution. This source also contains `manual_seed`, and `nn.Linear` performs default parameter initialization. Fixing the alias alone would not establish compliance with the deterministic contract. Because the first rejection masks later potential problems, the evidence does not support a claim that a one-line change would make all 161 candidates executable.

The Phase 2 handwritten builder used the full `torch.nn.Module` spelling and explicit parameter values, demonstrating detection of that positive fixture. A later identical-microbatch negative control received Pass. Neither result removes the Phase 3 frontend limitation.

**Failure classification.** Framework-contract and AST-validation limits, together with source syntax errors. The results do not establish that the LLM failed to conceive a plausible accumulation scenario or that the execution oracle missed an executed anomaly.

#### B6. 204033 — compiled alias: Internal views did not establish returned-alias applicability

The benchmark observes shared storage effects between a returned complex base and its conjugate imaginary view: mutating the view after return must affect the base. Agreement of initial values is insufficient to establish that property.

Of 98 candidates, 63 were rejected statically—37 manual_seed, 18 top-level-structure violations, and 8 syntax errors. Another 33 failed during execution: 18 returned local `forward` functions as data and could not be pickled, 9 lacked top-level `forward`, 3 did not return dictionaries from build, and the remainder had shape, key, or storage errors. None produced a completed target result counted by this metric.

Only [attempt-0043](../result/pytorch-correctness/763298/phase3/204033/attempt-0043/program.py) and [attempt-0067](../result/pytorch-correctness/763298/phase3/204033/attempt-0067/program.py) completed the compiled target. Both summaries record:

```json
{"initial_outputs_agree": true,
 "returned_alias": false,
 "compiled_graph_executed": true}
```

- Attempt 43 computes `y=x.view(5,2)`, then `z=y+1`, and returns only `{'output': z}`. Addition allocates new storage, and the returned collection contains one tensor.
- Attempt 67 computes `z=x+y` and `view_z=z.view(2,8)`, then returns only `{'view': view_z}`, without the base or another tensor sharing storage. Although mutation selects a subview, the MR's returned collection still contains only one observable member.

Both cases are therefore **Inapplicable, not Pass**. Eager `has_base=True` and compiled `has_base=False` do not alone establish a bug: this MR requires observable mutation propagation to another returned alias. A metadata difference does not by itself constitute a semantic violation.

Runtime evidence confirms Inductor graph execution. The generic configuration field `backend='eager'` does not indicate an absence of compilation because this handler explicitly selects Inductor. Neither case provided the negative/conjugate returned-alias trigger, so 204033 was not reproduced.

**Failure classification.** Builder and serialization contract failures, followed by returned structures that did not satisfy the MR's input and observable relations. These results do not establish correctness of all compiler alias behavior.

#### B7. 204365 — BatchNorm: Dense inputs did not exercise MKLDNN affine strides

Of 108 candidates, 81 were rejected statically—75 manual_seed and 6 syntax errors. The 27 completed workers included 8 transformation-inapplicable cases and 19 completed targets, all receiving Pass. No recorded target layout was MKLDNN.

[Attempt-0005](../result/pytorch-correctness/763298/phase3/204365/attempt-0005/program.py) constructed an ordinary dense `(4,8)` input, running_mean=0, running_var=1, weight=1, bias=0, and training=False. The sampler applied the negative action to the input, yielding stride `(16,2)`, offset 1, and the negative bit. However, [metadata](../result/pytorch-correctness/763298/phase3/204365/attempt-0005/execution/summary.json) show weight and bias with stride `(1,)` and input layout `torch.strided`.

The benchmark requires **an MKLDNN input together with stride-two affine tensors**. Changing the stride or negative bit of an ordinary dense input does not test MKLDNN weight/bias indexing. The sampler has no `.to_mkldnn()` action and selects only direct positional tensors rather than searching arbitrary keyword parameters. Although this example places the affine operands in args, its selected configuration transforms the input. The input's stride-two layout does not establish affine-stride coverage.

**Failure classification.** A joint layout/backend and operand-role coverage gap. Phase 2 adapters and the normalization oracle can process the constructed MKLDNN fixture, but generation did not reach that path. The 19 dense Passes do not indicate that the bug was fixed.

#### B8. 201268 — CSR add(out): Dense broadcasting did not exercise the sparse output buffer

Of 163 candidates, 131 were rejected for `manual_seed`. Of the 32 entering workers, 2 failed because of out-of-bounds storage or incompatible shapes. The remaining 30 completed, comprising 13 transformation-inapplicable cases and 17 completed targets, all receiving Pass. No CSR target metadata were recorded.

[Attempt-0001](../result/pytorch-correctness/763298/phase3/201268/attempt-0001/program.py) tested dense `(3,4)` plus dense `(4,)` broadcasting with `alpha=2` and no `out` keyword. This is a legitimate addition test but does not satisfy the trigger: non-contiguous CSR output values causing copy-out to write the wrong storage locations.

The strided action transforms ordinary positional tensors. It neither converts dense tensors to CSR nor constructs a CSR `out` and injects an `out.values()[::2]` layout. CSR support in metadata and materialization does not establish that the sampler provided that configuration.

**Failure classification.** Joint generation gaps in sparse layout, output operands, and values storage, compounded by static rejection. The 17 Passes do not contribute TN labels to the independent audit: the add auditor requires equal shapes and dtypes, and these executions lie outside its supported conditions. All post-hoc labels for this API remain unknown; production Pass does not supply independent negative ground truth.

#### B9. 197089 — std: Compilation and backward were covered, the tiny scale was not

Phase 1 uses FP32 `[-2**-100, 2**-100]`, whose squared terms can underflow even though the correct std is nonzero and its gradient is `[-.5,.5]`. Inductor execution is only one condition; numerical scale also identifies this benchmark trigger.

Of 62 candidates, 28 were rejected statically—8 attribute-store restrictions, 19 syntax errors, and 1 manual_seed call. The 34 completed workers included 12 transformation-inapplicable cases. The other 22 completed target backward, yielding 20 Passes and 2 suspects; 12 had runtime compiled-graph evidence. No saved result matched the tiny-scale condition.

[Attempt-0000](../result/pytorch-correctness/763298/phase3/197089/attempt-0000/program.py) and several later candidates used values 1..12, shape `(3,4)`, and correction=1. Their variance is outside the benchmark's underflow regime. Representation actions preserve logical values and therefore do not rescale 1..12 to an extreme magnitude. Configuration also lacks magnitude or exponent exploration. Compilation and backward coverage can thus coexist with a numerical-trigger gap.

**Two additional suspects do not count as benchmark reproductions.** [Attempt-0038](../result/pytorch-correctness/763298/phase3/197089/attempt-0038/program.py), in FP64, and [attempt-0052](../result/pytorch-correctness/763298/phase3/197089/attempt-0052/program.py), in FP32, retain values 1..12 while the sampler injects the negative bit. Inductor reverses the gradient sign. With mean=6.5 and std=`sqrt(13)`, the first element's derivative should be `-5.5/(11*sqrt(13))≈-0.138675`, but the observed sign is positive; the maximum discrepancy is approximately 0.27735. Python `math.fsum/hypot` corroborates both discrepancies, and the 20 ordinary cases receive independent TN labels.

These representation/backward anomalies differ from the tiny-input issue 197089. Whether they share a root cause with other compiled negative-bit problems remains unresolved. They form one candidate group for further analysis, without a claim of a new root cause. A suspected failure in the same API is not automatically reproduction of the benchmark issue.

**Failure classification.** Execution-mode and backward states were reached, but numerical-scale coverage was missing. No confirmed trigger for 197089 was shown to have been missed by the oracle.

#### B10. 200613 — copy_: Negative-bit injection on the CPU source exposed a known variant

Of 105 candidates, 63 were rejected statically—62 manual_seed and 1 syntax error. Another 22 failed during execution: 20 omitted the Tensor-method `other` argument, while the remainder had build-type or reference-operand errors. Of 20 completed workers, 15 were transformation-inapplicable. The 5 completed targets yielded 4 Passes and 1 suspect.

The LLM builder in [attempt-0011](../result/pytorch-correctness/763298/phase3/200613/attempt-0011/program.py) uses explicit arithmetic to construct CPU source values `11..22`, shape `(3,4)`, and a CUDA destination. It returns args `(dst, src)` with `non_blocking=False`. Repair retains configuration attempt=10, so `representation='negative'` and `subject=10//7=1` select the second Tensor, src. The LLM did not write a conjugate reproducer.

```text
Before injection: CPU real source, logical values 11..22
After injection: CPU source shape (3,4), stride (8,2), offset 1, neg=True
                 logical values remain 11..22
destination: CUDA, contiguous, neg=False
actual copy result: -11..-22
expected / materialized control: 11..22
max absolute error: 44
```

The [runtime summary](../result/pytorch-correctness/763298/phase3/200613/attempt-0011/execution/summary.json) and [readable observations](../result/pytorch-correctness/763298/phase3/reviewed-observations.json) retain this result. An independent Python logical-indexing reference for equal-dtype copying distinguishes the sign error without a tolerance-based argument; the audit labels the case TP. First automatic detection occurred at 33.97 seconds.

The Phase 1 fixture uses conjugation and broadcasting, whereas the generated case uses a negative, non-contiguous source with equal source/destination shapes. [Issue 200613](https://github.com/pytorch/pytorch/issues/200613) explicitly includes negative-view and sliced-layout tests. The saved [Copy.cu excerpt](../result/pytorch-correctness/research/upstream-source-excerpts.json) also shows source conj/neg flags being reset on the temporary-copy path. The sign reversal is consistent with that mechanism, supporting attribution to the same known issue rather than another bug. This attribution rests on outputs, metadata, the upstream description, and source evidence; it does not claim native branch tracing or upstream-patch verification.

**Why the original matcher missed the case.** The frozen rule required CPU source `conj=True`, so the oracle-detected `neg=True` case was not attributed to the benchmark, leaving an initial issue rate of 2/10. A benchmark-side correction admitted neg/conj and temporary-copy conditions, while preserving the [original matcher output](../result/pytorch-correctness/763298/phase3/frozen-matcher-evaluation.json); the reviewed rate became 3/10. This was a matcher false negative, not a correctness-oracle false negative. Generator inputs were unchanged.

#### C. Cross-cutting causes and limits of inference

| Layer | Observed limitation | Unsupported inference |
| --- | --- | --- |
| Syntax / static contract | Strict JSON can contain SyntaxError; manual_seed is rejected; Module aliases are unresolved. | Every rejected source is illegal PyTorch, or removing seed calls alone guarantees reproduction. |
| Runtime contract | Missing self/other, returned local functions, absent top-level forward, and out-of-bounds views. | An execution exception establishes a framework defect or proves a legal benchmark trigger was reached. |
| State sampling | No MKLDNN/CSR conversion, singleton-batch insertion, or numerical-scale sweep; only direct positional tensors are transformed. | A layout adapter establishes layout coverage, or any non-contiguous tensor satisfies a particular kernel trigger. |
| MR applicability | Compiled cases lack returned alias pairs; LU residual gates fail. | Completed target invocation warrants Pass, or different metadata warrants failure. |
| Oracle/reference | slogdet control disagreement remains unassessed; successful cases have mathematical or state evidence. | Reference disagreement alone proves a target bug, or every Suspected Failure is independently confirmed. |
| Attribution | The narrow copy matcher omits a known negative-view variant; std exposes a different discrepancy. | Any failure in an API reproduces its benchmark issue, or a matcher miss is an oracle miss. |

The common prompt prohibits RNG use but retains a `seed` parameter, and installed docstrings contain `randn` examples. The model repeatedly generated manual_seed calls. This motivates prompt ablation, but **no controlled experiment identifies which prompt component caused those calls or quantifies a benefit from revision**. Alias-aware validation, runtime-contract repair, and richer representation/scale sampling are likewise testable next steps, not completed results of this run.

The run establishes at least one independently corroborated generated witness for three known issues. Phase 1/2 confirmed the other seven locally, but Phase 3 encountered earlier-stage obstacles or lacked discriminating conditions. Absence of observed oracle misses applies only to the 19 generated cases with independent positive labels. Unexecuted, MR-inapplicable, unsupported, and unaudited cases remain unresolved.

#### D. Independent-audit denominators

| API issue label | TP | FP | TN | FN | Unknown, including unexecuted |
| --- | ---: | ---: | ---: | ---: | ---: |
| 203078 | 0 | 0 | 1 | 0 | 105 |
| 204407 | 3 | 0 | 5 | 0 | 114 |
| 200599 | 0 | 0 | 0 | 0 | 132 |
| 202706 | 13 | 0 | 4 | 0 | 80 |
| 181723 | 0 | 0 | 0 | 0 | 161 |
| 204033 | 0 | 0 | 0 | 0 | 98 |
| 204365 | 0 | 0 | 19 | 0 | 89 |
| 201268 | 0 | 0 | 0 | 0 | 163 |
| 197089 | 2 | 0 | 20 | 0 | 40 |
| 200613 | 1 | 0 | 1 | 0 | 103 |

The table reports existing [independent-audit.json](../result/pytorch-correctness/763298/phase3/independent-audit.json) results, not experiments added during documentation revision. Candidates without results receive no numerical adjudication. Completed executions outside the auditor's supported semantics, shapes, dtypes, or condition bounds also remain unknown. Examples include broadcast add outside equal-shape support and context builders using `min` outside the allowlist. LU, graph, and compiled-alias cases lack corresponding independent post-hoc implementations.

Audit references are computed from initial inputs rather than copied from production references. Their numerical uncertainty margins remain conservative review policies, not error theorems for all PyTorch kernels. Excluding unknown cases also prevents extrapolation of the observed 0/50 FPR to the full population. Of the 19 audited positives, 17 correspond to known benchmarks and 2 are the std candidates above. Three reproduced issue identities and 19 positive cases count different units.

### Timing, resources and cleanup

The ten nominal 300-second API budgets used **2,567.91 s** in generation/execution
loops (42.80 min), plus **21.29 s** of installed-API metadata discovery. Individual
loops ended after 255.01–263.64 s because of the predeclared 45-second dispatch
reserve. The complete experiment iteration took **2,597 s**. Completed/failed
worker intervals summed to **864.44 s**; this is worker elapsed time, not GPU
utilization. Provider-reported usage was **1,642,117 input tokens** and **325,341
output tokens**. Timings use the reused service and warmed compilation caches,
not a cold-start benchmark.

The model started once in **192 s**, with **zero restarts**. The full development
and evaluation allocation lasted **4,208 s**, with two GPUs: **8,416 allocated
GPU-seconds (2.338 GPU-hours)**. These include startup, idle/coordination time and
failed development iterations. Per-iteration timings are preserved separately;
overlapping service and testing step durations are not added as wall time.

The test scratch directory was removed, the interactive testing terminal closed,
and all owned steps exited. `squeue -j 763298` was empty and `sacct` reported the
job `COMPLETED`, exit `0:0`; the service step was deliberately cancelled during
shutdown. This job completion status is not used as an MR Pass. No task-owned
vLLM service or allocation remains running.

### Evidence index

All paths below refer to immutable inputs/results or derived reports in job 763298:

- [Final Phase 2 integration](../result/pytorch-correctness/763298/phase2-frozen/summary.json)
- [Frozen experiment protocol](../result/pytorch-correctness/763298/phase3/protocol.json)
- [Metric denominators and runtime coverage](../result/pytorch-correctness/763298/phase3/final-analysis.json)
- [Original frozen matcher results](../result/pytorch-correctness/763298/phase3/frozen-matcher-evaluation.json)
- [Reviewed issue attribution](../result/pytorch-correctness/763298/phase3/reviewed-attribution.json)
- [Independent audit and every retained unknown](../result/pytorch-correctness/763298/phase3/independent-audit.json)
- [Readable selected observations](../result/pytorch-correctness/763298/phase3/reviewed-observations.json)
- [Per-iteration timing and verified cleanup](../result/pytorch-correctness/763298/session-accounting.json)

The production engine checksum recorded by workers is
`d2c85c8d352abc278b4039685797027b03b79a357ed67160ac7564e170126708`.
The experiment driver required the frozen engine to match the validated Phase 2
Python sources. The benchmark-side audit and attribution correction were added
separately and never supplied to the generator.

### Independent post-hoc audit design

During the frozen run, a separate evaluator was added to audit saved observations.
It cannot feed information back to generation. It computes small determinant
inverses using exact Python rational elimination, scalar std using `math.fsum`
and `math.hypot`, exact bounded integer scatter sums, equal-dtype broadcast copying,
inference batch normalization, and bounded scalar/CSR addition. Context-state
checks require a builder that cannot change ambient context and compare with the
fresh worker defaults. Unsupported cases remain unadjudicated.

The audit reconstructs expected values from saved inputs and reads the framework
verdict only to classify TP/FP/TN/FN afterward. It does not rerun target APIs or
copy the production oracle's expected outputs. FPR and sensitivity from this
subset are explicitly scoped to independently adjudicated generated cases; they
are not estimates for the unadjudicated population. Audit code is benchmark-side
and was not present in the frozen generation snapshot.

### Scope of blindness and contract limitations

Oracle-family design was informed by the benchmark. Withholding reproducers from
the model tests generation within those general strategies; it is not a claim of
unbiased discovery or absence of model pretraining contamination. Representation
injection is performed by the sampler, so a detected trigger can be a joint result
of LLM input construction and runtime transformation.

The frozen contract accepts a local class spelled `torch.nn.Module` but does not
resolve `nn.Module` aliases for that check. A rejection of such a program is a
framework-contract limitation, not evidence that the Python/PyTorch program is
illegal. The strict RNG-call filter also rejects `manual_seed`, even if other
values are explicit. These limitations are retained in the measured run instead
of changing the protocol after observing outcomes. Static contract validity is
reported explicitly and must not be interpreted as semantic validity of every
rejected or accepted PyTorch program.
