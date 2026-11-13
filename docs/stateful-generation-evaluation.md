# Stateful generation evaluation

## Status and claim boundary

This report concerns the new `stateful` strategy (contract version 4; program format `stateful-1`). G1–G7 completed their recorded acceptance checks. The frozen 300-second protocol finished in job 771760; independent review, the audit correction and resource cleanup are complete. Development attempts, authored fixtures and formal generated tests are separate populations. The historical 10/10 fixture result and 3/10 guided-generation result in [the earlier evaluation](pytorch-correctness-evaluation.md) are not results for this experiment.

The implementation admits APIs independently of oracle support. It does not claim support for every PyTorch feature or an implemented TensorFlow backend. Unknown certificates produce inapplicability or unassessed coverage, not a correctness pass.

## Completed formal results

The frozen run completed all ten targets once under the declared 300-second per-API budget. The authored fixture/control gate detected and independently corroborated **10/10** benchmarks. Full-source generation yielded **0/10 independently confirmed benchmark reproductions** and **zero production Suspected Failures**. This is the measured confirmed count, not proof that every unadjudicated execution was correct. The [reviewed result](../result/stateful-generation/771760/stage-7/formal-01/evaluation-reviewed-v2.json) is authoritative for the metrics below; the [automatic result](../result/stateful-generation/771760/stage-7/formal-01/evaluation-automatic-v1.json) is retained unchanged.

| Issue / API | Attempts | Sources | Static valid | Source completed | Runtime-valid | Applicable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [203078](https://github.com/pytorch/pytorch/issues/203078) `torch.Tensor.scatter_add` | 40 | 40 | 40 | 31 | 21 | 9 |
| [204407](https://github.com/pytorch/pytorch/issues/204407) `torch.linalg.slogdet` | 61 | 61 | 61 | 32 | 32 | 23 |
| [200599](https://github.com/pytorch/pytorch/issues/200599) `torch.linalg.lu_solve` | 31 | 31 | 30 | 25 | 19 | 14 |
| [202706](https://github.com/pytorch/pytorch/issues/202706) `torch.autocast` | 53 | 53 | 53 | 52 | 22 | 22 |
| [181723](https://github.com/pytorch/pytorch/issues/181723) `torch.cuda.make_graphed_callables` | 37 | 37 | 37 | 24 | 8 | 0 |
| [204033](https://github.com/pytorch/pytorch/issues/204033) `torch.compile` | 47 | 47 | 47 | 0 | 0 | 0 |
| [204365](https://github.com/pytorch/pytorch/issues/204365) `torch.nn.functional.batch_norm` | 37 | 37 | 37 | 19 | 14 | 8 |
| [201268](https://github.com/pytorch/pytorch/issues/201268) `torch.add` | 42 | 42 | 42 | 27 | 20 | 17 |
| [197089](https://github.com/pytorch/pytorch/issues/197089) `torch.std` | 43 | 42 | 42 | 28 | 23 | 0 |
| [200613](https://github.com/pytorch/pytorch/issues/200613) `torch.Tensor.copy_` | 43 | 43 | 43 | 17 | 17 | 2 |
| **Total** | **434** | **433** | **432** | **255** | **176** | **95** |

Here, “Sources” means a response from which a source string was extracted, not necessarily syntactically valid Python. The static Valid Test Rate is **432/433 = 99.77%**; Executable Test Rate is **255/432 = 59.03%**. Runtime-valid sources account for **176/255 = 69.02%** of completed sources, or **176/433 = 40.65%** of extracted sources. A static-valid result does not establish legal training execution. One LU candidate had an unmatched delimiter; one std generation ended at the provider length limit before producing a usable source.

There were 172 repair attempts, including 52 input-coverage refinements, and 325 API-scoped source fingerprints. The raw syntax-valid count was 329; 103 candidates used the narrowly defined trailing-delimiter normalization. Completed source/follow-up executions were 255/177; context restoration uses a single observed execution, so it requires no artificial follow-up. All 95 applicable comparisons passed. Raw verdict counts were 95 Pass, 137 Inapplicable, 20 Invalid and 182 without a completed verdict. These categories are not interchangeable with the runtime-valid funnel.

| Issue | State attained | API-scoped signatures | Eligible mutation candidates* | Used seconds | Unused seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 203078 | 13 | 6 | 31 | 275.35 | 24.65 |
| 204407 | 0 | 0 | 32 | 287.13 | 12.87 |
| 200599 | 10 | 3 | 25 | 258.77 | 41.23 |
| 202706 | 21 | 4 | 0 | 288.75 | 11.25 |
| 181723 | 0 | 0 | 0 | 281.55 | 18.45 |
| 204033 | 0 | 0 | 0 | 287.94 | 12.06 |
| 204365 | 5 | 2 | 8 | 265.64 | 34.36 |
| 201268 | 8 | 5 | 27 | 274.96 | 25.04 |
| 197089 | 3 | 1 | 28 | 269.51 | 30.49 |
| 200613 | 0 | 0 | 17 | 286.69 | 13.31 |

*This column counts test candidates with at least one eligible mutation, not the number of transformation plans. Across the run there were 60 candidates attaining requested transition predicates and 21 API-scoped signatures. There were 168 candidates with eligible mutations. Saved evidence contains 233 verified mutation-application events across 134 saved members, including canonical materializations; these are not 233 distinct states or MR passes. Saved members can survive a later worker/inspection failure, so their count is not a replacement for completed-execution counters.

### Independent labels and review correction

The reviewed confusion counts are **TP=0, FP=0, TN=39, FN=0**. The 39 eligible negatives comprise 22 context-restoration candidates and 17 returned-value copy candidates. Thus the measured FPR is **0/39**, restricted to that adjudicated subset. **Correctness Violation Detection Rate is undefined (0/0)** because no eligible independent positive was established. An additional 30 independently negative candidates were outside the required state/path domain; 365 attempts remain audit-unknown. Unknowns are neither negatives nor confirmed oracle misses. Time to First Detection is censored for all ten APIs at their actual used times in the table.

The original automatic audit incorrectly treated whole FX graph outputs as individual API outputs. Review withdrew eight positive labels: scatter attempts 0004, 0022, 0031, 0032, 0038 and 0039, and add attempts 0014 and 0015. Three scatter labels had been counted as FN; the other five were outside the valid metric domain. All eight become unknown rather than negative. These were audit errors, not PyTorch defects. The review did not alter any generated source, production verdict or budget.

After the formal run, the evaluator was corrected to exclude uncertified whole-graph observations from single-target references and attribution. Independent eager positives are retained; partial eager negatives cannot certify excluded graph outputs. Nine focused benchmark tests passed in 2.351 seconds (`iteration-4ur2Pq`), including the saved fixture-control audit. A subsequent all-artifact crosscheck reproduced every reviewed candidate label/classification and the aggregate metrics. The frozen evaluator hash is `84c740ad69bbf9f75dd930475de276e32cf7d865161c7662628e207486b56a0d`; the corrected evaluator hash is `c5f94a3de368315c6731cce1d665484288ce043132458ecc25ad7fee18c02cf0`. The 43 production engine files remained unchanged.

### Why each benchmark was or was not reproduced

The fixed fixtures were all reproduced and corroborated. The paragraphs below distinguish that Phase 2 evidence from the generated Phase 3 results. Structural observations refer to saved runtime evidence, not merely source-text predictions. The [review evidence](../result/stateful-generation/771760/stage-7/formal-01/review-evidence-v2.json) supplies member-level descriptors and events.

#### 203078: `torch.Tensor.scatter_add`

The fixture combines an expanded index with stride `(1, 0)` and a source/destination width beyond the index width; Python integer scatter accumulation and a contiguous-index control corroborate the wrong row addressing. None of the 55 saved source/follow-up members in the formal run satisfied the recorded structural combination. There were 21 runtime-valid candidates and nine applicable comparisons, but no independently established trigger. This is an observed trigger-coverage gap, not an established oracle miss. The six compiled candidates that initially received positive audit labels were comparing a whole graph output with a single-target reference; those labels were withdrawn. [Run records](../result/stateful-generation/771760/stage-7/formal-01/203078/summary.json).

#### 204407: `torch.linalg.slogdet`

The fixture exercises a negative-bit real view obtained from a complex conjugate's imaginary component, followed by determinant backward; an inverse-transpose derivative and materialized control corroborate the error. Formal generation produced one saved negative-bit operand, but did not combine it with the necessary backward path. All 29 failed executions raised the missing-grad-function error; 32 runtime-valid candidates remained, with zero attained transition predicates. For example, attempt 0001 invokes backward on results from an input without gradient history. The 23 passing comparisons therefore do not test the benchmark's backward mechanism. [Run records](../result/stateful-generation/771760/stage-7/formal-01/204407/summary.json).

#### 200599: `torch.linalg.lu_solve`

The fixture has a singleton batch dimension with noncanonical stride; a rational reconstruction of the LU system and an independent solve corroborate the wrong matrix selection. Formal saved operands were rank-two matrices and rank-one pivot tensors, without the required rank-three-or-higher batched LU representation. Generic singleton coverage included ordinary matrix dimensions and does not establish a singleton batch dimension. Nineteen candidates were runtime-valid and fourteen comparisons applied, but the trigger was not observed. Shape-preserving mutation cannot invent the missing batch structure without a separate semantic relation. [Run records](../result/stateful-generation/771760/stage-7/formal-01/200599/summary.json).

#### 202706: `torch.autocast`

The fixture recursively reuses one autocast decorator; the public restoration contract and independent fresh-instance scopes corroborate leaked thread-local state. The formal run produced 52 completed source executions, but none had the recorded overlapping same-instance mechanism. Twenty-two runtime-valid context cases passed and independently received negative labels; thirty other completed cases missed a state predicate. For example, attempt 0002 restored two scopes but constructed only one observed target boundary when two were requested. Distinct legal nesting signatures do not imply recursive decorator reuse. All 52 saved builds changed initialization RNG through module construction, despite the prompt's explicit-initialization request. The context relation concerns entry/exit state, so this is not evidence of numerical determinism or compliance with the initialization prompt. [Run records](../result/stateful-generation/771760/stage-7/formal-01/202706/summary.json).

#### 181723: `torch.cuda.make_graphed_callables`

The fixture accumulates gradients from different microbatches after CUDA capture; the analytic sum for its linear model and eager control corroborate static gradient-buffer reuse. The formal run had eight runtime-valid candidates, zero attained transition predicates and zero applicable comparisons. Twelve saved members had capture plus multiple gradient events, but none of those members was runtime-valid, and these structural facts alone do not prove corruption. Early candidates passed the returned callable to `probe.call` instead of the configured factory; later failures included capture legality, exact event-count goals and missing semantic-execution certificates. This is a mixture of generation/state failures and oracle applicability gaps, not a confirmed false negative. [Run records](../result/stateful-generation/771760/stage-7/formal-01/181723/summary.json).

#### 204033: `torch.compile`

The fixture returns a conjugate imaginary view and mutates it after compilation; scalar complex arithmetic and eager alias propagation corroborate the mismatch. None of the 47 formal candidates completed a valid source execution. Thirty-seven failed the configured-target probe check, five used unsupported registered modules, and the remainder had unsupported function payloads, missing names/keys or invalid parameter registration. Attempt 0000 compiles a function and probes the returned callable, although `torch.compile` itself is the configured target. No generated alias-propagation experiment reached the oracle; the result cannot establish whether the framework would detect a correctly generated instance. [Run records](../result/stateful-generation/771760/stage-7/formal-01/204033/summary.json).

#### 204365: `torch.nn.functional.batch_norm`

The fixture combines MKLDNN inference input with strided affine tensors; a scalar normalization formula and contiguous-affine control corroborate the error. All saved formal operand descriptors were `torch.strided`, and no generated source requested MKLDNN. A weight-stride mutation did occur, for example in attempt 0018, but on an ordinary dense execution. Fourteen candidates were runtime-valid and eight comparisons applied. Early generation also registered non-trainable statistics in the framework's trainable-parameter dictionary. Those rejections concern Mirage's program contract, not proof that the PyTorch API call is illegal. The benchmark's layout/affine combination was not reached. [Run records](../result/stateful-generation/771760/stage-7/formal-01/204365/summary.json).

#### 201268: `torch.add`

The fixture uses CSR `out=` with noncontiguous values storage; scalar addition after independent CSR index expansion corroborates incorrect writes. Every saved formal operand descriptor was dense/strided. Attempt 0029 requested sparse COO, which neither established the required CSR output nor completed the target execution. Twenty candidates were runtime-valid and seventeen comparisons applied, but dense addition passes cannot assess this CSR mechanism. Two automatic positive labels from compiled graph outputs were withdrawn; neither was in the valid metric domain. Generated CSR write behavior remains unassessed. [Run records](../result/stateful-generation/771760/stage-7/formal-01/201268/summary.json).

#### 197089: `torch.std`

The fixture combines CPU Inductor compilation with tiny, nonuniform FP32 values and backward; stable scalar standard-deviation arithmetic and its analytic derivative corroborate the error. The formal run did observe small and large finite inputs in runtime-valid cases, so lack of extreme-value generation is not a sufficient explanation. However, all saved compiled candidates (0007, 0008, 0009 and 0037) failed their required runtime predicates. Attempt 0007 backpropagated through a separate eager call; attempt 0037 reached compiled backward but observed two target boundaries when one was requested, using ordinary-scale values. Twenty-three candidates were runtime-valid, but none had an applicable oracle; numerical-domain and execution certificates were the main limits. No independent positive was established for these generated gradient computations. [Run records](../result/stateful-generation/771760/stage-7/formal-01/197089/summary.json).

#### 200613: `torch.Tensor.copy_`

The fixture copies a CPU lazy-conjugate source into a CUDA destination with broadcasting; explicit logical complex values and a resolved-view control corroborate the view-bit error on the tested NVIDIA environment. Every saved formal tensor descriptor was on `cuda:0`; one negative-bit representation was observed, but no CPU-to-CUDA combination. Seventeen candidates were runtime-valid and independently negative for the returned-value contract; only two passed all production comparison requirements. Other candidates commonly attempted an in-place write to a leaf requiring gradients or used invalid parameter registration. Current value-preserving representation mutations retain device placement, so they cannot supply the absent cross-device setup. [Run records](../result/stateful-generation/771760/stage-7/formal-01/200613/summary.json).

### Edge coverage and failure attribution

Counts below are APIs out of the ten-target pool. A requested category may not apply to a context factory or to a particular API. Source/follow-up witnesses require runtime-valid members; relation-eligible coverage requires an applied production comparison. The columns describe different sets, so they are not an additive funnel.

| Category | Requested | Valid source | Valid follow-up | Relation-eligible |
| --- | ---: | ---: | ---: | ---: |
| ordinary | 10 | 8 | 7 | 6 |
| zero | 10 | 8 | 7 | 6 |
| small_finite | 10 | 5 | 4 | 3 |
| large_finite | 10 | 5 | 5 | 2 |
| repeated | 10 | 8 | 7 | 5 |
| nonuniform | 10 | 8 | 7 | 5 |
| mixed_sign | 10 | 5 | 4 | 3 |
| singleton | 10 | 6 | 6 | 4 |
| broadcast | 10 | 5 | 5 | 3 |
| supported_layout | 10 | 5 | 6 | 4 |

The recorded first-blocking-stage counters are unresolved runtime 177, state 54, backward 25, mutation application 3, parse 1, and none 174. “None” is a missing/default stage label, not proof of success: for example, the std provider-length failure has no such field. The separately reported provider, inspection and verdict fields take precedence. No independently established positive remained after review, so this run cannot quantify oracle misses or matcher misses; it does expose generation, state and applicability bottlenecks.

The results support the narrower claim that complete LLM-owned state programs can be inspected and that supplied deterministic benchmark triggers can be detected. They do not demonstrate autonomous discovery of the ten issues under this budget, universal API correctness assessment, or a causal benefit from edge prompting. Broader state predicates, factory-use compliance, layout combinations and sound numerical/semantic certificates remain coverage limits. The historical guided 3/10 result used a different generation task and is not a controlled baseline for this run.

### Cost, artifacts and cleanup

The formal driver took 2,825 seconds. Per-API generation/execution totaled 2,776.27 seconds of the nominal 3,000 seconds, leaving 223.73 seconds unused under the recorded dispatch reserve. Metadata took 23.25 seconds; automatic post-hoc audits took 19.46 seconds. Reviewed-v2 computation took 2.34 seconds (five seconds for its isolated iteration). Logged token usage totals at least 1,402,625 input and 140,385 output tokens; the std length-failure attempt lacks complete usage accounting, so these are not exact total billing counts.

The two reusable allocations were 770948 (`8gpus`, 12,177 seconds) and 771760 (`dev`, 3,526 seconds), each with two H200 GPUs. Service startup took 217 and 185 seconds respectively. There was one allocation replacement and no in-place failure restart. Their combined allocated time was 15,703 seconds and allocated GPU time was 31,406 GPU-seconds; this excludes earlier standalone validation jobs and is not GPU utilization. The [resource ledger](../result/stateful-generation/771760/resource-ledger.json) preserves 46 per-iteration time/status records. Overlapping serving and testing durations are not added as wall time.

Before shutdown, vLLM reported zero running and zero waiting requests. The owned test caches were removed, both interactive test sessions were closed or explicitly terminated, and both jobs and all their steps exited. Job 771760 and its testing step completed with exit 0; both service steps ultimately exited with signal 9 through owned shutdown, rather than a clean service exit. No owned job remains active. See [cleanup evidence](../result/stateful-generation/771760/cleanup-final.json) and [corrected-evaluator crosscheck](../result/stateful-generation/771760/stage-7/formal-01/corrected-evaluator-crosscheck.json). Scheduler completion is cleanup evidence, not an MR Pass.

## What changed

| Component | Implemented behavior | Current boundary |
| --- | --- | --- |
| Program generator | The LLM writes both `build(device, seed)` and the complete `run(ctx, probe)`, including all training actions. | Probes observe/call the target; they do not fill missing backward, contexts, optimizer updates, compilation or capture. |
| State sampler | Selects goals from observed canonical capabilities and explores joint predicates using a deterministic API-independent schedule. | A predicate request is not proof of reachability. Final state evidence comes from the inspector. |
| Operand mutation | Discovers tensors in actual positional/keyword/container operands; preserves repeated identities and checks storage aliases, writes and gradient connectivity. | No API or issue identifier enters mutation candidate selection. Distinct shared views and unsupported writable/gradient domains are rejected. |
| Backend adapter | Supplies dense representations, CSR value-storage transformations, MKLDNN logical materialization, compiler/context/capture evidence and numerical checks. | The backend-neutral interface is tested with a fake adapter; TensorFlow is not implemented. |
| Inspector | Records actual target/backward paths, checkpoint state, inputs, transitions and mutation evidence. | Configuration flags and generated assertions cannot establish coverage. |
| Oracle catalog | Conditional semantic families share obligations and comparisons across APIs. | A catalog entry alone does not implement its certificates; several family combinations remain unavailable. |
| Replay | Reloads immutable source/engine and checks reconstructed initial values, representation and environment, with no provider call. | It does not serialize arbitrary live autograd graphs. |

The initial/final state includes inputs, parameters, input-leaf and parameter gradients, module state, optimizer/scheduler/scaler state, portable checkpoint/extra data and execution context where supported. A saved pair checks its required initial state before executing the follow-up. Unsupported payloads are explicit contract failures.

The representation path supports cloning, noncontiguous storage, offsets, supported lazy conjugate/negative representations, singleton strides and certified expansions. Eligibility is derived from actual operand descriptors and semantic obligations. CSR transformations preserve index structure; MKLDNN comparisons observe logical values. Parameter replacement, ambiguous write propagation, overlapping writable storage and unverified alias preservation are not silently accepted. Compilation/capture does not silently fall back to eager mutation.

## Generation protocol and provenance

Each request contains the installed target API specification, its documentation hash and annotation, the source-only JSON contract, execution configuration, requested input categories, state predicates, applicable relation requirements, and—after a valid canonical—its generated source and previously observed transition signatures. No issue identifier, benchmark reproducer or issue-specific transition program is supplied to production generation.

The model owns every operation and its order. The sampler may request a number of observed boundaries or backward events; it never translates these counts into executable actions. For a context factory, construction and scope entry/exit are separate observables. A compiled boundary is an executed target-containing FX graph boundary, not a claim about the number of fused kernels.

Prompts evolved during development. Exact `request.json`, raw `response.json`, provider HTTP payloads, source and diagnostics are preserved per attempt. Prompt hashes distinguish revisions even when the same nominal version label appears in multiple development iterations. The frozen report identifies the final hash and reproduce the final prompt. Earlier pilots are not a controlled estimate of prompt improvement.

One syntax-only normalization is permitted: remove a single standalone final `}` only when Python diagnoses that exact unmatched delimiter and the remaining complete source parses. Both source versions and the normalization record are saved. No training statement is added or rewritten. Raw syntax-valid responses, normalized candidates and LLM repair responses are counted separately. Unknown runtime exceptions and suspected framework failures are not repaired into passes.

## Relations, observables and uncertainty

| Relation | Required evidence and observations | Interpretation |
| --- | --- | --- |
| Representation equivalence | Same required initial state, source, logical target inputs and checkpoints; certified effects/coordinates/numerical domain; target outputs and root gradients. | Value-preserving representations may be compared only in the certified domain. |
| Materialized value effects | Matching schema-derived writable roles and compatible initial/input conditions; returned logical target outputs. | This does not assert arbitrary storage identity or write propagation. |
| Context restoration | Balanced original scopes and a supported restoration contract; entry/exit context state. | Uncertified direct same-instance reentry is inapplicable. Supported decorator evidence is recorded separately. |
| Execution mode | Same authored computation, executed compiler-graph or capture evidence, aligned observations and compatible initial state. | A source-owned compiled/captured computation is compared with its certified control configuration; the inspector does not invent it. |
| Other catalog families | Differentiation, accumulation, state restoration and alias effects have declared obligations. | Missing certificates remain explicit; declaring these families is not full implementation support. |

`dtype_scaled_v1` is a screening policy. It uses dtype-aware, scale-relative comparisons rather than treating every difference as a violation; nonfinite or uncertain numerical conditions remain unassessed. Matrix certification includes a conditioning bound. Independent audits use their own mathematical references and uncertainty bands. A production Suspected Failure is not independent ground truth or a confirmed defect.

## Development evidence

The ten authored fixtures were detected through the new path in job `770759`; independent audit job `770811` corroborated their controls. Job `770911` checked the integrated audit and zero-provider replay. These establish earlier implementation evidence; the final engine subsequently passed fixtures-05 and the new-allocation fixtures-06 gate.

Development used allocation `770948`, with one H200 for Qwen/Qwen3-Coder-Next-FP8 and a separate GPU/Slurm step for tests. The testing environment is Python 3.14.10 and PyTorch 2.15.0+cu132. The serving environment is separate: vLLM 0.33.0, serving PyTorch 2.14.0+cu132 and Transformers 5.20.2. The model revision is `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`; requests use temperature 0 and a 4096-token output limit. Serving context length is 16384, with at most two sequences. Exact environment records are under `result/stateful-generation/770948/`.

The development targets are sin/backward count 2, no_grad/restoration count 2, square/target-connected backward events 2, and cos/executed compilation. Acceptance requires two distinct runtime-verified training signatures for each core predicate. Observation-only detach differences do not count. Optimizer updates and clearing are observed through hooks/profile events; the framework does not execute them on behalf of the model.

Early failures were informative: unnecessary module/optimizer construction, wrong parameter-object registration, discarded probe results, missing backward connectivity, unsupported checkpoint functions, random initialization copied from documentation, redundant closing delimiters and failure to create an executed compiled graph. They are generation/contract/state failures, not evidence of PyTorch defects. A later sin pilot met its diversity gate while no_grad still repeated one legal sequence; the joint-state sampler addresses this measured exploration gap without supplying code.

The final-source regression iteration, `iteration-rgSvHY`, passed all 155 tests without skips (87.953 seconds). The separate `fixtures-05` integration and independent-control audit passed all ten cases. Earlier skipped audits were subsequently executed with their artifact paths. Development results and all failed iterations remain in [the roadmap](roadmap-004-stateful-generation.md) and immutable allocation artifacts.

## Measurement definitions

The formal experiment uses 300 seconds per API. An unused 180-second protocol was temporarily frozen after Slurm declined an allocation extension; after the user authorized a partition change, the original 300-second budget was restored before any formal generation. Development budgets are recorded separately. The budget includes generation, LLM repair, source execution, mutation variants and both MR members. API metadata discovery, service startup and post-hoc independent auditing are measured separately. Admission reserves `max(10 seconds, 1.5 × longest completed candidate duration + 5 seconds)`; actual unused time is reported. This policy does not guarantee that infrastructure cannot stall.

Report response count, parsed source count, raw syntax validity, normalization count, contract validity, executable source/pair counts, runtime validity, state attainment, distinct transition signatures, mutation eligibility, MR applicability and independently audited outcomes. Valid Test Rate uses contract-valid candidates divided by parsed generated sources; Executable Test Rate uses completed source executions divided by contract-valid candidates. Additional runtime-valid rates keep state attainment separate from execution.

Input coverage has three levels: requested, actually target-observed and relation-eligible. Acceptance witnesses must additionally come from runtime-valid cases. Repeated-value inspection is bounded; absence of a repeated value in the inspected prefix does not prove global uniqueness. Broadcast coverage measures compatible unequal operand shapes, not an undocumented promise that every API broadcasts those operands.

Independent known-issue reproduction and automatic detection are separate metrics. A mathematically established positive that the framework did not report is not a detection. Required-path/state failures are excluded from oracle-miss denominators and reported separately. Unknown audits are never negatives; output-only negative audits cannot certify an entire gradient comparison. False Positive Rate applies only to the independently adjudicated negative subset. First-detection time requires both a production report and independent attribution; absence of detection is censored.

## Gate completion

G6 passed the diversity, input-coverage, fixture/control, regression, held-out reuse and freeze gates. G7 completed every API budget, versioned independent review, metric and artifact checks, the post-experiment audit regression checks, and owned-resource cleanup. Low discovery is an experimental outcome; the gate did not require a minimum reproduction rate.

## Implementation details and reusable scope

The new path is selected by `--strategy stateful` (contract version 4). The historical `training` and `properties` paths remain available, but neither supplies transition templates to this experiment. In particular, the old `gradient_reference` property generator is not the generator evaluated here.

The backend-neutral records live in `contracts/capabilities.py`; the framework boundary is `backends/base.py`. `runtime/operands.py` discovers actual argument trees and requests eligible plans from the adapter. Plans identify a structural argument path, an action, and a target-call boundary. Repeated tensor objects retain their identity group, and distinct shared views require an established alias obligation. Selection is a deterministic enumeration of eligible candidates, not an API-specific priority list. Numeric edge coverage changes LLM-generated initialization before the pair is formed; representation mutation preserves the relation's logical input values.

`contracts/exploration.py` selects goals using observed canonical capabilities. The initial goals cover backward invocation count, target-connected gradient events, balanced context restoration, and executed compilation; a joint goal varies the number of target boundaries. These describe states to attain, not action sequences to execute. Raw action traces remain available, while observation-only detach operations are excluded from the diversity signature. Optimizer steps and zeroing are observable even though optimizer-state goal sampling is not comprehensive.

`runtime/stateful.py` executes the generated `build` and `run`. `inspector/stateful.py` verifies the requested checkpoint, target execution, required backward connectivity, and each predicate. A compiled-state goal with required backward additionally needs gradient evidence attached to the executed target-containing graph. Eager backward elsewhere in the program cannot satisfy it. Context restoration observes original enter/exit methods; uncertain direct same-instance reentry does not receive a supported restoration certificate.

The PyTorch representation oracle has one capability registry organized into pointwise, reduction, linear-algebra, indexed-value, normalization, and writable-value families. Registry membership supplies a semantic capability, not an expected-output formula. For example, matrix comparison also requires measured conditioning; reduction membership alone does not establish the current numerical-domain certificate. Compilation additionally uses a conservative source certificate that excludes implementation-identity observations and unsupported control flow. Arbitrary executable programs can therefore remain useful for coverage while receiving no correctness assessment.

Every attempt stores `request.json`, `response.json`, `raw-program.py`, normalized `program.py`, `outcome.json`, execution traces and, when produced, `source.pt`/`followup.pt`. A replay manifest points to immutable engine/source hashes and reconstructs initial data. Replay now checks both pair members' semantic evidence, not just their verdict. A matching seed is neither a determinism certificate nor proof of equivalent initial state.

The PyTorch adapter uses framework-specific inspection metadata, including view/version/storage information and compiler graph observation. Some inspection hooks are internal PyTorch interfaces; they are version-sensitive and are not a portable public backend contract. The contract requires generated target calls and representation constructions to use public operations. Porting the adapter requires re-establishing these inspection guarantees, not merely changing import names.

Withholding benchmark reproducers from prompts does not establish that the pretrained model has never encountered a public issue. Development also uses authored fixtures to validate the inspector and oracle. The controlled boundary is that issue IDs, trigger programs and expected results do not enter production generation requests; this is not a claim of model-pretraining blindness or an unbiased estimate of unknown-bug discovery.

## Relationship to the original three phases

**Phase 1:** Reuse the previously collected ten deterministic benchmarks and their public issue references. This task does not claim a new issue search or ten newly discovered defects. Re-executing their authored programs on the installed environment validates the new integration; development re-executions after source changes are not statistical repetitions supporting a stability estimate.

**Phase 2:** Stages G1–G5 add the new source contract, backend/operand boundary, LLM-owned transitions, runtime evidence and conditional oracle families. G6 validates these components together using authored positive fixtures, independent controls, regression tests, generated development programs, input coverage and held-out API checks. A fixture detection tests the framework's capacity to recognize supplied conditions; it is not an LLM discovery.

**Phase 3:** G7 evaluated the frozen full-source generator under the declared per-API budget. Development programs—including failed held-out checks subsequently retired into development—are excluded from its discovery denominator. All ten formal target APIs use the same generation/mutation/oracle mechanisms; issue-specific attribution runs afterward and never supplies generation feedback.

The first held-out check (`heldout-01`) failed: sigmoid produced 8 parsed sources but only its canonical was runtime-valid and no transition predicate was attained; Tensor.mul received 8 responses, produced 7 sources and attained transitions in 3 of 4 runtime-valid cases. Sigmoid failures reused consumed upstream graph state. The revised generic guidance explains graph ownership without supplying an action sequence, and applies only when backward is required. The failed check remains evidence of limited initial generalization. The replacement held-out budget is 300 seconds/24 responses per API, so neither prompt changes nor additional opportunity can be credited independently for a later outcome.

## Exact frozen request example

The system message is exactly `Follow the supplied testing contract. Return only the requested JSON object.` The user message is `request.prompt + "\nContext:\n" + json.dumps(request.context)`. A repair additionally supplies the prior assistant response and `Repair this diagnosed problem only:\n` followed by its diagnostic. Thus the prompt below is only one component of the complete request, not a substitute for its saved context.

The following is the verbatim first formal canonical prompt for `torch.Tensor.scatter_add` (forward path, `stateful-2-edges`). Its prompt SHA-256 is `2e3f14399ae4ce9510783fff9cb6852b402897402f1233f3a4723e458dc94a0d`. API names, required paths, predicates, prior signatures and canonical source vary according to recorded scheduling and observed evidence.

```text
Target API: torch.Tensor.scatter_add. Generate a small deterministic test exercising this exact API.
Use explicit tensor/zeros/ones/arange/linspace initialization. Never sample random values, even with a seed; do not copy random examples from the API documentation.
Required target backward: False. Requested state predicates: [].
For an eager test, use the value RETURNED by probe.call for subsequent computation; calling the target separately and discarding probe.call output does not test its backward. If backward is not required, a legal target exercise needs no training loop or optimizer.
Return one JSON object with exactly source (a complete Python source string). Write concise code; omit explanatory comments and redundant snapshots.
1. Define build(device, seed) returning a dict with inputs and parameters dictionaries. Optional fields: modules, optimizer, scheduler, scaler and checkpoint_data. Use explicitly initialized tensors. A seed does not replace initialization. Replace all retained random module values. Registered module parameters must be the same objects as entries in parameters, not the tensors used to construct Parameters. Omit unused optional fields. optimizer/scheduler/scaler must be real objects or None, never dictionaries. inputs/parameters/modules are dictionaries and may be empty. Functions belong in source, not checkpoint_data or returned observations.
2. Define run(ctx, probe). You own all operations and their ordering: the framework inserts no backward, clearing, optimizer step, context, compilation or replay. Without canonical_source, write a minimal legal canonical use and the required backward. canonical_source is a starting point, not a mandatory program structure. Rewrite it as needed to reach the supplied predicates at their named checkpoints. Counts are exact, not lower bounds. Choose a different legal training transition structure from prior_transition_signatures; different values or observation copies alone do not establish diversity. No fixed action sequence is required.
3. For an eager target invocation, call probe.call(target_callable, *args, **kwargs) and use its returned result. ONLY the configured target goes through probe.call; call other operations directly. Tensor methods require their bound receiver or an explicit receiver with the unbound method. Use a returned context manager/callable according to its public specification. For required backward, differentiable target operands need gradient history before the call and the observed result must reach backward; an unrelated downstream parameter or separate unobserved call is insufficient.
4. A compiled_execution=True goal REQUIRES source-owned torch.compile and an actual invocation of the resulting callable. Call the target directly inside your compiled function; keep probes outside tracing/capture and invoke the compiled callable directly. For the supported compiled inspection contract, keep backward calls outside the compiled body. A compile flag or tracing-only call is not runtime evidence. Do not fake probe fields.
5. run returns a dictionary of observable tensors/primitives, never expected answers, modules, optimizers or the whole ctx. Return tensors directly; the inspector snapshots them. Optional probe.mark(name, data) adds unique intermediate checkpoints. The inspector owns initial/final; never use those reserved names. Probe has only call and mark methods. Expose related alias partners where an effect relation needs them.
6. No random sampling, NumPy/Tensor.numpy(), external I/O, assertions or hidden globals. Use public APIs. Keep each tensor within 4096 elements and at most four updates. Keep differentiable views in run. Generate legal values/shapes for the assigned input category. Replace canonical inputs that do not demonstrate it, consistently for both MR members; actual target values, not comments describing them as small/large, establish coverage. If a requested capability is unavailable, preserve legal coverage and do not invent evidence.

Consider edge cases from the assigned coverage category as well as ordinary
inputs across the run. Choose legal, explicitly initialized inputs using the
supplied API specification and backend capabilities.
For numerical coverage, consider exact zeros, repeated and nonuniform values,
mixed signs where legal, and small nonzero or large finite magnitudes relative
to the dtype's representable range. Derive scales from dtype metadata rather
than assuming one fixed floating-point format. Record whether conversion or
execution changes the intended magnitude category.
For structural coverage, consider legal singleton dimensions, batch dimensions,
broadcast-compatible unequal shapes, and alternative supported layouts. Use
empty dimensions only when permitted by both the API and the selected relation.
When relationships between operands or returned values matter, expose the
observables required by that relation. Include nontrivial, distinguishable
values where legal; do not assume that every edge case should fail.
Select a bounded subset for this program rather than combining every edge case.
Preserve required domains, initial conditions, differentiation coordinates and
source/follow-up input relations. If a category is not applicable, report that
limitation instead of forcing an invalid input or inventing an expected answer.
You still own the training-state operations and their ordering. The framework
may mutate tensor representations but will not construct your training history.
```

When backward is required, the request uses `stateful-3-edges` and adds the following generic graph-lifetime guidance. It does not supply a state-transition sequence.

```text
Graph lifetime is your responsibility: backward consumes saved graph state unless you explicitly preserve it. Losses sharing an upstream differentiable intermediate also share that graph, even if their target outputs were computed separately. Choose legal retention or recomputation yourself; reconstructing only a downstream target output does not rebuild a consumed upstream graph. A backward_count goal counts actual backward/grad invocations, whereas target_backward_count counts target-connected gradient events.
```

The context contains the installed API docstring and its hash, configuration, canonical source or null, state predicates and measurement units, input-category definitions, relation requirements, prompt version/hash, and prior transition signatures. Small and large finite inputs are defined by `sqrt(finfo(dtype).tiny)` and `sqrt(finfo(dtype).max)` respectively; these are coverage thresholds, not expected outputs or bug labels. Context factories are not required to have a backward path. The formal schedule otherwise alternates one forward proposal with two backward proposals, while repairs retain the diagnosed proposal.

For this first API the installed docstring is only a signature and a reference to `scatter_add_`; the saved specification was not expanded by retrieving that referenced documentation. This limits the supplied specification and should be considered when interpreting generation failures.

The frozen prompt-template source SHA-256 is `725bb45863b816d3100b1d12ae228cb989dd72c0f1dfabe2addc77624c12d98a`; the syntax-normalizer source hash is `57a46d4ab5c96cc2ad4bfcc4b47ed75be8e4039ea404988c1c741748259a02a0`. These identify implementation files, whereas each request hash identifies its concrete prompt.

Artifacts: [complete first request](../result/stateful-generation/771760/stage-7/formal-01/203078/attempt-0000/request.json), [backward request](../result/stateful-generation/771760/stage-7/formal-01/203078/attempt-0001/request.json), and [frozen protocol](../result/stateful-generation/771760/stage-7/formal-01/protocol.json).

## Final development acceptance evidence

These are development gates, excluded from formal discovery rates. All entries refer to the final production source. A gate can establish state reachability even when its correctness relation remains inapplicable.

| Check | Target | Responses | Parsed | Runtime-valid | State attained | Signatures | Applicable comparisons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot-11 | sin | 52 | 52 | 16 | 15 | 2 | 0 |
| pilot-11 | context | 56 | 56 | 23 | 22 | 4 | 23 |
| pilot-11 | square | 46 | 46 | 18 | 17 | 6 | 0 |
| pilot-11 | compiled | 24 | 24 | 8 | 7 | 3 | 7 |
| edges-04 | sin-base | 24 | 24 | 21 | 0 | 0 | 21 |
| edges-04 | sin-edges | 24 | 24 | 23 | 0 | 0 | 0 |
| edges-04 | add-base | 24 | 24 | 24 | 0 | 0 | 22 |
| edges-04 | add-edges | 24 | 24 | 24 | 0 | 0 | 0 |
| heldout-02 | tanh | 24 | 23 | 3 | 2 | 2 | 0 |
| heldout-02 | sub | 24 | 24 | 9 | 8 | 1 | 5 |

The input gate witnessed all ten defined categories across runtime-valid source/follow-up evidence. It does not show that every API supports every category, nor that all categories received an applicable oracle. Each gate also passed the recorded zero-provider replay of both members’ semantic evidence. The fresh held-out targets were `torch.tanh` and `torch.Tensor.sub`; sigmoid and Tensor.mul had already been exposed by the failed first check. The differing budgets and development revisions prohibit a causal claim about prompt improvement.

[Pilot evidence](../result/stateful-generation/770948/stage-6/pilot-11/model-checks.json), [input coverage evidence](../result/stateful-generation/770948/stage-6/edges-04/model-checks.json), and [held-out evidence](../result/stateful-generation/770948/stage-6/heldout-02/model-checks.json). The new `dev` allocation repeated the ten fixture/control checks in [fixtures-06](../result/stateful-generation/771760/stage-6/fixtures-06/independent-controls.json) before freezing the final 300-second protocol. All ten were detected and independently corroborated.
