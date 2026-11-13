# Roadmap 004: General representation mutation and LLM-generated training states

## Status and approved direction

The user approved implementation and automatic progression after each acceptance gate. Stages G1–G7 have completed their recorded acceptance checks within the documented implementation boundaries. The frozen experiment, independent review, audit-only correction and resource cleanup are complete. Historical planning entries below describe their original scope; the execution records and status table describe current progress. The [Traditional Chinese counterpart](../reports/roadmap-004-stateful-generation.md) records the same scope and evidence.

The design has three constraints: representation mutations operate on actual tensor inputs without API-specific recipes; the LLM generates training-state operations and their ordering; correctness oracles implement reusable semantic relations rather than one checker per API. The core must admit new API targets without requiring a new oracle branch and must separate framework-independent orchestration from a PyTorch adapter. A future TensorFlow adapter is an extension target, not a deliverable of this implementation.

The earlier [roadmap](roadmap-003-pytorch-correctness-benchmark.md) and [evaluation](pytorch-correctness-evaluation.md) remain historical evidence. Their 10/10 integration and 3/10 guided-generation results are not acceptance results for the new path. Existing unrelated working-tree changes must remain untouched.

## Responsibilities and non-negotiable boundaries

| Component | Responsibility | Boundary |
| --- | --- | --- |
| LLM | Generate initialization, complete state-transition programs, semantic checkpoints, and MR-compatible program pairs. | Does not provide expected answers, certify applicability, or assign verdicts. |
| Mutation engine | Discover actual operand structures, enumerate legal representation transformations, apply them, and verify their effects. | Receives tensor descriptors and contracts, not API names, issue IDs, or API-specific mutation priorities. |
| Inspector | Establish target execution, required backward connectivity, state attainment, mutation attainment, and checkpoint alignment from evidence. | A generated declaration or configuration flag is not proof. Missing training actions are never supplied automatically. |
| Oracle library | Check reviewed Preconditions, Input Relations, and Output Relations for semantic families. | No universal output-equality assumption and no requirement to author a checker for each target API. |
| Backend adapter | Supply framework-specific API metadata, operand inspection, mutation implementations, observations, and replay support. | Unsupported capabilities remain explicit; the core does not assume all backends expose PyTorch representations. |

Open API admission does not imply universal correctness assessment. A valid execution without a supported, established relation contributes verified coverage with correctness unassessed. API specifications and available schemas may inform legality and effects; they must not encode a failing-input recipe. LLM semantic classifications are proposals, not certificates.

## Stage sequence and gate policy

| Stage | Deliverable | Gate | Recorded status |
| --- | --- | --- | --- |
| 1 | Contracts, backend boundary, and frozen acceptance definitions | G1: responsibilities and evidence requirements are explicit | Passed (job 769979) |
| 2 | General operand discovery and representation mutation | G2: mutations generalize across input structures and preserve required relations | Passed (job 770003) |
| 3 | LLM-owned training program generation and execution | G3: executable state transitions come from generated source | Passed (job 770038) |
| 4 | Runtime inspection and state/transition evidence | G4: coverage cannot be credited without observed evidence | Passed (job 770050) |
| 5 | Semantic-family MR and oracle integration | G5: validated relations distinguish violations from inapplicability | Passed for implemented subset (job 770087) |
| 6 | End-to-end validation, generation diversity, and freeze | G6: the new path is independently validated and ready for measurement | Passed (frozen-02, job 771760) |
| 7 | Frozen LLM experiment and independent attribution | G7: results are complete, auditable, and correctly scoped | Passed (job 771760; reviewed-v2) |

Proceed in order. A failed gate returns work to the responsible earlier stage; preserve failed evidence and do not silently relax acceptance conditions. Once its checks pass, continue within the approved scope without requesting permission again for routine stage transitions. Material design or file-scope changes still follow AGENTS.md. Gate completion requires inspecting results, not merely a successful Slurm submission or job exit.

## Stage 1: Contracts and backend boundary

### Work

Define versioned contracts for operand trees, alias groups, mutation plans and obligations, abstract state predicates, generated programs, semantic checkpoints, execution records, and MR specifications. Separate planned predicates from observed facts and unsupported or unknown evidence. Track source provenance for operations: LLM source, representation mutation, instrumentation, or oracle computation.

Add a backend interface and the initial PyTorch adapter. Keep PyTorch imports and framework operations in the adapter or existing explicitly backend-specific modules; new core scheduling and verdict logic must consume framework-neutral records. Use a small fake adapter to test this boundary without claiming TensorFlow compatibility.

Introduce the new `stateful` strategy and an independent contract version while preserving existing strategies. The generated interface is `build(device, seed)` plus `run(ctx, probe)`. Probe methods observe or mark boundaries; they never implement backward, optimizer updates, context recursion, compilation, capture, or replay for the generated program.

Define the initial state dimensions: gradient state, context state, graph lifecycle, and execution mode. Record the observable predicates supported for each dimension; unknown graph facts remain unknown. Freeze acceptance cases, numerical policies, resource bounds, and explicit finite coverage denominators before measuring results. Start with a stable deterministic enumeration of eligible mutation candidates, retaining configuration and order; adaptive ranking is a separate change unless explicitly included in the approved contract.

### Evaluation-driven additions (E1, E3, E4, E5, E6, E8)

Define an API-independent input-coverage profile alongside the state predicate. Categories include finite numerical magnitude, value patterns, legal shape boundaries, dtype/layout capabilities, operand roles, and observable relationships. A category is a generation objective, not a prewritten reproducer. Record requested categories separately from observed properties and oracle eligibility. Shape or value generation occurs before constructing the MR pair; a representation mutation does not silently change semantic inputs to satisfy its own preconditions.

Reconcile deterministic-input policy with supplied documentation. Preserve the raw installed docstring and its hash, identify RNG-based examples as documentation rather than generation instructions, and save the exact annotated prompt actually sent. Treat an initialization seed setting separately from random sampling: a lone seed call is not proof that a test depends on randomness, and a seed does not certify explicit initialization. Require explicit values for inputs and registered parameters/buffers; account for implicit module initialization and any retained RNG-dependent state. Unresolved determinism remains a limitation rather than a successful deterministic case.

Define callable binding, checkpoint-data serialization, and required program-entry-point contracts. Distinguish bound methods from unbound Tensor methods and include receiver requirements when available from signatures or schemas. Unknown signature metadata remains unknown. Executable functions belong in saved source, while checkpoint data contain only supported values or registered reconstruction records.

### G1 acceptance

- Contracts represent unsupported capabilities, missing evidence, invalid execution, inapplicable relations, Pass, Suspected Failure, and coverage-only/unassessed outcomes without conflating them.
- Mutation selection has no API/issue identifier input. Semantic-oracle selection does not gate API admission.
- The fake adapter exercises the core record flow; the PyTorch adapter exposes only implemented capabilities.
- Initial state predicates, semantic-family certificates, mutation scope, budgets, and acceptance fixtures are written down. An unspecified semantic property cannot default to true.
- E1/E3/E4: contract fixtures distinguish seed setting from stochastic inputs, bound from unbound calls, and executable source from checkpoint payloads; missing information is not filled by an API-specific recipe.
- E5/E6/E8: freeze input-coverage category definitions and the three evidence levels: requested, observed, and relation-eligible. No category contains a benchmark constant, issue ID, or API-specific trigger combination.

Evidence: contract checks, dependency inspection, adapter capability inventory, and the fixed acceptance specification. No real-model discovery claim is made at this gate.

## Stage 2: General input discovery and representation mutation

### Work

Discover tensors in actual positional and keyword arguments and nested dictionaries, lists, tuples, and namedtuples. Use structural paths, such as `kwargs["out"]`, instead of a fixed positional subject. Preserve repeated object references and record storage alias groups. Treat sparse components through layout adapters; do not mutate structural indices as if they were ordinary values.

Reuse supported stride, offset, clone/materialization, negative-bit, conjugate-bit, singleton-stride, and value-preserving expansion primitives. Eligibility depends on tensor metadata and the selected relation's obligations. The first implementation certifies dense-strided transformations; additional layout operations enter only after their own legality, alias, gradient, and replay checks. Discovery of a layout is not a claim that mutations on it are supported.

Handle read/write roles, alias preservation, autograd connectivity, and parameter/optimizer identity explicitly. Do not unconditionally detach an existing computation graph. Reject transformations whose required relationships cannot be maintained. Unknown writable or alias behavior cannot receive an equivalence verdict.

For eager execution, inspect actual target operands at the call boundary. For compiled/captured execution, establish a supported preparation boundary outside tracing/capture and verify that the executed target consumes the transformed input. Instrumentation that changes graph execution, causes graph breaks, or only observes tracing cannot establish execution-mode coverage.

### Evaluation-driven additions (E3, E6, E7)

Replace blind cycling through actions with eligibility-filtered, deterministic enumeration over every discovered operand path and supported alias group. A real tensor does not receive a conjugate action, an integer tensor does not receive a negative-bit action, and a missing singleton axis does not count as a mutation attempt that reached the target. Record rejected candidates and reasons without spending a target-worker launch on a precondition already known to fail. Revalidate at application time because static eligibility is provisional.

Request missing generic prerequisites from the input generator when appropriate, such as a legal singleton dimension, repeated values compatible with expansion, or a supported layout. The LLM constructs a new canonical input, and both MR members start from that input; the mutator must not invent a batch axis or change values in only one member. Keep this coverage request separate from correctness-repair feedback.

Before claiming the full-ten G6 gate, add and validate reusable PyTorch layout adapters for MKLDNN logical materialization and CSR value-storage transformations. Exercise nested/keyword writable outputs and related read-only operands through the same path mechanism. CSR index structure and alias obligations remain intact. Layout eligibility comes from backend capabilities and verified semantic contracts, not an API-name-to-layout table. Cross-input combinations are enumerated from eligible candidates under a declared composition budget; no benchmark-specific layout/operand conjunction is scheduled.

### G2 acceptance

- Discover all tensors in the declared supported container forms, including keyword and nested inputs; reconstruction preserves container structure and repeated references.
- The same operand descriptors and obligations produce the same mutation candidates regardless of API identity. Representative APIs reuse the same mutation code without added name-based branches.
- Each supported action has a successful relation-preserving case and rejection cases for violated preconditions. Verify values, shape/dtype/device obligations, alias relationships, and required gradient connectivity.
- Unsafe parameter replacement, lost aliasing, invalid overlapping writes, and unsupported layouts are rejected or marked unresolved rather than counted as valid transformations.
- Save planned, eligible, applied, and observed mutation states separately. Only verified attainment contributes coverage.
- E3/E7: an eligibility test matrix covers real/integer/complex dtypes, missing singleton axes, legal and illegal expansions, and positional/keyword/nested operands; impossible actions are filtered and retained as rejected-candidate evidence.
- E6: adapter tests cover MKLDNN logical values and CSR values with writable outputs, including rejection of unverified alias/effect relationships. A layout observed or successfully serialized is not counted as a target layout unless the target consumed it.
- E6/E7: demonstrate generic multi-operand exploration within the declared budget and verify that any newly generated input is shared consistently by both MR members.

Evidence: operand discovery records, mutation plans, before/after metadata and values, relation checks, and rejection reasons. Compile/capture insertion that lacks reliable evidence remains explicitly unsupported rather than using an eager fallback.

## Stage 3: LLM-generated training transitions

### Work

Generate and validate a canonical program, then ask the LLM to generate complete legal transition programs and, where needed, MR-compatible pairs. Supply public API specifications, the program contract, abstract state goals, relation requirements, budgets, and prior program context. Withhold issue identifiers, MREs, known trigger sequences, and expected answers.

The LLM owns the placement and ordering of forward/backward, gradient clearing and accumulation, optimizer operations, contexts, checkpointing, and compile/capture/replay operations used by the candidate. No `depth` or `microbatches` field causes the runner to synthesize a hidden lifecycle. Shared syntax and probe interfaces do not prescribe a single transition sequence.

Resolve supported import aliases in validation. Preserve a source-only JSON response envelope where practical. Repair syntax, contract violations, confirmed illegal operations, and verified missing paths/states with bounded feedback. Unknown runtime exceptions and suspected correctness failures do not enter a repair-to-Pass loop. Preserve every attempt and its provenance.

### Evaluation-driven additions (E1, E2, E3, E4, E5, E8)

Add the versioned edge-case supplement below to canonical/input generation, selected from the generic coverage profile rather than the API identity. Balance ordinary cases with finite magnitude extremes, repeated and nonuniform values, symmetric and asymmetric structure where meaningful, and legal singleton/batch/broadcast shapes. Request supported sparse/opaque layouts and writable output operands through capability information. Avoid trivial inputs as the only population: symmetric matrices, identical microbatches, or identity computations may legitimately conceal differences. Do not require a candidate to fail, and do not reject a valid Pass because it is nondiscriminating.

Resolve ordinary import aliases, including module base classes, while still validating initialization and effects. Parse the Python source inside the JSON envelope. Validate advertised entry points and checkpoint-data shape before expensive target work; local helper functions may remain in source but must not be serialized as arbitrary data. Validate method-call binding before invocation when metadata is sufficient. The new contract does not impose a legacy top-level forward function unless that particular program interface requires one.

For relation construction, explain the required observables, not an implementation sequence. For example, an alias relation requires two observable related tensors; a local view that is never exposed cannot satisfy that relation. The LLM chooses how to construct and use them. Repair only diagnosed contract/legality/path failures within the bound. A failed numerical reference or unexplained target exception is not a prompt to simplify away a suspected defect.

**Proposed generic edge-case prompt supplement (not used in the historical experiment):**

```text
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

The supplement is a proposed design, not evidence of improved generation. It supplies no issue identifiers, exact failing values, or known lifecycle sequences. Requests use a frozen category schedule; generated explanations of category attainment remain untrusted until inspected.

### G3 acceptance

- The runner can execute different legal state-transition structures through the same contract, without adding a new execution template for each structure.
- Removing a backward, context entry, optimizer update, or other required action from source removes that action from execution; the runner does not replace it.
- Mock-provider and authored contract fixtures demonstrate alias handling, bounded repairs, malformed returns, missing target paths, and preservation of unknown exceptions.
- Saved records show the exact prompt, source, configuration, repair history, and operation provenance. Generation validity remains distinct from state attainment.
- E1/E2/E3/E4: regression fixtures cover strict JSON containing invalid Python, legal import aliases, missing method receivers/operands, implicit randomized initialization, unsupported checkpoint payloads, and genuinely missing entry points. Fixing the first diagnostic must not automatically certify the rest of the program.
- E5/E8: prompt tests verify the generic supplement, category schedule, and relation obligations without benchmark-specific material. Actual edge-case attainment and prompt effectiveness are tested at G4/G6, not inferred from prompt text.
- E4/E8: an alias relation lacking observable partners remains inapplicable; the runner neither manufactures the partner nor inserts the state-transition sequence.

Evidence: source/trace correspondence and contract/repair tests. Real LLM transition diversity is measured at G6, after the inspector is available; mock-provider tests are not LLM generation results.

## Stage 4: Inspector and verified state coverage

### Work

Collect backend observations at semantic checkpoints and actual API events. Record target entry versus completed return, gradient-path evidence, relevant parameters and optimizer state, context entry/exit state, and supported graph-execution evidence. Register resources needed for inspection without accepting generated claims as proof of their state.

Verify requested state predicates and summarize observed transitions. Keep a normalized source fingerprint and a separate runtime transition signature. Changing constants or variable names alone does not establish a different training-state path. Compare checkpoint meaning rather than raw operation counts.

Freeze the abstraction fields used for transition signatures and their finite candidate-set denominators. Count only evidence-supported transitions; report unsupported observations and partially observed executions separately.

### Evaluation-driven additions (E5, E6, E7, E8, E12)

Measure actual input coverage at the target checkpoint: dtype/layout, operand path and role evidence, shape/singleton/broadcast facts, supported alias relations, and value summaries such as zero/nonzero counts, repeatedness, sign distribution, and finite magnitude bins derived from dtype metadata. Detect when an intended tiny nonzero input rounded to zero or a large value became nonfinite. Persist exact data needed by the oracle, not only summary bins.

Record a candidate's path through parsing, static checks, initialization/snapshot, mutation eligibility/application, target entry/return, required backward, state attainment, MR applicability, and independent audit. Preserve the first blocking diagnosis as the exclusive funnel category and supplementary diagnostics separately. This avoids both counting multiple errors as multiple candidates and claiming every failed worker never entered the target.

Keep requested input diversity, observed input diversity, and verified training-transition diversity separate. A symmetric or identical-input case may be a valid Pass; reaching negative-bit, compiled, or accumulated state does not imply that a discrepancy must be observable.

### G4 acceptance

- False checkpoint labels, absent target calls, unrelated gradients, and missing state transitions cannot obtain full coverage credit.
- A configuration saying compiled or captured is insufficient without supported actual execution evidence; tracing alone fails the gate.
- Two programs with different names/constants but the same abstract observed path do not inflate transition diversity. Distinct verified paths remain distinguishable.
- Incomplete evidence, an unmet target predicate, an illegal program, and an oracle violation retain different diagnoses.
- E5: fixtures distinguish a requested tiny value from an actually nonzero representable input, and record output underflow separately without automatically declaring it a bug.
- E6/E8: layout and alias coverage require actual target observations; a single returned view cannot satisfy a two-observable alias relation.
- E12: a full per-stage funnel is reconstructible from events without double-counting repair candidates, API entry, completed execution, or secondary diagnostics.

Evidence: trace records, state snapshots, predicate evaluations, transition signatures, and intentionally incomplete-evidence controls.

## Stage 5: Reusable semantic-family oracles

### Work

Implement relation selection using semantic capabilities and explicit certificates rather than target-name branches. Initial relation families cover logical-value representation equivalence, differentiable computation, certified decomposition/accumulation, persistent-state restoration, context restoration, execution-mode equivalence, and observable alias/effect relationships. Add algebraic references only for a coherent semantic class with a documented defining relation; do not expand by writing one formula per API.

A family can remain unavailable for a capability combination until its certificate and observation requirements are implemented. All advertised relations require reviewed preconditions, input relations, allowed differences, semantic checkpoints, observables, tolerance policy, and nondeterminism handling. Dynamic observations establish measurable conditions but do not prove arbitrary program purity, differentiability, or equivalence by themselves.

Reuse sound existing comparisons and formulas as limited references where their conditions hold. Refactor only the required seams: the new oracle consumes evidence from LLM-generated transitions and never silently constructs the transition under test. For gradient checks, preserve the declared differentiation coordinates; do not detach and substitute a different question. Directional finite differences require domain, smoothness, step-size, and numerical-stability checks; ambiguous results remain unassessed.

### Evaluation-driven additions (E8, E9, E10)

Test numerical policies on ordinary, near-zero, extreme-scale, and poorly conditioned inputs before freezing them. Use scale-aware comparisons and justified reference precision within each semantic relation; do not simply increase absolute tolerance to suppress control disagreement, or let a fixed absolute tolerance erase a tiny expected value. Keep reference disagreement, nonfinite observations, and failed residual/conditioning certificates distinct from target violations. Input coverage outside a relation's domain remains useful without forcing a gradient or residual verdict.

Expand independent audit by supported semantic contracts rather than issue identities. Examples include valid broadcast-aware elementwise relations, bounded pure scalar helpers such as min/max on certified primitive data, and effect-aware context builders. Arbitrary overloaded calls or unknown effects remain unsupported. Audit expected values must be reconstructed independently of production expected outputs; sharing an input descriptor does not justify copying the oracle answer.

Add nondiscriminating controls, including symmetric inputs and identical microbatches where their relations permit them. These controls must remain Pass when observations agree. A requirement for multiple related observables is an applicability condition, not a license to fabricate missing observations.

### G5 acceptance

- Every advertised relation passes known-valid examples, rejects violated preconditions/input relations, and detects deliberately corrupted observations. Synthetic corruption validates the checker; it is not reported as a PyTorch defect.
- Demonstrate cross-API reuse for each advertised relation applicable to multiple APIs, without adding API-specific oracle branches. Relations without sufficient evidence remain explicitly unavailable.
- Nonfinite or unstable references, failed checkpoint alignment, uncertain effects, and control disagreement cannot become false confirmed failures.
- Well-defined coverage-only executions remain available even when no oracle applies. All automated failures remain suspected until independent corroboration.
- E8/E9: ordinary and nondiscriminating controls remain valid; near-threshold control disagreement and residual/conditioning failures stay unassessed or inapplicable as appropriate. Synthetic sign/scale corruption is detected only where the relation is numerically justified.
- E10: audit tests cover supported broadcast shapes and certified pure helper calls, plus deliberate unsupported variants. Report audit support and unknown reasons; a smaller unknown count is not evidence of soundness by itself.
- E9/E10: independent labels are not derived from production Pass/Suspected Failure, and tolerance changes cannot be justified solely by improved benchmark detection or fewer rejected controls.

Evidence: the relation/capability matrix, positive and negative controls, synthetic-corruption checks, applicability reasons, and numerical-policy validation.

## Stage 6: Integration, real-model diversity, and freeze

### Work

Connect generation, mutation, execution, inspection, oracle selection, evidence storage, and replay through the new workflow. Replay reruns saved source from supported initial state with the saved mutation plan; it does not invent arbitrary serialization of live autograd graphs or call the LLM. Unsupported state restoration is reported explicitly.

Replay all ten admitted benchmark fixtures through the new worker, with explicit source containing their transitions. Legacy handlers must not silently supply missing actions. Record each new-path detection, failure, unsupported feature, or unassessed result. The goal remains detection of all ten; a prior 10/10 is not evidence for this implementation, and remaining gaps are not hidden by weakening a relation.

Run a real-model development pilot to verify that the LLM produces multiple legal state-transition paths. For each of the four initial state dimensions, use a nontrivial, declared predicate and require at least two distinct runtime-verified transition signatures reaching that predicate. This is an engineering acceptance target, not a population-level diversity estimate. If a dimension lacks support or the model does not meet the target, keep that gate incomplete and diagnose the cause.

Use a predeclared development API set and a separate held-out API set drawn from supported semantic capabilities. After freezing, test held-out APIs without adding API-name logic. If their results guide code or prompt changes, reclassify them as development evidence and declare a fresh held-out set for a new freeze.

### Evaluation-driven additions (E1–E12)

Create an evaluation-remedy acceptance matrix from E1–E12 before the pilot. Use generic synthetic inputs and held-out API examples to validate the remedies; keep exact benchmark reproducer constants confined to benchmark fixtures. Include numerical extremes, singleton/broadcast shapes, supported layouts and writable outputs, observable aliases, and nontrivial state sequences. Measure requested-to-observed input-category conversion and relation applicability, not only valid-source rate.

Compare the base prompt with the same prompt plus the generic edge-case supplement in a separate, predeclared development comparison. Keep the engine, model, API set, category requests, and nominal per-arm budgets matched; preserve exact prompt versions, order, actual elapsed time, token usage, and cache/service conditions. Distinct generated programs are separate cases, not repeated benchmark reproducers. A single comparison is descriptive and supplies no confidence interval or guarantee of improvement. If edge categories remain absent, mark the gap rather than inserting a known issue's exact input.

Exercise independent audit and attribution during development before the formal freeze. Validate that equivalent documented mechanism variants are not excluded by an overly narrow matcher and that a different anomaly in the same API is not counted as the benchmark issue. Keep this logic outside generation. The final frozen protocol explicitly includes accepted prompt changes, input-category budgets, numerical policies, audit support, and attribution versions.

### G6 acceptance

- Integrated execution and relevant existing regression checks pass; deterministic saved-source replay uses zero provider calls and preserves required evidence.
- The ten-fixture result matrix is complete. A claim that all ten are supported requires 10/10 new-path detections with agreeing controls. An incomplete matrix cannot be presented as full benchmark integration; remaining capability gaps must be resolved before the planned full-ten experiment.
- The real-model pilot meets the declared per-dimension transition-diversity target without hidden runner-generated transitions. Record generation, repair, state-attainment, and applicability denominators.
- Hold-out testing demonstrates reuse without new API-specific mutation/oracle branches. Unsupported held-out cases remain explicit and do not count as assessed successes.
- Freeze source hashes, prompts, capability certificates, policies, model/environment settings, API pools, budgets, and attribution rules. Preserve development failures outside the frozen experiment population.
- E1–E12: every remedy has evidence or an explicit unresolved entry. Acceptance requires the advertised behavior to pass its tests; no claim that all earlier weaknesses are solved follows from a completed checklist alone.
- E5/E6/E8: the real-model pilot records at least one legal, target-observed witness for each declared applicable edge-category family in the acceptance set. Categories lacking a justified numerical oracle may count as observed coverage but not correctness Pass. Missing promised categories keep their generation acceptance item incomplete; do not repair the metric with handwritten benchmark inputs.
- E5: report the base-versus-edge prompt comparison with all denominators, including failures, repairs, and unassessed cases; preserve a non-improvement result. Do not attribute changes to the prompt if other experimental conditions changed.
- E10/E11: audit and attribution controls include supported known-mechanism variants and unrelated anomalies, with independent labels and versioned rules.

Evidence: new-path integration matrix, pilot prompts/programs/traces, diversity records, held-out results, replay checksums, regression logs, and frozen protocol. Failure does not justify inventing a benchmark-specific fallback.

## Stage 7: Frozen experiment and independent evaluation

### Work

Run the new LLM experiment with a nominal 300-second generation/execution budget per target API (restored before formal generation after the user-authorized partition change; see the budget-amendment history). Charge generation, repair, exploratory execution, mutation variants, both members of an MR pair, and in-loop validation to that budget. Measure metadata discovery, service startup, and later independent audit separately; report any dispatch reserve and actual used time. Do not describe equal nominal API budgets as exact equal amounts of generated work.

Use explicit initialized values and deterministic issue triggers. Execute each configured case once; source/follow-up members and distinct mutation variants are separate execution cases, not statistical repetitions. Preserve counts for generated programs and derived executions separately. Do not rerun deterministic cases solely to increase apparent confidence.

Perform benchmark attribution after execution, without feeding issue-specific information back into generation. Use independent evidence to distinguish benchmark reproduction from unrelated anomalies in the same API. Report unadjudicated cases and false-positive denominators explicitly. The old guided experiment remains a historical comparison, not a controlled causal baseline unless conditions are demonstrably matched.

### Evaluation-driven additions (E5, E6, E10, E11, E12)

Publish input-category coverage and a per-stage failure funnel alongside state and transition coverage. Separate generated-source candidates, repair responses, unique source fingerprints, canonical inputs, mutation-derived executions, completed target/backward paths, applicable comparisons, and independently adjudicated cases. Duplicate source is not new transition diversity, while a distinct legal input can still contribute input coverage.

Retain the frozen automatic attribution result. If later independent evidence changes issue attribution, publish a separate versioned reviewed result and its reason; do not overwrite the original or feed it back into the completed generation run. Report new anomalies separately from known-issue reproduction. Keep all audit unknowns visible; FPR applies only to independently adjudicated negatives.

Evaluate whether edge guidance produced intended inputs and whether those inputs reached a usable oracle. Distinguish generation omission, mutation infeasibility, unsupported layout, state non-attainment, relation inapplicability, reference uncertainty, independently established oracle misses, and matcher misses. An API Pass on ordinary dense values does not show that a sparse, opaque-layout, or extreme-scale issue is fixed.

### G7 acceptance

- Every target has a complete outcome or a documented infrastructure/budget failure. Low discovery rates are valid experimental outcomes, not grounds for altering the frozen protocol.
- Report valid/executable rates, state attainment, verified transition diversity, mutation eligibility/application/attainment, MR applicability, known-issue reproduction, audited violation detection/FPR, first-detection time, generation counts, and costs with explicit denominators.
- Distinguish failure to generate a legal program, failure to reach state, mutation limits, MR inapplicability, oracle misses established by independent positive evidence, and attribution misses.
- Preserve raw requests/responses, source, exact initial data where supported, mutation plans, traces, observations, environment, hashes, and audit results. Confirm service/process/allocation cleanup.
- E5/E6/E12: results include requested/observed/relation-eligible edge coverage and the complete execution funnel; categories and denominator definitions match the frozen protocol.
- E10/E11: automatic and reviewed attribution remain separately reproducible, audit unknowns are not treated as negatives, and unrelated same-API anomalies do not increase the benchmark reproduction rate.

Evidence: a new English evaluation report and matching Traditional Chinese report, together with complete immutable execution artifacts. No minimum issue-reproduction percentage is imposed on the frozen experiment.

## Evaluation findings and remedy traceability

The following findings come from the [existing per-issue evaluation](pytorch-correctness-evaluation.md#per-issue-analysis-reproduction-and-non-reproduction). E1–E12 identify planning requirements, not newly discovered bugs. Exact issue inputs remain benchmark evidence; the proposed remedies use general contracts and semantic categories. Historical observations motivate these changes but do not establish that a proposed prompt or implementation will improve the results.

| ID | Observed limitation and evidence | Responsible stages and planned remedy | Acceptance evidence |
| --- | --- | --- | --- |
| E1 | Repeated manual_seed rejection and RNG examples in installed documentation; implicit module initialization also needs review (B1–B5, B7–B10, C). | 1/3/6: explicit deterministic-input policy, raw-doc preservation with annotations, separate seed/sample diagnostics, initialization checks. | Contract tests and real-model rejection/validity measurements; no claim that docstring examples caused the behavior without a controlled comparison. |
| E2 | All 161 CUDA-graph candidates were statically rejected; 131 first failed the literal module-base spelling check and 30 had syntax errors (B5). | 3/6: import-alias resolution, Python parsing inside JSON, and subsequent initialization/effect checks. | Legal aliases accepted; malformed source and unresolved determinism still diagnosed; no assumption that one fix makes all candidates valid. |
| E3 | 18 scatter failures omitted the receiver; 20 copy failures omitted other (B1/B10). | 1/2/3: signature/schema binding and structural paths for receivers, positional, keyword, and nested operands. | Bound/unbound method and missing-argument tests; unavailable metadata stays unresolved. |
| E4 | Compiled-alias candidates returned unpicklable local functions, lacked required entry points, or returned the wrong build type (B6). | 1/3/6: distinguish executable source from checkpoint data and validate the actual new contract early. | Snapshot round trips, malformed-payload controls, bounded repairs; no compulsory legacy lifecycle template. |
| E5 | Ordinary magnitudes missed the tiny-scale trigger; ordinary LU shapes lacked the singleton batch axis (B3/B9). | 1/3/4/6/7: generic dtype-relative magnitude and legal shape edge guidance, observed-category checks, and a predeclared prompt comparison. | Actual legal target inputs cover declared edge categories; distinguish request, construction, target observation, and oracle eligibility. |
| E6 | Dense BatchNorm/add tests never supplied the MKLDNN affine-stride or CSR writable-values configurations; direct positional sampling was too narrow (B7/B8). | 1/2/3/4/6: capability-based layout/input generation, layout adapters, keyword/nested discovery, and bounded cross-operand combinations. | Cross-API adapter and operand tests plus target-observed layout coverage; no API-specific conjunction schedule. |
| E7 | 96 completed workers were transformation-inapplicable before recorded target completion (A). | 2/4/6: eligibility filtering, rejection evidence, and separate generic requests for missing input prerequisites. | Fewer knowingly impossible dispatches by construction; actual outcome rates measured rather than promised. |
| E8 | A symmetric negative-bit matrix legitimately passed; two compiled outputs lacked observable alias partners; identical batches can conceal accumulation differences (B2/B5/B6). | 1/3/4/5/6: discriminating-value diversity, explicit observable requirements, and nondiscriminating negative controls. | Valid Pass remains Pass; missing partners remain inapplicable; LLM constructs the state sequence and related outputs. |
| E9 | One slogdet result was unassessed for control disagreement and two LU cases failed reference residual gates (B2/B3). | 5/6: scale/conditioning-aware reference validation and predeclared uncertainty policies. | Preserve unassessed/inapplicable results; detect justified synthetic errors without weakening tolerances to increase detection. |
| E10 | Audit covered 69 of 1,154 candidates; 1,085 remained unknown, including 9 suspects. Broadcast shapes and pure helper calls were among support gaps (B4/B8/D). | 5/6/7: semantic-family audit support, certified helpers, independent expectations, and explicit support denominators. | Known answers and unsupported controls; no production-verdict-derived labels or assumed negative unknowns. |
| E11 | Narrow copy attribution originally counted 2/10 instead of reviewed 3/10; two std anomalies did not reproduce its benchmark issue (B9/B10). | 6/7: mechanism-level attribution tests, separate automatic/reviewed versions, and independent corroboration. | Known variants and unrelated anomalies classified separately; attribution never feeds generation. |
| E12 | There were 661 repair responses and 748 API-scoped source fingerprints among 1,154 candidates; failures occurred at different stages (A and results). | 4/6/7: exclusive stage funnel, separate secondary diagnoses, program/input/execution counts, and runtime diversity. | Reconstruct all denominators from artifacts; duplicate text does not inflate state-transition coverage. |

### Prompt comparison and scope safeguards

The planned prompt comparison asks whether the generic supplement changes observed input-category coverage and usable test yield, not whether it reproduces a particular issue. Record the two conditions as `base` and `base_plus_edges`. Keep the same category requests in both; the treatment is the additional guidance text. Declare equal nominal budgets, execution order, and resource policy before starting. Report actual budget use, prompt-token overhead, repair counts, and unassessed outcomes. Do not silently replace difficult categories with easier ones after inspecting results.

Development knowledge from the old benchmark is acknowledged. Cross-API and held-out checks evaluate reuse, not absence of benchmark influence or model pretraining contamination. No remedy is reported as successful until its specified checks and, where applicable, real-model experiment have executed. This planning update changes neither historical evaluation data nor its recorded prompts.

## Proposed implementation file map

This is the concrete planning scope for subsequent implementation, not a record that these files have changed. Reuse existing code where possible and do not refactor unrelated paths. Additional files require a scope update under AGENTS.md.

| Area | Files |
| --- | --- |
| Contracts and target sampling | Modify `src/mirage/contracts/programs.py`, `src/mirage/contracts/exploration.py`; add `src/mirage/contracts/capabilities.py`. |
| Backend boundary | Add `src/mirage/backends/__init__.py`, `src/mirage/backends/base.py`, `src/mirage/backends/pytorch.py`. |
| Operand discovery and mutation | Add `src/mirage/runtime/operands.py`; modify `src/mirage/runtime/representations.py`. |
| Generation and repair | Modify `src/mirage/generator/program.py`, `src/mirage/generator/repair.py`. |
| Execution and inspection | Add `src/mirage/runtime/stateful.py`, `src/mirage/inspector/stateful.py`; modify `src/mirage/runtime/probes.py`, `src/mirage/runtime/worker.py`. |
| Semantic relations | Add `src/mirage/rules/catalog.py`; modify `src/mirage/rules/conditional.py`, `src/mirage/rules/properties.py` at the required integration seams. |
| Workflow and artifacts | Modify `src/mirage/workflow.py`, `src/mirage/cli.py`, `src/mirage/artifacts/report.py`, `src/mirage/artifacts/bundle.py`. |
| Tests | Add `tests/test_capabilities.py`, `tests/test_operands.py`, `tests/test_stateful.py`, `tests/test_relation_catalog.py`; modify relevant `tests/test_program.py`, `tests/test_exploration.py`, `tests/test_properties.py`, `tests/test_cli.py`, `tests/test_artifacts.py`, `tests/test_pytorch_benchmark.py`. |
| Validation and experiment integration | Modify `benchmarks/pytorch_correctness/evaluate.py`, `sbatch/experiments/pytorch_correctness.sbatch`; add `sbatch/validation/stateful_generation.sbatch`. |
| Documentation | This roadmap and its Chinese counterpart; future `docs/stateful-generation-evaluation.md` and `reports/stateful-generation-evaluation.md`. |

No new dependency, Python installation, or project-configuration change is planned. TensorFlow implementation, distributed execution support, arbitrary live-graph serialization, and universal numerical certification are outside this first implementation. They must not be implied by the backend interface or open API pool.

## Execution, evidence, and completion policy

Use the prepared compute-node-accessible environment and dispatch all PyTorch computation, compilation, real-model validation, and integration suites through Slurm. Follow the account, partition, script locations, resource checks, and logging rules in [sbatch/README.md](../sbatch/README.md). Static inspection and submission preflight do not count as executed validation. Reuse a healthy vLLM service within the development task when repeated model validation is required; keep inference and testing steps separate with explicit resources and verify cleanup at task completion.

Store evidence under `result/stateful-generation/<run-id>/stage-<number>/`, with per-attempt source/configuration snapshots. Every gate record contains the implementation hash, commands, environment, actual pass/fail/unsupported counts, failed cases, and artifact paths. Update this roadmap and its counterpart with actual outcomes as work proceeds. A failed gate is not relabeled complete to meet a schedule.

## Planning update record

- Created the staged English roadmap and corresponding Traditional Chinese report.
- Implementation stages 1–7 remain planned; no source implementation, new PyTorch tests, Slurm jobs, or LLM experiments ran during this planning update.
- Document validation passed: both editions contain the same seven gates and heading counts, matching implementation/artifact paths, valid local links, and no trailing whitespace. The English document contains no Chinese text. This validates the planning documents only, not any implementation gate.

## Evaluation-driven planning update

The user requested remedies for the observed limitations in the earlier PyTorch evaluation, including generic edge-case guidance. This update is limited to this roadmap and its Chinese counterpart. Review the recorded failure modes, map them to stage work and acceptance checks, add a proposed prompt supplement, and validate bilingual coverage and links. Historical results and prompts remain unchanged; these remedies are planned, not implemented or measured improvements.

Completed this planning revision: mapped twelve evaluation findings to work and acceptance criteria across all seven stages; added the literal proposed edge-case supplement and its development comparison. Lightweight validation passed for matching stage/remedy counts, identical bilingual prompt text, matching file-path inventories, ten local links and applicable anchors, English-only prose in this document, balanced code fences, and whitespace. SHA-256 checks confirmed that both historical evaluation editions were unchanged. No implementation, PyTorch execution, Slurm job, or model experiment ran; G1–G7 remain planned.

## Execution authorization and progress

The user authorized implementation of the proposed file map and sequential execution of stages 1–7. After a stage passes its acceptance checks, continue automatically to the next stage. Record failures and resolve them before advancing; do not infer gate completion from code changes. Stage 1 is in progress: inspect existing contracts, implement framework-neutral capability/evidence records and the backend boundary, and validate contract behavior before proceeding. The approved validation sequence remains lightweight static checks, Slurm-backed runtime/integration checks, the real-model development pilot, and then the frozen experiment.

### G1 acceptance record

Stage 1 passed its foundational contract checks in Slurm job `769979`: 9/9 tests passed, including fake-backend evidence flow, unknown/LLM-only evidence rejection, aligned relation requirements, bound/unbound call binding, checkpoint payload rejection, seed/sample separation, contract-version separation and deterministic category enumeration. Source hashes and the immutable test snapshot are under `result/validation/769979/`. Static compilation and batch syntax checks also passed. Capabilities currently advertised are tensor descriptors and dense storage identity; mutation, execution and numerical certificates are not yet advertised. The initial candidate set has four state dimensions and ten input categories; mutation composition starts with single candidates, with a two-operand bound to be validated in G2. Numerical comparison remains unavailable until G5 validates a named policy. Stage 2 now proceeds automatically.

### G2 acceptance record

Stage 2 passed in job `770003`: all 23 contract/mutation tests passed. The preceding job `770001` passed 22 tests; review then added explicit CSR-values alias detection and overlapping-write rejection, tested in the final run. Coverage includes nested/namedtuple/keyword operands, repeated references, distinct shared-storage rejection, empty allocations, real/integer/complex eligibility, application-time revalidation, two-operand enumeration, derivatives through two unrelated APIs, CSR writable values and MKLDNN logical materialization. These are mutation-component checks, not evidence of generated target execution or G6 benchmark integration. Distinct shared views, differentiable CSR reconstruction, unknown write effects, and mutation inside capture/compilation remain unsupported. Stage 3 proceeds automatically.

### G3 acceptance record

Stage 3 passed in job `770038`: all 32 accumulated tests passed. Job `770018` first exposed a nonportable enum in saved evidence; the final implementation converts evidence to primitive records and passes weights-only artifact reload. Authored fixtures demonstrate retained and recomputed histories with equal accumulated derivatives, removal of backward without runner substitution, method binding, actual call-boundary mutation, legal import aliases, JSON/source contracts, literal base/edge prompt treatment, and preservation of unknown exceptions without repair. These authored/mock checks establish generator/runtime plumbing, not LLM discovery or real-model diversity. Compilation/capture execution certificates and numerical verdicts remain unavailable. Stage 4 proceeds automatically.

### G4 acceptance record

Stage 4 passed component acceptance in job `770050` (39/39 tests). Job `770048` first passed 38 tests; review identified instrumentation-created detach operations in transition signatures, so the final run additionally verifies that extra observation cannot increase training-transition coverage. Checkpoint facts are inspector-owned; generated dictionaries cannot assert state attainment. Repeated checkpoint names are ambiguous, unobserved compile execution remains unknown, and requested tiny values rounded to zero are not credited as small-nonzero coverage. Target input metadata and output alias groups are captured separately from snapshots. These checks establish the supported eager evidence path; compiled/captured state observation still requires implementation before G6. Stage 5 proceeds automatically.

### G5 acceptance record

Stage 5 passed the currently supported oracle-component checks in job `770087` (53/53 tests). The catalog declares seven semantic families with explicit obligations; declaring a family does not implement its certificates. Representation comparisons are enabled only for reviewed source/effect/numerical subsets. Other family combinations remain inapplicable until their evidence adapters exist. Controls cover true cross-API pairs, synthetic sign corruption, tiny and extreme finite values, nonfinite uncertainty, ill-conditioned matrix rejection, primitive min/max versus overloaded helpers, and independent broadcast arithmetic auditing. Integer-overflow audit domains remain unknown. Job `770084` exposed shared mutable configuration in execution records; `770086` verified the deep-copy fix (52/52), and `770087` includes the final helper control. No synthetic corruption is a PyTorch finding. Stage 6 now begins integration and capability completion; its ten-fixture and real-model gates have not passed.

### G6 development protocol (declared before model execution)

The current new-path fixture replay detected 10/10 cases in job `770759`; independent control auditing in job `770811` corroborated all ten without rerunning their target programs. Job `770911` additionally validated zero-provider replay and the integrated fixture/audit path. These are authored-fixture results, not LLM generation results. Later runtime changes require validation before the final freeze.

The first real-model development pilot uses the existing prepared Qwen/Qwen3-Coder-Next-FP8 service, temperature 0, and its existing serving configuration. Development state targets are: `torch.sin` with two observed backward calls; `torch.no_grad` with two completed restoration scopes (forward-only target); `torch.square` with two target-connected backward events; and `torch.cos` with an actually executed compiled graph. Each receives up to 12 source/repair responses and a 180-second generation/execution budget. The first legal program is canonical; subsequent transition requests receive its source, the state predicate, and prior observed transition signatures, without prescribed action sequences. Require two distinct verified signatures for each target predicate.

A separate descriptive prompt comparison uses `torch.sin` and `torch.add`, base followed by base-plus-edges for each API, 120 seconds and at most 12 candidates per arm. Order is fixed and reported; cache/service effects prevent a causal interpretation from this single comparison. Input-category requests and model/engine settings are matched. The initial held-out model-generation set is `torch.sigmoid` and `torch.Tensor.mul`, each with 120 seconds and at most 8 candidates, evaluated only after development checks pass. If those results guide changes, retire that set and declare a fresh one. Input coverage distinguishes generated source from mutation follow-ups; layout witnesses may originate from the mutation engine. Requested, observed and oracle-eligible categories remain separate.

The final ten-API experiment retains a 300-second budget per API and may start only after G6 passes, including state diversity, edge-category witnesses, held-out reuse, independent fixture controls, regression checks and a frozen engine/prompt protocol. Missing witnesses are recorded and investigated; fixture inputs are never inserted into generation to fill a coverage gap.

### G6 integration and first development iteration

The reusable two-GPU allocation is job `770948`: one GPU serves the prepared model and one is assigned to the attached testing step. The first full regression iteration `iteration-sQhMpu` ran 141 tests and failed one test-only provider import; its isolated child process could not import `test_stateful` under unittest discovery. After explicitly exporting the test module path in that test, `iteration-yVd22Q` ran 142 tests: 141 passed and one saved-fixture audit test was skipped because its artifact path was not supplied. This skip does not replace the separately executed ten-fixture audit. No model inference had been used as regression evidence.

Pilot `stage-6/pilot-01` uses an immutable code snapshot. Early observed failures include use of unavailable NumPy, calling non-target operations through `probe.call`, disconnected target backward paths, malformed source, reserved checkpoint names, and nonportable optimizer observations. These remain failed development attempts. Subsequent generic prompt clarifications explain the target-only probe boundary, differentiable target-result connectivity, direct tensor observations and optional optimizer use. They provide no concrete training sequence. Generic category definitions specify the already-declared dtype-relative measurement thresholds. These changes require a fresh pilot; they do not alter the running pilot or its results.

State-goal scheduling now uses canonical runtime capabilities rather than API identifiers: observed target backward permits gradient, target-connected graph-lifecycle and compiled-execution goals; observed context exits permit restoration goals. The LLM plans the actions and runtime inspection verifies each goal. A context-manager callable capability can remove an inappropriate backward requirement; it does not supply any scopes. Context identity and nesting-depth evidence are recorded for later independent mechanism attribution.

### G6 second-pilot diagnosis and diversity correction

`pilot-01` produced no runtime-valid programs for sin, context or square; compiled produced one legal canonical, but no target attained its requested transition predicate. `pilot-02` received 12 responses per target: sin had six runtime-valid programs, including five state-attaining transitions; context and square had none; compiled had two runtime-valid programs, including one state-attaining transition. Saved-source replay matched its verdict with zero provider calls. These results do not pass G6.

Review found that the second pilot's four raw sin signatures reduced to one core sequence after removing detached observation copies. That count cannot establish diversity. The revised signature retains raw events for inspection but excludes all detach/detach_ events from diversity, and observes actual optimizer updates through public step hooks and completed zero_grad returns. These observers execute no training operations.

The next `pilot-03` uses a concise `stateful-2` prompt: it leads with the target API, backward requirement and exact state predicates, requests a minimal canonical, and clarifies module-parameter object identity, target-only probe calls and direct tensor observations. The LLM still chooses the entire training sequence; no benchmark input or action template is supplied. The generic formal experiment will alternate forward and backward paths, so an unsupported backward input domain cannot exclude legal forward tests; this schedule is identical across APIs. Held-out and formal evaluation have not started.

### G6 next development protocol: broader response budget

Pilot 04 still failed the diversity gate, despite successful state-attaining examples. Before pilot 05, keep the same four APIs and core predicates, but allow 300 seconds and at most 24 responses per API. This increases the development opportunity budget; it does not lower the required two distinct verified training signatures. The held-out set has not been used.

Syntax handling now removes only one standalone trailing `}` when Python reports that exact unmatched delimiter and removing it makes the complete source parse. It never adds or rewrites a training statement. Save raw response, raw source, normalized source and the normalization record; report raw syntax-valid and post-normalization counts separately. Invalid optional optimizer/scheduler/scaler dictionaries and nonportable checkpoint helpers receive explicit contract diagnostics rather than unresolved attribute errors. Context comparison rejects uncertified direct same-instance reentry. The unused-optimizer constructor certificate does not certify optimizer updates.

Budget admission reserves the greater of 10 seconds or 1.5 times the largest completed candidate duration plus 5 seconds. This is a cost-based dispatch policy, not a guarantee against infrastructure delays. Actual reserve and used time are recorded. All source, repair, execution and pair costs remain inside each nominal API budget. These development changes will be fixed before the prompt comparison, held-out run and final freeze.

### G6 joint-state diversity remedy

Pilot 05 reached the two-signature target for sin, but its 21 state-attaining no_grad transitions still used one sequence. More responses alone did not solve this measured diversity gap. Before pilot 06, retain the same core predicates, 300-second budget and 24-response bound, and add a deterministic, API-independent auxiliary predicate: one or two observed target boundaries at the final checkpoint. This is a joint abstract-state request, not an operation sequence or source template. The LLM chooses how to satisfy it. The sampler receives predicates and an attempt index, not an API or issue identifier.

Boundary units are explicit: one completed eager target probe, or one executed target-containing compiler-graph boundary. For a context factory, construction is separate from entering/exiting the returned context; graph boundaries do not claim a count of fused backend kernels. Raw events, attained predicates and distinct action signatures are retained. The original per-dimension two-signature gate remains required. This protocol change is development evidence and will not be compared causally with earlier pilots.

### G6 pilot acceptance, oracle review and input-feedback protocol

Pilot 06 passed the four core diversity checks on its snapshot: sin/context/square/compiled produced 3/5/2/2 distinct verified signatures. This does not complete G6. Final review added controls against failed compiled invocations, arbitrary custom backends, backward on an unrelated eager path, and implementation-identity observations. Compiled-state evidence now requires a successful supported target graph; required backward is tied to its target boundary. Execution-mode equivalence also requires a conservative source certificate. Arbitrary callbacks, implementation metadata and data-dependent Python branches remain inapplicable. These changes require new validation on the final engine.

The first base/edge comparison had 12 responses per arm and did not establish small_finite or large_finite witnesses. One generated source described 1e-8 float32 values as small, below the intended stress level but above the declared small-magnitude threshold. No prompt improvement is inferred from this result. Generic measured-category feedback now reports missing categories and actual dtype thresholds to the LLM. A refinement requests a new legal initial input shared by both MR members; it does not change the original verdict, provide expected answers, mutate numeric values in the runner, or refine a Suspected Failure. Coverage refinements have separate provenance and use the existing two-response bound. Category scheduling advances for fresh requests, not repair responses; realized category counts are reported per arm.

Before the next model calls, declare pilot 07 with the same four core/joint state goals, 300 seconds and at most 24 responses per API. Declare a new descriptive comparison `edges-02`: sin and add, base then base-plus-edges, 300 seconds and at most 24 responses per arm, identical engine and coverage-feedback policy. Retain `edges-01` as an earlier failed acceptance comparison; do not pool it into `edges-02`. The original held-out set remains unused. Full regression and ten-fixture/control validation must pass before this final development iteration is accepted.

### G6 final validation protocol: separate state and input objectives

Pilot 07 failed the unchanged diversity gate (sin/context/square/compiled: 3/2/1/1 verified signatures). Its numeric-coverage refinements consumed responses in the state-diversity check. Before pilot 08, retain the four core/joint predicates, 300 seconds and 24 responses per API, disable optional input-coverage refinement only for this focused state check, and advance joint-state scheduling on fresh proposals rather than repair responses. The independent edges-03 check retains coverage refinement, the same sin/add base/edge ordering, and 300 seconds/24 responses per arm. Both gates remain mandatory; formal evaluation combines state sampling and input-coverage refinement. Neither response repairs nor coverage refinements establish diversity without runtime evidence.

Replay acceptance now requires equality of saved source and follow-up observations, state/checkpoint evidence, target-path evidence and signatures, as well as the verdict; verdict agreement alone is insufficient. Revalidate the regression suite, all ten fixtures with independent controls, development checks, and an unused held-out set on the same source snapshot before freezing. Earlier pilot/edge results remain development records and cannot satisfy the final source-hash gate.

`edges-02` completed all four arms (24 responses each): runtime-valid counts were 24/24/22/23 for sin-base/sin-edges/add-base/add-edges. Their union covered all ten declared input categories. This passed the earlier snapshot's edge gate, but its verdict-only replay cannot satisfy the final evidence-replay gate. It remains a descriptive development result, not evidence of a causal edge-prompt benefit. The next full regression iteration is `iteration-bNraBG`.

Final-engine regression `iteration-bNraBG` passed 154 tests with one artifact-path-dependent audit skipped (155 total, 81.172 seconds test time). `fixtures-03` / `iteration-oMtHkk` then detected all ten authored fixtures and independently corroborated all ten controls; the fixture gate passed in 58 seconds wall time. This is integration/oracle evidence, not LLM discovery. Pilot 08 and edges-03 now run on the matching production source. A focused invocation with the saved fixture path will execute the previously skipped audit before freeze.

Pilot 08 completed successfully in `iteration-AdOIV1`: sin/context/square/compiled received 24/24/24/21 responses, with 9/15/7/7 runtime-valid cases and 4/4/2/3 distinct verified transition signatures. Compiled stopped at its time-admission limit; this is not 24 completed responses. Saved-source and saved-follow-up evidence replay passed with zero provider calls. This accepts the final-engine diversity subgate. The same-source edges-03 comparison is running; held-out evaluation and G7 remain unstarted.

`edges-03` / `iteration-mmlxBP` passed on the final source: all four arms produced 24 runtime-valid cases from 24 responses, their union established all ten input categories, and semantic-evidence replay passed with zero provider calls. This is descriptive evidence of achieved coverage, not a causal prompt-effect estimate. The previously skipped saved-fixture audit was executed in `iteration-pBSKv7`; all seven benchmark tests passed (2.337 seconds). With development checks complete, the predeclared unused held-out APIs, `torch.sigmoid` and `torch.Tensor.mul`, now run in `heldout-01` with 120 seconds and at most eight responses each. No production source changes are planned after this held-out check.

### G6 held-out failure and graph-lifetime clarification

Heldout-01 failed the reuse gate: sigmoid had one runtime-valid canonical and no attained transition among eight responses; Tensor.mul attained transitions. Sigmoid generated losses that reused a consumed upstream graph, even when downstream target outputs were rebuilt. This is a generation-legality failure, not an oracle miss or a confirmed framework defect. The failed held-out set is now development evidence and will not be reused as an unseen set.

Before further model calls, introduce stateful-3 prompt clarification of generic autograd graph ownership: shared upstream intermediates share graph lifetime; the LLM must choose legal retention/recomputation; backward invocations differ from target-connected gradient events. No executable sequence or API-specific example is supplied. Keep all verdict rules and gates unchanged. Revalidate full regression, fixtures-04, pilot-09 (same four goals, 300 seconds/24 responses), and edges-04 (same four ordered arms, 300 seconds/24 responses). Revisit sigmoid only as a development check with 300 seconds/24 responses. Then use a newly declared, unused held-out set, torch.tanh and torch.Tensor.sub, with the same 300-second/24-response budget. Prompt and budget changes are jointly evaluated; no causal benefit is assigned to either alone. Retain both the original held-out failure and the new result in the final report.

Pilot 09's context group failed the unchanged diversity gate: only one transition signature was attained. Its early canonical responses wrapped a computation instead of invoking the configured context factory. Generic backward guidance was irrelevant to this forward-only request. Before the queued edges-04 run begins, condition the new graph-lifetime paragraph on the existing requires_backward flag, with no API-name check. Forward-only requests retain their earlier exact prompt/version; backward requests retain stateful-3 guidance. Pilot 09 remains failed development evidence. Validate the resulting source with fixtures-05, full regression, pilot-10 and the already declared edges-04 protocol before using any fresh held-out APIs. This changes prompt applicability, not state predicates, numerical rules or diversity thresholds.

The conditional-guidance engine passed edges-04 (`iteration-NUea40`, 586 seconds): runtime-valid counts were 21/23/24/24, all ten categories were observed, and evidence replay passed. Full regression `iteration-rgSvHY` passed all 155 tests with no skips (87.953 seconds test time). Fixtures-05 passed all ten detections and independent controls. Pilot-10, the retired-sigmoid development revisit, and heldout-02 are scheduled sequentially with machine-checked prerequisite gates; a failed prerequisite prevents the unused held-out APIs from being queried. The formal experiment remains unstarted.

### G6 use the full declared generation budget

Pilot 10's square group reached only one verified signature before the 24-response cap, so the whole diversity gate fails even though other groups have attained diversity. Before pilot 11, keep the current source, all four core/joint goals, the same 300-second per-API budget, and the same oracle conditions; raise the response ceiling to 100000 so time admission, rather than an early response cap, bounds every group. This is a development opportunity adjustment applied uniformly to all four APIs. Do not pool or relabel pilot 10 as passed. Existing final-source edge, fixture and regression evidence remains valid because production source is unchanged. The sigmoid revisit and fresh held-out calls remain gated on the new pilot's measured result.

### Formal-budget amendment before any formal generation

Slurm did not accept an attempted extension of job 770948 from four to four-and-a-half hours (`Unspecified error`; the verified limit remains four hours). Submission preflight estimated a new allocation only on the following morning; preflight is not an executed job. Under the user's original allowance to adjust the suggested five-minute budget when necessary, predeclare a uniform **180 seconds per API** for G7 before any formal model call. Keep all ten APIs, model settings, state/mutation policies and oracle thresholds unchanged. The freeze command now records its explicit --budget value (default 300; positive values required), and the experiment must match it exactly. Development checks keep their separately recorded budgets. Do not compare these 180-second results causally with the historical 300-second experiment. Reserve allocation time for post-hoc evidence and service cleanup; if remaining time cannot support the complete protocol, use normal Slurm scheduling rather than bypassing allocation limits.

Pilot 11 passed in iteration-XKxYJw on the final engine: sin/context/square/compiled produced 52/56/46/24 parsed sources, 16/23/18/8 runtime-valid cases, and 2/4/6/3 distinct verified transition signatures. Both-member semantic replay passed with zero provider calls. The same source had already passed all 155 regression tests, ten fixtures/independent controls, and edges-04. The retired sigmoid check is now running; the new held-out set remains gated on its result. The higher response ceiling establishes development capability, not a controlled improvement in generation rate.

### G6 final acceptance and G7 start

Heldout-02 / iteration-ceILJE passed on the unchanged production engine: tanh received 24 responses, produced 23 parsed sources, and had three runtime-valid cases including two attained transitions; Tensor.sub received 24 responses and had nine runtime-valid cases including eight attained transitions. Evidence replay passed. The retired sigmoid revisit separately had 12 runtime-valid cases and 11 attained transitions from 24 responses; it is development evidence, not held-out evidence.

Freeze iteration-sdzBp9 succeeded. The protocol is result/stateful-generation/770948/stage-6/frozen-01/protocol.json. All 43 production Python source hashes agree across regression iteration-rgSvHY, fixtures-05, pilot-11, edges-04 and heldout-02. The frozen evaluator hash is 84c740ad69bbf9f75dd930475de276e32cf7d865161c7662628e207486b56a0d; the uniform formal budget is 180 seconds per API. G6 is accepted for the documented capability subset. G7 started immediately in stage-7/formal-01. Do not change frozen source, prompts, thresholds or attribution code during this experiment; low reproduction is an admissible outcome.

### G6 accepted; user-authorized partition change before G7

The user explicitly authorized changing partition to shorten queue time. Heldout-02 passed: tanh had 3 runtime-valid cases and 2 attained transitions from 24 responses (23 parsed sources); Tensor.sub had 9 valid cases and 8 attained transitions from 24 responses. Strict replay passed. The retired sigmoid development revisit passed with 12 valid cases and 11 attained transitions from 24 responses. Freeze-01 passed all source-hash gates with its 180-second budget, but **no formal generation was executed under that protocol**.

The dev partition has H200 GPUs, a four-hour maximum and a higher scheduling priority tier; submission preflight estimated a substantially shorter wait than 8gpus. Following the user's authorization, use an explicit dev override with two GPUs, eight CPUs and a 90-minute allocation for the final experiment. Close the completed development allocation normally, validate the new environment/fixtures, and create a new frozen protocol with **300 seconds per API**, restoring the originally suggested budget before any formal model call. Keep freeze-01 as a superseded, unused protocol. No production generator, mutation or oracle source changes accompany this resource change. Record both service startups, allocated time and cleanup evidence separately. A preflight estimate is not a submitted or executed job.

Replacement job 771760 was submitted to dev with an afterany:770948 dependency, then started on H200 node 25a-hgpn015 after cleanup; actual scheduling did not require the preflight's predicted wait. Development job 770948 completed with exit 0 after 12,177 allocated seconds. Before stopping it, service metrics verified zero running/waiting requests and the testing cache was removed. The interrupted unified-exec terminal was no longer attachable; its idle owned step ignored TERM and was killed. Service/terminal steps ended as cancelled (signal 9), while test iteration exit statuses remain separately preserved. All old job steps exited. Cleanup evidence is saved in cleanup-before-stop.json and cleanup-final.json. The replacement service is starting; no formal model calls have begun.

### G6 final acceptance and G7 launch

Replacement service startup took 185 seconds. Testing software/environment fields matched the development job except hostname, partition and job ID; serving fields differed only in host, job ID and GPU UUID. Fixtures-06 / iteration-t0VrEz detected all ten cases and independently corroborated all ten controls on the replacement node (74 seconds). Freeze-02 / iteration-a0JvIk passed the complete source-hash gate and records 300 seconds per API. Its protocol is result/stateful-generation/771760/stage-6/frozen-02/protocol.json; the old 180-second freeze remains unused.

G6 is accepted. G7 now executes the ten frozen targets under result/stateful-generation/771760/stage-7/formal-01. The generator receives only installed API specifications, generic configurations and its own generated/observed history. No production source or evaluator changes are permitted during this run. Independent attribution remains post-hoc; low discovery is a valid result and will not cause protocol changes.

### G7 report provenance and post-hoc review

The formal run uses `stage-7/formal-01` in allocation 771760. Generation and production evaluation remain frozen. The evaluation report now includes a verbatim formal prompt, its backward-only graph-lifetime addition, context fields, file/request hashes, and final development gate counts. Historical roadmap 003 and its evaluation are unchanged.

Review of the automatic attribution code found a scope mismatch: compiled target records store the entire FX graph output, whereas the single-target reference expects the API output. A tuple-versus-tensor difference is insufficient evidence of a framework defect. Preserve `evaluation-automatic-v1.json` and, after all ten API budgets, execute the separately recorded `posthoc_review.py` under Slurm to produce `evaluation-reviewed-v2.json` and `review-evidence-v2.json`. The review excludes uncertified whole-graph outputs from single-target audits and retains uncertainty for the excluded scope. It changes no generated program, production verdict, budget, or future request. Final metrics must use reviewed labels and disclose any withdrawals.

After formal generation and the versioned review finish, correct the same output-scope guard in `benchmarks/pytorch_correctness/evaluate.py` and add focused regression cases in the already-approved `tests/test_pytorch_benchmark.py`. Validate unknown whole-graph outputs, partial negative evidence, and independently positive eager evidence. This is a post-experiment audit-only correction; retain the frozen evaluator snapshot and distinguish its hash from the corrected file. No generator or engine changes are planned.

## Final G7 acceptance and handover

The formal run completed in `result/stateful-generation/771760/stage-7/formal-01`, using the accepted `stage-6/frozen-02/protocol.json`. The driver invocation was `experiment-stateful --output <formal-01> --gate <frozen-02/protocol.json> --budget 300`; `iteration-UYFdmZ` records the exact command and immutable source. It exited 0 after 2,825 seconds. All ten API budgets completed. The 180-second frozen-01 protocol was never used for formal generation.

### Phase 1: retained benchmark evidence

Reused the ten previously collected deterministic upstream issues and authored minimal programs; no new issue search or new-defect discovery is claimed. The final production engine passed the ten fixture/control checks in fixtures-05 and, after moving to `dev`, fixtures-06. Every fixture produced a production Suspected Failure with independently corroborated controls. These are framework-capability checks, not generated discoveries. The existing roadmap 003, reproduction report and historical evaluation remain unchanged.

### Phase 2: generalized implementation and validation

G1–G5 implemented the source/state contracts, backend boundary, actual-operand mutation, full-source LLM program path, runtime state evidence and supported semantic-family relations. G6 verified runtime diversity, all ten input categories across valid evidence, fresh held-out API reuse and exact semantic replay of both pair members. The accepted pre-formal source passed 155 tests without skips in 87.953 seconds. The final development table and limitations are in [the evaluation](stateful-generation-evaluation.md#final-development-acceptance-evidence). These results establish the implemented subset; they do not implement all training-state dimensions, all PyTorch APIs or TensorFlow.

The generator receives an installed API specification, a full-source contract, runtime-derived state goals, input categories and relation obligations. It owns every training transition; mutations operate on discovered tensors. The exact final [forward prompt and backward-only addition](stateful-generation-evaluation.md#exact-frozen-request-example), [complete request](../result/stateful-generation/771760/stage-7/formal-01/203078/attempt-0000/request.json), and frozen hashes make this boundary reviewable. Issue IDs and benchmark programs are confined to benchmark execution/post-hoc attribution, not production requests.

### Phase 3: measured outcome and reasons

The formal experiment recorded 434 generation attempts, 433 extracted sources, 432 static-valid candidates, 255 completed source executions, 177 completed follow-ups, 176 runtime-valid candidates, 60 attained transition predicates and 95 applicable comparisons. All applicable comparisons passed; no production Suspected Failure was reported. The independently confirmed known-issue reproduction count is **0/10**, compared with **10/10** supplied fixtures in Phase 2. These measure different tasks.

The reviewed independent counts are TP=0, FP=0, TN=39 and FN=0. FPR is 0/39 only on the adjudicated negative subset; violation-detection rate is undefined because no eligible positive was established. Thirty additional negatives are outside the required state/path domain; 365 attempts remain audit-unknown. All first-detection times are censored. The report provides each API's trigger, actual evidence, generation/state/mutation/applicability failure reasons and links to raw records. It specifically distinguishes std's generated extreme inputs from missing numerical/execution certificates, and distinguishes ordinary dense passes from absent sparse, MKLDNN or cross-device triggers.

### Review correction, validation and artifact preservation

`posthoc_review.py` ran against the frozen evaluator in `iteration-v85oR2` (exit 0, five seconds), preserving automatic-v1 and creating reviewed-v2 plus member evidence. Eight automatic positive labels based on whole-graph outputs were withdrawn to unknown, including three eligible FN labels. No generated program, production verdict or budget changed.

After formal generation, `benchmarks/pytorch_correctness/evaluate.py` was corrected to exclude uncertified FX whole-graph outputs from single-target mathematical audits and attribution. Two focused tests were added to `tests/test_pytorch_benchmark.py`. The initial audit guard passed nine tests in `iteration-YBxIUt`; after adding the matching attribution guard, the final nine-test suite passed in `iteration-4ur2Pq` (2.351 seconds, no skips). `iteration-0AVDgP` then crosschecked every saved candidate against reviewed-v2 with zero label/classification or metric differences. This correction changes the evaluator hash after the experiment; the frozen evaluator snapshot remains available, and all 43 production engine hashes remain unchanged. The final-source full 155-test result refers to the pre-formal implementation, not a claim that a 157-test full suite was rerun after the audit correction.

Evidence: [automatic-v1](../result/stateful-generation/771760/stage-7/formal-01/evaluation-automatic-v1.json), [reviewed-v2](../result/stateful-generation/771760/stage-7/formal-01/evaluation-reviewed-v2.json), [member review evidence](../result/stateful-generation/771760/stage-7/formal-01/review-evidence-v2.json), and [corrected-evaluator crosscheck](../result/stateful-generation/771760/stage-7/formal-01/corrected-evaluator-crosscheck.json). The [English evaluation](stateful-generation-evaluation.md) and [Traditional Chinese evaluation](../reports/stateful-generation-evaluation.md) contain matching results and verbatim prompt artifacts.

### Resource closure and acceptance decision

Job 771760 used the explicitly authorized `dev` partition with two GPUs and a 90-minute limit. It completed after 3,526 allocated seconds; job 770948 had used 12,177 seconds. Startup times were 185 and 217 seconds respectively. There was one allocation replacement, no in-place failure restart, and no change to formal budgets after generation started. Both owned jobs and all steps have exited; the testing terminal closed, outstanding inference counts were zero, and job-specific test caches were removed. Both service steps ended with signal 9 during owned shutdown; this is recorded rather than represented as a clean service exit.

The [resource ledger](../result/stateful-generation/771760/resource-ledger.json) records 46 iteration timings/statuses and separates startup, formal work, audits and allocated GPU time. [Cleanup evidence](../result/stateful-generation/771760/cleanup-final.json) confirms no active owned jobs. G7 is accepted because all targets, metrics, uncertainty, raw evidence and cleanup are accounted for. Its acceptance never required a minimum discovery rate. No remaining compute or inference service is left running for this task.
