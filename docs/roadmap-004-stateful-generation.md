# Roadmap 004: General representation mutation and LLM-generated training states

## Status and approved direction

The user accepted the architectural direction and requested implementation steps with corresponding acceptance gates. This roadmap organizes that design; it does not report implementation or experimental completion. The present update creates only this English roadmap and its [Traditional Chinese counterpart](../reports/roadmap-004-stateful-generation.md).

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

| Stage | Deliverable | Gate | Initial status |
| --- | --- | --- | --- |
| 1 | Contracts, backend boundary, and frozen acceptance definitions | G1: responsibilities and evidence requirements are explicit | Planned |
| 2 | General operand discovery and representation mutation | G2: mutations generalize across input structures and preserve required relations | Planned |
| 3 | LLM-owned training program generation and execution | G3: executable state transitions come from generated source | Planned |
| 4 | Runtime inspection and state/transition evidence | G4: coverage cannot be credited without observed evidence | Planned |
| 5 | Semantic-family MR and oracle integration | G5: validated relations distinguish violations from inapplicability | Planned |
| 6 | End-to-end validation, generation diversity, and freeze | G6: the new path is independently validated and ready for measurement | Planned |
| 7 | Frozen LLM experiment and independent attribution | G7: results are complete, auditable, and correctly scoped | Planned |

Proceed in order. A failed gate returns work to the responsible earlier stage; preserve failed evidence and do not silently relax acceptance conditions. Once its checks pass, continue within the approved scope without requesting permission again for routine stage transitions. Material design or file-scope changes still follow AGENTS.md. Gate completion requires inspecting results, not merely a successful Slurm submission or job exit.

## Stage 1: Contracts and backend boundary

### Work

Define versioned contracts for operand trees, alias groups, mutation plans and obligations, abstract state predicates, generated programs, semantic checkpoints, execution records, and MR specifications. Separate planned predicates from observed facts and unsupported or unknown evidence. Track source provenance for operations: LLM source, representation mutation, instrumentation, or oracle computation.

Add a backend interface and the initial PyTorch adapter. Keep PyTorch imports and framework operations in the adapter or existing explicitly backend-specific modules; new core scheduling and verdict logic must consume framework-neutral records. Use a small fake adapter to test this boundary without claiming TensorFlow compatibility.

Introduce the new `stateful` strategy and an independent contract version while preserving existing strategies. The generated interface is `build(device, seed)` plus `run(ctx, probe)`. Probe methods observe or mark boundaries; they never implement backward, optimizer updates, context recursion, compilation, capture, or replay for the generated program.

Define the initial state dimensions: gradient state, context state, graph lifecycle, and execution mode. Record the observable predicates supported for each dimension; unknown graph facts remain unknown. Freeze acceptance cases, numerical policies, resource bounds, and explicit finite coverage denominators before measuring results. Start with a stable deterministic enumeration of eligible mutation candidates, retaining configuration and order; adaptive ranking is a separate change unless explicitly included in the approved contract.

### G1 acceptance

- Contracts represent unsupported capabilities, missing evidence, invalid execution, inapplicable relations, Pass, Suspected Failure, and coverage-only/unassessed outcomes without conflating them.
- Mutation selection has no API/issue identifier input. Semantic-oracle selection does not gate API admission.
- The fake adapter exercises the core record flow; the PyTorch adapter exposes only implemented capabilities.
- Initial state predicates, semantic-family certificates, mutation scope, budgets, and acceptance fixtures are written down. An unspecified semantic property cannot default to true.

Evidence: contract checks, dependency inspection, adapter capability inventory, and the fixed acceptance specification. No real-model discovery claim is made at this gate.

## Stage 2: General input discovery and representation mutation

### Work

Discover tensors in actual positional and keyword arguments and nested dictionaries, lists, tuples, and namedtuples. Use structural paths, such as `kwargs["out"]`, instead of a fixed positional subject. Preserve repeated object references and record storage alias groups. Treat sparse components through layout adapters; do not mutate structural indices as if they were ordinary values.

Reuse supported stride, offset, clone/materialization, negative-bit, conjugate-bit, singleton-stride, and value-preserving expansion primitives. Eligibility depends on tensor metadata and the selected relation's obligations. The first implementation certifies dense-strided transformations; additional layout operations enter only after their own legality, alias, gradient, and replay checks. Discovery of a layout is not a claim that mutations on it are supported.

Handle read/write roles, alias preservation, autograd connectivity, and parameter/optimizer identity explicitly. Do not unconditionally detach an existing computation graph. Reject transformations whose required relationships cannot be maintained. Unknown writable or alias behavior cannot receive an equivalence verdict.

For eager execution, inspect actual target operands at the call boundary. For compiled/captured execution, establish a supported preparation boundary outside tracing/capture and verify that the executed target consumes the transformed input. Instrumentation that changes graph execution, causes graph breaks, or only observes tracing cannot establish execution-mode coverage.

### G2 acceptance

- Discover all tensors in the declared supported container forms, including keyword and nested inputs; reconstruction preserves container structure and repeated references.
- The same operand descriptors and obligations produce the same mutation candidates regardless of API identity. Representative APIs reuse the same mutation code without added name-based branches.
- Each supported action has a successful relation-preserving case and rejection cases for violated preconditions. Verify values, shape/dtype/device obligations, alias relationships, and required gradient connectivity.
- Unsafe parameter replacement, lost aliasing, invalid overlapping writes, and unsupported layouts are rejected or marked unresolved rather than counted as valid transformations.
- Save planned, eligible, applied, and observed mutation states separately. Only verified attainment contributes coverage.

Evidence: operand discovery records, mutation plans, before/after metadata and values, relation checks, and rejection reasons. Compile/capture insertion that lacks reliable evidence remains explicitly unsupported rather than using an eager fallback.

## Stage 3: LLM-generated training transitions

### Work

Generate and validate a canonical program, then ask the LLM to generate complete legal transition programs and, where needed, MR-compatible pairs. Supply public API specifications, the program contract, abstract state goals, relation requirements, budgets, and prior program context. Withhold issue identifiers, MREs, known trigger sequences, and expected answers.

The LLM owns the placement and ordering of forward/backward, gradient clearing and accumulation, optimizer operations, contexts, checkpointing, and compile/capture/replay operations used by the candidate. No `depth` or `microbatches` field causes the runner to synthesize a hidden lifecycle. Shared syntax and probe interfaces do not prescribe a single transition sequence.

Resolve supported import aliases in validation. Preserve a source-only JSON response envelope where practical. Repair syntax, contract violations, confirmed illegal operations, and verified missing paths/states with bounded feedback. Unknown runtime exceptions and suspected correctness failures do not enter a repair-to-Pass loop. Preserve every attempt and its provenance.

### G3 acceptance

- The runner can execute different legal state-transition structures through the same contract, without adding a new execution template for each structure.
- Removing a backward, context entry, optimizer update, or other required action from source removes that action from execution; the runner does not replace it.
- Mock-provider and authored contract fixtures demonstrate alias handling, bounded repairs, malformed returns, missing target paths, and preservation of unknown exceptions.
- Saved records show the exact prompt, source, configuration, repair history, and operation provenance. Generation validity remains distinct from state attainment.

Evidence: source/trace correspondence and contract/repair tests. Real LLM transition diversity is measured at G6, after the inspector is available; mock-provider tests are not LLM generation results.

## Stage 4: Inspector and verified state coverage

### Work

Collect backend observations at semantic checkpoints and actual API events. Record target entry versus completed return, gradient-path evidence, relevant parameters and optimizer state, context entry/exit state, and supported graph-execution evidence. Register resources needed for inspection without accepting generated claims as proof of their state.

Verify requested state predicates and summarize observed transitions. Keep a normalized source fingerprint and a separate runtime transition signature. Changing constants or variable names alone does not establish a different training-state path. Compare checkpoint meaning rather than raw operation counts.

Freeze the abstraction fields used for transition signatures and their finite candidate-set denominators. Count only evidence-supported transitions; report unsupported observations and partially observed executions separately.

### G4 acceptance

- False checkpoint labels, absent target calls, unrelated gradients, and missing state transitions cannot obtain full coverage credit.
- A configuration saying compiled or captured is insufficient without supported actual execution evidence; tracing alone fails the gate.
- Two programs with different names/constants but the same abstract observed path do not inflate transition diversity. Distinct verified paths remain distinguishable.
- Incomplete evidence, an unmet target predicate, an illegal program, and an oracle violation retain different diagnoses.

Evidence: trace records, state snapshots, predicate evaluations, transition signatures, and intentionally incomplete-evidence controls.

## Stage 5: Reusable semantic-family oracles

### Work

Implement relation selection using semantic capabilities and explicit certificates rather than target-name branches. Initial relation families cover logical-value representation equivalence, differentiable computation, certified decomposition/accumulation, persistent-state restoration, context restoration, execution-mode equivalence, and observable alias/effect relationships. Add algebraic references only for a coherent semantic class with a documented defining relation; do not expand by writing one formula per API.

A family can remain unavailable for a capability combination until its certificate and observation requirements are implemented. All advertised relations require reviewed preconditions, input relations, allowed differences, semantic checkpoints, observables, tolerance policy, and nondeterminism handling. Dynamic observations establish measurable conditions but do not prove arbitrary program purity, differentiability, or equivalence by themselves.

Reuse sound existing comparisons and formulas as limited references where their conditions hold. Refactor only the required seams: the new oracle consumes evidence from LLM-generated transitions and never silently constructs the transition under test. For gradient checks, preserve the declared differentiation coordinates; do not detach and substitute a different question. Directional finite differences require domain, smoothness, step-size, and numerical-stability checks; ambiguous results remain unassessed.

### G5 acceptance

- Every advertised relation passes known-valid examples, rejects violated preconditions/input relations, and detects deliberately corrupted observations. Synthetic corruption validates the checker; it is not reported as a PyTorch defect.
- Demonstrate cross-API reuse for each advertised relation applicable to multiple APIs, without adding API-specific oracle branches. Relations without sufficient evidence remain explicitly unavailable.
- Nonfinite or unstable references, failed checkpoint alignment, uncertain effects, and control disagreement cannot become false confirmed failures.
- Well-defined coverage-only executions remain available even when no oracle applies. All automated failures remain suspected until independent corroboration.

Evidence: the relation/capability matrix, positive and negative controls, synthetic-corruption checks, applicability reasons, and numerical-policy validation.

## Stage 6: Integration, real-model diversity, and freeze

### Work

Connect generation, mutation, execution, inspection, oracle selection, evidence storage, and replay through the new workflow. Replay reruns saved source from supported initial state with the saved mutation plan; it does not invent arbitrary serialization of live autograd graphs or call the LLM. Unsupported state restoration is reported explicitly.

Replay all ten admitted benchmark fixtures through the new worker, with explicit source containing their transitions. Legacy handlers must not silently supply missing actions. Record each new-path detection, failure, unsupported feature, or unassessed result. The goal remains detection of all ten; a prior 10/10 is not evidence for this implementation, and remaining gaps are not hidden by weakening a relation.

Run a real-model development pilot to verify that the LLM produces multiple legal state-transition paths. For each of the four initial state dimensions, use a nontrivial, declared predicate and require at least two distinct runtime-verified transition signatures reaching that predicate. This is an engineering acceptance target, not a population-level diversity estimate. If a dimension lacks support or the model does not meet the target, keep that gate incomplete and diagnose the cause.

Use a predeclared development API set and a separate held-out API set drawn from supported semantic capabilities. After freezing, test held-out APIs without adding API-name logic. If their results guide code or prompt changes, reclassify them as development evidence and declare a fresh held-out set for a new freeze.

### G6 acceptance

- Integrated execution and relevant existing regression checks pass; deterministic saved-source replay uses zero provider calls and preserves required evidence.
- The ten-fixture result matrix is complete. A claim that all ten are supported requires 10/10 new-path detections with agreeing controls. An incomplete matrix cannot be presented as full benchmark integration; remaining capability gaps must be resolved before the planned full-ten experiment.
- The real-model pilot meets the declared per-dimension transition-diversity target without hidden runner-generated transitions. Record generation, repair, state-attainment, and applicability denominators.
- Hold-out testing demonstrates reuse without new API-specific mutation/oracle branches. Unsupported held-out cases remain explicit and do not count as assessed successes.
- Freeze source hashes, prompts, capability certificates, policies, model/environment settings, API pools, budgets, and attribution rules. Preserve development failures outside the frozen experiment population.

Evidence: new-path integration matrix, pilot prompts/programs/traces, diversity records, held-out results, replay checksums, regression logs, and frozen protocol. Failure does not justify inventing a benchmark-specific fallback.

## Stage 7: Frozen experiment and independent evaluation

### Work

Run the new LLM experiment with a nominal 300-second generation/execution budget per target API. Charge generation, repair, exploratory execution, mutation variants, both members of an MR pair, and in-loop validation to that budget. Measure metadata discovery, service startup, and later independent audit separately; report any dispatch reserve and actual used time. Do not describe sequential 300-second API budgets as exact equal amounts of generated work.

Use explicit initialized values and deterministic issue triggers. Execute each configured case once; source/follow-up members and distinct mutation variants are separate execution cases, not statistical repetitions. Preserve counts for generated programs and derived executions separately. Do not rerun deterministic cases solely to increase apparent confidence.

Perform benchmark attribution after execution, without feeding issue-specific information back into generation. Use independent evidence to distinguish benchmark reproduction from unrelated anomalies in the same API. Report unadjudicated cases and false-positive denominators explicitly. The old guided experiment remains a historical comparison, not a controlled causal baseline unless conditions are demonstrably matched.

### G7 acceptance

- Every target has a complete outcome or a documented infrastructure/budget failure. Low discovery rates are valid experimental outcomes, not grounds for altering the frozen protocol.
- Report valid/executable rates, state attainment, verified transition diversity, mutation eligibility/application/attainment, MR applicability, known-issue reproduction, audited violation detection/FPR, first-detection time, generation counts, and costs with explicit denominators.
- Distinguish failure to generate a legal program, failure to reach state, mutation limits, MR inapplicability, oracle misses established by independent positive evidence, and attribution misses.
- Preserve raw requests/responses, source, exact initial data where supported, mutation plans, traces, observations, environment, hashes, and audit results. Confirm service/process/allocation cleanup.

Evidence: a new English evaluation report and matching Traditional Chinese report, together with complete immutable execution artifacts. No minimum issue-reproduction percentage is imposed on the frozen experiment.

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
