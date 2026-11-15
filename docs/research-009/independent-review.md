# Phase 009 independent source review

This review compares the candidate with the preserved phase-008 engine at
`result/general-state-research/baseline-v8/code`. It is a source inspection, not a
runtime validation result, a proof of Python purity, or evidence of improved bug
discovery. Each Slurm regression uses its own immutable candidate snapshot;
follow-up source changes require another snapshot and executed checks. The
argument-normalization revision was reviewed after root recorded protocol ledger
revision `01b4929`, before its reported runtime outcome was available here.

## Scope and independence

The generator implementation author independently reviewed changes to
`workflow.py`, `sampler/policy.py`, `artifacts/report.py`, and the backend source
certificate. A separate backend author reviewed the generator's binding hints
and structured planning protocol. The generator author's own implementation
checks are not described as an independent review of that implementation.

The review examined family request allocation and cached plans; explicit
planned/unresolved responses and examined IDs; repair evidence drawn from the
actual generated source; the bounded no-progress policy; aggregate goal/MR
metrics; and binding/effect restrictions in the expanded source certificate.
No Torch execution, inference, compilation, or experiment was performed by this
reviewer. AST parsing and `git diff --check` passed for the generator files and
their tests before the runtime handoff. Executed test outcomes belong in the
roadmap and immutable Slurm logs, not inferences from this inspection.

## Concrete counterexamples and dispositions

1. **Malformed planning responses bypassed the parent failure budget.** The
   empty-plan branch incremented `planning_failures`, while invalid JSON or an
   inconsistent examined-ID envelope raised before that branch. Repeated such
   replies could revisit the same parent without reaching the intended budget.
   The workflow now counts dispatched state-planning failures after exception
   handling, without double-counting the empty-plan branch. The added fixture
   requests a canonical case, two malformed planning replies, and a fresh case;
   it asserts failure counts, absence of patch dispatch, and parent reselection.
   This is a bounded scheduling policy, not a statement that the parent's goals
   are unreachable. See `workflow.py:913` and
   `StatePlanTests.test_two_malformed_planning_replies_exhaust_parent_budget_then_generate_fresh`.

2. **A framework namespace could be mistaken for a Tensor receiver in static
   hints.** With configured `Tensor.add`, the expression
   `probe.call(torch.add, x, x)` passed a receiver guard that excluded `torch.`
   prefixes but missed the exact name `torch`. The guard now excludes the
   namespace root and a negative control checks the mismatch. Runtime already
   rejects the wrong configured target, so the issue concerned misleading repair
   guidance rather than falsely certified target execution. See
   `source_operand_bindings` and
   `SourceOperandGuidanceTests.test_bound_receiver_unbound_and_keyword_paths_match_runtime_convention`.

3. **Aliased callable factories could hide an earlier runtime boundary from
   source ordinals.** A source assignment such as `factory = torch.compile`,
   followed by calling the alias and then a direct `probe.call`, escaped the
   original scan for direct factory calls. A compiled internal target can have
   runtime boundaries that do not correspond to that direct probe ordinal.
   The cross-review led to conservative unresolved hints for any recognized
   compiler/capture symbol reference, including local and top-level import
   aliases. Rebinding the `probe` parameter now also leaves hints unresolved.
   The backend reviewer implemented these controls and added
   `SourceOperandGuidanceTests.test_factory_symbol_aliases_leave_call_ordinals_unresolved`
   and `test_probe_rebinding_leaves_source_bindings_unresolved`. These later
   controls require a fresh validation snapshot; the first snapshot was not
   silently updated.

4. **A compatibility gate exposed overly broad source-certificate rejection.**
   Root's frozen fixed-fixture gate exposed rejection of explicitly tracked
   argument containers and native complex constants. The initial blanket ban on
   argument expansion suppressed otherwise reviewed programs. The correction is
   generic: normalize known sequence/result-tuple positional expansions and
   string-key owned-dictionary keyword expansions, and admit native complex
   constants as primitive values. Unknown or callback-bearing arguments,
   duplicate keywords, non-string keyword keys and expanded helper writes remain
   uncertified. No benchmark API, input recipe, expected value or tolerance was
   added. The source review checked the correction at
   `backends/pytorch.py:546` through the native receiver checks, and inspected
   `CertificateBindingTests.test_known_container_expansions_and_complex_literals_are_certified`
   and `test_unknown_or_unsafe_expanded_arguments_remain_uncertified`. Their
   presence and the source review are not a claim that a runtime gate passed.

The protocol review found no additional blocking counterexample in the inspected
normal and malformed replies. Legacy replies retain provisional planning status.
A structured unresolved reply names its examined goals and blockers; malformed
proposals do not consume the entire catalogue. Compatible cross-family
conjunctions remain possible, but runtime must establish every obligation.

## Certificate and oracle separation

`PyTorchBackend.explain_certificate` tracks reviewed binding kinds through local
assignments and the build context. The review checked that a method name alone
does not establish a Tensor receiver, opaque target returns do not acquire
arbitrary Tensor methods, imported/interface bindings cannot be overwritten, and
container aliases do not conceal a certified write or cycle. Explicit helper
write-capable `out`/`inplace` arguments, unknown argument expansions, unreviewed writes,
arbitrary helper bodies, classes and representation-sensitive metadata reads
(such as stride/layout/contiguity) are rejected from this source subset. Reviewed
shape, dtype and device bindings are handled separately; this is not a blanket
ban on every tensor attribute.

The normalized argument review found that duplicate keys are rejected before
admission, all normalized argument values still receive binding checks, helper
`out`/`inplace` checks run after expansion, and unbound Tensor helpers still
require a reviewed Tensor first positional value. Target argument effects remain
subject to their runtime schemas and separate MR conditions. A known dictionary
containing `out=None` remains conservatively uncertified because the expanded
binding does not retain the literal origin; explicit `out=None` is supported.
This is a support limitation, not an effect bypass.

No concrete newly admitted helper/effect counterexample was found in the reviewed
paths. This limited negative finding is not a completeness guarantee. The source
certificate is only one input to conditional MR applicability. The target still
requires a reviewed logical-value/effect class, appropriate numerical-domain
checks, valid executions, preserved coordinates, equal required logical inputs
and initial state, and aligned checkpoints. A metadata target may execute and
receive a certificate for its surrounding source while remaining inapplicable
to the value-representation MR. Its differing metadata result is not thereby a
framework defect. The existing semantic and numerical gates must remain
separate; successful execution or a source certificate cannot replace them.

Both planning and generation guidance now explicitly treats an unknown parent
certificate as missing MR proof, not an invalid program. Selected tensor/training
states and path obligations must be preserved; deleting a difficult state to
obtain a certificate is not an acceptable substitute. Legal coverage-only cases
remain useful and have correctness unassessed.

## Remaining limits and interpretation

- Static source hints are bounded guidance, labelled separately from runtime
  evidence. Dynamic/helper/callable-factory boundaries remain unresolved; last
  assignment hints do not prove an edit survives later assignments or attains a
  predicate. This work does not add general symbolic execution or new call-site
  instrumentation.
- Family balancing allocates requests among available goal families. It does not
  guarantee equal accepted proposals, equal execution cost, equal attainable
  states, or improved discovery. The phase-008 displacement analysis motivates
  the change but does not establish its causal benefit.
- Four unary and four joint units bound the planning context, with explicit
  total/shown/omitted counts. Full recorded coverage remains intact. Neither
  those prompt counts nor the bounded first-four-operand/edge-axis measurement
  slice is an exhaustive feasible-state denominator.
- Stopping a repair without new attained obligations can abandon a reachable
  state that needs several coordinated edits. It is an explicit resource policy;
  it does not certify impossibility. The comparison uses the same predicate
  identities, and suspected correctness failures are not repaired away.
- The reviewed source subset still excludes substantial training histories and
  arbitrary module/control-flow effects. A missing certificate and unavailable
  MR are distinct from invalid execution. Numerical stability and independent
  defect corroboration remain necessary.
- The local literature synthesis motivates runtime-evidenced exploration of
  memory representation together with training transitions and conditional MRs.
  It does not support a firstness claim for memory-layout testing, nor does this
  source review establish novelty or experimental superiority. See
  `literature/synthesis.md` for the reviewed positioning and its limits.

## Final source and workspace review before the held-out campaign

The final read-only review checked the latest family-selection integration,
planning status handling, request-specific schema and generator binding scope.
No new blocking source issue was found in those inspected paths. This does not
predict the outcome of the running development comparison or the held-out
campaign, and it does not authorize tuning on held-out outcomes.

- Family selection uses only families present in the remaining catalogue,
  seeded inverse request counts per API, and recorded selection probabilities.
  Cached plans retain their exact IDs; selected and explicitly unresolved IDs
  are handled separately from the rest of the catalogue. Request balancing is
  not a guarantee of equal successful executions. Feedback transfers at the
  predicate-family level while keeping each exact parent/goal identity separate.
- Empty/unresolved planning replies now remain `planning_unresolved`; malformed
  or incompatible replies are `planning_rejected`, not invalid Python programs.
  Only explicit examined IDs consume the corresponding planning candidates.
  Two unsuccessful dispatched planning calls can exhaust a parent budget without
  claiming its remaining goals are unreachable.
- `planning_schema(catalog)` deep-copies the global template, constrains both ID
  arrays to exact catalogue IDs, and uses disjoint planned/unresolved object
  branches. An empty catalogue permits empty arrays only. The unchanged
  validator still owns exact union/uniqueness checks, family membership,
  contradictory predicates and required edit regions. The schema does not prove
  semantic feasibility, and legacy replies remain provisionally valid.
- Both planning and generation text scope unknown operand-binding hints to
  predicates that actually select operand paths. Unscoped training counts and
  context predicates do not require tensor bindings; their legal transitions
  and runtime evidence remain required. Factory/helper ambiguity remains
  unresolved rather than inferred from source ordinals.
- Duplicate repair payloads/ASTs are stopped before another worker execution;
  predicate-progress checks still compare the same obligations. Full prior
  artifacts remain available, and these resource decisions do not convert an
  unknown execution into a correctness result.

The source/test inventory comparison against
`result/general-state-research/baseline-v8/code` found exactly these **13 changed
files**, with no source/test additions or deletions:

| File | Task change |
| --- | --- |
| `src/mirage/artifacts/report.py` | Planning-family/status and applicable-goal/cost reporting |
| `src/mirage/backends/pytorch.py` | Binding/effect-aware source-certificate diagnostics and generic compatibility corrections |
| `src/mirage/generator/http_provider.py` | Request-specific planning schema |
| `src/mirage/generator/program.py` | Bounded source binding/repair guidance and predicate-scoped prompts |
| `src/mirage/generator/state_plan.py` | Family request protocol, structured statuses, schema and balanced bounded context |
| `src/mirage/runtime/api_metadata.py` | Public Torch lazy-submodule resolution with preserved dependency errors |
| `src/mirage/sampler/policy.py` | Transfer feedback across exact IDs in the same predicate family |
| `src/mirage/workflow.py` | Family planning, explicit status/budget accounting and bounded repair handling |
| `tests/test_adaptive.py` | Sampler/repair/report integration controls |
| `tests/test_http_provider.py` | Planning schema payload controls |
| `tests/test_program.py` | Source binding, lazy resolution and prompt-scope controls |
| `tests/test_state_plan.py` | Family/protocol/schema-scope and workflow planning controls |
| `tests/test_stateful.py` | Source-certificate binding/effect and compatibility controls |

All **97 files** in the frozen baseline inventory still matched their archived
hashes. The current staged binary diff was byte-identical to
`baseline-v8/preexisting-staged.patch`; both SHA-256 values were
`469fefe5ce4ed24a6007e627d90c7f13da4b442600abfa7fca8f2637436865d0`.
Thus the user's staged work was not replaced by this task. Existing phase-007/008
contracts, stateful/numerical inspectors, stateful runtime and worker, corpus,
and their capability/corpus/numerical/relation tests were additionally checked
as byte-identical to the task baseline. Within changed files, prior top-level production
function and class-method definition names remained present. The one intentionally
replaced test was the old mandatory relation-priority test; its replacement
checks family rounds and explicit selection constraints.

These inventory and AST-name checks provide evidence against accidental broad
replacement; they do not prove behavioral equivalence of edited files. This
review made no production edits and reran no Torch/model tests. Root's executed
Slurm checks and immutable snapshot records remain the validation authority.

## Separate postformal review: local-VJP source effects

This section was added after the frozen `final-01` campaign. It does not replace
the preceding pre-campaign source inventory, hashes or review history. Formal
metrics remain zero autonomous suspected failures in both arms. The motivating
saved-evidence miss and independent analytic/reduced-experiment corroboration are
recorded in [final-slogdet-analysis.md](final-slogdet-analysis.md); that manual
posthoc evidence is not a new upstream bug or autonomous-discovery result.

The independently reviewed patch is limited to six files:
`backends/pytorch.py`, `runtime/stateful.py`, `inspector/oracles.py`, and
`test_stateful.py`, `test_oracle_routing.py`, `test_numerical_gradients.py` in the
existing source/test directories. The numerical-gradient implementation itself
is unchanged. It introduces the distinct
`explicit_straight_line_local_vjp_v1` source-effect certificate and records it as
`numerical_source_certificate`.

The additional source admission is narrowly scoped to native Tensor
`requires_grad_` in `run` with an explicitly written AST literal `True`. Bound
methods require the existing proven Tensor receiver; unbound methods additionally
require the explicit first positional receiver to have that proven kind. Direct
positional and `requires_grad=True` forms are supported. Omitted, false, numeric,
dynamic, expanded, duplicated or wrongly bound flags do not gain admission.
Namespace aliases retain their binding proofs; namespace/method rebinding,
custom receivers, hooks, unreviewed overrides/effects and helper mutations remain
outside the existing whole-source reviewed subset. This is an oracle-specific
extension of that subset, not removal of the whole-source effect check.

Runtime stores the new tag separately. Only `audit_execution` consumes the exact
new tag as an alternative source prerequisite. The original
`program_certificate` and representation-MR deterministic condition are
unchanged: the newly admitted run-time setup still yields old certificate
`None`, leaving its representation MR inapplicable. Completed valid execution,
actual direct backward evidence, unchanged RNG, eager/read-only native target,
supported storage coordinates, reconstructed baseline output/VJP equality and
unchanged numerical stability conditions remain independent requirements. No
replay success, deterministic flag or new certificate establishes full training
history equivalence; numerical agreement still does not manufacture an MR Pass.

Cross-author source review found no blocking newly admitted binding/effect case.
The eight added test methods cover native/literal positive forms, dynamic and
expanded flag rejection, namespace/hook/custom-receiver rejection, MR separation,
exact-tag routing/legacy compatibility, correct unary/binary VJPs, and unchanged
read-only/compiled/baseline-mismatch abstention. Six-file `git diff --check`
passed. These are static review findings and descriptions of test intent.
**At the time of this appendix, root's frozen Slurm validation was running; no
runtime pass is claimed here.** Its saved snapshot, logs and subsequently
reported results remain authoritative.

A second cross-author source review also checked the separate manual analytical
artifacts. It found the Fraction inverse-transpose and observed logabsdet
cotangent use consistent; the real nonsingular sign term has zero derivative.
The reported physical pullback applies the negative-bit sign once and is labeled
inferred. In the reduced complex/conjugate/imaginary construction, the mapping
from the original real base to the logical operand is identity, so comparing the
base gradient with the logical analytic derivative is appropriate. This review
performed no Torch execution or additional computation and does not turn the
manual known-mechanism evidence into a formal discovery.

Subsequent root-executed Slurm validation completed:130relatedtests,10/10fixedfixtures,andtheoriginal-source/configposthocreplayallpassed;MRremainsinapplicablewhilethelocalnumericaloracleflagsfollowup.Thislaterexecutionevidenceissavedin[validation.md](validation.md)anddoesnotchangethefrozenformal0-suspectoutcome.
