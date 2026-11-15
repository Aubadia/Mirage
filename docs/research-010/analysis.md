# Why observed coverage did not establish stronger training-state testing

Roadmap009 increased observed joint tensor units, but its final paired study did
not establish more actual memory × optimizer transition cases. Those results are
preserved in `../research-009/report.md`. This phase addresses that particular
measurement and execution gap; it does not reinterpret those original results.

## The missing connection

The previous search could observe target operands, attributable backward events,
optimizer snapshots and generic step actions in different places. These facts
were not a proof that a selected target produced a particular registered
parameter's gradient, that the optimizer consumed that gradient in the requested
memory representation, or that an applicable relation compared the resulting
update. More ordinary operand units or backward counts could therefore improve a
proxy while the desired training-state capability remained absent.

The new chain makes the connection explicit: direct target operand identity,
actual target Node contribution, the same completed backward invocation,
post-accumulate event on the registered leaf, unchanged gradient identity/version/
logical values at the pre-update boundary, and a completed native optimizer step.
The source owns backward and step. The harness changes only the consumed
gradient's representation after proving its provenance.

```mermaid
flowchart LR
    A[Registered target leaf] --> B[Actual target Node]
    B --> C[Completed backward and post-accumulate event]
    C --> D[Unchanged gradient at pre-update]
    D --> E[Exact-value layout action]
    E --> F[Completed native SGD]
    F --> G[Aligned parameters and momentum slots]
    G --> H[Conditional optimizer MR]
```

## General solutions implemented in this slice

1. **Executable configurations are part of a test case.** A verified source that
   already performs the required update can be reused with a new gradient-layout
   configuration. Requiring an unrelated Python edit for every transition wastes
   generation and invites loss of the valid training history. Reuse retains a
   checked source manifest, exact companion obligations and runtime revalidation;
   it never promotes parent evidence into child attainment.
2. **Conjunctions must survive planning.** A gradient-layout goal always carries
   its same-parameter completed-update obligation, including when the parent
   already satisfied that obligation. Otherwise an edit can obtain the layout by
   deleting the optimizer update it was supposed to test.
3. **Observation and proof need separate types and versions.** The new
   `optimizer-chain-1` units are separate from `state-semantic-2`. Reserved observer
   checkpoints and optimizer facts are excluded from the old unit vocabulary.
   Missing measurements, unknown attribution and verified zero counts remain
   distinct. Corpus admission and feedback preserve new verified chain units.
4. **Correctness is conditional at the relevant boundary.** The new MR compares
   post-update parameters and optimizer slots only after equal initial/pre-update
   logical state, native dispatch, source effects, gradient provenance and the
   permitted representation difference are established. Parameter change is an
   observation, not a validity requirement. Genuine zero VJPs and rounded-to-zero
   updates remain testable.
5. **Producer/consumer tests must use actual records.** Native descriptor paths
   are tuples while JSON paths are lists. Initial metadata-only tests missed a
   comparison that rejected all native chains. Typed path normalization and
   native/serialized controls fix this integration gap without relaxing proof.

These mechanisms do not contain target API names or the known slogdet trigger.
They apply to any supported target for which the runtime can establish the same
direct parameter/backward/update chain.

## Development01: execution evidence and the next bottleneck

The completed development study used `sin` and `add`, seed0, and 90 seconds per
API/arm. Its [comparison](../../result/optimizer-gradient-chain/779954/development-01-report/comparison.md)
and [mechanism audit](../../result/optimizer-gradient-chain/779954/development-01-mechanisms/mechanisms.md)
preserve the original records and hashes. All four searches completed one fresh,
source-valid case; the comparison integrity gates passed.

| Measurement | Reference | Candidate |
|---|---:|---:|
| Attempts / source-valid cases | 2 / 2 | 2 / 2 |
| Verified source chain cases / units | 0 / 0 | 1 / 1 |
| Optimizer MR applicable / Pass / suspected failure | 0 / 0 / 0 | 1 / 1 / 0 |
| Planning requests / selected gradient-update goals | 0 / 0 | 0 / 0 |
| Explicit configuration reuse records | Unmeasured | None observed |
| Numerical member-audit statuses | 4 agreement | 4 unsupported |

The candidate's verified chain and optimizer MR Pass came from its fresh `sin`
case. This is evidence that one generated program exercised the new chain, not
evidence that planning or configuration reuse improved exploration. Every search
stopped after its first case: 43.40–52.72 seconds remained, below the recorded
whole-round dispatch reserves of 60.37–74.40 seconds. No planning request was
sent. These results therefore cannot establish LLM refusal, failed goal supply,
or the success of a hypothetical extra attempt with a smaller reserve.

Both candidate sources used `momentum=0.0`. The `sin` MR Pass therefore does not
establish autonomous coverage of a nonempty momentum slot. Authored native
controls with positive momentum exercise slot comparison support; that validated
capability is separate from what these generated cases actually reached.

The candidate `add` source executed legally, but its optimizer source certificate
was unknown. A [source-only counterfactual audit](../../result/optimizer-gradient-chain/779954/development-01-certificate-audit.json)
against the exact saved engine isolated three authored forms outside the reviewed
fragment: indexed tensor initialization writes, a loop enabling gradients, and
`params.values()` as the SGD parameter collection. Replacing them with a tensor
literal, explicit per-leaf `requires_grad_(True)` statements, and an explicit
`[a, b]` parameter list obtained the optimizer certificate statically. None of
these rewritten programs was executed. This is source-subset admission, not
runtime equivalence, proven reachability, MR applicability or a search discovery.

All four candidate numerical audit records were unsupported because source
effects were outside the existing execution certificate; the reference's four
records reported agreement. The denominator is source/follow-up member audit
records, not unique programs or finite-difference calls. Even the statically
rewritten `add` source retained no local numerical certificate. An optimizer MR
Pass does not compensate for this missing independent derivative assessment.

## Development-driven changes adopted before final evaluation

Protocol revision3, recorded in the experiment ledger at `ecd3ff7` before final
searches, adopts the following bundled changes. Their bounded final effect and tradeoffs are reported below.

1. **Offer known configuration actions directly.** A bounded transition-library
   branch constructs typed gradient-layout plans only for a verified, reusable
   parent and an exact same-parameter completed-update companion. It adds one
   candidate branch per parent to avoid weighting a parent by its goal count.
   Child execution must still revalidate every obligation; cached proposals are
   not cached attainment.
2. **Separate examined goals from selected goals.** An LLM's examined or
   unresolved goal does not mean the corresponding library action was executed.
   Only actual selection consumes a goal across proposal routes. This preserves
   legal known actions without relabeling unresolved planning as success.
3. **Estimate known execution cost from its own parent.** Only this configuration
   branch can use the matching valid parent's completed worker time, with the
   existing reserve rule `max(10 seconds, 1.5 × worker seconds + 5 seconds)`.
   Missing or invalid measurements fall back to the whole-round reserve.
   Bootstrap and fresh-generation quotas apply before affordability filtering;
   general generation, unknown work and repair retain their existing reserve.
   The sampler formula and remaining-time execution deadline are unchanged.
4. **Explain the existing certificate fragment to the generator.** Generic
   guidance requests straight-line literal/pure initialization, explicit gradient
   setup and an explicit SGD parameter list. It retains unsupported legal cases
   as coverage and does not expand source-effect or numerical-oracle admission.

The final comparison evaluates this bundle, not the isolated causal contribution
of each component. The locked API panel, seed count, 120-second final budgets and
shared runtime/oracle remain unchanged. The completed development evidence below
does not substitute for the final comparison or establish a new framework defect.

## Development02: the generated update chain and configured transition executed

The second development study is complete. Its [comparison](../../result/optimizer-gradient-chain/779954/development-02-report/comparison.md)
and [mechanism audit](../../result/optimizer-gradient-chain/779954/development-02-mechanisms/mechanisms.md)
passed their integrity gates with no missing inputs. The paired study retained
`sin`/`add`, seed0 and 90 seconds per API/arm.

| Measurement | Reference | Candidate |
|---|---:|---:|
| Source-valid cases | 3 | 4 |
| Verified source chain cases / units | 0 / 0 | 4 / 4 |
| Planned gradient-layout + completed-update attained / attempted | 0 / 0 | 2 / 2 |
| Explicit configuration reuse records | Unmeasured | 2 |
| Optimizer MR applicable / Pass / suspected failure | 0 / 0 / 0 | 4 / 4 / 0 |
| Numerical member-audit statuses | 4 agreement, 2 unknown | 4 unsupported, 4 unknown |

Each candidate API produced one fresh source chain and one library-selected
configuration case. Both configured children attained their own gradient-layout
and completed-update obligations after execution, yielding two explicit
memory-representation × optimizer-update tests. The two plans came from the
bounded transition library; the candidate sent no LLM planning requests. The
reference made one `training_state` planning request for `sin`, whose catalog
contained no optimizer goal. Neither observation supports a claim that an LLM
accepted or refused the new optimizer goals. Library inventory entries are
metadata proposals, not additional reached states.

At the two successful library dispatches, the saved remaining time was above the
parent-worker reserve but below the old whole-round reserve:

| API | Saved remaining seconds | Parent-worker reserve | Old reserve on actual preceding rows |
|---|---:|---:|---:|
| `sin` | 55.23 | 49.31 | 56.52 |
| `add` | 54.95 | 49.30 | 56.91 |

Thus the old reserve condition would reject both dispatches **at those saved
selection instants with the observed history fixed**. This is a measured control
condition, not a replay of the old scheduler: a complete alternative run could
have different timing, parents, proposals and outcomes. The paired results
describe the bundled system, not separate causal effects of the library, prompt
or reserve change, and not a causal development01-to-development02 gain.

The candidate's four unsupported audit records remained outside the existing
source-effects certificate. Its four unknown records and the reference's two
unknown records were not selected by the bounded numerical-audit schedule. These
are member-record counts; unknown is not a numerical agreement. All four
candidate optimizer MR cases passed with nonzero recorded parameter updates,
but both generated sources used `momentum=0.0` and their configured children
reused that source. Autonomous nonempty momentum-slot coverage is therefore
still unestablished. The native positive-momentum controls support that testing
capability separately.

The current affected regression snapshot passed 172 tests. The final per-test log audit found 365 distinct related test IDs across partitioned
snapshots: 360 have pass evidence and five CUDA-only controls were skipped. This
corrects the earlier shorthand based on runner totals; it is not a single-snapshot
full-suite result, and repeated executions are not added again.
`controls-02` passed 13 native controls per arm; `fixtures-03` passed all ten
known cases and independent control checks per arm. Earlier failures remain
preserved. Formal source was frozen before launch at experiment ledger `cc513b5`
under protocol revision3. The final seed0 study completed all 12 searches and common scoring without
post-launch implementation changes or outcome reruns. Owned resources were
released after the final evidence audit; see the completed results below.

## Position within Mirage

This preserves the intended memory-representation × training-state testing
direction: identical logical gradients are consumed through different layouts at
an actual training transition, with measured reachability and conditional MR
evidence. It is not arbitrary gradient-value injection. This implementation tests
optimizer consumption after backward; it does not establish that backward wrote
into a transformed gradient buffer.

This is a bounded implementation of the research direction, not a claim of being
the first layout tester or of demonstrating cross-framework generality. The first
reviewed domain is exact native SGD, CPU float64 dense tensors, cold optimizer
slots, one step and directly bound trainable leaves. Warm state, multiple steps,
Adam, AMP, compilation and distributed execution require additional provenance
and MR domains. They remain unknown or coverage-only here.

## Experiment interpretation

The paired reference retains phase009 generation/planning while sharing the new
runtime, inspector, contracts, MR and numerical oracle with the candidate. Both
arms use fresh corpora, seed0, the same service and fixed per-API budgets. The
common scorer preserves original member validity and does not rerun programs or
rescue missing goals. Results are descriptive for one seed. Authored controls and
known fixtures establish implementation behavior; they are not autonomous bug
discoveries. Development results above do not substitute for the separately reported final
comparison.

Before the first model call, protocol revision2 symmetrically requires backward
for both arms on the six differentiable APIs. The mixed forward/backward policy
could otherwise start with forward-only cases and inherit those obligations in
short runs. This is an optimizer-capability panel, not a claim about the whole API
pool. Seed count, targets and time budgets were unchanged by the amendment.

The [numerical-audit boundary](numerical-audit-boundary.md) is a separate limitation:
the optimizer MR does not establish independent VJP correctness, and its new source
certificate deliberately does not widen the existing numerical certificate.

## Final single-seed result and unresolved causes

The [completed comparison](../../result/optimizer-gradient-chain/779954/final-seed0-report/comparison.md)
passed all29 integrity gates. Reference13 attempts yielded10 valid cases and0
verified optimizer chains; candidate18 attempts yielded15 valid cases,15 chains
and17 versioned chain units. The candidate reached10/10 selected layout+update
conjunctions (five strided and five offset), with15 optimizer MR Passes and no
suspects across any oracle. It used five fresh programs and ten configuration
reuses across five APIs. Both arms had0 valid `linear` cases; API attainment was
5/6, not6/6. Missing reference reuse/library fields remain unmeasured.

This is a redistribution of exploration, not a uniform coverage improvement.
The unchanged old semantic observer measured40→17 units (unary16→8, joint24→9)
and10→5 transition units. The reference's five ordinary training-state plans
reached repeated-backward goals, whereas the candidate spent those opportunities
on consumed-gradient layouts. Five of ten successful library dispatches would
fail the old reserve condition at their saved histories; no complete old-policy
counterfactual or component attribution was executed.

Numerical member records were reference8 agreement/2 unsupported/11 unknown and
candidate0 agreement/10 unsupported/22 unknown. New source-owned updates fall
outside old numerical source admission. The MR can also miss a shared wrong VJP
or a post-parameter difference within its existing numerical tolerance. A
post-hoc Slurm read of the53 saved raw members confirmed native/common coverage
identity and unchanged hashes; none of the15 valid candidate sources or15
follow-ups had nonempty slots or positive momentum. This was saved-evidence
inspection, not additional searches or derivative validation.

The [linear audit](final-linear-limitation.md) identifies independent failures:
composite/unobserved input edges outside direct Node attribution, unknown static
native return typing at `out.sum`, duplicate repairs, and one reference literal
shape error. Actual generic backward execution is not missing in the completed
sources; the required target proof is missing. A bounded composite-call proof,
separate return-type/effect evidence and exact static shape diagnostics are
possible general extensions; accepting output hooks or leaf.grad alone would
not solve the attribution problem soundly.

The [next-step plan](next-steps.md) now prioritizes call-local numerical admission,
composite attribution and explicit budgets across separate coverage/oracle
objectives before broader warm/multi-step state domains. These are future work,
not changes applied after seeing final outcomes. The [novelty review](novelty-positioning.md)
also found a direct PyTorch sparse/dense gradient-update comparison precedent.
The basic MR is not a novelty claim; the testable research hypothesis is the
incremental value of integrating predicates, attribution, reachability,
conditional MRs and coverage-guided exploration across APIs.
