# Final slogdet miss: generation reached the trigger, oracle admission stopped it

The frozen `final-01` experiment reported **zero autonomous suspected failures in both arms**. That result remains unchanged. Manual examination subsequently identified a saved candidate with an incorrect backward result and corroborated it using an independent algebraic reference and a new reduced reproduction. This is a diagnostic of a missed detection and a locally known mechanism, not a new autonomous discovery or a new upstream bug.

## Frozen campaign evidence

| Arm | Attempts | Runtime-valid source cases | Required target backward reached among valid cases | Applicable MR comparisons | Autonomous suspected failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 19 | 11 | 5 | 6 | 0 |
| Candidate | 14 | 11 | 8 | 8 | 0 |

Sources are the saved [reference summary](../../result/general-state-research/778971/final-01/reference/204407/summary.json) and [candidate summary](../../result/general-state-research/778971/final-01/candidate/204407/summary.json). These single-seed observations do not establish comparative discovery effectiveness.

Candidate `attempt-0008` generated `A=[[1,2],[3,4]]` and applied the negative representation to the actual slogdet operand. Both members executed the required target backward and met their state predicates. The actual followup had strides `[4,2]`, offset `1`, and negative bit true, while preserving the logical matrix. The saved direct-node evidence records sign and logabsdet output cotangents of `1` each. Thus this case did reach the relevant memory × backward combination.

Its original MR verdict was **inapplicable**, with `deterministic` the only missing condition. The source certificate diagnostic identifies `A.requires_grad_(True)` in `run`, line 14, as `unsupported_call`. The frozen analyzer admits this operation only in `build` ([backend source](../../src/mirage/backends/pytorch.py), original lines 623–627). Both local numerical audits were **unsupported**, because [oracle admission](../../src/mirage/inspector/oracles.py) first requires the same whole-source certificate. Attempt 8 was selected by the every-four-attempt audit schedule ([worker](../../src/mirage/runtime/worker.py), lines 110–115). This particular miss is an admission failure, not a lack of backward, lack of trigger generation, or a missed scheduling slot. The `deterministic` label does not establish that this program was actually nondeterministic.

Reference `attempt-0008` also created a negative followup, but on its second call, `B=diag(5,1)`. Saved source and followup VJPs agree with the analytic gradient. The nonsymmetric `A` on its first call remained ordinary. A diagonal inverse equals its transpose, so this particular operand cannot distinguish the observed off-diagonal transposition pattern. Reference `attempt-0018` instead used an expanded singular matrix and produced nonfinite values; its followup was invalid. The nonsingular analytic derivative is explicitly unknown there. These are distinct limitations, not evidence of additional confirmed defects.

Across the 22 source/followup audit slots belonging to the 11 valid source cases per arm, reference had 14 unscheduled, 4 unresolved forward-only slots, 2 agreements, and 2 certificate-unsupported slots; candidate had 14 unscheduled, 2 unresolved forward-only slots, 2 agreements, and 4 certificate-unsupported slots. This restricted denominator differs from the saved aggregate `search_metrics`, which also include records outside these valid-source pairs. Scheduling is a broader limitation but did not cause candidate 8's miss.

## Independent posthoc evidence

The [bounded extraction](../../result/general-state-research/778971/final-slogdet-saved-01/evidence.json) loaded only four exact trusted artifact directories under Slurm, without executing their generated source. Original program/summary/source.pt/followup.pt hashes were checked before and after. Tensor snapshots contain logical values; original negative-bit evidence comes from the recorded call metadata, not the copied snapshot's own bit.

The [independent review](../../result/general-state-research/778971/final-slogdet-independent-01/review.json) uses exact `Fraction` inverse-transpose on the recorded binary float values and actual output cotangents. For a real nonsingular matrix, sign is locally constant and the logabsdet gradient is `A^{-T}`. For candidate 8, determinant is exactly `-2` and infinity-condition is `21`:

| Evidence | Flattened target VJP | Result |
| --- | --- | --- |
| Exact analytic reference | `[-2, 1.5, 1, -0.5]` | Reference |
| Original saved source | `[-2, 1.5, 1, -0.5]` | Agreement |
| Original saved negative followup | `[-2, 1, 1.5, -0.5]` | Violation; normalized discrepancy `0.25` |

The predeclared rejection bound was `0.0025634765625`; no finite-difference output supplies this reference. Physical signed-coordinate pullbacks are reported separately and labeled as chain-rule derivations, not separately observed physical gradients. The old phase-008 source agrees and its negative followup violates the same independent reference. The singular reference-18 pair remains unknown.

A [new standalone reduced reproduction](../../result/general-state-research/778971/final-slogdet-repro-01/repro.json), using PyTorch `2.15.0+cu132` on CPU, exactly reproduced both saved VJPs. Ordinary, `resolve_neg`, and contiguous controls agreed with the analytic gradient in both direct operand VJP and original-base gradient; the negative view violated both. All four forward outputs were identical. The script imports no Mirage code, performs no finite differences, and does not replay the original training history. It is manually reduced, not proven globally minimal. The combined evidence supports **independent corroboration of the locally known negative-view slogdet backward mechanism**.

## Provenance and consequence

| Original source | SHA256 |
| --- | --- |
| Candidate final attempt 8 | `863e12b5636ca43dc21715acb956fb47101e4f6398ad391419da1a93d5a5ef63` |
| Reference final attempt 8 | `68a4a1e410e232eb90f1d4167e79b3a5b1fcf1d73cbbc31a65b0018cf90a81bc` |
| Reference final attempt 18 | `9d5190a98ffed043947b82f9e80b82aa41f6173ef43261a8ccf725557015568b` |
| Phase-008 baseline attempt 8 | `64734dcbdb6c1d5defc05081fcc72abec27e3b4f16133ee1b223f6a9571f8c89` |

Extraction SHA256 is `f9aae2d426b48ec30261d3de4d827ca35fcf62f865b0becd92c6fa3a88f9e1ac`; review/repro outputs record their own script, input, environment, and Slurm identifiers. Complete original artifact hashes are in the review JSON.

The generic improvement motivated by this miss is to distinguish proof needed for a **local reconstructed target VJP** from proof needed for a **whole-program MR**. A bounded native gradient-enablement transition may be admissible for the former without proving whole-history equivalence. Any expansion still needs native binding, literal argument, effect, hook/override, alias, direct-gradient, reconstruction, and numerical-domain checks; successful replay alone cannot certify it. A separately authorized postformal implementation and validation must remain outside the frozen comparison. No numerical thresholds, formal inputs, campaign summaries, or formal discovery counts were changed by this diagnosis.

## Postformal correction and same-source replay

The separately authorized correction now records
`numerical_source_certificate=explicit_straight_line_local_vjp_v1`. It reuses the
strict source binding/effect analysis and only additionally admits a proven
native Tensor receiver with explicit literal `True` in a `run`
`requires_grad_` call. Dynamic or expanded flags, disabling gradients, custom
receivers, namespace replacement, hooks and backend-effect calls remain outside
this certificate. The old `program_certificate` is still absent for candidate 8;
the representation MR and numerical algorithm/thresholds are unchanged.

Under the separately frozen `iteration-qTg5zq/code` engine, the
[postformal validation](../../result/general-state-research/778971/postformal-validation-02/summary.json)
passed 130 tests and retained 10/10 supplied fixed-fixture detections. It then
reran the saved candidate-8 worker request with the same source, configuration
and audit schedule, replacing only its engine path. The
[replay summary](../../result/general-state-research/778971/postformal-validation-02/exact-posthoc-replay/summary.json)
records:

| Evidence | Postformal result |
| --- | --- |
| Representation MR | `inapplicable`; `deterministic` still missing |
| Source local VJP audit | `agreement`, baseline verified |
| Negative followup local VJP audit | `violation`, baseline verified; both off-diagonal directions violate while the diagonal directions agree |
| Overall verdict | `suspected_failure`, attributed only to `followup` |

Both audits retained stable-step and one-sided-consistency checks. The
[replay checks](../../result/general-state-research/778971/postformal-validation-02/replay-checks.json)
all passed; original program/request/summary/tensor-record hashes and the new
frozen source/test snapshot remained unchanged. These provenance checks are not
a claim of byte-identical replay traces. No model inference or fresh search seed
was used.

This is a **manually selected posthoc recovered detection**, corroborated by the
independent analytic review and reduced reproduction above. It is not a formal
autonomous discovery or an estimated recall improvement. The frozen formal
comparison still reports zero suspects in both arms. The
[validation inventory](validation.md#postformal-local-vjp-correction--separate-version)
also retains the first postformal run's stale test-assertion failure and the
subsequent passing rerun.
