# Phase 009 validation inventory — Slurm 778971

This inventory was read from saved commands, exit-status files, test logs and
structured summaries. The original sections record **pre-final** validation in
Slurm allocation `778971`; the appended postformal section records a separately
versioned correction after the frozen campaign. No tests were rerun to write this document, and production code was
not changed. Every iteration retains its command and frozen code or engine
snapshot under `result/general-state-research/778971/`.

Test invocations overlap and use different development snapshots. **Do not add
the following counts as unique tests or present them as one final full-suite
run.** A successful process exit, fixture detection, structural control, static
certificate admission and an applicable MR result are different observations.
Elapsed values below are wrapper wall seconds from each iteration's
`status.json`; test-runner time is stated separately where useful. They are not
GPU utilization or a sum of allocated resource time.

## Related unit and integration test invocations

| Iteration | Scope from saved command | Executed result | Wrapper seconds |
| --- | --- | --- | ---: |
| [Wt8yHF command](../../result/general-state-research/778971/iteration-Wt8yHF/command.txt) | `regression-01`: ten modules listed below | 263 tests: 262 passed, one HTTP schema assertion failed; exit 1 | 323 |
| [JPhpOq log](../../result/general-state-research/778971/iteration-JPhpOq/stderr.log) | `test_program`, `test_state_plan`, `test_http_provider` after schema assertion and source-binding corrections | 74 passed; exit 0; runner 71.901 s | 76 |
| [xxs5ov log](../../result/general-state-research/778971/iteration-xxs5ov/stderr.log) | `test_stateful`, `test_numerical_gradients`, `test_relation_catalog` after generic known-container/complex-literal certificate correction | 134 passed; exit 0; runner 21.836 s | 26 |
| [lnW13k log](../../result/general-state-research/778971/iteration-lnW13k/stderr.log) | `test_program`, `test_state_plan`, `test_stateful`, `test_adaptive`, including public lazy API resolution, dtype aliases and planning-status handling | 190 passed; exit 0; runner 205.262 s | 209 |
| [pAdiNY log](../../result/general-state-research/778971/iteration-pAdiNY/stderr.log) | `test_state_plan`, `test_program`, `test_http_provider`, `test_adaptive`, including exact-ID/discriminated schemas, binding scope, encoding guidance and bounded repair | 114 passed; exit 0; runner 196.573 s | 200 |

The first regression's [structured summary](../../result/general-state-research/778971/regression-01/summary.json)
records module counts of capabilities 14, state-plan 27, program 37, stateful 88,
corpus 14, adaptive 25, oracle-routing 6, numerical-gradients 24,
relation-catalog 20, and HTTP-provider 8. The sole
[failure log](../../result/general-state-research/778971/regression-01/test_http_provider.py.log)
expected only `plans` in the response schema while the revised protocol requires
`plans` and `examined_goal_ids`. That assertion was corrected and rerun; the
original failure remains archived. The later schema changes have their own
114-test invocation rather than a retroactive pass assigned to regression-01.

## Fixed fixtures and structural observer controls

| Iteration / artifact | Scope and outcome | Exit / wrapper seconds |
| --- | --- | --- |
| [4cWnTO / fixtures-01](../../result/general-state-research/778971/fixtures-01/summary.json) | Ten supplied fixtures per arm; reference 4/10 detected, candidate 4/10. Compatibility gate failed because blanket argument-expansion rejection and missing complex-literal support withheld certificates. | 1 / 126 |
| [Ic30tP / fixtures-02](../../result/general-state-research/778971/fixtures-02/summary.json) | After generic certificate correction: reference 10/10, candidate 10/10; both fixture gates complete. | 0 / 112 |
| [ncHKYa / fixtures-03](../../result/general-state-research/778971/fixtures-03/summary.json) | Latest frozen engines: reference 10/10, candidate 10/10; both fixture gates complete. | 0 / 108 |
| [7iPQiL / controls-01](../../result/general-state-research/778971/controls-01/summary.json) | Both arms aborted before control execution: experimental driver attempted to create an output directory without its missing parents. This supplied no control evidence. | 1 / 8 |
| [1TtoMv / controls-02](../../result/general-state-research/778971/controls-02/summary.json) | Fresh rerun after driver directory fix: four structural cases per arm, all passed. | 0 / 37 |

For fixtures-03, inspect the per-arm
[reference](../../result/general-state-research/778971/fixtures-03/reference/fixed/summary.json)
and [candidate](../../result/general-state-research/778971/fixtures-03/candidate/fixed/summary.json)
summaries: `detected=10`, `total=10`, `fixture_gate_complete=true`, and
`controls_independently_audited=true`. The corresponding
[reference controls](../../result/general-state-research/778971/fixtures-03/reference/fixed/independent-controls.json)
and [candidate controls](../../result/general-state-research/778971/fixtures-03/candidate/fixed/independent-controls.json)
each contain ten corroborated fixed cases. Their broader `gate_complete` field
remains false; it is not interchangeable with `fixture_gate_complete`.
Both arms share backend SHA-256
`f0a09c2781c482a07085f92b1d682d239a03924a6ee553e88d048735daeb4163`
in the saved fixture engine inventories. The exact supplied inputs and existing
MR/numerical criteria were retained. These are known fixed controls, **not ten
autonomous discoveries, a false-positive estimate, or proof of search recall**.

Controls-02 covers `expanded_distinct`, `shared_disjoint`, `separate_clone`, and
`repeated_operand`. Both
[reference](../../result/general-state-research/778971/controls-02/reference/controls/summary.json)
and [candidate](../../result/general-state-research/778971/controls-02/candidate/controls/summary.json)
verify source validity, v2 unary/joint evidence, rejection of resurrected invalid
coverage, and unknown storage handling. The repeated-operand case additionally
checks that ambiguous operand-gradient attribution is not claimed. These are
observer/control assertions; their success does not give coverage-only cases a
correctness pass. The controls-02 driver elapsed time is 36.062 s; fixtures-03 is
107.512 s, separate from the rounded wrapper durations above.

## Saved-program certificate audits

These audits compare static certificate decisions on prior saved phase-008
programs. They execute no program replay, issue no new MR verdict and establish
no recovered defect. All use the archived original backend as the counterfactual.

| Iteration / artifact | Input and outcome | Exit / wrapper seconds |
| --- | --- | --- |
| [Uu7cSP / audit-01](../../result/general-state-research/778971/certificate-audit-01/audit.json) | Correct `paired-01` path; 445 programs examined. Revealed 20 removed certificates, including ten originally runtime-valid programs; the latter were diagnosed as missing native dtype alias support. | 0 / 3 |
| [uc5LFv / audit-02](../../result/general-state-research/778971/certificate-audit-02/audit.json) | Incorrect parent campaign path: `rows=[]`, `counts=[]`, **zero programs**. Invalid audit evidence despite process success. | 0 / 1 |
| [VCqNMO / audit-03](../../result/general-state-research/778971/certificate-audit-03/audit.json) | Corrected `paired-01` path with the final backend: 445 programs, 63 added / 164 kept / 218 still unknown / zero removed certificates. | 0 / 2 |

Audit-03's 445 records comprise 230 prior baseline programs and 215 prior new-arm
programs. Among the **226 originally runtime-valid** records, 19 gain certificates
(baseline 9, new 10), 103 retain them (58, 45), 104 remain unknown (65, 39), and
none loses a certificate. Of the 63 added certificates overall, 44 belonged to
originally invalid records; a static certificate does not make those executions
valid. The zero-program audit-02 is excluded from every substantive count.
Audit-03 supersedes audit-02 for the corrected path; it does not erase the failed
attempt or rerun the original cases.

## Fixed-request model checks and development runs

| Iteration / artifact | Scope and observed completion | Exit / wrapper seconds |
| --- | --- | --- |
| [WMXqzs / encoding-ablation-01](../../result/general-state-research/778971/encoding-ablation-01/summary.json) | Fifteen recorded requests, zero completed provider calls. Every record has `FileNotFoundError` for a missing provider artifact directory. No encoding effect can be inferred. | 1 / 1 |
| [sBEgQD / encoding-ablation-02](../../result/general-state-research/778971/encoding-ablation-02/summary.json) | Fifteen completed provider calls: fourteen paired source-encoding calls plus one separate planning-schema smoke. Three of four historical parse failures recover differentially; zero regressions among three controls. The exact-ID/schema smoke passes. | 0 / 22 |
| [I6VtLH / development-01](../../result/general-state-research/778971/development-01/summary.json) | Four seen APIs × two arms from fixture-02 engines, plus common scoring: nine completed steps. This is a development experiment, not a unit-test pass count. | 0 / 559 |
| [rZblUQ / development-02](../../result/general-state-research/778971/development-02/summary.json) | Same bounded development panel from fixture-03 engines, plus common scoring: nine completed steps. Search effectiveness is analyzed separately. | 0 / 568 |

Encoding-ablation-02's driver duration is 21.029 s and its recorded complete token
usage totals 97,957 tokens. Its preregistered adoption condition concerns Python
syntax/static contracts only: at least two of four differential recoveries, no
regressions in three controls, and all paired provider calls complete. The
condition was met, but the generated programs were **not executed** in this
ablation. The schema smoke checks service/schema/validator compatibility, not
state reachability or MR applicability. The failed first ablation cannot be
counted as 15 completed model calls.

Development process success likewise does not mean every generated case is
valid or every MR passes. Native and common-observer metrics, generation cost,
unknowns and failure funnels remain in their own experiment artifacts. No
partial final-campaign outcome is included here, and these development runs are
not independent held-out evidence.

## Postformal local-VJP correction — separate version

The frozen `final-01` engines and their zero autonomous suspected-failure counts
remain unchanged. After inspecting a saved miss, the authorized correction added
`explicit_straight_line_local_vjp_v1`: a separate source-effects certificate that
admits reviewed native Tensor `requires_grad_(True)` in `run`. The existing
representation-MR certificate, numerical algorithm, thresholds, rules and worker
were unchanged. This is a postformal implementation and replay, not another
search run or a corrected formal metric. Phase 009 now touches 17 unique
`src`/test files relative to its phase-008 baseline; the six-file correction
overlaps two files from the earlier 13-file scope.

| Iteration / artifact | Executed result | Exit / wrapper seconds |
| --- | --- | --- |
| [IXNFIw / postformal-validation-01](../../result/general-state-research/778971/postformal-validation-01/summary.json) | 130 tests: 129 passed, one stale prompt-suffix assertion failed in `test_stateful`; fixtures and replay did not run. | 1 / 29 |
| [qTg5zq / postformal-validation-02](../../result/general-state-research/778971/postformal-validation-02/summary.json) | 130 passed: 96 stateful, 8 oracle-routing, 26 numerical-gradient tests; then ten candidate fixtures and the saved-source replay completed. | 0 / 91 |

The [first failure log](../../result/general-state-research/778971/postformal-validation-01/test_stateful.py.log)
preserves the outdated assumption that the edge-case paragraph must be the last
prompt suffix. The assertion was changed to check its exact literal insertion
and equality of the remaining base prompt. Comparing the two saved protocol
inventories verifies identical production-source hashes and only
`test_stateful.py` changed among test hashes. No production change was made to
obtain the second pass. The two invocations overlap; they are not 260 unique
tests. Validation-02 driver time was 90.324 s, separate from the rounded wrapper
time. Its test-process durations were 22.897, 0.195 and 6.515 s respectively.

The candidate-only [fixed-fixture summary](../../result/general-state-research/778971/postformal-validation-02/fixed/summary.json)
records `detected=10`, `total=10`, `fixture_gate_complete=true`, and independent
controls audited; the broader `gate_complete` remains false. This preserves the
known fixed controls, not ten new discoveries. It is not a new paired-arm gate:
the postformal oracle differs from the frozen formal oracle. Fixture execution
took 56.366 s.

The [replay checks](../../result/general-state-research/778971/postformal-validation-02/replay-checks.json)
confirm candidate slogdet attempt 8 was rerun with its original source,
configuration and audit schedule, changing only the engine path. The source
numerical audit agreed; the negative followup violated the scoped VJP relation.
The MR remained **inapplicable**, with only `deterministic` missing, while the
overall verdict became **suspected_failure** for the followup alone. Replay
took 4.274 s. All six process/verdict/provenance checks passed, including hashes
of the original program, request, summary and both tensor records, and the
frozen source/test snapshot. This does not assert byte-identical replay traces
or rerun the search. Details are in the [miss analysis](final-slogdet-analysis.md).

The [postformal protocol](../../result/general-state-research/778971/postformal-validation-02/protocol.json)
identifies frozen engine `iteration-qTg5zq/code`; its backend SHA-256 is
`22e42aae970b7feb3e453f1e04c3691a0d707ad684699aac3cdbea4ac71542d9`.
Allocation/service cleanup is verified separately; successful validation alone
does not establish that Slurm jobs or service processes have exited.
