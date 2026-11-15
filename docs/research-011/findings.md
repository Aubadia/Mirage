# Framework representation traversal and scoped oracle evidence

## Historical motivating run

Latest single-API slogdet run generated 27 attempts, 23 valid target-backward executions, 16 optimizer MR Passes, no suspect and no effective numerical judgment. All 50 target operand entries were contiguous and lacked negative view. This demonstrates missing input representation exploration and oracle admission, not absence of a framework defect. One seed cannot isolate causal contributions.

## Final observed results (2026-11-15)

32/32 frozen seed0 searches completed without execution failure;233/233 artifact/flags/hash checks passed. All arms verified7/8 APIs (functional.linear lacks attributable composite backward evidence). A/B/C/D representation units18/33/14/29; training-state units14/13/12/12; numerical effective cases0/25,0/33,12/26,28/34; at-least-one-MR-or-numerical cases19/25,14/33,25/26,34/34. Tokens173567/100213/162323/83118. H1/H2 have observed mechanism evidence; H3 autonomous-discovery gain was not demonstrated: all arms0suspects.

Slogdet was not autonomously found. C/D performed4/8 numerical member agreements, while no saved target input in any arm had negative-view. B/D negative candidates remained pending; D tested2of15 applicable candidates, leaving13pending. D already contained a nonsymmetric matrix, so this run's missed known negative-view issue is more directly an unvisited-representation/budget problem than missing all semantic variation. No final tuning/rerun.

Remaining generic bottlenecks: incomplete per-source traversal within budget; repeated common representation classes over new source configurations; representation×training-predicate compatibility; direct-edge-only attribution for composite autograd; optimizer/MR certificate scope. These remain limitations/directions, not changes applied after final freeze. One seed does not establish statistical superiority or exhaustive correctness.

All owned jobs/steps terminated, job780825COMPLETED0:0; requests drained and task-local caches removed. Main index and final engine hashes unchanged. Full comparisons, accounting and engineering sync are in report.md.

## Approved hypotheses

H1: Traversing backend-enumerated representations for each legal source increases verified target representation coverage per generated token, while preserving LLM training-state and semantic-input exploration.
H2: Bounded local snapshot/Node evidence with scoped optimizer effects enables valid numerical judgments on ordinary training programs without weakening rejection of unsafe effects or numerical instability.
H3: Their combination increases opportunities for autonomous defect discovery. More opportunities alone do not establish more defects.

## Protocol

Final four arms A old, B traversal only, C oracle only, D both; eight fixed APIs, seed0 once per API/arm, 180s per search. Frozen engines, driver, controls and order precede final calls. Development uses sin/mm, four arms and 90s; separate artifacts. Same saved candidate-generated records are reassessed under baseline and candidate oracles without rerunning programs. Missing witnesses in old records are not invented.

## Lessons and constraints

- Target input representation is separate from optimizer gradient representation.
- Representation candidate counts are source-specific finite inventories; coverage is not universal.
- Old followup inspection may be absent: report as unmeasured, never zero evidence.
- Unsupported or unscheduled numeric records are not agreements.
- All computation via Slurm; immutable snapshots per iteration; no outcome-dependent final retries.

## Development history

Latest development-06 actual stateful96/96 and numerical39/39 pass. Cross-API development controls8/8 native SGD programs across four API families produce agreement. Their historical folder is named heldout, but these cases exposed native-schema/repeated-object issues and informed development fixes; they are not untouched holdouts or an unbiased generalization estimate. Reassessing the identical eight saved witnessed records under the immutable baseline yields8 unsupported; current oracle yields8 agreement. This directly supports scoped oracle availability on these controlled inputs, not autonomous bug discovery or universal correctness.

Initial broad regression349 had348 pass and1 faulty positive-control failure, retained. The control was corrected to keep the same faulty target backward during both original and reconstructed execution, preserving baseline gates. Development-03 traversal8/artifacts12/numerical34 passed. Later native schema and exact object binding fixes were necessary for cross-API development multi-input operators; original failed/incomplete validations remain.

Fixed10 baseline completed10/10 detection and10/10 independent positive controls (530.278s). The fixture_gate_complete flag is true; the separate overall gate_complete remains false by evaluator design. Candidate fixed controls also completed10/10 detection and10/10 independent positive controls (395.986s). These known fixtures are regression evidence, not autonomous discoveries. Development eight searches completedexit0: A/B/C/D validcases3/4/4/4, effective numericalcases0/0/2/3, suspects0each. Before formal freeze, generic dedup of already-witnessed initial followups was added and focused traversal10/artifacts13 tests passed. Final32 searches ran after preregistration ledger1b80bb9 and completed with233 integrity checks. TensorFlow excluded.
