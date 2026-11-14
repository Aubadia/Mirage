# General numerical gradient audit

Approved scope: common directional finite differences and observed VJP evidence;
no API-specific derivative, gradient checker, or expected-answer branch. Preserve
historical experiment artifacts and current MR semantics. API issue IDs may only
appear in benchmark attribution, never in the numerical checker.

Stages (continue automatically):
A. Bounded, versioned call/output/operand/backward numerical evidence and replay.
B. Representation-preserving reconstruction, baseline validation, deterministic
   multi-step directional differences, finite/nonsmoothness guards and budgets.
C. Workflow/benchmark integration, authored controls, frozen held-out API set,
   existing candidate re-execution with new evidence, regressions and cleanup.

Authorized files: src/mirage/contracts/capabilities.py;
src/mirage/runtime/{stateful,worker,representations}.py;
src/mirage/backends/pytorch.py; src/mirage/inspector/{oracles,numerical_gradients}.py;
src/mirage/workflow.py; src/mirage/artifacts/report.py;
benchmarks/pytorch_correctness/evaluate.py; benchmarks/oracle_generalization.py;
tests/{test_numerical_gradients,test_oracle_routing,test_stateful,
test_pytorch_benchmark,test_capabilities,test_adaptive}.py;
sbatch/validation/oracle_audit.sbatch; this roadmap;
docs/pytorch-correctness-evaluation.md. No dependencies or unrelated refactors.

Validation: static checks on login node; all Torch/computational validation via
Slurm. Freeze snapshots for each execution. First run authored positive/negative
controls (injected wrong gradients are checker tests, not framework defects),
coordinate/representation and missing-evidence tests. Then use eight held-out
APIs in four families; at least one usable numerical judgment per family is
required. Re-execute the historical generated candidate, validate reconstruction,
automatically audit and replay without changing the historical 0/10 report.
Retain unknown and explicit reasons when scope, correspondence, numerical
stability or resource limits prevent assessment. Numerical agreement is scoped
consistency, not universal semantic correctness. Stop owned jobs/steps on exit.

Frozen held-out set (not used to tune checker): torch.expm1, torch.log1p,
torch.lerp, torch.addcmul, torch.prod, torch.logsumexp, torch.baddbmm,
torch.linalg.solve. Families: unary, multi-input, reduction, matrix.
If these results inform checker changes, relabel them development and select a
new held-out set before reporting generalization.

Status: complete. Initial repository working tree was clean; approved implementation and validation completed.

Progress: Phase A numerical evidence and replay fields implemented. Generic router
keeps MR and numerical verdicts separate, schedules suspects/every fourth case,
and requires same-call complete evidence for attribution. Storage-coordinate
reconstruction and generic multi-step checker are implemented; authored controls
are being completed. No Torch validation has run yet. Default 8gpus preflight
reported next-day availability; explicit dev override retains one GPU/eight CPUs.

Development validation 777292 (dev, 1 GPU/8 CPUs): evidence 59, routing 6,
capabilities 12, adaptive 14 tests passed; benchmark tests passed with fixture
audit deferred to the dedicated integration stage. Numerical controls failed
on tuple/list shape comparisons in reconstruction. Failure logs retained;
fixing representation metadata normalization before the next frozen run.
Additional review requires stable directional evidence for routing, and keeps
MR sampling novelty separate from a numerical-only suspicion.

Second development snapshot 777299 passed all six unittest groups: 120 tests
selected, 118 passed and 2 deferred (CUDA opt-in and fixed-fixture artifact audit).
777304 separately passed the CUDA evidence test; its fixture driver then failed
on a sibling-module import path, before executing any fixture. Correcting only
the new validation driver and rerunning that stage.

Historical candidate re-audit 777303 passed: initial states and observations
exactly match the archived source/follow-up, source numerical agreement, negative
view follow-up numerical violation in 2/4 coordinate directions. Immutable
replay verified captured evidence and independently repeated numerical
corroboration. No API derivative formula; historical 0/10 remains unchanged.

Final validation: 777308 re-executed and independently corroborated 10/10 fixed
fixtures. 777321 passed CUDA float32/float64 chart/VJP control and the frozen
held-out gate: 6/8 agreement, 2/8 unknown, 0 numerical violations; all four
families represented. addcmul/baddbmm repeated-operand calls retained unknown
because original backward bindings were ambiguous. No checker changes after
held-out results. Current production files exactly match the 777321 snapshot.

Results: result/general-oracle/summary.json, candidate 777303/run/summary.json,
fixtures 777308/run/fixed/{summary,independent-controls}.json, held-out
777321/run/summary.json. All six owned jobs and steps exited; no service or
interactive allocation remains. Failed iteration artifacts retained. Aggregate
allocation: 287 GPU-seconds (one GPU/job), not utilization or elapsed wall time.

Limits: eager, bounded dense real FP32/FP64 and certified read-only effects.
Original pre-target leaf/history is not replayed. Pinned-memory flags and full
storage capacity are not captured by the existing descriptors; the chart
certifies observed occupied coordinates and listed metadata only. Unsupported
states and ambiguous evidence remain unknown. Numerical agreement covers
observed cotangents and tested directions, not forward semantic correctness.
The original autonomous 0/10 result is unchanged; no new LLM campaign ran.
