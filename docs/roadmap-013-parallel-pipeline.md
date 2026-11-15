# Roadmap 013: Eight-worker pipeline and one-pass quality acceptance

## Authorized scope

User explicitly requested integrating8 workers, evaluating whether one detected API/representation defect justifies skipping subsequent candidates, and running exactly one fresh180-model-second search for slogdet and one benchmark API. Choose torch.linalg.slogdet and torch.Tensor.scatter_add, seed0, CPU eager, required backward, existing Qwen3.8-27B MTP3 provider. The180s budget is cumulative end-to-end model-request wall time per API; framework execution, service startup and backpressure are separate. No repeated campaign to confirm speedup. Individual suspect replays and independent correctness review remain quality validation.

The proposed API×representation blacklist is not sound as a quality-preserving rule:equal representation labels do not establish equal inputs, dtype/shape, state, paths, oracle conditions or defect mechanism. A confirmed defect on one case does not prove later cases redundant. Do not implement this conditional request's blanket skip; retain all eligible candidates and distinguish suspect-case counts from corroborated mechanisms. No new general defect classifier or MR weakening.

Files in scope:src/mirage/runtime/pipeline.py, src/mirage/workflow.py, src/mirage/contracts/programs.py, src/mirage/cli.py, tests/test_pipeline.py, tests/test_cli.py, README.md, this roadmap, and validation/protocol/report artifacts under result/parallel-pipeline/. Use existing Slurm/session scripts with explicit resource overrides; do not modify unrelated code, dependencies or prior evidence. Preserve all existing worktree/index changes. This roadmap is the first edit after approval.

## Implementation

1. Default stateful pipeline to8 workers while preserving explicit --workers and other strategies' default1. Wire the existing workers option through workflow.
2. Dispatch multiple representation tasks concurrently. Use a separate bounded representation task pool and a shared execution pool so source workers, representation workers, exception triage and replay together never exceed the configured subprocess count. Prevent nested-pool deadlock and retain unique artifacts, initialization checks and full finite traversal.
3. Serialize traversal/ledger updates and completed-feedback publication. Requests consume a versioned snapshot of only completed evidence; completion order may change adaptive search, so preserve actual order/version. Keep the source backlog bounded.
4. Measure cumulative worker time, union of framework-active intervals, overlap with model intervals, observed concurrency, queue/backpressure and wall time separately. Persist worker count and backpressure accounting across resume; identify incomplete legacy timing instead of reporting a false zero.
5. Handle cancellation and drain all in-flight tasks before shutdown; retain dispatched candidates for resume and all original evidence. Extend focused tests for actual parallel dispatch, shared execution cap including source/replay, cancellation/resume, full traversal, feedback and interval arithmetic.

## Validation and acceptance

Run static checks, then focused deterministic and real-PyTorch regressions under Slurm. Tests may use fixtures; final LLM generation receives only API names and generic contracts, never benchmark triggers. Freeze source and final protocol only after required tests pass. Run each API once with8 workers,180 cumulative model-request seconds, worker timeout120s, one automatic suspect replay and full post-generation drain.

Reuse one healthy model service and one allocation across validation and acceptance. Explicitly request12 CPUs and2 GPUs with the prepared environment:4 CPUs/1GPU for vLLM and8 CPUs/1GPU for testing. Test computation remains CPU eager with one intra-op compute thread per worker. Preserve scheduler failures; use the verified dev partition explicitly if its preflight is preferable. Reserve allocation time for evidence and cleanup.

Report every existing nested search metric plus concurrency/timing, generation-only legality, full-cohort legality, coverage, representation terminal states, unknown/inapplicable reasons, request/token accounting, CPU/RSS, replay/corroboration and service/allocation costs. Verify frozen source, zero pending/dispatched work, max8 concurrent framework operations, feedback causality and model-budget accounting. No speedup claim against unmatched earlier campaigns. Record unavailable metrics and distinctions between dispatch completion, attainment and correctness.

Before completion, verify no owned inference requests remain, stop service/tests, release allocation/terminal connections, remove owned scratch and record terminal job/step states.

## Progress

- Read current serial pipeline, worker isolation, CLI/options, relevant tests and Slurm service split. Recorded the correctness reason for rejecting blanket API/representation skipping. Implementation pending.
