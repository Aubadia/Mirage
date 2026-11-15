# Roadmap 011: Framework representation traversal and local VJP oracle

## Approved scope (2026-11-15 Asia/Taipei)

User explicitly approved framework-owned traversal of applicable target input representations; LLM owns programs, semantic inputs and training states. Implement together with the previously approved oracle improvements, modifying necessary files. TensorFlow is excluded. Preserve existing worktree and index.

## Design

- Enumerate bounded existing backend-supported single representation actions per actual target call/operand. Runtime legality and alias/identity/gradient-coordinate obligations apply. Maintain per-source candidate inventory and fair resumable traversal; never equate optimizer-gradient layout with target-input coverage. Every child starts independently with the same MR-required initial conditions. No API-specific triggers or new mathematical API recipes.
- Preserve fresh/state program exploration while draining representation candidates. LLM does not author target representation actions. Record eligible, rejected, dispatched, reached, remaining and costs. Candidate denominators are bounded observed candidates, not exhaustive theoretical coverage.
- Admit local VJP checking using bounded source effects and runtime target snapshots/Node evidence independent of subsequent certified optimizer updates. Preserve restrictions against arbitrary hooks/global mutation, RNG effects, ambiguous bindings, writable target calls, unsupported reconstruction. No blanket certificate bypass. Baseline reconstruction, stability and numerical uncertainty gates remain mandatory.
- Audit scheduling uses evidence eligibility/novelty plus an exploration floor and suspect priority; cost stays within search budget. Unsupported/unknown never becomes Pass. MR and numerical verdicts remain distinct.

## Implementation steps

1. Freeze pre-change production source, index hash and protocol ledger; establish persistent research state/heartbeat.
2. Implement representation traversal/controller/prompt integration and tests.
3. Implement scoped oracle admission/evidence/routing and controls.
4. Integrate metrics, run relevant static checks and Slurm regressions, same-input old/new oracle reassessment and fixed controls.
5. Freeze final code/config/protocol before four-arm search, then execute once and analyze without outcome-dependent reruns.
6. Reproduce/corroborate any suspects separately; report outcomes/limitations; drain and close owned service/allocations.

## Final experiment (locked structure)

Four arms A=current baseline, B=traversal only, C=oracle only, D=both. Each API in each arm uses exactly seed0 once, 180s search budget, CPU eager/backward-required, same model/runtime/worker limit120s, fresh corpora. APIs: torch.sin, torch.tanh, torch.add, torch.mul, torch.mm, torch.bmm, torch.nn.functional.linear, torch.linalg.slogdet. 32 searches total. No known trigger sources supplied. Freeze exact execution order, snapshots and flags before final calls. Development/control runs are separate. No statistical significance claims from one seed.

Metrics: runtime validity, target representation and training state units separately, candidate traversal completion/attainment, API×state/MR units, effective numerical judgments and reason-specific unassessed counts, MR verdicts, distinct replayable/corroborated suspects, coverage/judgments per second and per token, generation/audit/serving/allocation costs separately.

## File scope

Necessary production and corresponding test changes within src/mirage and tests; experiment driver benchmarks/exploration_oracle.py; categorized Slurm scripts if needed; this roadmap and docs/research-011; artifacts and frozen ledger under result/exploration-oracle. No dependencies, TensorFlow implementation or unrelated refactoring.

## Progress

- Roadmap created before implementation. Existing code inspected: backend mutation candidates and graph-connected actions already exist; current controller selects one followup, and global source certificate blocks optimizer-containing programs from local VJP audits.

- Baseline49 production files and index hash frozen; isolated preregistration ledger init8fd1051. Heartbeat records task state/owned jobs. Production/test AST and service shell syntax passed before service load. Submitted job780825 using explicit dev partition,400G,2GPU,8CPU,4h,port18511; default8gpus preflight predicted23:21 vs dev16:01. Reuse one service across validation and final searches.

- First frozen regression iteration-lUnVO1: CLI8, artifacts8, oracle routing8, stateful96 passed; adaptive still executing with all six new traversal tests observed ok. One new numeric positive control failed because only saved VJP was altered; reconstruction correctly rejected mismatched evidence. Test corrected to a consistent synthetic wrong backward without production/threshold changes; fresh numerical rerun pending. Service healthy at16:20 server-start log; session readiness environment checks wait for the testing step to release resources, so session startup time includes this wait and is not pure model startup.

- Original adaptive49 and program49 regressions passed. Review subsequently found dynamic-arrival starvation in the index-based source round robin: continual source arrivals could prevent cursor wrap. Fixed to FIFO pending-source deque, with dynamic-arrival and budget-recovery controls; focused8 traversal rerun pending. Development01/02 protocols remain as unexecuted superseded freezes; development03 locks corrected traversal and robust terminal-failure accounting before controls/search. No formal calls yet.

- Initial full related regression349:348 ok+1 faulty new positive-control assertion. Focused fresh controls then passed traversal8,artifacts12,numerical34. Heldout8 all correctly remained unsupported because source typing could not establish unknown target Tensor returns. Added scoped-only native Torch schema return-kind proof (all associated overloads single Tensor), preserving Python-wrapper/mixed-return/unknown rejection; no API list or derivative recipes. Development04 preregistered b2a55bb before new numeric/heldout/fixed controls. Formal searches still unstarted.

- Scoped native-schema validation initially rejected scalar-overload families; constrained schema compatibility by proven Tensor arguments (including bound self), preserving mixed Tensor/tuple return rejection. Heldout then6agreement+2unknown due repeated same-object Node edges. Added numerical-only exact-object summed-coordinate binding, retaining semantic/optimizer edges unchanged and rejecting distinct objects/metadata or slot mismatch. Development06 frozen7e031c9: stateful96/96 and numerical39/39 passed; heldout8/8 agreement; exact same saved records old oracle8unsupported vs new8agreement. Fixed10 both-engine controls now running. No thresholds/API recipes changed.

- Final read-only integration audit: all50 production Python SHA256 match development06; feature-off behavior, source initialization, actual runtime action attainment and backward obligations reviewed without blockers. Fixed baseline10/10 detected with all10 independent controls corroborated; candidate pending. Final artifact auditor now validates actual per-run/per-worker treatment flags as well as snapshots/seed/order/case hashes.

- Development06 validation bundle finished exit0 in iteration-auZSeo: stateful96, numerical39, heldout8, same-record8 all passed. Fixed baseline10/10 and candidate10/10 detected; both independent-controls all10 corroborated. Fixed fixtures are not autonomous discoveries. Began preregistered development sin/mm four-arm90s searches against same healthy service.

- All8 development searches completed successfully. Integration reviews found no oracle/correctness blocker, but observed a generic traversal inefficiency: an initial pair can already verify the exact first representation action, then the queue dispatches it again. Before final freeze, inspect and eliminate only runtime-proven duplicate call/path/action work, recording reused evidence separately from newly dispatched candidates; retain all development06 results and validate dispatch/denominator controls before final. No API-specific ordering or outcome-dependent final rerun.

- Prefinal dedup: exact verified initial followup mutation plus preserved required backward calls now marks previously_reached, excluded from newly dispatched counts. First negative-control fixture incorrectly shared source/followup witness object; retained failed logs, corrected test-only alias. Fresh iteration-sHYJI7 passed traversal10/10 and artifacts13/13, including actual clone-skip then strided child with no provider call. Final protocol preregistered ledger1b80bb9 (SHA256 fd97dd3c774227cacbda5d15737865c882fbcd64658be6495fdab71728dd6080);32 searches started against frozen engines, no further production changes or outcome-driven reruns.

- Final complete:32/32 searches all terminal/successful,233/233 frozen artifact/feature/seed/hash checks passed. A/B/C/D representationunits18/33/14/29; training-stateunits14/13/12/12; effective numericalcases0/25,0/33,12/26,28/34. Tokens173567/100213/162323/83118; allarms0autonomoussuspects, slogdet not found. B/Dnegativecandidates remainpending with no targetnegativeview in rawtraces. functional.linear allarms retain Invalid due composite/unobserved Nodebinding; verifiedAPI coverage7/8, not8/8. No production/scorer/protocol edits or search reruns after finalfreeze.
- Full report, engineer sync, complete219-column perAPI/285-column perarm CSVs and supplemental oracle union/costs written in docs/research-011/report.md and final artifacts. D versusA improves representationunits61.1%, numericalavailability82.4%, tokens-52.1%; trainingstates-14.3%, searchtime+12.0%, no autonomous-discovery gain. Report explicitly distinguishes MR/numerical validity and knownfixture controls.
- Cleanup verified: service running/waiting requests0 twice, all owned disposable caches removed, mainjob780825COMPLETED0:0 and allsteps ended at19:29:52; owned allocation12777s,2GPU=25554allocatedGPU-seconds, restart0. Final production hashes and originalmainindex unchanged; heartbeat stop requested. All approved steps complete.

- Post-result read-only review clarified that D's two unverified clone cases lost attributable parameter-update evidence, not proof that the concrete state disappeared. Report now separates MR-inapplicable from numerically unassessed, and records action-order/reserve depth limits plus the build-phase grad-reset admission boundary. All65 negative candidates remained pending; no production/scorer/protocol changes or reruns.
