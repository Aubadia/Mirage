# Research log

| Entry | Type | Decision |
| --- | --- | --- |
| 1 | bootstrap | User authorized autonomous overnight work. Invoked autoresearch; preserve one final seed per treatment, Mirage state-space contribution and all correctness gates. Create roadmap before edits; use isolated protocol ledger rather than touching staged work. |

## 2026-11-14T18:37:10.610904+00:00 — Mechanism audit and implementation

Frozen V8 baseline has 97 inventoried files. Isolated protocol ledger initialized at 0bb8bce3e1da91ad0ea2dbcefb69fb144a614af0. Audit finds proposal-supply crowdout (training proposals missing in 41/52 new planning requests despite available goals) and certificate false rejections plus unsafe static acceptances. Root implements family allocation, transferable feedback and runtime no-progress stopping; two agents handle binding/planning and certificate regressions. Primary literature notes saved; memory-layout firstness rejected. No new compute yet.

## 2026-11-14T18:39:16.918688+00:00 — Slurm allocation

Default 8gpus preflight accepted but predicted08:02 start; explicit dev preflight predicted02:43. Submitted778971 using existing dev_session script, dev partition,2GPU,8CPU,400G,4h. Prepared Qwen3.8-27B BF16 MTP3 environment. Owned output root result/general-state-research. Unrelated job778934 left untouched.

## 2026-11-14T18:48:41.663496+00:00 — First frozen validation

Protocol/driver and10-file implementation inventory preregistered at113b4ca. Running10 related test modules under ownedSlurm778971/testingPTY14977. Model startup191seconds, restart0. Static AST and source reviews passed; no runtime pass claims yet. Generator reviewer found malformed-planning failures bypassed two-call parent budget; fixed before this snapshot and added integration regression.

## 2026-11-14T18:55:32.854448+00:00 — Regression01 complete

263tests,262passed. One old HTTP payload assertion expected only plans while new protocol requires examined_goal_ids. Updated that exact schema assertion plus status/blockers requirements; no production HTTP change. Independent review identified aliased compiler/capture symbol and rebound probe source-hint ambiguity; bounded unknown-preserving fix in program.py/test_program.py is being finalized. Frozen regression01 remains unchanged. Reporting/static-audit script preregistered at9a81b44.

## 2026-11-14T19:01:49.999418+00:00 — Fixed-control compatibility regression

74affected tests passed after source-hint/schema correction. Fixed10 reference detects4/10: six missing deterministic, because tracked build args/kwargs expansion was blanket-rejected, with two additionally rejecting Python complex constants. Diagnosed from source/AST, no fixture changes. Generic binding normalization and complex scalar support authorized in backend/test_stateful, with unsafe expansion/effect negative controls. Formal and development searches have not begun.

## 2026-11-14T19:05:54.713586+00:00 — Certificate compatibility correction validated

134certificate/MR/numerical tests passed in26seconds; generic complex literals and proven call-container expansion supported, unknown/duplicate/effect-bearing helper expansions rejected. Independent reader found no concrete effect bypass and documented remaining conservative out=None expansion limitation. Running fixtures02 against identical backend in both arms; no search outcomes yet.

## 2026-11-14T19:08:39.383296+00:00 — Fixed gate restored

Fixtures02: reference10/10 andcandidate10/10,111.13seconds,unchangedtriggers/MR/numericalconditions. Bothengines useidenticalcorrectedbackend. Runningstructuralcontrols, thenstaticcertificatecounterfactual. Freezeenginehashesfromfixtures02 beforedevelopment.

Structuralcontrols01 aborted before any case: driver omitted arm parent directories. Fixed output.mkdir(parents=True) only in experimental driver, preserved failed logs and will use fresh controls02. Production/fixtures unchanged.

## 2026-11-14T19:13:24.985913+00:00 — Development starts and historical static audit

Controls02 passed4perarm,36.06sec. Staticcertificateaudit445savedprograms:14original-valid newlyadmitted,10removed; everyvalidremoval caused by unrecognized torch.long. These are conservativefalseexclusions, not unsafe historicalsources. Generaldtypealias support assigned beforeformal; noissue-specificrules. Development01 alreadyfrozen/started8searches(seed0,90sec/API); retainitunchanged andlabelknowncertificatealiaslimitation. Noheldoutsearches yet.

## 2026-11-14T19:20:43.395264+00:00 — Preformal general compatibility and status fixes

Native dtype aliases verified against installed stubs/officialdocs; public lazy submodule resolver implemented without API-specific exceptions. Both final arms will share these fixes; frozen development01 unchanged. Root also distinguishes planning_unresolved/planning_rejected from invalid_program and adds focused workflow regression. These corrections do not use heldout search outcomes; none have run. Pending Slurm revalidation and fresh fixture gates.

Encoding-ablation01 failed before any HTTP request (all0.002sec; missing provider artifact subdirectory). No model observations/tokens were produced. Fixed experimental-driver mkdir, preserved failed outputs, retry same preregistered15requests as encoding-ablation02. Adoption criterion unchanged.

Encoding-ablation02 executed15requests in21.03sec:3/4 differential syntax/static-contract recoveries,0/3control regressions, allproviderscomplete, schema smoke accepted3planswith exactIDs. Meets preregisteredadoption rule; addedexactgenericencodinghinttoallstatefulfull/patchprompts. No generatedprograms wereexecuted bythisablation; no runtimecorrectnessclaim.

## 2026-11-14T19:51:31.804835+00:00 — Final candidate preflight

Latest114schema/binding/encoding/duplicate-repair tests passed. Fixtures03both10/10 in107.512sec with identical backend+public resolver overlays. Development02started from exact fixture03 engines,8searches90sec/APIseed0. Policy/report metadata labels corrected before freeze; no algorithm change after latest tests. Formal heldout searches not yet started.

## 2026-11-14T20:01:51.439552+00:00 — Development02 complete; final freeze

Development02passed9/9steps,567.131sec. Reference19valid/12MR/3qualifiedjoint vs candidate25/18/26; traininggoals1/4 vs4/7; explicitmemory×training0/0both. Candidate7planning-stage refusals remain, reference token usage incomplete. Single-seed descriptive positive indication only; no concrete correctness blocker, no further tuning. Final uses exactfixture03 engines,32searches120sec/APIseed0, then existingaudits+commonscore. Static certificate-audit02 had incorrect parentpath and0programs; it is invalid evidence despite exit0, rerun correctpaired-01path as audit03.

## 2026-11-14T21:12:49.423159+00:00 — Postformal eligibility diagnosis

Formal complete35/35steps,3233.52sec; all15gates true. Mixed results retained. Manual inspection candidate204407attempt8 found actual negative-view nonsymmetric target+backward with differing savedVJP; selectedaudit unsupported due whole-source requires_grad_ setup rejection. Saved-only extraction ran Slurm R5w5lv, no source execution or original edits. Independent review cautions against simply removing sourceeffectgate. Scoped distinct local-numerical effectcertificate correction recorded before edits; formalresults/engines remainfrozen, no newsearchseed.

Postformalvalidation01:129/130tests pass; the onlyfailure is a pre-existing stateful prompt assertion that assumes EDGE_CASE_PROMPT is the absolute suffix. The preregistered/adopted encoding hint now follows it. Correct the test to require the exact literal edge addition and byte-identical remaining base prompt; production prompt/formal snapshot remainunchanged. New oracle-specific tests all passed. Fixtures/replay intentionally not started after regression failure. Fresh validation02 will run the same protocol.

## Completion

Postformalvalidation02passed130tests,fixed10/10,exactmanualreplayflagsfollowupwhileMRinapplicable;90.324sec. Firstfailedpostvalidationretained. Formalengineandoriginalcandidateartifactsunchanged. Current17source/testfilesmatchvalidatedsnapshot;originalindex/stagedpatchandbaseline97filespreserved. Owned778971completed0:0,servingstepintentionallycancelledatcleanup;inferencedrained,scratchremoved,PTYclosed,heartbeatstopped. Resource9922sec×2GPU=5.512GPUhours,startup191sec,restarts0. Noadditionalsearchseed. Fullreportincludesnegativeplanned-memoryresultsandseparatelocaloraclefix.
