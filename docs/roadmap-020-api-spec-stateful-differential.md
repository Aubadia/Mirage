# Roadmap 020 — 從 API 名稱自動執行 training state 測試與 CPU/GPU 差分

## 接手摘要（每次交接先更新本節）

| 欄位 | 2026-11-18（Asia/Taipei）交接狀態 |
|---|---|
| 目標 | 在準備好的計算環境只提供一個公開 API 名稱，自動取得 Spec、執行案例、判定或跳過、保存終態報告。 |
| 目前位置 | **P0–P6 已完成。** 78/78 個盲測 cell 已對帳。完整回歸 565 項通過，2 項歷史資料控制另行補跑，兩項通過。 |
| 起點 | 分支 `feat/POC-Training-State-Not-Loop`，規劃時 HEAD 為 `2085344b1d30f69d77704856aee64eae949af745`。此值是比對起點，切換 session 時先查看實際工作樹。 |
| 已同意範圍 | 使用者授權持續實作與驗證至 P6。P2–P5 修改 cli、workflow、runtime/{api_metadata,stateful,worker}、generator/{program,state_plan,http_provider}、inspector/{stateful,comparison}、backends/pytorch、rules/catalog、artifacts/{report,replay} 與 tests/test_api_check；沿用既有 sampler 的單 operand 遍歷及既有 Slurm launcher。P6 的獨立實驗入口是 benchmarks/api_check_study.py。 |
| 下一步 | 本 roadmap 沒有待執行工作。先讀末節的分母、結果與適用範圍，再從 verified-metrics.json 的 per_api 查逐 seed 證據。後續新研究需另行凍結協定；勿重送本次完成的 campaign。 |
| 進行中工作 | 無。803432、803443、803549 皆已結案；803443.0 服務 step 依停止程序終止，803443.4 測試 step 已退出。停機前 running/waiting 兩次均為 0，squeue 查無本次工作。 |

接手順序：讀本節與最下方交接紀錄，執行 `git status --short --branch`、`git rev-parse HEAD`，比對最後一次證據與第一個未完成里程碑。若紀錄已有 job 或 run，先查 Slurm 狀態、該 run 的 manifest、最後持久化的階段與終態，再決定要收結果、續跑或修正。避免重送同一批工作。需求、授權或工作樹有更新時，以最新紀錄為準，不回退使用者的改動。

## 交付契約與既有基線

正式入口為 `mirage check <public-api>`，例如 `mirage check torch.add`（P1 已實作）。新增獨立入口可保留既有 `mirage run --strategy training|stateful` 作對照。準備好的 Nano4 配置負責 provider、執行裝置、時限、重試與輸出位置；每次執行的使用者輸入只需 API 名稱。缺少必要環境、無法解析名稱或預算耗盡時，仍須有機器可讀的終態與原因。引擎以 stateful 負責執行與觀測、training 提供能力與合法路徑。歷史 `properties` 標示棄用，僅供歷史報告與人工對照。

| 凍結證據 | 規劃時的實際結果 | 用途 |
|---|---|---|
| [固定 benchmark 與獨立對照](pytorch-correctness-reproduction.md) | 已知條件的人工固定案例 10/10 在該環境重現 | 正向控制與事後比對；issue 條件、固定輸入與答案均不進入通用規劃器或 prompt。 |
| [Training 792706](../result/training-e2e/20261117-parallel-002/metrics-summary.json) | 4/10 API 有已驗證案例，共 255 例；違反重現 0/10 | 此 4/10 表示該策略有案例通過驗證。 |
| [Stateful 792821，full-001](/work/asciibase64/Mirage/results/stateful-repair/20261117-full-001/measurement/metrics-summary.json) | 10/10 API 達到 runtime valid（4,398 例）；4/10 有 source optimizer-chain；1/10 有可重現數值違反（slogdet 13/13 replay，MR 違反 0）；顯式 tensor_relation 目標 0 次 | 固定原分母；數值差異、MR、完整 chain、API 執行分列。原兩種策略的時間預算口徑不同。 |
| [上期修正與回歸紀錄](/work/asciibase64/Mirage/results/stateful-repair/20261117-accounting-regression-002/repair-summary.json) | 前期持久化、模型預算與報表修正 | 供相容性與原始快照追溯；020 需留下自己的驗證證據。 |

已知接點：[describe()](../src/mirage/runtime/api_metadata.py) 已取本機版本與文件，現行 run 仍需指定 provider；[traversal](../src/mirage/sampler/representations.py) 從程式規劃候選移出 representation／tensor_relation；[inspector](../src/mirage/inspector/stateful.py) 對兩個具名 callable factory 驗證產物；[寫入比較](../src/mirage/rules/catalog.py) 已辨識 writable paths，但該分支只觀察回傳邏輯值。優先補足上述能力，沿用 [既有架構與證據契約](Architecture.md)。

## 執行與判定規則

1. **取得 Spec。** 解析公開路徑、alias、版本/build、可用 overload／signature、安裝 docstring、dispatcher schema；必要時自動取同版本官方文件或已快取版本。保存來源識別、內容雜湊與是否截斷。離線、版本不合、資料缺漏或衝突須可診斷，可用已知部分繼續。規格保證、模型候選、特定配置的 runtime 觀察分欄。一次合法探測只建立當次能力。私有或無法解析的 target 自動結案。
2. **從主張規劃片段。** 每項主張具體指定呼叫／overload、狀態承載者、predicate、checkpoint、必要歷史與觀測量。同一呼叫的 forward、alias、backward 等主張獨立計分。可直接構造的初態直接構造；要求來源歸因或跨步歷史才生成 backward、累積、step 或還原後續步驟。採用可組合的六類執行方式：直接呼叫與讀寫、作用域進出、建立並使用 callable、註冊後觸發、推進／還原狀態、多參與者與非同步同步。分類依能力、配置及語義，不依十題名單。未知領域保留原因；跨框架時由 backend 提供梯度、身分、同步等證據。
3. **規劃與派發分工。** 已驗證 source 交由 backend 列舉可行關係、合法轉移及表示變換，記錄候選身分、先決條件與排程狀態；模型補足程式與語義初態。覆蓋不足時優先派發尚未驗證且有可用觀測的關係，保留無 oracle 的合法探索。source/follow-up 各自驗證規定的初態、輸入與 checkpoint。送出前核對模型上下文及輸出預算，保留最後成功 source 與有界修復紀錄。
4. **依主張路由。** 先確認執行有效、target 與必要歷史達成。匹配通用 MR 或數值 oracle，動態驗證全部前提與輸入關係後才判定。適用時的 Pass／一致或疑似違反即保留。無可用判定（含 MR 不適用）才考慮 CPU/GPU 差分，保存原 oracle 的不適用原因。差分準入失敗則結案為 correctness unassessed，保留已驗證覆蓋。失效案例、worker 故障、timeout、provider 失敗與資源不足各記自己的狀態。
5. **CPU/GPU 差分。** 同一凍結程式、共同邏輯值、dtype、overload、必要狀態與語義 checkpoint，在各側獨立重建並驗證 layout、offset、alias 拓撲、初始參數及需要的 cotangent。只比較契約允許的共同觀測量，寫入檢查目的物件與已知需保留的位置，alias 比較各側內部關係。記錄硬體、backend、精度模式、確定性設定及同步事件。精確離散值、浮點容差、合法多解分別使用預先版本化的策略；原始差值保留。相同 seed 不構成跨裝置 RNG 等價；GPU-only、無共同 dtype、合法多解無比較準則、初態無法重建或可疑數值域等均可自動跳過。不得藉由搬移時連續化、改 dtype 或移除待測狀態取得 agreement。
6. **保存與 replay。** primary MR、數值 audit、CPU/GPU agreement、CPU/GPU mismatch、不可比較、coverage-only 各自保留來源與理由。差異自動凍結並在相同配置重跑各側，分出穩定差異、側內不穩定及未完成。CPU/GPU agreement 不計入 MR Pass；穩定差異不單獨指認哪側錯。標為框架缺陷前需要獨立對照等進一步證據。原子發布與唯一 run/case/claim 身分支援中斷與重試，缺舊欄位顯示未量測。

## 里程碑與進度

**狀態依據為下列驗收證據。** 編寫程式、排進 Slurm、成功退出都不足以單獨勾選階段。路徑是擬接入處；每階段開始時在交接紀錄列出確切的程式與測試檔案範圍。

- [x] **P0　規劃與盤點。** 記錄工作樹起點、歷史基線、正式入口差距、驗收矩陣及交接方式。
- [x] **P1　名稱到終態報告。** 接入 [CLI](../src/mirage/cli.py)、[API resolver](../src/mirage/runtime/api_metadata.py)、[RunOptions](../src/mirage/contracts/programs.py)、[workflow](../src/mirage/workflow.py) 及 [report](../src/mirage/artifacts/report.py)。先打通 name-only 入口、版本化 Spec、隔離探測與終態報表。核對驗收 A1、A2。
- [x] **P2　按主張生成與派發。** 接入 [能力契約](../src/mirage/contracts/capabilities.py)、[合法候選](../src/mirage/sampler/candidates.py)、[表示遍歷](../src/mirage/sampler/representations.py)、[state plan](../src/mirage/generator/state_plan.py) 與 [pipeline](../src/mirage/runtime/pipeline.py)。讓可行關係有單一明確的派發者；按身分、預算與 checkpoint 追蹤。核對 A3、A4。
- [x] **P3　通用觀測與主 oracle。** 接入 [PyTorch adapter](../src/mirage/backends/pytorch.py)、[事件探針](../src/mirage/runtime/probes.py)、[state inspector](../src/mirage/inspector/stateful.py)、[規則庫](../src/mirage/rules/catalog.py)、[oracle 路由](../src/mirage/inspector/oracles.py)。建立物件來源與 invocation 關係，保留編譯／capture 的專屬證據；補寫入目的與 alias 效果。核對 A4、A5。
- [x] **P4　CPU/GPU fallback。** 接入 [隔離 worker](../src/mirage/runtime/worker.py)、[環境紀錄](../src/mirage/runtime/environment.py)、[比較器](../src/mirage/inspector/comparison.py)、[replay](../src/mirage/artifacts/replay.py)。交付雙側重建、比較準入、同步、穩定性 replay 與理由碼。核對 A5–A7。
- [x] **P5　正式 E2E、相容與成本。** 走正式 CLI／report／resume／replay 入口，以舊封存資料、故障注入及真實 Slurm CPU/GPU run 核對 A1、A7、A8。CLI／報表標記 properties 已棄用，歷史模式與新策略的報表分列。
- [x] **P6　比較與泛化。** 凍結程式、模型、規則與比較協定，執行 A9。在相同 seed 下，同時記錄因探索回饋而異的工作量。保留每一組完整結果與尚未驗證的範圍。

## 驗收矩陣

| ID | 操作／對照 | 通過條件與應留下的證據 |
|---|---|---|
| A1 入口與無人值守 | 在配置好的 allocation 用正式入口只提供公開名稱；另試名稱不存在、provider 不可用、沒有 GPU、預算耗盡。 | 正例自動取得 Spec、執行並輸出 report。負例在有界時間內生成可解析終態與明確原因；CPU-only 環境仍可執行適用的主 oracle。退出碼可區分營運失敗與正常跳過；無互動輸入。保留 argv、stdout、exit、report。 |
| A2 Spec 可信度 | 涵蓋 Tensor 方法、Python wrapper、帶 overload 的 operator；測缺 schema、缺網路、快取版本不符與模型誤判能力。 | 正確記錄來源、版本、配置與未知欄位；不以模型提案或其他版本文件授予 correctness Pass。保存 spec、證據與判定。 |
| A3 真正派發關係 | 用已知可行的 view/base 與 writable operand 控制，加一個前提不成立的候選；在足以執行該固定控制的預算下運行。 | 正例至少一個關係能從 proposed → dispatched → reached 走完；負例有拒絕或未達原因。被動觀察到的關係另計。重試不重複計數。保存候選 ledger 與 runtime checkpoint。 |
| A4 片段與必要歷史 | 用六種執行方式的代表控制驗證。單次 forward、兩次 backward 累積、callable 實際使用、退出作用域、hook 觸發、非同步完成各測缺失必要事件的情形。 | forward 案例不強制 optimizer；只建立 callable、只註冊 hook、未同步或手動填 grad 均不得通過需要相應歷史的主張。保留事件身分、順序與 inspector 判定；尚無資源實測的類別標為待驗。 |
| A5 主 oracle 與路由 | 以可注入的正負控制覆蓋「回傳值相同但目的儲存效果錯誤」、alias 契約未知、主 oracle 已 Pass／違反、主 oracle 不適用。 | 可適用的控制能分出 pass 與違反；未知不誤判。只有無可用主判定的有效主張走差分。差分結果不可覆寫既有違反。分列 MR、數值、差分的證據。 |
| A6 雙側可比性 | 對同一凍結樣本跑真實 CPU/CUDA；再測搬移造成 layout 不符、隨機行為不可對齊、GPU-only、ties 缺等價準則、單側不支援。 | 可比案例有兩側同等初態、實際 target、設定、同步與對齊後的觀測。允許合法多解或自動跳過。不可比皆有原因，無任何錯誤 agreement／mismatch。數值門檻在比較前凍結。 |
| A7 差異與復原 | 用可控比較樣本測穩定差異與單側波動；模擬一側執行中斷，從持久化資料重試；再用凍結 manifest replay。 | 穩定差異與單側不穩定分開；單側完成不產生雙側結論。重試不覆寫原始證據、不重複計分；重播可用原始程式、輸入、配置與環境對照。 |
| A8 整合與報表 | 查看成功、混合、全部跳過、基礎設施失敗的報表，核對 raw rows；讀舊 run。跑適用回歸及真實 name-only E2E。 | 所有派發主張都有終態或明確未完成標記；階段分母能回推，unknown 不當負例、差分一致不計 MR Pass。記錄 wall、模型時間/token 完整性、執行、差分/replay 與資源。保存回歸摘要、實際 job/step、退出與清理證據。 |
| A9 泛化與歸因 | 固定案例作校驗；原十題盲測對照 stateful、整合版（關閉 fallback）、整合版（開啟 fallback）。使用相同模型、硬體、預算口徑、配對的多個 seed；凍結規則後從公開 API 分層抽保留集（排除開發樣本）。 | 事前持久化名單、抽樣方法、seed、資源、數值政策與停止條件。全名單保留在分母；報告零人工修改完成率、達成率、有效主／差分判定率、需專屬處理數、可重現問題數、E2E 與每小時有效判定。另核對原 slogdet 發現與各組漏失階段。未達 10/10 是研究結果，不能據此加入 issue 特例或更換分母。 |

## 驗證、證據與交接維護

- 輕量靜態檢查可在登入節點進行。整合套件、推論、CUDA、較重的重播與比較實驗均透過 Slurm；沿用 [Slurm 說明](../sbatch/README.md) 和 [full suite](../sbatch/validation/full_suite.sbatch)。`bash -n`、`sbatch --test-only` 只記為 preflight。實測保存實際命令、快照雜湊、環境、job/step、結果及退出碼。
- 後續實驗建議依使用者既有儲存安排放在 `/work/asciibase64/Mirage/results/api-spec-stateful-differential/<run-id>/`，每輪獨立並保留失敗。比較各組時記錄 cache／服務狀態與預算定義，不把重疊時間加為 wall time。開發迭代沿用同一健康服務與獨立測試 step，結束查證所擁有的服務、步驟及 allocation 均退出。
- 每完成一項、遇到阻塞、送 job 或準備切換 session，**同時更新檔首摘要、里程碑勾選及下方紀錄**。紀錄格式為「本地日期／P、A 編號／實際變更與授權檔案／驗證命令與結果／artifact 或 job、step、狀態／阻塞與已嘗試項目／下一個具體動作」。尚未驗證就維持未勾選。長工作用 run manifest 或結果路徑接續。
- 凍結時再填具體的保留集大小、多 seed 清單與共同比較預算；這些是實驗設計參數。初版 backend 為 PyTorch；其他框架移植與單一 issue 的手工專用規則不列入本次實作。

**交接紀錄**

| 日期（台北） | 進度、證據與下一步 |
|---|---|
| 2026-11-18 | P0。已核對 CLI、resolver、planner、inspector、report、Slurm 入口及上列原始量測；數字來自舊 run。已建立 P1–P6 與 A1–A9。僅有此 roadmap 的規劃工作；020 尚無實作測試或 job。下一個動作見檔首 P1。 |

- 2026-11-18／P1、A1–A2：使用者要求繼續。程式起點與規劃 commit 相同，工作樹僅有本 roadmap。當前無本人 Slurm job。本輪 scope 見檔首；P1 沿用既有 stateful 引擎，優先驗證名稱解析、Spec 來源、未知能力與持久終態；P2–P6 另列待驗。

### P1 實作與驗證紀錄（2026-11-18）

- 已新增正式 `mirage check <api> [--output <fresh-directory>]`。環境設定為 `MIRAGE_PROVIDER`（預設 vllm，使用既有服務設定）、`MIRAGE_CHECK_OUTPUT_ROOT`（預設 result/check）、`MIRAGE_CHECK_TIME_LIMIT`（300 秒整體牆鐘）、`MIRAGE_CHECK_WORKER_TIMEOUT`（60 秒）、`MIRAGE_CHECK_CASES`（4）、`MIRAGE_CHECK_SEED`（0）、`MIRAGE_CHECK_DEVICE`（auto/cpu/cuda）。P1 的 auto 先在 CPU 執行，forward 不強制 optimizer；跨裝置及主張特定的自動探索接 P2–P4。
- 實際變更為 `src/mirage/{cli,workflow}.py`、`runtime/{api_metadata,worker}.py`、`artifacts/report.py` 與新增 `tests/test_api_check.py`。沿用既有 RunOptions、原子 write_json、run_worker、stateful 與 Slurm launcher；未修改既有測試或專案依賴。
- Spec 保存安裝版本/build、signature、文件雜湊、可取得的 schema 與寫入角色。缺 schema 的 wrapper 可繼續，無文件時有限時／大小的同 release 官方文件 fallback；來源與未知能力保留。外部文件目前只保存為 generation context 候選，P2 將接入規格驅動規劃；未從文件文字自動授予語義證書。
- 原始碼 AST、bash -n、git diff --check 已通過。001 快照 job 803400（8gpus、8 CPU、1 GPU、4 小時 template）執行 101 tests / 31.555 秒，exit 0；002 明確縮至同 partition／CPU／GPU、32 GiB、30 分鐘，job 803407 執行 102 tests / 30.069 秒，exit 0。兩者 scheduler 全部 step 終態、cleanup cache_removed=true。排程 preflight 的開始時間預估未用作實際開始證據。
- 001、002 的 `name-checks/` 保存真實 CLI 產物，包括 torch.sin 的 Spec、target 執行、報告重讀及未知名稱跳過。provider 為明確標示 authored fixture，PyTorch 與隔離流程實際執行；這些控制不計為真實模型生成或 benchmark 發現。缺 GPU 用控制證據測試，真實 GPU 差分仍待 P4。
- 003 在凍結時的 AST 檢查發現報表括號錯誤，保留該不完整快照；已修正，未提交實際 job。004 凍結與 preflight 成功，最終 job 803413 執行相同回歸加空 verdict 負例。最終 103 tests / 34.231 秒全部通過，job 與各 step 均 COMPLETED、exit 0；cleanup cache_removed=true。118 個凍結檔案與工作樹逐一 SHA-256 相符，git diff --check 通過。P1 已勾選。

- 2026-11-18／P1 完成、A1–A2 已有對應控制與真實 CLI 證據：004 的 `validation-summary.json`、`jobs/803413/tests.log`、`jobs/803413/cleanup.json` 及 `name-checks/*/summary.json`。終態包含 completed、api_unavailable、provider_preflight_exception、budget_exhausted，成功案例 runtime_valid=1、MR Pass=1、數值 agreement 成員=2；固定 provider 控制不計為模型或 benchmark 成績。沒有本次工作殘留 job／服務。P2 是第一個未完成項；最終原始碼含 P1 reporting 修正，後續以 004 雜湊為比較點。

### P2–P6 連續實作交接

- 使用者授權連續執行至 P6。P2 接入 workflow、generator/{program,state_plan}、sampler/representations、runtime/stateful、inspector/stateful 與 tests/test_api_check。P3 以相同 runtime/inspector、backends/pytorch、rules/catalog、inspector/oracles 及其控制測試記錄來源、必要歷史與 write effect。P4 使用 runtime/worker、inspector/comparison、artifacts/replay 與正式 check 路由，逐側保存證據。P5 整合 CLI、report、resume/replay、成本與真實服務。P6 使用 benchmarks 下的實驗 driver 及現有 Slurm launcher 凍結比較協定與執行結果。驗收依 A3–A9，所有里程碑依實際證據更新。
- 起點工作樹保有 P1 全部變更，Slurm 查無本帳號工作。P2 第一項為使用 api-spec-1 提供生成上下文與單一派發來源。後續產物存於原 /work 樹。

- 2026-11-18 P2/P3，job 803432：凍結 20261118-p23-control-001，5 tests / 30.872 秒，scheduler 全 step 完成、exit 0，cache_removed=true。relation-ledger 的同 call 兩個具名 operand 從獨立變為共用儲存，checkpoint=final 有 runtime witness。正式名稱入口取 api-spec-1；scope、factory/使用、hook/觸發、Future/wait、同一 leaf 梯度跨 backward 的正負控制通過。provider 是 fixture。A3 尚補負候選、A5–A8 尚待 GPU/report/recovery 驗收。開發服務 job 803443，目錄 20261118-development/803443。接手先查 job 與 ready、iteration 狀態。

- P2/P3 驗收。803443/iteration-4nUvY9 的 60 tests / 67.109 秒通過，含目的值錯誤、guard 改變、缺目的證據、主 oracle 優先、真實 CUDA agreement/replay、逐 API 報表與舊策略控制。803443/iteration-5Ut7d2 的 A3 負候選證明 predicate 實測為 false，A4 六類正負控制通過。共用儲存以實際 operand path 分派與核對；回傳 alias 拓撲是觀測，未知契約保留 unassessed。必要歷史目前支援 dense leaf 累積、連結 scope/callable/registered callback 與 Future/Work。編譯/capture 沿用原生 backend 觀測。
- P4/P5 進度。803443 上 Qwen3.8-27B 真實入口 first smoke（iteration-jhaW5G）0/3，原因為 ctx 誤解與保留 checkpoint；已用通用的 ctx/probe 契約修正。iteration-6Qa5i8 的 torch.add 有 3/3（forward、operand_relation、backward）且 MR Pass=3。iteration-3IeNrz 在啟動測試前因 squeue 查詢 timeout 結束，未耗用模型請求。fixture 差分用 torch.bitwise_not 的含 offset/stride 樣本，CPU/CUDA agreement，MR Pass=0。故障注入發現重試時新產生 pyc 使 inventory 改變，已將程式凍結範圍修正成 Python source；待新快照驗證。
- P6 資料隔離。803443/iteration-vTIdws 跑固定控制，十個結果全數 reproduced，程序 exit 3 表示觀察到違反，實際 58 秒。另訂盲測為 13、37 兩個 seed、每 cell 最多 4 attempts／120 秒、worker 75 秒，原 10 題＋三個分層保留 API，各三個 arm；stateful 以 P1 最終快照的未融合引擎做對照。凍結前只用開發樣本，完整名單／配置與數值協定由 driver 在分派前保存。

- 803443/iteration-5Ut7d2 全套首次 567 tests / 761.093s，2 failures、2 skips；其中一項為已修正的 pyc manifest 問題，另一項因附加 srun shell 缺少本次實際 SLURM_SUBMIT_DIR。新 iteration-hv4S3j 用同一服務、補回實際 submit 目錄，170 tests / 80.075s 全通過；包含實際 CPU-only 半輪中斷、保留舊 trial、正式 resume/replay、輸入幾何、RNG／未知域負例、persistent optimizer step 正負例。P4 勾選；只對通過有限 domain 的 direct forward/operand 關係開放 fallback，梯度與非唯一解自動記錄不具可比性。最終完整 frozen regression 803549 使用獨立 1 GPU、8 CPU、32 GiB、30 分鐘；根目錄 20261118-p5-final-suite-001。
- P6 已啟動。development/803443 的 iteration-qh4jBa 凍結 20261118-p6-paired-001/protocol.json + protocol.sha256、engine/、baseline/ 及 benchmarks/。保留集從未在 development 程式、測試、benchmark 或 docs 出現的 basename 池分層抽出：torch.special.shifted_chebyshev_polynomial_u (49)、torch.fft.hfftn (21)、Tensor.erfc (30)。凍結模型 Qwen3.8-27B BF16/MTP3，H200、每組 2 個 seed、4 attempts、共 78 cells，模型服務準備 215 秒。A9 依正式 cell 證據判讀；prepare 或提交不代表已通過。續跑入口為已凍結 benchmarks/api_check_study.py campaign <root>，保留已完成或中斷的 cell 並從未送出的部分接續。實驗初始比較兩組 check 的 generation budget 相同，stateful 使用相同整體預算，不保證相同生成工作量。

### 已實作入口與證據位置

在 Slurm 測試 step 載入 session.env 並使用準備好的 Python。

```bash
export MIRAGE_PROVIDER=vllm
export MIRAGE_CHECK_OUTPUT_ROOT=/work/asciibase64/Mirage/results/api-spec-stateful-differential/my-checks
mirage_iteration -m mirage check torch.add
```

- 預設啟用差分，在同一 generation 預算下以 `MIRAGE_CHECK_DIFFERENTIAL=0` 關閉；`MIRAGE_CHECK_DEVICE=auto` 對公開 CUDA namespace 選 CUDA，其餘先在 CPU 建立 source。一般入口的輸入只有 API 名稱；P6 的共同裝置條件則事前固定在 protocol。
- Run 根目錄的 `api-spec.json` 保存完整來源，`execution/attempt-*/request.json` 保存實際送出的精簡 Spec 與 claim。vLLM 使用服務 tokenizer 的 chat 計數及實際 context limit 決定 output cap。原始模型輸出、程式、已執行的觀測、修復、MR／VJP 及既有 stateful replay 保留在 attempt 下。
- `execution/relation-ledger.json` 可依 API、seed、device、source 指紋與 predicate 身分追蹤 proposed/dispatched/reached/not_reached 與實際 checkpoint。單 operand 表示仍由原遍歷負責。已派發必要歷史以回傳物件身分／callback／wait／連續 backward 等事件為準。額外觀測回傳 tensor 與 operand 的 alias 分區；MR 有足夠證據才比較目的值、可觀測 alias partner 值與 adapter 已初始化的 guard。
- `differential-ledger.json` 與 `differential/attempt-*/manifest.json` 保存路由、policy、凍結引擎和原始成員指紋。兩側各自執行同一凍結程式，檢查具名 overload、初態、目標輸入值、stride/offset/alias 拓撲、執行裝置及 checkpoint；在 CUDA 同步後保存成員。浮點使用預先設定的有限 pointwise 域與對稱容差；已支援的離散邏輯／位元運算精確比較。無可用數值政策、未對齊 cotangent、混合裝置、隨機與多解等案例可保留 coverage。
- `mirage report RUN` 讀原始終態；`mirage resume RUN` 為有完整 generation summary 的半完成差分建立新的 trial；`mirage replay RUN/differential/attempt-NNNN/manifest.json` 不呼叫模型。只有雙側完整、可比才發布 agreement 或 mismatch，首次 mismatch 會重跑兩側。每次 trial 獨立保存 process、原始 member、同步、判定與原因。舊式 stateful replay 仍讀其原始 manifest；property 報告只供參考。

- 803549 與所有 step 已完成 exit 0，567 tests / 849.676 秒（565 通過、2 個需要封存資料的測試跳過），清理 cache_removed=true。已定位 training-slogdet/20261117-500s 與 stateful-generation/771760/stage-6/fixtures-06，於 803443 的獨立目錄準備了歷史報表副本及原始 tensor 的唯讀取用途，已排在 campaign 後補跑兩個控制。803443/iteration-vFFEtL 正在執行 P6，第一個 seed 39/39 cell 有終態。


### P5/P6 最終驗收與交接（2026-11-18）

| 測試集、兩個 seed 合併 | 方法 | 實際目標回傳 API | 達到有效案例的 API | 有效案例／嘗試 | 有效主判定 | 差分 agreement | 每 cell E2E 中位／P95 秒 | 每小時有效判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 原 benchmark | Stateful 對照 | 10/10 | 10/10 | 50/80 | 30 | — | 22.17／32.51 | 221.6 |
| 原 benchmark | 融合、關閉 fallback | 9/10 | 8/10 | 36/80 | 21 | — | 18.62／28.86 | 193.9 |
| 原 benchmark | 融合、開啟 fallback | 9/10 | 9/10 | 35/80 | 21 | 1 | 20.92／33.41 | 188.8 |
| 保留 API | Stateful 對照 | 3/3 | 2/3 | 14/24 | 0 | — | 21.77／23.65 | 0.0 |
| 保留 API | 融合、關閉 fallback | 3/3 | 3/3 | 14/24 | 2 | — | 19.97／28.81 | 59.6 |
| 保留 API | 融合、開啟 fallback | 3/3 | 3/3 | 13/24 | 2 | 0 | 23.20／33.16 | 50.4 |

- 兩個 seed 各跑 13 個名稱 × 三組方法。每組 benchmark 20 cell、保留集 6 cell；全部程序正常結案，逐 cell 人工修改 0。目標回傳指 source 確有目標回傳證據；有效案例還需通過指定 path、state 等執行條件。一個 API 有任一有效案例才計入該欄，主判定合併可用 MR 與相應的數值主判定，每 case 去重。每小時分母為該組各 cell 啟動至退出時間合計。
- 融合＋fallback 實際有效的 9 個 benchmark API 為 Tensor.scatter_add、linalg.slogdet、linalg.lu_solve、autocast、cuda.make_graphed_callables、compile、add、std、Tensor.copy_。兩組融合的 batch_norm 候選使用受限的 randn 初始化，修復重送相同程式，在 target 回傳前停止。關閉 fallback 的 scatter_add 有回傳案例，但後續 path 未通過，另有 index 越界與 graph 重用錯誤。兩組分別生成程式，覆蓋差距不能歸因給只在執行後介入的差分。
- 在原十題，啟用組有 14 案進入差分，1 案 torch.add（seed 37）agreement，13 案不可比。原因為 RNG／歷史 6、claim 歷史或可觀測量 5、缺適用 pointwise 政策 2。保留集 11 案全數不可比，RNG／歷史 6、無精確 native overload 5。真實盲測無 stable mismatch。固定／注入控制另有 mismatch 與單側不穩定的證據。
- **盲測三組各自為 0/10 benchmark 問題重現**，三組也都沒有可重播的疑似違反。相同環境的獨立固定控制為 10/10。舊版 full-001 的 slogdet 數值發現未在這次小預算下再出現：Stateful 有 8 個有效 slogdet 案例，兩個融合組各有 7 個；各個 seed 的獨立 scorer 都記錄機制未命中。部分其他 benchmark 沒有可用的歷史機制 matcher，逐 cell 標為 mechanism_unmeasured。完整結果只支持這個版本、預算和保留集下的觀察。
- A1–A5 由 P1 入口負例、803432 的 relation/control、803443 的 170 項針對性驗證與真實模型 smoke 支持。A6–A8 另有跨裝置重建、不可比、單側 timeout 後保存再 resume、凍結 replay、穩定差異與側內波動控制。最終 803549 的 567 項為 565 pass、2 skip；在 803443/iteration-aFol0N 帶入真正封存資料後，兩項皆 pass（4.039 秒）。使用報表副本，保留原始封存報表。
- A9 的凍結規則與名單、78 個獨立 cell、6 組摘要以及原生 row 與 source trace 對帳均已保存。基準 stateful 凍結自 P1 的未融合引擎；比較使用相同模型、硬體、配對 seed 與每 cell 上限。它與早期 180 model-seconds 的歷史 campaign 有不同的工作量與預算定義。P6 campaign 1695 秒、模型服務準備 215 秒、重啟 0 次。三個本次 job 合計分配 8,125 GPU-seconds；此值衡量分配時間。
- 證據入口在 [凍結協定與名單](/work/asciibase64/Mirage/results/api-spec-stateful-differential/20261118-p6-paired-001/protocol.json)、[逐 API／seed 與延遲對帳](/work/asciibase64/Mirage/results/api-spec-stateful-differential/20261118-p6-paired-001/audit/verified-metrics.json)、[每個 cell 的原始索引與漏失原因](/work/asciibase64/Mirage/results/api-spec-stateful-differential/20261118-p6-paired-001/results.json) 及 [驗證與清理紀錄](/work/asciibase64/Mirage/results/api-spec-stateful-differential/20261118-p6-paired-001/validation-summary.json)。凍結引擎 56 個 Python 檔逐一與工作樹及最後全套回歸版本相符。P5、P6 已勾選。
