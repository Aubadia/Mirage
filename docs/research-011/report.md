# Mirage 表示遍歷與局部梯度 oracle

目前狀態：實作、控制驗證、四組共32次正式搜尋、233項證據檢查與資源清理均完成。每API/arm只有seed0一次。合併版表示coverage與數值oracle可用性提高、token成本下降，但訓練狀態coverage下降；自主suspect四組均0，slogdet本輪沒有自主找到。

## 改動範圍

- LLM 負責程式、語意輸入及訓練狀態；框架根據實際 target operand，枚舉既有 backend 適用的單一表示轉換。
- 每個來源保留有界候選清單並公平派送；來源／變體獨立執行，保留 checkpoint、alias、identity 與梯度座標條件。候選未執行或未到達均明確記錄。
- 以有界來源 effect 認證、target 獨立快照及實際 Node VJP 證據，允許已認證的後續 optimizer 更新不阻擋局部數值檢查。未證實的副作用與數值穩定性仍拒絕／保留 unknown。
- audit 排程結合資格、新觀測組合、均勻探索與 suspect 優先；MR 判定與數值 oracle 判定分開。
- TensorFlow 未實作；未加入 slogdet 專用觸發程式、導數公式或搜尋權重。

## 工程師同步：目前整個框架

- **入口與 backend**：CLI 提供 training、properties、stateful 三種策略；目前 backend 為 PyTorch，API 依實際安裝版本解析 signature／文件，不限於 acceptance list。TensorFlow 尚未實作。[cli.py](/home/asciibase64/Aubadia/Mirage/src/mirage/cli.py)、[backends/pytorch.py](/home/asciibase64/Aubadia/Mirage/src/mirage/backends/pytorch.py)
- **Layer 1：探索與選擇**：重用 canonical／corpus，建立 runtime-derived 狀態目標、LLM 合法轉移提案及已知 configuration actions；依有效率、coverage novelty、成本及 fresh 配額選擇，保留 unresolved／inapplicable 原因。[workflow.py](/home/asciibase64/Aubadia/Mirage/src/mirage/workflow.py)
- **Layer 1：框架表示遍歷**：從實際 call／operand 枚舉適用單動作，來源佇列輪流派送；驗證前置條件、初始資料及變體實際到達，紀錄 pending／reached／not_verified。保留 stride、offset、clone、canonical、合法 conjugate／negative 等既有動作；不宣稱所有組合或任意 alias 聯合轉換都已支援。[sampler/representations.py](/home/asciibase64/Aubadia/Mirage/src/mirage/sampler/representations.py)
- **Layer 2：程式生成與修復**：stateful 使用 `build(device, seed)`／`run(ctx, probe)`；LLM 負責程式、語意輸入與訓練狀態，局部 patch 保留來源義務，依語法、靜態契約與執行診斷修復。表示變體重用來源、不再呼叫 LLM；明確拒絕無變化修復與未證實路徑。[generator/program.py](/home/asciibase64/Aubadia/Mirage/src/mirage/generator/program.py)
- **Layer 2：模型與執行邊界**：支援 vLLM／自訂 provider，隔離推論與 worker；保存 request／response／token，受生成、worker 與搜尋時間預算約束，取消未確認時停止重用服務。
- **Layer 3：執行觀測**：追蹤實際 target call、output／直接 Node backward、operand 梯度貢獻、context、graph lifecycle、表示與資料屬性；有 compile／CUDA graph callable 專用觀測，無法歸因的路徑維持 unknown。[runtime/stateful.py](/home/asciibase64/Aubadia/Mirage/src/mirage/runtime/stateful.py)、[inspector/stateful.py](/home/asciibase64/Aubadia/Mirage/src/mirage/inspector/stateful.py)
- **Layer 3：MR**：依策略具備表示／物化、context restoration、execution-mode／capture、optimizer 梯度表示等比較；先驗證 Pre／Input Relation／checkpoint，再比較必要輸出與容差。MR catalog 的規格不代表每個 API 都有適用實作。[rules/catalog.py](/home/asciibase64/Aubadia/Mirage/src/mirage/rules/catalog.py)
- **Optimizer 觀測**：建立 target → backward → 累積梯度 → optimizer 消費 → 參數／slot 更新證據；梯度 layout MR 與 target 輸入 layout 分開。已審查領域為原生 SGD、小型 CPU float64 dense、無外部 hook／共享參數 storage；不是任意 optimizer／AMP 的通用認證。[runtime/optimizer.py](/home/asciibase64/Aubadia/Mirage/src/mirage/runtime/optimizer.py)
- **數值 oracle**：獨立保存局部輸入與 Node VJP，以受限來源 effects、原生唯讀 schema、版本／綁定證據允許後續已認證更新；重建原點後才做有界方向有限差分、收斂與單側穩定性檢查。輸出 agreement／violation／unsupported／unknown；不承諾任意 dtype、context 或副作用皆可判定。[inspector/oracles.py](/home/asciibase64/Aubadia/Mirage/src/mirage/inspector/oracles.py)
- **其他既有策略**：training 仍有 layout、storage、checkpoint、mutation、saved-tensor lossless roundtrip、accumulation 六類有條件 MR；properties 有受限 CSR／MKLDNN、梯度參考、context／capture／compiled alias probes。這些策略的特定控制不等於 stateful 對任意程式的支援。
- **回饋與 coverage**：保存驗證過的 API、semantic／joint state、transition、API × MR、MR applicability、optimizer chain、生成／執行成本；新報表分開 target 表示與訓練狀態，分開 source／followup、候選分母與 oracle 各階段，沒有理論全集覆蓋率。[artifacts/report.py](/home/asciibase64/Aubadia/Mirage/src/mirage/artifacts/report.py)
- **判定與重現**：Invalid、Inapplicable、Pass、Suspected Failure 與 coverage-only／correctness unassessed 分開；suspect 有界自動 replay、不呼叫模型，核對來源、初始資料、環境與證據。尚無通用任意程式自動最小化，也不自動將 suspect 升級 confirmed defect。
- **產物、驗證與部署**：保存原始程式、規劃、prompt、trace、tensor、manifest、engine hashes、replay 與成本；固定 benchmark 與生成路徑隔離。Nano4 計算均經 Slurm，重用有界 vLLM session，保留非零失敗並驗證清理。Scheduler／GradScaler／DataLoader／distributed state 的完整通用探索與認證仍未完成；CPU offload 也不作已全面支援的主張。

**CLI 現況**：`mirage run` 預設仍是 `--strategy training`；本輪功能需選 `--strategy stateful`。stateful CLI 預設開啟 `--representation-traversal` 和 `--oracle-improvements`，可分別以 `--no-representation-traversal`、`--no-oracle-improvements` 關閉。其他策略會將這兩個旗標關閉。直接呼叫 Python `run_stateful(...)` 及 `RunOptions` 的兩旗標預設仍為 False，實驗必須明確指定，避免改變舊呼叫者。

CLI 其他預設為 CPU、eager、seed0、100 cases、14400秒、worker600秒、replay1；`--provider` 必填。正式實驗另固定180秒／worker120秒／每組每API seed0，不能把 CLI 預設預算混入報告。stateful 預設要求 target backward；`--forward-only` 用於明確允許不要求 target backward 的工作。[contracts/programs.py](/home/asciibase64/Aubadia/Mirage/src/mirage/contracts/programs.py)

## 固定比較設計

| 組別 | 表示遍歷 | Oracle 改善 | 狀態 |
|---|---|---|---|
| A | 無，凍結原版 | 無 | 8/8搜尋完成 |
| B | 有 | 無 | 8/8搜尋完成 |
| C | 無 | 有 | 8/8搜尋完成 |
| D | 有 | 有 | 8/8搜尋完成 |

每組每個 API 只用 seed0 一次，180 秒搜尋上限、CPU eager、target backward 必要、worker120 秒、fresh corpus。固定 API：sin、tanh、add、mul、mm、bmm、linear、slogdet，共32次搜尋。以 API 分組、組內旋轉 A/B/C/D 順序；版本、驅動程式、控制樣本與順序在正式執行前凍結。開發搜尋為 sin/mm、四組各90秒，另存成果，不併入正式指標。

## 驗證紀錄

Oracle／執行控制使用 `result/exploration-oracle/development-06/`；正式凍結前另完成表示遍歷去重的 focused regression，正式版本位於 `result/exploration-oracle/final/`。

- **最新關鍵回歸**：stateful 96/96、numerical 39/39 通過。這些是 development-06 實際執行的測試，不是把歷史結果合併成同一版本的全套測試。
- **跨API開發控制（含SGD）**：8/8 API 得到數值 agreement，涵蓋 unary、multi-input、reduction、matrix 四類。API 為 expm1、log1p、lerp、addcmul、prod、logsumexp、baddbmm、linalg.solve，皆在 target backward 後執行 native SGD。這些案例曾暴露schema與重複object綁定問題，並用於開發修正，因此不是未接觸的holdout，也不是無偏泛化估計；`heldout/`僅保留歷史目錄名稱。
- **同一份保存記錄重審**：相同8份新 witness 記錄，舊版 oracle 為8 unsupported；新版為8 agreement。沒有重新生成／執行案例或補造舊證據，支持「在這8個控制案例上局部oracle可用性提升」，不屬於自主發現或未接觸API的泛化率。
- **先前遍歷／報表驗證**：development-03 的 traversal 8/8、artifacts 12/12、numerical 34/34 通過；artifacts 包含4項純 metadata 的 driver 失敗保留與續跑測試。
- **初次廣泛回歸**：12模組共349項，348通過、1個新正向控制失敗。該控制原本只在第一次執行注入錯誤，重建執行沒有相同錯誤，因而被 baseline gate 正確拒絕；控制已修正並在後續 numerical 驗證通過，沒有取消重建一致性檢查。原始失敗記錄保留於 result/validation/780825/iteration-lUnVO1/regression/。
- **開發期限制修正**：跨API開發控制曾暴露來源 effects／native schema 及多輸入 Node 綁定的適用範圍問題；latest snapshot 使用實際原生 schema 證據與精確 operand object 綁定。原本失敗結果保留，不改寫為通過。
- **固定10 benchmark**：baseline 已完成10/10 detection，10/10皆有獨立 positive、controls_agree 與 production_reported 證據；fixture_gate_complete=true，階段耗時530.278秒。candidate 也完成10/10 detection及10/10獨立佐證，階段耗時395.986秒。這是固定已知 fixture 的能力保留檢查，不是自主發現或改進後新增10個問題。既有 evaluator 的 gate_complete=false 代表未完成其整體研究 gate，與 fixture_gate_complete=true 不矛盾。

最新控制與回歸詳見 development-06/validation-bundle/summary.json、heldout/summary.json、same-record/{baseline,candidate}.json。

- **開發搜尋**：sin/mm、四組各seed0與90秒，共8次搜尋全部exit0；A/B/C/D的嘗試與有效案例分別為3/3、4/4、4/4、4/4，有效數值判定案例為0、0、2、3，自主suspect皆0。B/D各有2個遍歷變體到達，但各20個已知適用候選僅完成2個，不能宣稱已全部遍歷。這是正式凍結前的開發資料，與最終32次結果分開；正式8個API在開發後固定，亦不稱為未接觸holdout。
- **正式前去重**：框架現在只在完整runtime mutation witness吻合時，將來源初次followup已測到的同一候選記為`previously_reached`，避免重派送；`tested_candidates`包含已驗證先前測試與本次到達，派送數另列。focused regression為traversal10/10、artifacts13/13，原始較早版本失敗保留；去重不是已知bug專用規則。
- **正式凍結**：protocol建立時間為2026-11-15 17:50:56（Asia/Taipei），ledger commit `1b80bb9`。固定32次、各API/arm只有seed0，正式開始後不調參、不改source/scorer/protocol、不重跑以取代不佳結果。


驗證通過後才凍結正式版本；不將成功提交 Slurm 或工作 exit0 等同 MR Pass。

## 正式結果

**32/32 搜尋全部正常終結，233/233 artifact／snapshot／seed／旗標／case hash 檢查通過。四組自主 suspect 均為0；slogdet 這次沒有被框架自行找到。** 每API/arm仍只有seed0一次，沒有依結果替換、調參或重跑。搜尋於2026-11-15 17:51:22至19:26:31（Asia/Taipei）執行；共同scoring及driver總耗時5734秒。

| 指標 | A 原版 | B 僅遍歷 | C 僅 oracle | D 合併 |
|---|---:|---:|---:|---:|
| 完成搜尋 / 預定 | 8/8 | 8/8 | 8/8 | 8/8 |
| 具有效 target backward 的 API / 8 | 7/8 | 7/8 | 7/8 | 7/8 |
| 嘗試 / runtime-valid | 37 / 25 | 42 / 33 | 40 / 26 | 42 / 34 |
| Runtime-valid 比例 | 67.6% | 78.6% | 65.0% | 81.0% |
| Target 表示 units（source/followup 聯集，各API加總） | 18 | 33 | 14 | 29 |
| 訓練狀態 units（同上） | 14 | 13 | 12 | 12 |
| 既有 semantic units（來源；含 unary/joint） | 37 | 52 | 41 | 36 |
| Transition units | 9 | 9 | 8 | 8 |
| Joint semantic units | 18 | 26 | 20 | 18 |
| 適用 MR 比較案例 / 有效案例 | 19/25 | 14/33 | 25/26 | 17/34 |
| API × MR units | 5 | 6 | 7 | 7 |
| State × MR units | 27 | 34 | 37 | 34 |
| 本輪實際適用的不同 MR family | 1 | 1 | 1 | 1 |
| 有效數值判定案例 / 有效來源案例 | 0/25 (0.0%) | 0/33 (0.0%) | 12/26 (46.2%) | 28/34 (82.4%) |
| 至少一項 MR／數值 oracle 完成判定 | 19/25 (76.0%) | 14/33 (42.4%) | 25/26 (96.2%) | 34/34 (100.0%) |
| 數值 member agreement / violation / unsupported / unknown | 0 / 0 / 22 / 33 | 0 / 0 / 28 / 42 | 24 / 0 / 0 / 32 | 54 / 0 / 0 / 18 |
| MR inapplicable 有效案例（仍可能有數值判定） | 6 | 19 | 1 | 17 |
| 自主 suspect / 自動重現 suspect | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 程式生成 calls / 規劃 calls / provider 總 calls | 24 / 7 / 31 | 21 / 2 / 23 | 21 / 7 / 28 | 19 / 1 / 20 |
| 完整 input+output tokens | 173,567 | 100,213 | 162,323 | 83,118 |
| 搜尋時間（秒，含預算內生成／執行／oracle） | 1085.712 | 1199.171 | 1142.853 | 1216.426 |
| API metadata 時間（秒，搜尋預算外） | 129.386 | 129.762 | 132.403 | 130.002 |
| 單輪內部計時加總（秒，排除初始Torch import） | 1220.169 | 1334.426 | 1280.338 | 1351.557 |
| 數值 audit 核心時間（秒） | 0.000 | 0.000 | 2.062 | 4.489 |
| Target 表示 units / 搜尋秒 | 0.01658 | 0.02752 | 0.01225 | 0.02384 |
| Target 表示 units / 1000 tokens | 0.1037 | 0.3293 | 0.0862 | 0.3489 |

「數值有效率」只計至少一個source/followup member得到agreement或violation的有效來源案例。舊版仍有MR判定，不能將數值有效率0%說成舊版完全沒有oracle。至少一項oracle的比例是MR與數值案例的聯集，不相加；D的100%只代表34個有效來源案例各有至少一項有限範圍判定，不代表所有42個嘗試有效、全部訓練語意正確或framework沒有bug。聯集另由保存紀錄直接計數，見[補充計數](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/supplemental-oracle-union.json)。

實際適用MR family在四組都只有`optimizer_gradient_representation`；API×MR的5/6/7/7不是新增5/6/7/7條MR。表示變體即使MR inapplicable，也保留驗證後coverage；C/D另可進行局部數值判定。

### 表示遍歷完成度

| 指標 | A 原版 | B 僅遍歷 | C 僅 oracle | D 合併 |
|---|---:|---:|---:|---:|
| 來源配置數 | 未建立清單 | 19 | 未建立清單 | 18 |
| 全部候選（適用＋不適用） | 未建立清單 | 288 | 未建立清單 | 232 |
| 適用候選 | 未建立清單 | 180 | 未建立清單 | 145 |
| 已測總數（先前證據＋新到達） | 未建立清單 | 19 | 未建立清單 | 15 |
| 先前已到達 | 未建立清單 | 5 | 未建立清單 | 1 |
| 本次派送 | 未建立清單 | 14 | 未建立清單 | 16 |
| 本次實際到達 | 未建立清單 | 14 | 未建立清單 | 14 |
| 本次未證實到達 | 未建立清單 | 0 | 未建立清單 | 2 |
| 待測 | 未建立清單 | 161 | 未建立清單 | 128 |

B新派送14/14到達，D為14/16到達。D兩個clone變體的mutation witness與target backward都成立，但現有直接parameter identity綁定無法將clone後的operand連回註冊leaf parameter，因此缺少原配置要求的target-gradient-update證據，梯度offset與update狀態維持未證實。這不能解讀成clone失敗或實際狀態必然不存在；故保留not_verified，不計入完整目標到達。來源局部數值檢查與followup狀態有效性分開。全部30個框架遍歷子案例均未新增LLM程式生成呼叫。B/D僅測完各來源已知適用候選的19/180（10.6%）與15/145（10.3%）；其餘161/128個仍待測。這些是來源／配置特定分母，不能跨組視為相同理論全集，也不能宣稱已遍歷所有表示。

### 每API比較

每格為「runtime-valid/嘗試；target表示units；有效數值判定案例」。

| API | A 有效/嘗試；表示；數值判定 | B | C | D |
|---|---|---|---|---|
| torch.sin | 4/5；1；0 | 4/5；4；0 | 4/5；1；2 | 4/5；4；4 |
| torch.tanh | 4/5；1；0 | 5/5；3；0 | 4/5；1；2 | 5/5；4；4 |
| torch.add | 4/5；4；0 | 5/5；6；0 | 2/3；2；1 | 5/5；4；4 |
| torch.mul | 3/3；5；0 | 5/5；7；0 | 4/5；5；1 | 5/5；6；4 |
| torch.mm | 4/4；2；0 | 5/5；4；0 | 4/5；2；2 | 5/5；4；4 |
| torch.bmm | 3/5；2；0 | 5/5；5；0 | 4/5；2；2 | 5/5；4；4 |
| torch.nn.functional.linear | 0/7；0；0 | 0/7；0；0 | 0/7；0；0 | 0/7；0；0 |
| torch.linalg.slogdet | 3/3；3；0 | 4/5；4；0 | 4/5；1；2 | 5/5；3；4 |

四組functional.linear均有實際forward及output-gradient事件，但直接Node輸入歸因為`composite or unobserved input edge`，無可歸因target Node事件；框架因無法證實要求的target backward而保守判Invalid。各組7次嘗試包含4次backward證據失敗與3次重複／無變化repair拒絕。因此完成8個API搜尋不等於8個API有效coverage；本輪有效target-backward API皆7/8。這是既有composite-autograd觀測缺口，未在正式凍結後修改或重跑。

### 改善與取捨

- **表示coverage提升**：D對A為18→29（+61.1%），B為33。D每搜尋秒表示units提升43.8%，每1000tokens為0.1037→0.3489（3.36倍）。這些是觀測到的各API單位加總，不是理論空間覆蓋百分比。
- **Oracle可用性提升**：數值可判定案例由0/25→28/34（82.4%）；至少一種oracle的案例由19/25→34/34。C為12/26（46.2%）數值可判定。D數值member為54agreement、0violation、18unknown，其中16個是排程未選、2個是followup狀態未驗證；不能將未選或unknown當作Pass。
- **提升不全面**：訓練狀態units為14→12（-14.3%），既有來源semantic units為37→36，transition為9→8；適用MR案例19→17。表示遍歷保留更多表示測試機會，也占用原本的狀態探索預算。C沒有遍歷，表示units只有14，雖然數值可用性已改善。
- **Token成本降低，執行成本上升**：D tokens為173,567→83,118（-52.1%），provider calls31→20；程式生成calls24→19。搜尋時間1085.7→1216.4秒（+12.0%），單輪內部計時加總1220.2→1351.6秒（+10.8%）。worker process時間859.9→1093.6秒（+27.2%，已包含在搜尋時間，不能再相加）。D數值audit核心僅4.489秒，不能把worker啟動／trace收集等全部成本都稱為數值audit時間。未記錄貨幣價格，也未測GPU利用率。
- **找bug結果未提升**：四組自主suspect、自動重現suspect均0，本輪沒有進入獨立佐證的新候選。固定已知10benchmark仍為10/10→10/10；該控制結果不能併入自主發現。單一seed只支持本輪觀察與具體機制，不能作統計顯著或普遍優越主張。

### slogdet：這次沒有自主找到，原因有直接證據

| 指標 | A | B | C | D |
|---|---:|---:|---:|---:|
| 有效/嘗試 | 3/3 | 4/5 | 4/5 | 5/5 |
| 表示units | 3 | 4 | 1 | 3 |
| 數值可判定案例 | 0 | 0 | 2 | 4 |
| 數值agreement members | 0 | 0 | 4 | 8 |
| 自主suspect | 0 | 0 | 0 | 0 |

D已能對含native SGD更新的slogdet程式做數值檢查，8個member得到agreement，因此這輪並非「slogdet完全無oracle可用」。但四組的全部保存target輸入trace都沒有negative-view（negative bit均為false）；D只派送clone與strided，15個適用候選中2個到達、13個pending，包含3個來源／配置的negative候選。B也有4個negative候選全部pending，雖然已測到strided與offset。

D的LLM甚至已生成非對稱矩陣`[[1,2],[3,4]]`，所以不能將本輪漏測一律歸因為只生成對稱語意輸入。針對先前已知negative-view slogdet問題，這次證據更直接指向：**有限預算與既定派送順序沒有走到相應表示**。新來源／訓練配置繼續進入FIFO佇列，部分機會也保留給fresh／狀態探索；候選會保留為pending，而非被宣稱已測過。已知fixture中的slogdet偵測成功是給定控制案例，不是本輪自主發現；本輪也沒有事後人工發現的新slogdet候選。

下一步可比較通用的「未覆蓋表示類別優先／每來源至少完成一輪／程式生成與表示執行分開配置預算」，並處理composite-autograd歸因與表示×狀態條件相容性。這些是後續設計方向，本輪正式版本及結果未依slogdet結果調整。

更細的成本證據：B的slogdet搜尋在剩46.47秒時停止，候選來源的派送reserve約47.42–48.54秒；D剩25.08秒，reserve約47.43–48.75秒。實際子案例worker約28.4–28.8秒，但單次耗時不保證下一次必然完成。全部B/D清單共有65個negative候選，均仍pending；固定動作順序與新增來源共同限制深度，並非僅slogdet的特例。後續可通用地比較表示類別新穎性排程、安全攤提初始化成本，以及沿已驗證轉換傳遞parameter provenance。

來源effect認證也仍有邊界：C的mul attempt-0004在`build`中設`a.grad=None`／`b.grad=None`，目前審查過的reset許可位於`run`，因此其兩個member在admission保持unsupported。這項理由與排程未執行的unknown分開保存；本輪沒有放寬門檻把它改判Pass。

### 完整指標與證據

- [共同32輪比較](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/comparison.json)、[233項檢查及各組加總](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/final-audit.json)。原版25筆followup inspection僅從原先保存的tensor紀錄還原，用同一凍結scorer重新彙整，沒有重跑或補造證據。
- [各API全部指標CSV：32列／219欄](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/metric-exports/metrics-by-api.csv)、[各組全部指標CSV：4列／285欄](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/metric-exports/metrics-by-arm.csv)、[欄位與缺值說明](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/metric-exports/README.md)。CSV保留缺值、null、0、false差異，以及各加總欄位的實際API貢獻數，並非補0。
- [成本補充](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/cost-supplement.json)、[凍結protocol](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/protocol.json)。完整requests、程式、trace、tensor、replay manifest及每輪成本均在該protocol對應搜尋目錄。

## 指標解讀與比較限制

1. 使用同一凍結 scorer 重新彙整四組。舊版缺少 summary.followup_inspection 時，僅從原先保存的 followup.pt 讀回既有 inspection，避免因新版多輸出欄位而虛增 coverage；不重新執行或捏造新 witness。
2. Source/followup 表示與訓練狀態分別呈現，再算成員聯集；不把 optimizer 的梯度 layout 計入 target 輸入 layout。
3. 遍歷分母是特定來源觀測到的單動作候選。沒有遍歷庫存的 A/C 標記未測量，不能用0候選推論其能力。
4. 有效數值案例代表該 runtime-valid 案例至少一個 member 得到 agreement/violation。member 數不是獨立案例數；admission、排程、數值判定是不同漏斗，不能相加。
5. 比例以原始分子／分母彙總，不平均不同分母的比例。效率用總工作量除總搜尋時間；serving startup、allocation GPU-seconds 分開，後者不是 GPU utilization。
6. 同記錄重審、固定 fixture 與人工解析發現不計入自主發現。suspect 是待驗證違反，不等於已確認框架缺陷；不同問題數需重現、最小化及獨立佐證後去重。
7. 一個 seed 只支持觀察到的差異與具體機制，不支持統計顯著或普遍優越主張。更多 coverage／有效判定不自動等於找到更多 bug。

## 資源與清理

同一Slurm allocation780825／vLLM服務重用全部開發與正式測試，restart_count=0。正式搜尋driver5734秒；32輪completion.wall_seconds合計5186.491秒（各process載入Torch後才開始，非完整process lifetime），其中搜尋4644.163秒、API metadata521.553秒；這些時間部分包含／重疊，不可再與整個driver wall相加。每次迭代獨立status/log/code保存在result/validation/780825/iteration-*。

- Allocation：2026-11-15 15:56:55至19:29:52（Asia/Taipei），12777秒，2 GPU，合計25554 allocated GPU-seconds（7.098 GPU-hours），含啟動、開發、等待及正式工作；不是GPU利用率，也不是正式實驗單獨GPU耗時。
- Server日誌16:20:01開始提供服務，相對allocation約1386秒；session.json的startup_seconds=3026還包含測試step占用資源、延遲環境／GPU隔離確認的等待，因此不能當成純模型載入時間。
- 兩次／相隔2秒的service metrics均為running=0、waiting=0；任務擁有的三個node-local scratch/cache目錄均已刪除，先前測試cache亦有獨立清理紀錄。模型及共用persistent cache保留。
- 正常stop marker後，batch／job均COMPLETED 0:0，所有owned steps已終結；serving step在清理時CANCELLED屬正常停止。原互動terminal於上下文切換後不可用，所有正式搜尋完成且status保存後才關閉idle step780825.5，另用同allocation新step780825.7完成audit及清理，沒有中斷／重跑任何搜尋。
- [節點清理證據](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/cleanup-node.json)、[Slurm終態／時間](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/slurm-accounting.psv)、[清理後空佇列](/home/asciibase64/Aubadia/Mirage/result/exploration-oracle/final/slurm-queue-after.txt)。主repository index hash與baseline相同；production與正式凍結candidate source hashes相同。
