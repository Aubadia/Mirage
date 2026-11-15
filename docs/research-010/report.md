# 梯度記憶體表示與真實 optimizer 更新鏈

狀態：兩輪開發實驗、受影響回歸、控制 gate 與正式 seed0 比較均完成；資源清理與證據核對完成。本輪沒有新的疑似失敗或已確認缺陷。

本輪把「目標 API 的 backward → 註冊參數的梯度 → optimizer 實際消費的梯度表示 → 更新後參數與 momentum」接成可驗證的鏈。這修補上一輪只增加一般張量／joint units，卻未建立更多真實 memory × optimizer 測試的缺口。完整範圍見 [roadmap010](../roadmap-010-optimizer-gradient-chain.md)。

## 為什麼舊指標不等於找到更多問題

原本各處有 backward 事件、optimizer 快照與一般表示測試，但它們沒有共同證明「這個 target 的梯度，以這個布局被這次更新消費」。因此增加一般 coverage，可能只改善代理指標。另一方面，生成器可能取得布局卻丟掉原本的更新步驟；合法程式只有執行設定需要改變時，又會被 AST 無修改的規則攔下。

本輪以明確來源鏈、必保留的完成更新義務、設定變更重用，以及獨立版本的 coverage 解決這些問題。新增 MR 比較對齊更新邊界的參數與 optimizer slots，沒有放寬數值門檻，也沒有加入 slogdet 的特例。詳見 [原因與泛用方案](analysis.md)。

## 已實作的能力

- Runtime 核對直接 target Node、同一次完成的 backward、參數 post-accumulate 事件，以及更新前未被覆寫的梯度。
- Harness 在 backward 完成後，以保持精確邏輯值的 copy 產生 canonical、strided 或 offset 梯度，交由 source 自己的原生 SGD 消費。
- 新梯度表示目標附帶同參數的 `optimizer.completed` 義務；已驗證的原程式可只改執行設定，重新執行確認，不需要無關的程式碼修改。
- Corpus 與回饋獨立保存 `optimizer-chain-1`；不把新 checkpoint／facts 混進舊的 `state-semantic-2`。未觀測、證據不足、已驗證為零分開記錄。
- 新 MR 僅在來源、原生更新範圍、初始／更新前狀態及允許的表示差異都成立時，比較更新後參數和 momentum slots。

## 驗證與實驗

開發後修正的 `regression-05` 目前受影響 172 項全部通過。最終逐項 log 核對跨快照共有 365 個不同相關測試：360 個有通過證據，5 個 CUDA 專用測試跳過；重跑不重複加總，也不是同一 final 快照完整重跑。先前以 runner 的測試總數概括為 365 通過的進度數字在此更正。先前 exception-rerun 等失敗紀錄仍保留。

目前 `controls-02` 兩組相同 runtime/oracle 下的原生 MR 控制各 13 項通過；`fixtures-03` 兩組固定 10 案均完成 10/10 偵測與獨立控制檢查。先前 `controls-01`、`fixtures-02` 的成功及 `fixtures-01` 參考組編譯 std 案例在 180 秒 timeout 的失敗均保留。後續只將兩組固定案例的驗證 timeout 對稱延長至 600 秒；搜尋預算未變。這些是實作與已知案例 gate，不是自主發現。

模型服務在 Slurm job779954 內健康重用至本輪結束。前一個 job779931 因服務就緒超過 15 分鐘等待上限而失敗，其結果與非零狀態均保留；本次就緒等待上限明確延長至 30 分鐘。服務的 session-ready 計時另包含測試 GPU 的環境驗證等待，不可當作純模型載入時間。

正式 source 在啟動前以實驗 ledger `cc513b5` 凍結，沿用 protocol revision3。兩組共用 runtime／oracle／觀測器，參考組保留上一版生成與規劃；候選組使用本輪功能。每組只用 seed0，固定 `sin`、`add`、`mm`、`linear`、`tanh`、`bmm` 六個 API，每 API 120 秒。兩組均要求 backward；這是首次模型呼叫前登錄的 optimizer 能力面板設定。開發搜尋只用 `sin`、`add`，每 API／組 90 秒。正式 12 次搜尋與共同評分共 13 步皆 exit0；沒有依結果修改程式、替換 API 或重跑搜尋。四個 transfer API 是搜尋留出，並非實作或單元測試從未接觸。

## 正式結果：更新鏈增加，同時有其他覆蓋的取捨

[正式比較](../../result/optimizer-gradient-chain/779954/final-seed0-report/comparison.md)與[機制稽核](../../result/optimizer-gradient-chain/779954/final-seed0-mechanisms/mechanisms.md)完整性檢查皆通過，沒有缺失的必要結果。有效案例與 coverage 以 source 計算；MR 是配對判定，數值 audit 另以 source／follow-up 成員紀錄計算。

| 指標 | 參考組 | 新版 |
|---|---:|---:|
| 嘗試／有效案例 | 13／10 | 18／15 |
| 有效 API／固定面板 API | 5／6 | 5／6 |
| 已驗證更新鏈案例／`optimizer-chain-1` 單位 | 0／0 | 15／17 |
| 梯度布局＋完成更新目標達成／嘗試 | 0／0 | 10／10 |
| 明確記錄的設定重用案例 | 未量測 | 10 |
| Optimizer MR 適用／Pass／suspect | 0／0／0 | 15／15／0 |
| 全部 oracle 的疑似失敗 | 0 | 0 |
| 舊 `state-semantic-2` 單位 | 40 | 17 |
| 舊 transition 單位 | 10 | 5 |
| 數值 audit：agreement／unsupported／unknown | 8／2／11 | 0／10／22 |
| 保存的搜尋 planning／generation tokens | 105,320 | 29,868 |
| 搜尋耗時合計（秒） | 530.09 | 558.68 |

新版在 `sin`、`add`、`mm`、`tanh`、`bmm` 各有 3 個有效鏈案例：1 個 fresh 加 2 個 library 設定轉換。10 個子案例都重新執行並達成自身的布局與更新義務；沒有 LLM optimizer planning request。17 是版本化、去重的鏈單位，不是 17 個不同程式或缺陷。10 次成功 library dispatch 中有 5 次在其實際歷史／剩餘時間下會被舊 reserve 條件攔下；此固定歷史診斷不能取代整個 scheduler 的因果比較。

**能力增益有明確代價。** 舊語義單位的 unary 部分由 16 降到 8，joint 由 24 降到 9。更多預算進入相同 source 的梯度表示轉換，並不保證一般 operand／其他訓練狀態也探索更多。新增與舊單位的定義不同，不能相加抵銷。未來需以可適用 oracle 與分開的覆蓋目標配置預算；維持 fresh quota 本身仍不足以維持 oracle 多樣性。

Slurm 內對全部保存 raw records 的[補充觀測](../../result/optimizer-gradient-chain/779954/final-observations.json)確認：新版15個有效 source 與15個有效 follow-up 均沒有非空 post-update slot，也沒有正 momentum；不能宣稱正式搜尋已覆蓋非空 momentum。這次只讀取已保存張量，沒有重跑程式或改判定，所有原始 hashes 保持一致。

數值 audit 的分母是 source／follow-up 成員紀錄。新版 10 unsupported 都因 source effects 超出既有 numerical certificate；22 unknown 中 21 是未被有界排程選定，1 是原執行／路徑未驗證。參考組 11 unknown 包含 10 個未排程與 1 個路徑未驗證，另有 2 unsupported。**新版沒有數值 agreement，不可把 optimizer MR Pass 當成獨立導數驗證。** 表中零 suspected failure 也不等於框架正確。

兩組 `linear` 均 3 次嘗試、0 有效案例。部分程式確實執行 target 與 backward，但無法通過目前直接 Node 歸因域（`composite or unobserved input edge`）；另有重複修復回覆、參考組輸入建構錯誤及候選組來源 certificate 的獨立限制。這不是「沒有執行 backward」或 LLM 拒絕 optimizer goal。詳見 [linear 的證據與泛用解法](final-linear-limitation.md)。正式結果保留此缺口，未事後更換 API 或放寬證據。

所有保存 member 的 native／共同評分 coverage 逐筆一致：參考組 source11、follow-up10；新版 source17、follow-up15。這些數量包含已保存但無效的 member，與有效案例分母不同。參考組缺少 `source_reused`／library inventory 等新欄位，不能把未量測的設定重用或候選供應改寫成零；這與已驗證的更新鏈為零是不同問題。參考組 10 個有效 source 的更新 count 已驗證為零；整體另有 2 個未觀測案例。新版整體另有 1 個未觀測與 2 個 count unknown，不能改寫成零。

Driver 總 wall time 為 1360.24 秒，包含 metadata、子程序啟動與共同評分；搜尋耗時及 worker／audit／rerun 分項互相包含，不相加。Token 完整性僅指保存的搜尋 planning／generation usage，未包含 metadata 呼叫，也不是整個任務的全部模型成本。本研究只有單 seed、短時預算與整組策略改動，不支持統計優勢、元件單獨因果或整體 bug-finding 提升。

## 首輪開發結果：瓶頸在規劃呼叫之前

[Development01 摘要](../../result/optimizer-gradient-chain/779954/development-01-report/comparison.md)與[機制稽核](../../result/optimizer-gradient-chain/779954/development-01-mechanisms/mechanisms.md)的完整性 gate 通過，原始結果與 hashes 均保留。

| 指標 | 參考組 | 候選組 |
|---|---:|---:|
| 嘗試／source 有效案例 | 2／2 | 2／2 |
| source 更新鏈案例／單位 | 0／0 | 1／1 |
| Optimizer MR 適用／Pass／suspect | 0／0／0 | 1／1／0 |
| Planning requests／選定梯度＋更新目標 | 0／0 | 0／0 |
| 明確設定重用紀錄 | 未量測 | 未觀測到 |
| 數值 audit 成員紀錄 | 4 agreement | 4 unsupported |

候選組的一條鏈與 MR Pass 來自 fresh `sin` 案例。四次搜尋都在首個案例後停止，當時仍有 43.40–52.72 秒名目預算，但低於記錄的 60.37–74.40 秒整輪 dispatch reserve。**沒有送出任何 planning request，不能說 LLM 拒絕 optimizer 目標**，也不能從這批結果評價規劃或設定重用的效果。降低 reserve 是否能產生更多有效案例，仍須實驗驗證。

候選組 `sin`、`add` 都使用 `momentum=0.0`，唯一的 `sin` MR Pass 沒有證明搜尋觸發了非空 momentum slot。正 momentum 的人工原生控制驗證了 slot 比較能力，須與自動搜尋實際達到的 coverage 分開。

候選組 `add` 合法執行，但來源超出既有 optimizer certificate 可證明的程式片段。[靜態稽核](../../result/optimizer-gradient-chain/779954/development-01-certificate-audit.json)依序辨識 indexed tensor writes、以迴圈設定梯度，以及 `params.values()` 三種限制。改為 tensor literal、逐個 leaf 明確設定 `requires_grad_(True)`，再以 `[a, b]` 傳入 SGD 後，靜態 optimizer certificate 成立；**改寫未執行**，不代表執行等價、鏈達成或 MR 適用，也未取得局部數值 certificate。

候選組 4 筆數值 audit 均因來源效果超出既有執行 certificate 而 unsupported，參考組 4 筆為 agreement。分母是 source／follow-up 成員的 audit 紀錄，不是四個不同程式或四次有限差分呼叫。新增更新鏈伴隨這項導數審查適用率限制，不能只報 MR Pass。

## 開發後已採用的通用方案

Protocol revision3 已在最終搜尋前以實驗 ledger `ecd3ff7` 記錄下列整組修正，未改 final API、seed0、120 秒預算或共用 oracle 範圍。

- **直接提供已知設定轉換。** 對已有完整更新鏈的有效 parent，以有界 transition library 提供梯度布局設定候選，每個 parent 只加入一個候選分支；子案例仍須重新證明同參數完成更新與全部義務，不把 parent 證據當成達成。
- **分開 examined 與 selected。** LLM 曾檢視或標為 unresolved，不等於已執行該 library action；只有實際選定才跨提案路徑消耗該目標。
- **依已知 parent 的 worker 成本估計。** 設定分支採同一有效 parent 已完成的 worker 耗時，保留 `max(10 秒, 1.5 × 耗時 + 5 秒)`；缺資料則回退整輪 reserve。先套用 bootstrap／fresh quota，再做可負擔性篩選；一般生成、未知工作與 repair 保留原 reserve，sampler 公式與執行 deadline 不變。
- **提示既有可證明範圍。** 以泛用提示要求直線式 literal／pure 初始化、明確梯度設定及 SGD 參數列表；不擴張 certificate 或數值 oracle，不把不支援的合法案例判為錯誤。

這些改動是整組系統方案。首輪開發不提供效果證據，第二輪的實測支持有限範圍內的執行能力；仍不能辨識任一元件的個別因果貢獻。

## 第二輪開發結果：設定轉換確實接上更新鏈

[Development02 摘要](../../result/optimizer-gradient-chain/779954/development-02-report/comparison.md)與[機制稽核](../../result/optimizer-gradient-chain/779954/development-02-mechanisms/mechanisms.md)已完整通過資料 gate，無缺失檔案。兩組維持 seed0、`sin`／`add`、每 API 90 秒。

| 指標 | 參考組 | 候選組 |
|---|---:|---:|
| source 有效案例 | 3 | 4 |
| source 更新鏈案例／單位 | 0／0 | 4／4 |
| planned 梯度表示＋完成更新達成／嘗試 | 0／0 | 2／2 |
| 明確設定重用紀錄 | 未量測 | 2 |
| Optimizer MR 適用／Pass／suspect | 0／0／0 | 4／4／0 |
| 數值 audit 成員紀錄 | 4 agreement、2 unknown | 4 unsupported、4 unknown |

候選組每個 API 都產生一個 fresh 更新鏈，再由 library 選擇一個設定重用案例。兩個子案例重新執行後，均達成自身的梯度表示與完成更新義務，形成兩個明確的 memory representation × optimizer update 測試。這兩個 typed plans 來自有界 library，候選組沒有送出 LLM planning request；參考組 `sin` 有一個 `training_state` 規劃呼叫，但 catalog 沒有 optimizer 目標。不能將此結果描述為 LLM 接受或拒絕新目標，亦不把 library 候選數當成可達狀態數。

兩次成功 library dispatch 的剩餘時間分別為 55.23、54.95 秒；新 parent-worker reserve 為 49.31、49.30 秒，而依實際先前案例耗時重算的舊整輪 reserve 為 56.52、56.91 秒。因此，**固定實際歷史與保存的選擇時點**，舊 reserve 條件會擋住這兩次 dispatch。這只證明該控制條件的差異；完整舊 scheduler 可能產生不同時間、parent 與提案，不能據此宣稱整個方法的因果增益，也不把 dev01→dev02 當成單一元件的因果實驗。

四個候選 optimizer MR 都 Pass，且記錄到非零參數更新，沒有 suspect。但兩個生成來源仍為 `momentum=0.0`，子案例重用相同來源，尚未證明搜尋涵蓋非空 momentum slot。候選組的 4 unsupported 仍受 source-effects certificate 限制；其 4 unknown 與參考組 2 unknown 都是 bounded audit schedule 未選定的成員紀錄。unknown 不是 agreement，新增更新鏈沒有補足獨立導數審查。

## 現有界線

這一版證明的是 **optimizer 消費梯度的表示差異**，不是 backward 寫進非連續梯度 buffer。範圍為 CPU float64、原生 SGD、cold state、單次更新與直接註冊的 target leaf；warm state、多步、其他 optimizer、AMP／compile／distributed 仍需新的證據與 MR 範圍。

Optimizer MR Pass 不等於梯度數學正確。含 `step()` 的來源目前仍可能不適用既有局部數值 audit；兩邊共有的錯誤 VJP 也可能產生相同更新。因此會分開報告更新 MR 與數值 audit 的適用率，不宣稱整體找 bug 能力已提升。後續泛用方案是具時間與效果證據的 call-local 憑證，詳見 [數值 audit 邊界](numerical-audit-boundary.md)。

單一 seed 的結果只能支持描述性的能力比較。已知案例與合成錯誤控制是驗證材料，不是自動發現的新框架缺陷。

研究定位與先行文獻的有限核對見 [novelty positioning](novelty-positioning.md)。PyTorch 原生測試已有梯度表示差異經真實更新後比較參數的基本先例，因此這個 MR 本身不能當成新穎性。可保留的是跨 API predicates、target 歸因、runtime reachability、conditional MR 與 coverage-guided exploration 的整合價值假說；本輪不能證明優先權或勝過原生測試。


## 執行資源、失敗保存與完成檢查

[完整驗證與資源紀錄](../../result/optimizer-gradient-chain/779954/validation-resource-01/validation.md)保留每輪耗時、退出碼、原始 hashes、跳過測試和歷史缺失。六個缺失的 lifecycle／summary 檔案來自先前啟動失敗或遭 timeout 中斷的輪次，不屬於正式比較；沒有以後續成功覆寫。現行172項受影響測試全數實際通過，跨快照的5項 CUDA-only 跳過限制另列。

本輪重用一個健康 vLLM service 至結束；此前有一次啟動失敗並替換服務。779954 的 session-ready 為1985秒，其中包括測試GPU環境驗證等待。正式 driver 為1360.24秒；逐輪測試時間另列，不能與 startup 相加成 wall time。兩個頂層 Slurm allocations 合計配置26,158 GPU-seconds，約7.27 GPU-hours，包含失敗啟動及閒置配置時間，**不是 GPU 利用率**。

[清理前證據](../../result/optimizer-gradient-chain/779954/cleanup-before-stop.json)確認推論 running／waiting 都為0、沒有 cancellation-unconfirmed，且本輪兩個 job 的已識別節點暫存均刪除；共用持久模型／服務快取保留。測試 PTY exit0，779954 於2026-11-15 14:49:19臺北時間完成，779931 的失敗紀錄保留；兩個 job 及所有步驟均已終止。Heartbeat 亦退出，PID已不存在。

[最終凍結與清理核對](../../result/optimizer-gradient-chain/779954/final-freeze-cleanup-audit.json)確認72個 source／test 檔雜湊仍等於正式啟動前版本，root Git index 沒有改動。僅實驗用獨立 ledger 記錄研究協定與結果，沒有提交或暫存使用者主倉庫的既有變更。

下一輪具體方案見 [next-steps](next-steps.md)：優先處理 numerical admission、複合 target 歸因與 oracle／覆蓋預算配置，再擴 warm／多步與 backward-buffer 域。本輪已完成核准切片；這些擴充尚未實作。
