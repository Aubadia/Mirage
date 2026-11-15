# 完整 Mirage 架構與目前實作的差距

本文件只對照原始碼與使用者要求的三層架構；未讀取本輪正式搜尋結果，未執行 Torch 或修改 production。核心差距是：**能執行生成的訓練程式、能保存狀態、能選擇／驗證狀態轉移、能判定轉移後正確性，是四種不同能力。** 第009輪改善提案與修補，尚未補齊後三者的完整訓練生命週期。

## 目前已有的能力，及其上限

新的 `stateful` 路徑讓程式擁有所有操作順序，且真實觀察 target、backward node、operand gradient、context、compiled/captured callable。它並非只有 forward layout 測試。不過，[STATE_DIMENSIONS](../../src/mirage/contracts/capabilities.py#L13) 只有 gradient、context、graph_lifecycle、execution_mode、representation、data、tensor_relation；沒有 optimizer、scheduler、AMP scaler、RNG、DataLoader 或 distributed 維度。

自動 training goal 實際只有 backward_count、target_backward_count、target_boundary_count、restoration_count 從一變二，以及 compiled_execution；見 [goal_catalog:136](../../src/mirage/generator/state_plan.py#L136)。`training_state` 分類名稱不會產生新的測量能力。使用者可以提供既有維度內的其他 field，但 [inspect_state:49](../../src/mirage/inspector/stateful.py#L49) 只有在 inspector 自有 facts／operand witness 存在時才判定達成；把名稱寫進 JSON 並不足夠。

| 面向 | 已實作、可指認的證據 | 距離完整狀態探索的缺口 |
|---|---|---|
| 梯度與 graph lifecycle | 完成 backward invocation、target node、operand contribution 及 callable boundary 有分開證據；[inspect_state:68](../../src/mirage/inspector/stateful.py#L68) 驗證必要路徑 | repeated backward/count 不等於 optimizer 更新生命週期；梯度 none/zero facts 只聚合註冊 parameters，不能精確指定每個 parameter／input gradient 的階段 |
| Optimizer | [training_snapshot:31](../../src/mirage/runtime/stateful.py#L31) 保存 state_dict；[OptimizerObserver:1122](../../src/mirage/backends/pytorch.py#L1122) 觀察單一註冊 optimizer 完成 step、zero_grad，事件進入 transition_signature | 沒有 parameter-group／slot 的穩定 selector、momentum/state-initialized predicate、step 前後證據或更新計數 facts。stateful 只檢查 optimizer 型別，沒有 legacy 路徑的「每個 optimizer parameter 必須註冊」檢查；不能據此聲稱完整訓練狀態已保存 |
| Scheduler／AMP scaler | optional ctx 可帶 stateful object；初始／最後 state_dict 會保存；[execute:387](../../src/mirage/runtime/stateful.py#L387) 要求可呼叫 state_dict | 沒有 scheduler.step、unscale、overflow、skipped-update、scale.update 的明確事件、順序合約與 predicate；保存 blob 不證明使用了正確物件或順序 |
| Autocast／grad context | [context_state:764](../../src/mirage/backends/pytorch.py#L764) 記錄 CPU/CUDA autocast、dtype/cache、grad mode；特定 context target 有 enter/exit restoration MR | 不是完整 AMP 更新生命週期；[Probe.mark:136](../../src/mirage/runtime/stateful.py#L136) 的 facts 沒有 scaler 或完整 autocast 狀態目標 |
| RNG | [rng_state:21](../../src/mirage/runtime/state.py#L21) 保存 Python、CPU、當前 CUDA RNG；stateful 保存 run 前後並檢查 unchanged | stateful [生成／解析合約](../../src/mirage/generator/program.py#L227) 禁止已識別的 random sampling，未提供 RNG 轉移目標；沒有具名 Generator、所有 device、DataLoader worker RNG 的完整模型。unchanged 是限制條件，不是 RNG 行為測試 |
| Checkpoint／restore | `checkpoint_data` 有 portable payload 驗證；initial/final snapshot 包含註冊 modules、optimizer、scheduler、scaler 與 extra_state；[execute:403](../../src/mirage/runtime/stateful.py#L403) 對 replay 重新 build 的初始值／metadata 做一致性檢查 | 這是初始重建／證據重播，不是任意訓練中斷後的 checkpoint→load→continue。沒有 complete-state inventory、恢復事件、下一批次位置或中間 state_dict 比較 |
| Saved tensors／offload／recompute | stateful 有 forward operand metadata 與 backward 可歸因證據 | [semantic_coverage:161](../../src/mirage/inspector/stateful.py#L161) 明確將 backward_representation 標為 unsupported：尚未把保存／取回的實體 Tensor 與 target、autograd node、version、storage、device 串起來。CPU offload、pinned placement 與重算材料化尚無完整 stateful predicate/MR 鏈 |
| DataLoader／distributed | 一般 source 可能自行呼叫公開 API；portable extra_state 可以攜帶使用者整理的資料 | 無 iterator/sampler epoch/cursor、worker queue／seed、rank/world/process-group、collective sequence、同步或分片 optimizer 的測量模型。live DataLoader／iterator 不能直接放入 portable extra_state；[run_worker:38](../../src/mirage/runtime/process.py#L38) 是單 worker 啟動，不是分散式測試協調器 |

不能將以上寫成「整個 repository 完全沒有 optimizer／AMP／checkpoint」。舊 `training` 路徑已有 [bounded history validator](../../src/mirage/contracts/exploration.py#L46)，實際執行 zero/forward/backward/retain/step，並以 optimizer hook 記錄成功更新；見 [runtime/program:378](../../src/mirage/runtime/program.py#L378)。它也建立 GradScaler、執行 [activation checkpoint](../../src/mirage/runtime/program.py#L342)、[lossless saved-tensor clone hooks](../../src/mirage/runtime/mr_actions.py#L21)、受限的 FP32 microbatch accumulation，且 [conditional MR](../../src/mirage/rules/conditional.py#L104) 比較 loss、梯度、parameters、modules、optimizer、scaler，checkpoint MR 另比 RNG。這些是可重用的既有垂直能力，但受舊固定程序契約約束，尚未整合成新 stateful 的通用目標／轉移／證據模型。

## 三層之間尚未閉合的接口

**Layer 1：缺少可測量的訓練狀態 vocabulary。** 沒有 optimizer／scheduler／AMP 等 facts，就無法建立可靠的 API × 狀態未探索集合、合法轉移候選或成本回饋。現有 family balancing 只能分配已存在的候選。更多 prompt token 不會使不存在的 predicate 變得可達。

**Layer 2：程式擁有操作，但 action 尚未形成效果合約。** [stateful_request:480](../../src/mirage/generator/program.py#L480) 明確不插入 step/backward/context；planned actions 仍是待生成文字，沒有通用 read/write sets、依賴與完成事件。合法 source 可以碰巧完成 scheduler 或 scaler 操作，但目前無法證明該操作屬於選定 parameter/optimizer、位於正確 phase，或造成預期狀態變化。

**Layer 3：有快照，尚無完整語意 checkpoint 及 MR 路由。** `probe.mark(name, values)` 保存任意觀察值、註冊 parameter gradients 與少量 facts，沒有每個 checkpoint 的完整訓練狀態；重複名稱被視為 ambiguous，尚無 iteration/microstep/phase occurrence selector。[representation_pair:68](../../src/mirage/rules/catalog.py#L68) 比較 target outputs 與梯度；[execution_mode_pair:128](../../src/mirage/rules/catalog.py#L128) 另比 parameters/optimizer，但不含 scheduler/scaler/RNG 的完整結束狀態。[catalog](../../src/mirage/rules/catalog.py#L24) 雖宣告 accumulation、state_restore、alias_effect，stateful [worker 路由](../../src/mirage/runtime/worker.py#L79) 實際選 context、capture、compile、representation，再加輔助 VJP；宣告 relation 名稱不等於有 pair constructor、動態前提證據與執行路徑。

此外，source certificate 仍拒絕多數 stateful mutation／控制流程。正確方向是為訓練效果建立專屬合約與對應 oracle，保留 coverage-only，而不是要求 generator 刪掉訓練步驟以換取純函數證書。詳見 [certificate 邊界](certificate-boundaries.md)。

## 建議優先順序與驗收條件（提案，尚未實作）

| 優先 | 通用設計與最小閉環 | 必要正負控制／完成標準 |
|---|---|---|
| P0：讓真實更新可選、可證 | 註冊 optimizer↔parameter↔group 的穩定身分；明確追蹤完成 update、zeroing、parameter/slot 狀態；建立 `(logical_step, microstep, phase)` checkpoint。先加入少量 exact predicates，並將 memory selector 延伸到 gradient 與 optimizer slot | 同一數值但不同 storage/layout 的 gradient 經過同一合法更新可形成 MR pair；目標 backward 必須連到被更新參數。未註冊參數、未完成／skipped step、錯 optimizer、僅改計數器都不能取得 coverage |
| P1：訓練狀態恢復與 AMP／scheduler | 建立完整 persistent-state manifest，包含 parameters/buffers/grads、optimizer groups/slots、scheduler、scaler、RNG 及 logical data position；構造連續執行 vs save/load/continue 的合法 pair，只比較對齊後所需狀態與剩餘語意輸入 | 少存任一必要項目應使 Pre/In 未成立；省略 restore 或錯下一批次不可算 Pass。成功與 overflow/skipped-update 需分開證據；scheduler 順序依相應公開合約判定，不能強加單一通用順序 |
| P1：保存表示與訓練歷史交會 | 將 pack/unpack/recompute 事件連到保存者 node、取用 backward invocation、版本及 storage/device；先重用既有 clone-hook/checkpoint 能力，再擴充受控 CPU offload／pinning | 值相同但錯 dtype/device/version/history 不可取得 roundtrip certificate；重算必須實際執行且對齊相同語意輸入。量測 layout/alias × 保存策略 × backward/update phase 的組合，而非只數 hooks |
| P2：資料流程與分散式 | 先做受限、可恢復的資料序列／sampler，再做明確多 rank launcher、collective trace、全 rank completion、分片與全域 checkpoint manifest | 相同樣本語意／權重／順序有證據；iterator advancement、worker seed 或 rank state 遺失必須被辨識。單 rank 提早失敗／超時不可被其餘 rank 的 Pass 掩蓋，並驗證整組 cleanup |

每一項擴充都必須同時具備 **predicate schema → 合法 action/preconditions → source generation → runtime witness → MR/其他 oracle → 成本與未知原因回饋**，才能稱為完整新增能力；只有 enum、prompt 或 state_dict 欄位不算。驗證先以通用正負 controls 建立可靠性，再以預凍結的跨 API／狀態案例檢查泛化，不能依已知 bug 配方或放寬 tolerance 追求命中。

建議第一個驗收目標是「某記憶體表示的梯度，經由可歸因 target backward，進入完成的真實 optimizer update，並在對齊 checkpoint 比較所需 parameter／slot 狀態」。它直接保留 **記憶體表示 × 真實訓練狀態轉移** 的研究核心；先完成這個小而完整的閉環，比繼續擴充純 layout／數值初始化白名單更能補足使用者架構。
