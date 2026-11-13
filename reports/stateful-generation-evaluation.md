# Stateful generation 評估

## 狀態與主張範圍

本報告評估新的 `stateful` 策略（contract version 4；程式格式 `stateful-1`）。G1–G7 已完成記錄中的驗收；每 API 300 秒的凍結協議已在工作 771760 完成，獨立審閱、audit 修正與資源清理亦已完成。開發嘗試、人工撰寫的固定案例與正式生成案例屬於不同母體。[先前評估](pytorch-correctness-evaluation.md)的 10/10 固定案例與 3/10 引導生成結果，不是本實驗的結果。

框架接受 API 與是否具有 oracle 支援分開處理。目前不宣稱涵蓋所有 PyTorch 特性，也沒有實作 TensorFlow backend。憑證未知時，結果為不適用或未評估的覆蓋，不能算作正確性通過。

## 正式實驗完成結果

凍結執行依宣告的每 API 300 秒預算，完成全部十個目標的一次實驗。人工案例／控制驗收偵測並獨立佐證 **10/10** 個 benchmark；完整來源生成得到 **0/10 個獨立確認的 benchmark 重現**，且 **production Suspected Failure 為零**。這是實測的已確認數量，不代表所有未裁定執行都正確。下列指標以[審閱結果](../result/stateful-generation/771760/stage-7/formal-01/evaluation-reviewed-v2.json)為準；[自動結果](../result/stateful-generation/771760/stage-7/formal-01/evaluation-automatic-v1.json)保留原樣。

| Issue／API | 嘗試 | 來源 | 靜態有效 | 來源完成 | 執行證據有效 | 適用比較 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [203078](https://github.com/pytorch/pytorch/issues/203078) `torch.Tensor.scatter_add` | 40 | 40 | 40 | 31 | 21 | 9 |
| [204407](https://github.com/pytorch/pytorch/issues/204407) `torch.linalg.slogdet` | 61 | 61 | 61 | 32 | 32 | 23 |
| [200599](https://github.com/pytorch/pytorch/issues/200599) `torch.linalg.lu_solve` | 31 | 31 | 30 | 25 | 19 | 14 |
| [202706](https://github.com/pytorch/pytorch/issues/202706) `torch.autocast` | 53 | 53 | 53 | 52 | 22 | 22 |
| [181723](https://github.com/pytorch/pytorch/issues/181723) `torch.cuda.make_graphed_callables` | 37 | 37 | 37 | 24 | 8 | 0 |
| [204033](https://github.com/pytorch/pytorch/issues/204033) `torch.compile` | 47 | 47 | 47 | 0 | 0 | 0 |
| [204365](https://github.com/pytorch/pytorch/issues/204365) `torch.nn.functional.batch_norm` | 37 | 37 | 37 | 19 | 14 | 8 |
| [201268](https://github.com/pytorch/pytorch/issues/201268) `torch.add` | 42 | 42 | 42 | 27 | 20 | 17 |
| [197089](https://github.com/pytorch/pytorch/issues/197089) `torch.std` | 43 | 42 | 42 | 28 | 23 | 0 |
| [200613](https://github.com/pytorch/pytorch/issues/200613) `torch.Tensor.copy_` | 43 | 43 | 43 | 17 | 17 | 2 |
| **合計** | **434** | **433** | **432** | **255** | **176** | **95** |

「來源」指已從回覆取出 source 字串，不保證是語法有效的 Python。靜態 Valid Test Rate 為 **432/433 = 99.77%**；Executable Test Rate 為 **255/432 = 59.03%**。執行證據有效的來源占完成來源 **176/255 = 69.02%**，或占取出來源 **176/433 = 40.65%**。靜態有效不代表訓練執行合法。一個 LU 候選有不匹配的括號；一次 std 生成達到 provider 長度上限，未產生可用來源。

共有 172 次 repair 嘗試，其中 52 次是 input-coverage refinement；依 API 分開計算共有 325 個來源指紋。原始語法有效數為 329，103 個候選使用嚴格限定的結尾括號正規化。完成的來源／follow-up 執行數為 255／177；context restoration 使用單次觀察執行，不需人為補上 follow-up。95 次適用比較全部 Pass。原始 verdict 分布為 95 Pass、137 Inapplicable、20 Invalid，以及 182 個沒有完成 verdict 的案例；這些分類不能取代 runtime-valid 的執行漏斗。

| Issue | 達成狀態 | API 內不同簽章 | 有合格 mutation 的候選* | 使用秒數 | 未用秒數 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 203078 | 13 | 6 | 31 | 275.35 | 24.65 |
| 204407 | 0 | 0 | 32 | 287.13 | 12.87 |
| 200599 | 10 | 3 | 25 | 258.77 | 41.23 |
| 202706 | 21 | 4 | 0 | 288.75 | 11.25 |
| 181723 | 0 | 0 | 0 | 281.55 | 18.45 |
| 204033 | 0 | 0 | 0 | 287.94 | 12.06 |
| 204365 | 5 | 2 | 8 | 265.64 | 34.36 |
| 201268 | 8 | 5 | 27 | 274.96 | 25.04 |
| 197089 | 3 | 1 | 28 | 269.51 | 30.49 |
| 200613 | 0 | 0 | 17 | 286.69 | 13.31 |

*此欄計算至少有一項合格 mutation 的測試候選，不是 transformation plan 數量。整輪有 60 個候選達成所要求的轉移述詞，依 API 分開計算共有 21 種簽章。168 個候選有合格 mutation；保存證據中，134 個成員合計有 233 個驗證通過的 mutation application 事件，包含 canonical materialization，不能解讀成 233 種不同狀態或 MR Pass。保存的成員可能出現在後續 worker／inspection 失敗之前，因此不能用其數量取代完成執行計數。

### 獨立標籤與審閱修正

審閱後的 confusion counts 為 **TP=0、FP=0、TN=39、FN=0**。39 個符合指標範圍的負例包含 22 個 context-restoration 候選與 17 個回傳值 copy 候選；因此實測 FPR 為 **0/39**，只適用於這個已裁定子集。由於沒有建立符合條件的獨立正例，**Correctness Violation Detection Rate 未定義（0/0）**。另有 30 個獨立負例不符合必要狀態／路徑範圍，365 次嘗試維持 audit-unknown。Unknown 不能算 negative，也不能算已確認的 oracle 漏報。十個 API 的 Time to First Detection 均在表列實際使用時間處設為 censored。

原始自動 audit 把整張 FX graph 輸出誤當作單一 API 輸出。審閱撤回八個 positive 標籤：scatter 的 0004、0022、0031、0032、0038、0039，以及 add 的 0014、0015。其中三個 scatter 標籤原先計為 FN，另外五個不在有效指標範圍；八個均改為 unknown，而非 negative。這些是 audit 錯誤，不是 PyTorch 缺陷。審閱沒有修改任何生成來源、production verdict 或預算。

正式執行後，evaluator 已修正為將未認證的整圖觀察排除於單一目標參考與歸因。獨立 eager positive 仍保留；部分 eager negative 無法認證被排除的整圖輸出。九項針對性 benchmark 測試在 2.351 秒內全部通過（`iteration-4ur2Pq`），包含保存的 fixture-control audit。後續對全部 artifacts 的交叉檢查，重現了每個審閱後候選標籤／分類及整體指標。凍結 evaluator 雜湊為 `84c740ad69bbf9f75dd930475de276e32cf7d865161c7662628e207486b56a0d`；修正後為 `c5f94a3de368315c6731cce1d665484288ce043132458ecc25ad7fee18c02cf0`。43 個 production engine 檔案沒有變動。

### 各 benchmark 為何重現或未重現

固定案例全部得到重現與佐證；以下區分此 Phase 2 證據與 Phase 3 生成結果。結構觀察依據保存的 runtime 證據，而非只看來源文字推測。[審閱證據](../result/stateful-generation/771760/stage-7/formal-01/review-evidence-v2.json)提供各成員 descriptors 與事件。

#### 203078: `torch.Tensor.scatter_add`

人工案例結合 stride `(1, 0)` 的 expanded index，以及超出 index 寬度的 source／destination；Python 整數 scatter 累加與 contiguous-index 控制共同佐證錯誤的列位址計算。正式執行保存的 55 個來源／follow-up 成員中，沒有任何一個滿足記錄中的結構組合。共有 21 個執行證據有效的候選、9 次適用比較，但沒有獨立確認觸發條件。這是已觀察到的觸發覆蓋缺口，不是已證實的 oracle 漏報。六個編譯候選原先得到 positive audit 標籤，是因整張圖輸出被拿來與單一目標參考比較；這些標籤已撤回。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/203078/summary.json).

#### 204407: `torch.linalg.slogdet`

人工案例使用複數共軛之虛部所形成的 negative-bit 實數 view，再執行 determinant backward；逆矩陣轉置導數與實體化控制共同佐證錯誤。正式生成保存過一個 negative-bit operand，卻沒有將它與必要 backward 路徑結合。29 次失敗執行皆出現缺少 grad function 的錯誤；其餘 32 個候選執行證據有效，但沒有達成任何轉移述詞。例如 attempt 0001 的輸入沒有梯度歷史，卻對其結果呼叫 backward。因此，23 次通過的比較並未測到 benchmark 的 backward 機制。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/204407/summary.json).

#### 200599: `torch.linalg.lu_solve`

人工案例包含非標準 stride 的 singleton batch 維度；以有理數重建 LU 系統並獨立求解，佐證讀取了錯誤矩陣。正式保存的 operands 是二維矩陣及一維 pivot tensor，沒有必要的三維以上 batched LU 表示。一般 singleton 覆蓋包含普通矩陣維度，不能代表 singleton batch 維度。19 個候選執行證據有效、14 次比較適用，但未觀察到觸發條件。保持 shape 的 mutation 若沒有另一個語義關係，不能自行加入缺少的 batch 結構。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/200599/summary.json).

#### 202706: `torch.autocast`

人工案例遞迴重用同一 autocast decorator；公開的狀態恢復契約與獨立的新 instance scopes 佐證 thread-local state 洩漏。正式執行完成 52 次來源執行，但沒有記錄中的同 instance 重疊機制。22 個執行證據有效的 context 案例通過比較，並獨立判為 negative；另有 30 個完成案例未滿足狀態述詞。例如 attempt 0002 恢復了兩個 scope，卻只建構一個被觀察的 target boundary，而要求是兩個。不同合法 nesting 簽章不代表已產生遞迴 decorator 重用。52 份保存的 build 都因 module 建構改變了初始化 RNG，未遵守 prompt 的明確初始化要求。Context relation 檢查的是進出狀態，因此這不構成數值確定性或初始化 prompt 遵循程度的證明。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/202706/summary.json).

#### 181723: `torch.cuda.make_graphed_callables`

人工案例在 CUDA capture 後累加不同 microbatch 的梯度；線性模型的解析輸入總和與 eager 控制佐證 static gradient buffer 重用。正式執行有 8 個執行證據有效的候選，沒有達成轉移述詞，也沒有適用比較。12 個保存成員具有 capture 與多個梯度事件，但這些成員都不符合 runtime-valid 條件；結構事實本身也不能證明資料損壞。早期候選把回傳 callable 傳給 `probe.call`，而非指定的 factory；後續限制包含 capture 合法性、精確事件數目標與缺少語義執行憑證。因此，結果混合了生成／狀態失敗與 oracle 適用性缺口，不能算成已確認的 false negative。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/181723/summary.json).

#### 204033: `torch.compile`

人工案例回傳 conjugate imaginary view，並在編譯後修改它；純量複數運算與 eager alias 傳播佐證差異。47 個正式候選沒有任何一個完成有效來源執行。其中 37 個未通過指定目標的 probe 檢查，5 個登錄不支援的 module，其餘包含不支援的 function payload、缺少名稱／key 或錯誤的參數登錄。Attempt 0000 先編譯函數，再對回傳 callable 使用 probe，但指定目標是 `torch.compile` 本身。沒有生成的 alias 傳播實驗進入 oracle，因此不能從本結果判定框架是否會偵測正確生成的案例。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/204033/summary.json).

#### 204365: `torch.nn.functional.batch_norm`

人工案例結合 MKLDNN inference input 與 strided affine tensors；純量 normalization 公式與 contiguous-affine 控制佐證錯誤。正式保存的所有 operand descriptors 都是 `torch.strided`，也沒有生成來源要求 MKLDNN。確實發生過 weight-stride mutation，例如 attempt 0018，但其執行仍為普通 dense 表示。14 個候選執行證據有效，8 次比較適用。早期生成還將非訓練 statistics 放入框架要求 trainable parameter 的字典；這些拒絕屬於 Mirage 程式契約，不代表 PyTorch API 呼叫本身非法。Benchmark 所需的 layout／affine 組合並未達成。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/204365/summary.json).

#### 201268: `torch.add`

人工案例使用 values storage 不連續的 CSR `out=`；獨立展開 CSR 索引後做純量加法，佐證錯誤寫入。正式保存的所有 operand descriptors 都是 dense／strided。Attempt 0029 雖要求 sparse COO，卻沒有建立所需的 CSR output，也沒有完成目標執行。20 個候選執行證據有效、17 次比較適用，但 dense addition 的 Pass 無法評估此 CSR 機制。兩個來自編譯圖輸出的自動 positive 標籤已撤回，兩者原本也都不在有效指標範圍。生成的 CSR 寫入行為仍未評估。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/201268/summary.json).

#### 197089: `torch.std`

人工案例結合 CPU Inductor 編譯、極小且不均勻的 FP32 值與 backward；穩定的純量標準差公式與解析導數佐證錯誤。正式執行確實在 runtime-valid 案例觀察到小／大有限輸入，因此不能只用「沒有生成極端值」解釋結果。然而，所有保存的編譯候選（0007、0008、0009、0037）都未滿足要求的 runtime predicates。Attempt 0007 對另一個 eager 呼叫做 backward；attempt 0037 達成 compiled backward，但要求一個 target boundary 時觀察到兩個，且使用普通量級輸入。23 個候選執行證據有效，卻沒有適用 oracle；主要限制是數值範圍與執行憑證。這些生成的梯度計算沒有建立獨立 positive。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/197089/summary.json).

#### 200613: `torch.Tensor.copy_`

人工案例將 CPU lazy-conjugate source 以 broadcasting 複製到 CUDA destination；明確的邏輯複數值與 resolved-view 控制，在本次 NVIDIA 環境共同佐證 view-bit 錯誤。正式保存的所有 tensor descriptors 都位於 `cuda:0`；雖觀察到一次 negative-bit 表示，卻沒有 CPU-to-CUDA 組合。17 個候選執行證據有效，且就回傳值契約獨立判為 negative；只有 2 個滿足所有 production 比較條件。其他候選常對需要梯度的 leaf 做 in-place 寫入，或使用不符合契約的參數登錄。目前保值 representation mutation 保留 device placement，因此不能補出缺少的跨裝置設定。 [執行記錄](../result/stateful-generation/771760/stage-7/formal-01/200613/summary.json).

### Edge coverage 與失敗歸因

下表以十個目標 API 為分母。要求的類別未必適用於 context factory 或特定 API；來源／follow-up witness 必須來自 runtime-valid 成員，relation-eligible 覆蓋則要求實際適用 production 比較。各欄描述不同集合，不是可相加的漏斗。

| 類別 | 曾要求 | 有效來源 | 有效 follow-up | Relation 適用 |
| --- | ---: | ---: | ---: | ---: |
| ordinary | 10 | 8 | 7 | 6 |
| zero | 10 | 8 | 7 | 6 |
| small_finite | 10 | 5 | 4 | 3 |
| large_finite | 10 | 5 | 5 | 2 |
| repeated | 10 | 8 | 7 | 5 |
| nonuniform | 10 | 8 | 7 | 5 |
| mixed_sign | 10 | 5 | 4 | 3 |
| singleton | 10 | 6 | 6 | 4 |
| broadcast | 10 | 5 | 5 | 3 |
| supported_layout | 10 | 5 | 6 | 4 |

記錄的 first-blocking-stage 計數為 unresolved runtime 177、state 54、backward 25、mutation application 3、parse 1，以及 none 174。「None」是缺省／未填寫的 stage 標籤，不代表成功；例如 std 的 provider 長度失敗沒有此欄位，應以另外保存的 provider、inspection 與 verdict 欄位為準。審閱後沒有獨立確認的正例，因此本輪無法量化 oracle miss 或 matcher miss，但確實揭露了生成、狀態及適用性瓶頸。

結果支持較窄的主張：完整由 LLM 撰寫的狀態程式可以被檢查，而且框架能偵測已提供的確定性 benchmark 觸發案例。本輪沒有證明在此預算下自主發現十個 issue、通用 API 正確性評估，或 edge prompting 的因果效益。更廣的狀態述詞、factory 使用遵循程度、layout 組合，以及可靠的數值／語義憑證，仍是覆蓋限制。歷史引導生成的 3/10 使用不同生成任務，不能視為本輪受控 baseline。

### 成本、證據與清理

正式 driver 使用 2,825 秒；各 API 生成／執行合計使用名目 3,000 秒中的 2,776.27 秒，依已記錄派工保留政策留下 223.73 秒未使用。Metadata 使用 23.25 秒，自動事後 audit 使用 19.46 秒；reviewed-v2 計算使用 2.34 秒，其隔離 iteration 共五秒。已記錄 token 用量至少為 input 1,402,625、output 140,385；std 的長度失敗 attempt 缺少完整用量，因此不能將此當作精確總計費數。

兩個重用 allocation 分別為 770948（`8gpus`，12,177 秒）與 771760（`dev`，3,526 秒），各配置兩張 H200。服務啟動分別為 217 與 185 秒；有一次 allocation 更換，沒有 allocation 內的失敗重啟。合計配置時間 15,703 秒、GPU 配置時間 31,406 GPU-seconds；這不包含更早的獨立驗證工作，也不是 GPU 使用率。[資源紀錄](../result/stateful-generation/771760/resource-ledger.json)保存 46 個 iteration 的時間／狀態；重疊的 serving 與 testing 時間不相加成 wall time。

關閉前 vLLM 回報 running 與 waiting requests 均為零。已移除本工作測試快取，關閉或明確終止兩個互動測試 session，兩個 jobs 及全部 steps 均已退出。Job 771760 與其 testing step 以 exit 0 完成；兩個服務 step 最後都經本工作的關閉流程以 signal 9 結束，並非服務正常 exit。目前沒有仍執行的本工作 job。參見[清理證據](../result/stateful-generation/771760/cleanup-final.json)與[修正 evaluator 交叉檢查](../result/stateful-generation/771760/stage-7/formal-01/corrected-evaluator-crosscheck.json)。Scheduler completion 只證明清理完成，不是 MR Pass。

## 實作變更

| 元件 | 已實作行為 | 目前邊界 |
| --- | --- | --- |
| 程式生成器 | LLM 撰寫 `build(device, seed)` 與完整的 `run(ctx, probe)`，包含所有訓練操作。 | Probe 只觀察／呼叫目標，不補上 backward、context、optimizer update、編譯或 capture。 |
| 狀態 sampler | 依 canonical 的已觀察能力選擇目標，以 API 無關的確定性排程探索聯合 predicates。 | 要求狀態不等於證明可達；final state 必須由 inspector 驗證。 |
| Operand mutation | 發現實際位置／關鍵字／容器輸入中的 tensor，保存重複 identity，檢查 storage alias、寫入與梯度連接。 | Mutation 候選選擇不接收 API 或 issue 識別資訊；拒絕不同共享 view 與未支援的寫入／梯度條件。 |
| Backend adapter | 提供 dense representation、CSR values storage、MKLDNN 邏輯實體化，以及 compiler／context／capture 與數值證據。 | 框架無關介面已以 fake adapter 測試，但未實作 TensorFlow。 |
| Inspector | 記錄實際 target／backward 路徑、checkpoint state、輸入、轉移及 mutation 證據。 | 設定旗標或生成的 assertions 不能證明覆蓋。 |
| Oracle catalog | 多個 API 共用具條件的語義家族義務與比較。 | 宣告 catalog entry 不等於實作其憑證；部分家族組合仍不可用。 |
| Replay | 重載不可變的來源／engine，檢查重建的初始值、表示及環境，不呼叫 provider。 | 不宣稱序列化任意活躍的 autograd graph。 |

在支援範圍內，初始與最終狀態包括輸入、參數、input leaf 與 parameter gradients、module state、optimizer／scheduler／scaler state、可攜 checkpoint／額外資料及執行 context。Follow-up 執行前會檢查配對所需的初始狀態；不支援的 payload 會明確標記為契約失敗。

表示路徑支援 cloning、noncontiguous storage、offset、已支援的 lazy conjugate／negative 表示、singleton stride 及已驗證的 expansion。資格依實際 operand descriptor 與語義義務決定。CSR 轉換保留索引結構，MKLDNN 比較觀察邏輯值。不會默默接受 Parameter 替換、模糊的寫入傳播、重疊 writable storage 或未驗證的 alias 保存；compile／capture 也不會默默改用 eager mutation。

## 生成協定與來源追蹤

每份 request 包含已安裝目標 API 的 specification、文件 hash 與註記、僅含 source 的 JSON 契約、執行設定、輸入類別、state predicates、relation requirements，以及合法 canonical 建立後的生成來源與既有 transition signatures。Production generator 不接收 issue 編號、benchmark reproducer 或 issue-specific transition 程式。

每個操作及其順序均由模型決定。Sampler 可以要求已觀察邊界數或 backward 事件數，但不會把計數轉成可執行動作。對 context factory，建構與 scope 進出分開觀察；compiled boundary 指實際執行且包含目標的 FX graph 邊界，不代表融合後 kernel 的數量。

Prompt 曾在開發期間修改。每次嘗試都保留精確的 `request.json`、原始 `response.json`、provider HTTP payload、來源與診斷。即使名目版本標籤相同，prompt hash 仍可區分開發修訂。凍結報告已識別最後 hash 並列出最終 prompt；早期 pilots 不是受控的 prompt 改善估計。

只允許一種語法正規化：Python 明確診斷最後一行單獨的 `}` 不匹配，且移除後整份來源可解析時，移除這一個括號。兩份來源及處理記錄均保存，不新增或改寫訓練操作。原始語法有效、正規化後候選及 LLM repair 回覆分別計數；未知 runtime exception 與疑似框架失敗不會進入修成 Pass 的流程。

## Relations、觀察值與不確定性

| Relation | 所需證據及觀察值 | 解讀 |
| --- | --- | --- |
| Representation equivalence | 相同所需初始狀態、來源、邏輯 target inputs 與 checkpoints；已認證的 effects／coordinates／數值條件；target outputs 與 root gradients。 | 只在已認證範圍比較保值表示。 |
| Materialized value effects | 相符的 schema-derived writable roles 與初始／輸入條件；返回的 target 邏輯值。 | 不主張任意 storage identity 或寫入傳播。 |
| Context restoration | 原始 scope 平衡完成且具有支援的 restoration contract；進出時的 context state。 | 未認證的同一 instance 直接重入不適用，decorator 證據另行記錄。 |
| Execution mode | 相同作者程式、已執行 compiler graph／capture 證據、對齊觀察值及相容初始狀態。 | 比較來源自行建立的 compiled／captured computation 與已認證 control；inspector 不代為建立。 |
| 其他 catalog 家族 | Differentiation、accumulation、state restoration、alias effects 均有宣告義務。 | 缺少憑證會明確保留；宣告不等於全面支援。 |

`dtype_scaled_v1` 是篩選政策，採 dtype-aware、相對量級比較，不把每個差異都當成違反；非有限值或不確定數值條件維持未評估。矩陣認證包含 conditioning bound。獨立 audit 使用自己的數學參考與不確定區間；production Suspected Failure 不等於獨立 ground truth 或已確認缺陷。

## 開發證據

十個人工固定案例在 job `770759` 經新路徑偵測；獨立 audit job `770811` 驗證其 controls。Job `770911` 檢查整合 audit 及零 provider replay。這些是先前實作版本的證據；最終 engine 後續已通過 fixtures-05 與新 allocation 的 fixtures-06 驗收。

開發使用 allocation `770948`，一張 H200 提供 Qwen/Qwen3-Coder-Next-FP8，另一張 GPU／獨立 Slurm step 執行測試。測試環境為 Python 3.14.10、PyTorch 2.15.0+cu132；serving 環境另為 vLLM 0.33.0、PyTorch 2.14.0+cu132、Transformers 5.20.2。模型 revision 為 `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`，request 使用 temperature 0 與 4096-token 輸出上限；serving context length 為 16384，最多兩個 sequences。精確環境記錄位於 `result/stateful-generation/770948/`。

開發目標為 sin／backward count 2、no_grad／restoration count 2、square／target-connected backward events 2，以及 cos／實際編譯執行。每個核心 predicate 必須有兩種不同且經 runtime 驗證的 training signatures；僅 observation detach 的差異不計入。Optimizer update 與 clearing 透過 hooks／profile events 觀察，框架不代模型執行。

早期失敗揭露了不必要的 module／optimizer 建構、parameter 物件登錄錯誤、丟棄 probe 結果、缺少 backward 連接、不支援的 checkpoint function、從文件抄入隨機初始化、多餘結尾括號，以及未建立實際執行的 compiled graph。這些是生成／契約／狀態失敗，不是 PyTorch 缺陷證據。後續 sin pilot 達到多樣性門檻，但 no_grad 仍重複同一合法序列；聯合狀態 sampler 針對這個已量測的探索缺口提供條件，不提供程式。

最終來源版本的完整回歸 `iteration-rgSvHY` 通過全部 155 項測試，沒有略過（87.953 秒）。獨立執行的 `fixtures-05` 整合與控制稽核通過全部十個案例。先前略過的稽核後來已指定資料路徑補跑。開發結果及失敗版本保留於[路線圖](roadmap-004-stateful-generation.md)與不可變 allocation artifacts。

## 指標定義

正式實驗採用每 API 300 秒。Slurm 未接受延長 allocation 後，曾暫時凍結未使用的 180 秒協議；使用者授權更換 partition 後，在任何正式生成前恢復原 300 秒預算。開發預算另行記錄。此預算包含生成、LLM repair、來源執行、mutation variants 與 MR 兩個成員。API metadata、服務啟動、事後獨立 audit 分別計時。派工保留 `max(10 秒, 最長已完成 candidate 時間 × 1.5 + 5 秒)`，報告實際未用時間；此政策不保證基礎設施不會停滯。

報告回覆數、可解析 source 數、原始語法有效數、正規化數、契約有效數、可執行 source／pair 數、runtime validity、state attainment、不同 signatures、mutation eligibility、MR applicability 與獨立 audit 結果。Valid Test Rate 為 contract-valid candidates／已解析生成來源；Executable Test Rate 為完成來源執行／contract-valid candidates。另列 runtime-valid rate，以區分狀態達成與執行。

輸入覆蓋分為 requested、實際 target-observed、relation-eligible。驗收 witness 還必須來自 runtime-valid 案例。Repeated-value 檢查有範圍限制，已檢查前綴沒有重複不代表全體唯一；broadcast 覆蓋表示相容但不同的 operand shapes，不代表所有 API 都承諾對其 broadcasting。

獨立已知問題重現與自動偵測分開計算。數學上確認為 positive、但框架未回報者不能算 detection。未達必要路徑／狀態者排除於 oracle-miss 分母，另行報告。未知 audit 不算 negative；只驗證輸出為 negative 的 audit 不能認證整個梯度比較。False Positive Rate 僅適用於獨立 adjudicated negative 子集。First-detection time 同時要求 production 回報與獨立歸因；無偵測則保留 censored 狀態。

## 階段驗收完成

G6 已通過多樣性、輸入覆蓋、案例／控制、回歸、保留集重用與凍結驗收。G7 已完成所有 API 預算、版本化獨立審閱、指標與證據檢查、實驗後 audit 回歸檢查及本工作資源清理。低發現率屬於實驗結果；此 gate 不要求最低重現率。

## 實作細節與可重用範圍

新路徑使用 `--strategy stateful`（契約版本 4）。既有 `training` 與 `properties` 路徑仍可使用，但不會向本實驗提供狀態轉移模板。尤其，舊版 `gradient_reference` property generator 並不是本次評估的生成器。

框架中立的紀錄定義位於 `contracts/capabilities.py`，後端介面位於 `backends/base.py`。`runtime/operands.py` 探索實際引數樹，並向 adapter 要求符合條件的計畫。每個計畫記錄引數結構路徑、操作與目標呼叫邊界。重複的張量物件保留其身分群組，不同的共享 view 則必須先建立 alias 條件。選擇方式為確定性的合格候選列舉，不是 API 專用優先順序。數值 edge coverage 由 LLM 在建立執行配對之前修改初始化；representation mutation 則保留關係所要求的邏輯輸入值。

`contracts/exploration.py` 依已觀察的 canonical 能力選擇目標。初始目標涵蓋 backward 呼叫數、連到目標的梯度事件、平衡的 context 恢復與實際編譯執行；聯合目標則改變目標邊界數。這些目標描述應達到的狀態，不規定可執行的操作序列。原始操作 trace 會保留，但僅用於觀察的 detach 不計入多樣性簽章。即使 optimizer-state 目標抽樣尚不完整，實際 optimizer 更新與清除梯度仍可被觀察。

`runtime/stateful.py` 執行生成的 `build` 與 `run`。`inspector/stateful.py` 驗證指定檢查點、目標執行、必要 backward 連通性與各項述詞。如果編譯狀態目標要求 backward，還必須取得連到已執行且包含目標之圖的梯度證據；程式其他位置的 eager backward 不足以滿足此條件。Context 恢復觀察原始 enter/exit 方法；語義不確定的同實例直接重入不會獲得恢復契約證明。

PyTorch representation oracle 使用一個依 pointwise、reduction、linear-algebra、indexed-value、normalization 與 writable-value 分類的能力表。分類提供語義能力，不提供各 API 的預期輸出公式。例如，矩陣比較還要求實測條件數符合限制；屬於 reduction 類別也不代表已建立目前的數值適用條件。編譯比較另使用保守的來源證明，排除實作身分觀察與未支援控制流程。因此，合法可執行程式仍可能只有覆蓋價值，沒有正確性判定。

每個 attempt 保存 `request.json`、`response.json`、`raw-program.py`、正規化後的 `program.py`、`outcome.json`、執行 trace，以及成功產生時的 `source.pt`／`followup.pt`。重播 manifest 指向不可變的引擎／來源雜湊，並重建初始資料。重播現在檢查配對兩側的語義證據，而不僅是判決。一樣的 seed 既不是確定性證明，也不代表初始狀態等價。

PyTorch adapter 使用框架專屬的檢查資訊，包括 view／版本／storage 中繼資料與編譯圖觀察。部分檢查 hook 屬於 PyTorch 內部介面，依賴版本，不能視為可攜的公開後端契約。契約要求生成的目標呼叫與 representation 建構使用公開操作。移植 adapter 時必須重新建立這些檢查保證，不能只替換 import 名稱。

不在 prompt 提供 benchmark 重現程式，不代表預訓練模型從未接觸公開 issue。開發也使用人工案例驗證 inspector 與 oracle。此實驗控制的界線是：issue ID、觸發程式與預期答案不會進入正式生成請求；這不構成預訓練資料盲測保證，也不是未知 bug 發現能力的無偏估計。

## 與原三個 Phase 的關係

**Phase 1：** 沿用先前收集的十個確定性 benchmark 及其公開 issue 參考。本任務不宣稱重新搜尋了 issue，也不宣稱發現十個新缺陷。在目前安裝環境重新執行人工程式，是為了驗證新整合；程式變更後的開發重跑，不是用來估計穩定性的統計重複實驗。

**Phase 2：** G1–G5 加入新來源契約、後端／operand 介面、由 LLM 撰寫的狀態轉移、執行證據及條件式 oracle 類別。G6 使用人工正例、獨立控制、回歸測試、生成的開發程式、輸入覆蓋與保留 API 檢查，驗證整體組合。偵測人工案例只驗證框架辨識已提供條件的能力，不是 LLM 發現率。

**Phase 3：** G7 已在預先宣告的每 API 預算下評估凍結的完整來源生成器。開發程式——包含失敗後轉為開發用途的保留集——不納入正式發現率分母。十個正式目標 API 使用相同的生成／mutation／oracle 機制；issue 專屬歸因在執行後進行，不向生成器提供回饋。

第一輪保留集（`heldout-01`）失敗：sigmoid 產生 8 個可解析來源，但只有 canonical 的執行證據有效，沒有達成轉移述詞；Tensor.mul 收到 8 次回應、產生 7 個來源，4 個執行證據有效的案例中有 3 個達成轉移。Sigmoid 的失敗重用了已消耗的上游圖狀態。修訂後的通用說明解釋圖所有權，不提供操作序列，且只在要求 backward 時適用。失敗結果保留為初始泛化能力有限的證據。替換保留集的預算為每 API 300 秒／24 次回應，因此後續結果不能單獨歸功於 prompt 調整或增加的生成機會。

## 凍結請求原文範例

System message 原文為 `Follow the supplied testing contract. Return only the requested JSON object.`。User message 為 `request.prompt + "\nContext:\n" + json.dumps(request.context)`。修復請求另外提供先前 assistant 回覆，以及 `Repair this diagnosed problem only:\n` 加上診斷。因此，下列 prompt 只是完整請求的一部分，不能取代保存的 context。

以下為正式實驗第一個 `torch.Tensor.scatter_add` canonical prompt 原文（forward 路徑、`stateful-2-edges`）。其 prompt SHA-256 為 `2e3f14399ae4ce9510783fff9cb6852b402897402f1233f3a4723e458dc94a0d`。API 名稱、必要路徑、述詞、先前簽章與 canonical 來源，依已記錄排程及執行證據而變動。為保留實際送入模型的內容，prompt 原文維持英文。

```text
Target API: torch.Tensor.scatter_add. Generate a small deterministic test exercising this exact API.
Use explicit tensor/zeros/ones/arange/linspace initialization. Never sample random values, even with a seed; do not copy random examples from the API documentation.
Required target backward: False. Requested state predicates: [].
For an eager test, use the value RETURNED by probe.call for subsequent computation; calling the target separately and discarding probe.call output does not test its backward. If backward is not required, a legal target exercise needs no training loop or optimizer.
Return one JSON object with exactly source (a complete Python source string). Write concise code; omit explanatory comments and redundant snapshots.
1. Define build(device, seed) returning a dict with inputs and parameters dictionaries. Optional fields: modules, optimizer, scheduler, scaler and checkpoint_data. Use explicitly initialized tensors. A seed does not replace initialization. Replace all retained random module values. Registered module parameters must be the same objects as entries in parameters, not the tensors used to construct Parameters. Omit unused optional fields. optimizer/scheduler/scaler must be real objects or None, never dictionaries. inputs/parameters/modules are dictionaries and may be empty. Functions belong in source, not checkpoint_data or returned observations.
2. Define run(ctx, probe). You own all operations and their ordering: the framework inserts no backward, clearing, optimizer step, context, compilation or replay. Without canonical_source, write a minimal legal canonical use and the required backward. canonical_source is a starting point, not a mandatory program structure. Rewrite it as needed to reach the supplied predicates at their named checkpoints. Counts are exact, not lower bounds. Choose a different legal training transition structure from prior_transition_signatures; different values or observation copies alone do not establish diversity. No fixed action sequence is required.
3. For an eager target invocation, call probe.call(target_callable, *args, **kwargs) and use its returned result. ONLY the configured target goes through probe.call; call other operations directly. Tensor methods require their bound receiver or an explicit receiver with the unbound method. Use a returned context manager/callable according to its public specification. For required backward, differentiable target operands need gradient history before the call and the observed result must reach backward; an unrelated downstream parameter or separate unobserved call is insufficient.
4. A compiled_execution=True goal REQUIRES source-owned torch.compile and an actual invocation of the resulting callable. Call the target directly inside your compiled function; keep probes outside tracing/capture and invoke the compiled callable directly. For the supported compiled inspection contract, keep backward calls outside the compiled body. A compile flag or tracing-only call is not runtime evidence. Do not fake probe fields.
5. run returns a dictionary of observable tensors/primitives, never expected answers, modules, optimizers or the whole ctx. Return tensors directly; the inspector snapshots them. Optional probe.mark(name, data) adds unique intermediate checkpoints. The inspector owns initial/final; never use those reserved names. Probe has only call and mark methods. Expose related alias partners where an effect relation needs them.
6. No random sampling, NumPy/Tensor.numpy(), external I/O, assertions or hidden globals. Use public APIs. Keep each tensor within 4096 elements and at most four updates. Keep differentiable views in run. Generate legal values/shapes for the assigned input category. Replace canonical inputs that do not demonstrate it, consistently for both MR members; actual target values, not comments describing them as small/large, establish coverage. If a requested capability is unavailable, preserve legal coverage and do not invent evidence.

Consider edge cases from the assigned coverage category as well as ordinary
inputs across the run. Choose legal, explicitly initialized inputs using the
supplied API specification and backend capabilities.
For numerical coverage, consider exact zeros, repeated and nonuniform values,
mixed signs where legal, and small nonzero or large finite magnitudes relative
to the dtype's representable range. Derive scales from dtype metadata rather
than assuming one fixed floating-point format. Record whether conversion or
execution changes the intended magnitude category.
For structural coverage, consider legal singleton dimensions, batch dimensions,
broadcast-compatible unequal shapes, and alternative supported layouts. Use
empty dimensions only when permitted by both the API and the selected relation.
When relationships between operands or returned values matter, expose the
observables required by that relation. Include nontrivial, distinguishable
values where legal; do not assume that every edge case should fail.
Select a bounded subset for this program rather than combining every edge case.
Preserve required domains, initial conditions, differentiation coordinates and
source/follow-up input relations. If a category is not applicable, report that
limitation instead of forcing an invalid input or inventing an expected answer.
You still own the training-state operations and their ordering. The framework
may mutate tensor representations but will not construct your training history.
```

要求 backward 時，請求使用 `stateful-3-edges`，並加入下列通用圖生命週期說明；它不提供狀態轉移序列。

```text
Graph lifetime is your responsibility: backward consumes saved graph state unless you explicitly preserve it. Losses sharing an upstream differentiable intermediate also share that graph, even if their target outputs were computed separately. Choose legal retention or recomputation yourself; reconstructing only a downstream target output does not rebuild a consumed upstream graph. A backward_count goal counts actual backward/grad invocations, whereas target_backward_count counts target-connected gradient events.
```

Context 包含已安裝 API 的 docstring 與雜湊、設定、canonical 來源或 null、狀態述詞與量測單位、輸入類別定義、relation requirements、prompt 版本／雜湊，以及先前轉移簽章。小／大有限輸入的定義分別為 `sqrt(finfo(dtype).tiny)` 與 `sqrt(finfo(dtype).max)`；這些是覆蓋門檻，不是預期輸出或 bug 標籤。Context factory 不要求 backward 路徑；其他正式提案採一次 forward、兩次 backward 的排程，修復則保留被診斷的提案。

第一個 API 的已安裝 docstring 只有簽名與指向 `scatter_add_` 的參考；保存的 specification 沒有進一步取得被參考的文件。這是提供給模型的規格限制，解讀生成失敗時應一併考慮。

凍結的 prompt-template 來源 SHA-256 為 `725bb45863b816d3100b1d12ae228cb989dd72c0f1dfabe2addc77624c12d98a`；語法正規化來源雜湊為 `57a46d4ab5c96cc2ad4bfcc4b47ed75be8e4039ea404988c1c741748259a02a0`。它們識別實作檔案，各 request hash 則識別具體 prompt。

證據：[完整第一份請求](../result/stateful-generation/771760/stage-7/formal-01/203078/attempt-0000/request.json)、[backward 請求](../result/stateful-generation/771760/stage-7/formal-01/203078/attempt-0001/request.json)、[凍結協議](../result/stateful-generation/771760/stage-7/formal-01/protocol.json)。

## 最終開發驗收證據

這些是開發驗收資料，不納入正式發現率；各項均對應最終 production 來源。驗收可以證明狀態可達，即使該案例的正確性關係仍不適用。

| 檢查 | 目標 | 回覆 | 解析成功 | 執行證據有效 | 達成狀態 | 簽章 | 適用比較 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot-11 | sin | 52 | 52 | 16 | 15 | 2 | 0 |
| pilot-11 | context | 56 | 56 | 23 | 22 | 4 | 23 |
| pilot-11 | square | 46 | 46 | 18 | 17 | 6 | 0 |
| pilot-11 | compiled | 24 | 24 | 8 | 7 | 3 | 7 |
| edges-04 | sin-base | 24 | 24 | 21 | 0 | 0 | 21 |
| edges-04 | sin-edges | 24 | 24 | 23 | 0 | 0 | 0 |
| edges-04 | add-base | 24 | 24 | 24 | 0 | 0 | 22 |
| edges-04 | add-edges | 24 | 24 | 24 | 0 | 0 | 0 |
| heldout-02 | tanh | 24 | 23 | 3 | 2 | 2 | 0 |
| heldout-02 | sub | 24 | 24 | 9 | 8 | 1 | 5 |

輸入驗收在執行證據有效的來源／follow-up 中合計觀察到十種已定義類別；這不代表每個 API 支援所有類別，也不代表每種類別都有適用 oracle。各驗收亦通過已記錄的零 provider、配對兩側語義證據重播。新的保留目標為 `torch.tanh` 與 `torch.Tensor.sub`；sigmoid 與 Tensor.mul 已在失敗的第一輪被使用。不同預算與開發修訂使結果無法支持 prompt 改善的因果結論。

[Pilot 證據](../result/stateful-generation/770948/stage-6/pilot-11/model-checks.json)、[輸入覆蓋證據](../result/stateful-generation/770948/stage-6/edges-04/model-checks.json)、[保留集證據](../result/stateful-generation/770948/stage-6/heldout-02/model-checks.json)。新的 `dev` allocation 在凍結最終 300 秒協議之前，透過 [fixtures-06](../result/stateful-generation/771760/stage-6/fixtures-06/independent-controls.json) 重做十個案例與控制檢查；全部被偵測，並得到獨立佐證。
