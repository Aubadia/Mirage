# PyTorch 正確性：第 2–3 階段評估

本評估於 2026-11-13 完成。第 2 階段檢出 **10/10** 個確定性基準；凍結的 LLM 實驗從 **1,154 個生成候選**中重現 **3/10** 個已知問題。下列獨立審核指標僅適用於明確完成審核的子集。所有本任務所屬 Slurm 程序與模型服務均已退出。

詳細說明見[逐案重現分析](#per-issue-analysis)；[路線圖](roadmap-003-pytorch-correctness-benchmark.md#phase-3-details)記錄實作流程與完整提示詞。

## 範圍與證據

十個固定輸入的上游基準及第 1 階段佐證，記錄於[重現報告](../docs/pytorch-correctness-reproduction.md)。第 2 階段衡量的是：給定觸發條件後，Mirage 能否辨識失敗。第 3 階段衡量的是：不提供問題敘述、重現程式、預期答案或問題編號，真實模型能否產生觸發條件。兩者衡量不同能力。

本評估在既有訓練程式策略之外加入 `--strategy properties`，重用已安裝 API 探查、隔離的 provider、執行 worker、追蹤紀錄、數值比較及引擎快照。LLM 建構輸入與小型模組／函式；經審查的執行策略建構來源與後續執行，並負責正確性判定。這是受引導的性質測試程式生成，而非不受限制的完整訓練迴圈生成。既有訓練策略仍可使用。

## 可重用的測試性質

| 類別 | 前提與輸入關係 | 觀測值與輸出關係 |
| --- | --- | --- |
| 邏輯值實體化 | 經認證的 API、支援的張量布局、保持邏輯值與選項；可寫入的 `out` 不得與輸入共用儲存空間。Batch normalization 須使用推論模式及累積統計量。 | 比較原始目標輸出與實體化執行。scatter 加總、LU 求解、batch normalization、稀疏加法與複製的額外語義參考須與對照組一致；LU 參考還須滿足定義該解的線性方程。 |
| 解析梯度 | 小型、可微分的實數運算元；std/var 須為全元素純量歸約，具有正自由度與非零變異數；行列式輸入須為條件良好、無批次且非奇異的方陣。須實際執行目標反向傳播；編譯模式須有計算圖執行紀錄。 | 將純量輸出及第一個運算元的導數，與 float64 公式及實體化 eager 對照組比較。運算元為新建立的獨立微分座標，不測試既有建構歷史。 |
| 上下文還原 | 支援的上下文／裝飾器、有界遞迴、相同參數與初始執行緒區域狀態；無執行緒並行或競爭。 | 重入同一裝飾函式後，梯度模式、推論模式、autocast 啟用狀態、dtype 與快取設定須精確符合進入前狀態及新建上下文對照組。 |
| 捕捉圖的梯度累積 | 小型、經認證的確定性 FX 圖，無可變緩衝區；2–4 個簽章相同的 CUDA microbatch；相同初始參數，累積前將梯度設為 None。 | 比較同一 microbatch 加總結束時，捕捉圖與 eager 的參數梯度。eager 是差異式參考，並非獨立數學證明。 |
| 編譯後回傳別名 | 相同初始邏輯輸入、實際執行編譯圖、初始回傳值一致；選定的 eager 回傳張量與另一回傳張量共用儲存空間；`add_` 修改合法且為有限值。 | 相同修改後，eager 與編譯執行的回傳值須一致。檢查的是可觀察別名行為，不是跨執行的原始位址相等。 |

預設篩選容差為 `atol=1e-5, rtol=1e-4`；解析梯度使用 `atol=0, rtol=1e-4`，避免抹去極小尺度的違反。上下文狀態採精確比較。誤差計算會提升浮點／複數值的精度，並檢查形狀與 dtype。非有限值或語義參考與對照組不一致時，結果為**正確性未評估**，而非失敗。容差是篩選政策，不是通用誤差上界或缺陷證明。

確定性表示循環涵蓋原始表示、stride-two、非零位移、negative bit、conjugate bit、singleton stride 與 expanded 表示。操作必須對該運算元合法且保持邏輯值，否則判為不適用。目標呼叫記錄實際中繼資料；未支援的 API／性質組合不會取得正確性通過判定。CSR values 與不透明 MKLDNN 張量具備明確轉接器，但並非支援所有稀疏或量化布局。

快照重播保存輸入、程式與校驗值，不呼叫 LLM。MKLDNN 使用邏輯資料加布局標記。張量序列化會遞增版本計數器，因此重播比較所有已記錄的中繼資料，但排除絕對版本數；這些性質建立新計算圖，不測試已保存計算圖的版本計數器行為。其他中繼資料變化會使重播遭拒。此能力刻意限於較窄範圍，不涵蓋任意 autograd 歷史還原。

## 實驗協定

- 使用一個本機 Qwen/Qwen3-Coder-Next-FP8 服務，temperature 0、top-p 1，最多生成 4096 tokens；已準備的模型版本記錄於服務產物。
- 依序對每個 API 執行；另行計時的已安裝 API 中繼資料探查完成後，生成與執行合計最多 300 秒。剩餘時間低於 45 秒時停止派送新請求；該保留時間明確報告，不計作已生成工作。
- 使用通用 API 類別設定與確定性表示循環，嘗試索引會改變生成 seed／設定。正式程式不匯入基準測試資料。請求產物完整保留已安裝的公開文件字串及其範例。
- 對建構函式的靜態契約錯誤最多修復兩次；執行錯誤、判定結果與基準觸發檢查不提供特定問題的修復提示。
- 每個候選在隔離 worker 中執行一次。計時失敗、不支援程式與非法輸入，均與判定違反分開。未以重複試驗估計機率式問題重現率。
- 第 2 階段驗證後凍結實作與提示詞。初始冒煙／驗證失敗保留為開發證據，不納入凍結的第 3 階段樣本。
- 使用一個 Slurm 配置、兩張 GPU，分別供服務與測試步驟使用。每次迭代的程式快照不可變；編譯快取為工作專屬的可拋棄暫存。

確定性基準不代表模型在不同硬體／軟體環境中的生成也完全確定。此處記錄單次實際實驗，沒有跨獨立生成實驗的信賴區間。

## 指標定義

`generated` 計算已解析且包含原始程式的回應數。`contract_valid / generated` 為靜態有效測試率，不代表執行時合法性。`completed target executions / contract_valid` 為可執行測試率。原始狀態另外區分正常結束但不適用的程式，以及實際完成的目標執行。

只有在獨立審查生成案例的基準機制與判定證據後，才確認問題重現。僅符合結構觸發條件者另記為 `benchmark_consistent_detection`。第 2 階段偵測率以十個獨立納入的正向基準案例為分母；第 3 階段敏感度則需要獨立裁定的生成正例。誤報率須以獨立裁定的負例計算 FP/(FP+TN)；未裁定案例排除，絕不預設為負例。若計算已觀察的錯誤發現比例，其公式為 FP/(TP+FP)，不可誤稱為 FPR。

首次偵測時間自該 API 預算開始計算，包含生成、修復與執行。這些時間指自動判定器的偵測事件，其問題歸屬於後續獨立確認；不包含後續審查延遲。沒有偵測的目標，依實際經過時間作設限處理，不填入虛構偵測時間。生成案例數包含重複；另提供唯一原始碼指紋供檢視。

## 已完成的執行結果

Slurm 工作 **763298** 測試 PyTorch **2.15.0+cu132**，環境為 Python 3.14.10、NVIDIA H200。獨立模型服務環境使用 vLLM 0.33.0／PyTorch 2.14.0+cu132；服務端版本**並非**受測框架版本。模型快照為 `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`。

第 2 階段最終檢查透過正式性質測試 worker，檢出 **10/10** 個納入案例。完整套件於 **63.277 秒內通過 70 項測試**。兩項已保存輸入的檢查式重播——negative view 及帶跨步 affine values 的 MKLDNN——在環境未變、provider 呼叫為零的條件下完成。凍結實驗之後，另通過 **12 項性質測試**，包含普通編譯 view 與相同 microbatch 的 CUDA-graph 負向對照，以及 **4 項基準／審核測試**。這些結果驗證的是已測案例與檢查機制，並非所有可能程式的正確性證明。

凍結實驗產生 **1,154 個候選**，包含 **661 個修復回應**，以及 **748 個依 API 區分的原始碼指紋**。可解析程式採 AST 指紋，其餘採原始碼雜湊。十個 API 預算均已完成。

| 問題 / API | 生成數 | 靜態有效 | 目標已執行 | 適用比較 | 疑似失敗 | 首次基準偵測 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [203078](https://github.com/pytorch/pytorch/issues/203078) `torch.Tensor.scatter_add` | 106 | 32 | 1 | 1 | 0 | 未偵測（設限） |
| [204407](https://github.com/pytorch/pytorch/issues/204407) `torch.linalg.slogdet` | 122 | 29 | 10 | 9 | 3 | 8.78 s |
| [200599](https://github.com/pytorch/pytorch/issues/200599) `torch.linalg.lu_solve` | 132 | 38 | 20 | 18 | 0 | 未偵測（設限） |
| [202706](https://github.com/pytorch/pytorch/issues/202706) `torch.autocast` | 97 | 41 | 34 | 34 | 22 | 10.73 s |
| [181723](https://github.com/pytorch/pytorch/issues/181723) `torch.cuda.make_graphed_callables` | 161 | 0 | 0 | 0 | 0 | 未偵測（設限） |
| [204033](https://github.com/pytorch/pytorch/issues/204033) `torch.compile` | 98 | 35 | 2 | 0 | 0 | 未偵測（設限） |
| [204365](https://github.com/pytorch/pytorch/issues/204365) `torch.nn.functional.batch_norm` | 108 | 27 | 19 | 19 | 0 | 未偵測（設限） |
| [201268](https://github.com/pytorch/pytorch/issues/201268) `torch.add` | 163 | 32 | 17 | 17 | 0 | 未偵測（設限） |
| [197089](https://github.com/pytorch/pytorch/issues/197089) `torch.std` | 62 | 34 | 22 | 22 | 2 | 未偵測（設限） |
| [200613](https://github.com/pytorch/pytorch/issues/200613) `torch.Tensor.copy_` | 105 | 42 | 5 | 5 | 1 | 33.97 s |

「適用比較」表示實際產生 Pass 或 Suspected Failure。「目標已執行」要求目標完成執行，但後續判定仍可能不適用或未評估。總計 226 個 worker 完成、130 個目標完成、125 個適用比較、97 個 Pass、28 個疑似失敗。生成數包含契約無效程式與重複嘗試。

- **靜態有效測試率：**310/1,154 = **26.86%**。
- **可執行測試率：**130/310 = **41.94%**，分母為靜態有效程式。以全部生成候選為分母，目標執行比例為 11.27%。
- **經審查的已知問題重現率：** **3/10 = 30%**，包含 204407、202706、200613。
- **第 2 階段正向基準偵測率：** **10/10 = 100%**。
- **獨立生成案例審核：**TP=19、FN=0、FP=0、TN=50。敏感度為 **19/19 = 100%**，FPR 為 **0/50 = 0%，僅適用於此審核子集**。只有 69/1,154 個候選完成獨立裁定；其餘 1,085 個包含生成失敗、不支援案例及未裁定觀測。28 個疑似案例中有九個未被此保守審核裁定。

獨立正例子集包含 17 個與已知基準相關的案例，以及兩個額外 std 異常；後者不增加已知問題數。接近數值門檻的差異因不確定性檢查而保持未裁定；相同 dtype 的複製及上下文狀態採精確語義比較。此子集不支持整體零誤報或無漏報的主張。

### 問題歸屬與額外觀察

原始凍結比對器只計入 **2/10 個問題**，其 copy 規則僅識別 conjugate source。獨立審查確認，[問題 200613](https://github.com/pytorch/pytorch/issues/200613) 明確包含 negative view 及切片布局的暫存複製案例。因此，生成的 CPU negative strided source 屬於同一問題：邏輯值 11..22 複製到 CUDA 後成為 -11..-22，最大絕對誤差 44。修正後的事後比對器與獨立 copy 判定支持 **3/10**。原始 `frozen-matcher-evaluation.json` 保留不變；僅修正歸屬，未改變生成、執行、來源輸入或正式程式。

第一個 slogdet 案例使用 `[[2,1],[3,4]]`，預期導數為 `[[0.8,-0.6],[-0.2,0.4]]`，實際非對角元素卻互換，最大誤差約 0.4。第一個 autocast 案例深度為 2，返回後 autocast 仍啟用、快取停用，未還原為原本的停用與啟用狀態。兩案均有通過的對照組與獨立參考。

std attempts 38 與 52 使用 1..12、negative-bit 運算元及 Inductor。標準差正確，但梯度符號反轉；FP64 與 FP32 的最大誤差皆約 0.27735。Python 純量數學獨立佐證兩例差異。它們**未重現問題 197089 的極小輸入下溢條件**，保留為一組相關候選，根因與新穎性尚未確認，不視為兩個新確認缺陷，也不增加十案基準的成功數。

<a id="per-issue-analysis"></a>
### 逐案分析：重現與未重現

本節區分觀察結果、程式碼支持的阻礙，以及尚未測試的改善方向。十個問題均已在第 1 階段固定案例與第 2 階段正式 worker 整合中重現。以下未重現專指**第 3 階段自動生成**，不表示缺陷已修復，也不表示判定器無法檢出。

[第 3 階段路線圖](roadmap-003-pytorch-correctness-benchmark.md#phase-3-details)記錄實際提示詞、五個類別後綴、完整請求範例、修復訊息與取樣規則；[第 2 階段說明](roadmap-003-pytorch-correctness-benchmark.md#phase-2-details)界定實作範圍。模型並未取得問題後改寫 MRE，而是依經審查的性質協定生成建構函式，再由取樣器注入表示。

#### A. 各執行階段的分母

| 問題 | 生成數 | 靜態拒絕 | 執行失敗 | Worker 完成 | 目標完成 | 通過 | 疑似失敗 | 不適用 | 未評估 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 203078 | 106 | 74 | 18 | 14 | 1 | 1 | 0 | 13 | 0 |
| 204407 | 122 | 93 | 0 | 29 | 10 | 6 | 3 | 19 | 1 |
| 200599 | 132 | 94 | 2 | 36 | 20 | 18 | 0 | 18 | 0 |
| 202706 | 97 | 56 | 7 | 34 | 34 | 12 | 22 | 0 | 0 |
| 181723 | 161 | 161 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 204033 | 98 | 63 | 33 | 2 | 2 | 0 | 0 | 2 | 0 |
| 204365 | 108 | 81 | 0 | 27 | 19 | 19 | 0 | 8 | 0 |
| 201268 | 163 | 131 | 2 | 30 | 17 | 17 | 0 | 13 | 0 |
| 197089 | 62 | 28 | 0 | 34 | 22 | 20 | 2 | 12 | 0 |
| 200613 | 105 | 63 | 22 | 20 | 5 | 4 | 1 | 15 | 0 |

不適用案例既包含目標呼叫前轉換前提不成立，也包含目標完成後 MR 適用性檢查失敗。因此整欄不能解讀為有效執行卻被判定器漏報。完整執行漏斗如下：

```text
1154 個生成候選
 ├─ 844 個靜態契約／語法拒絕
 └─ 310 個靜態有效
     ├─ 84 個執行失敗
     └─ 226 個 worker 完成
         ├─ 96 個在記錄目標完成前，表示轉換即不適用
         └─ 130 個目標完成執行
             ├─ 4 個 MR 不適用（2 個 LU 殘差檢查、2 個編譯別名檢查）
             ├─ 1 個正確性未評估（slogdet 參考／對照組不一致）
             └─ 125 個適用比較
                 ├─ 97 個通過
                 └─ 28 個疑似失敗
```

`execution_failed` 表示整個 worker 失敗；錯誤可能出現在建構函式、快照或參考計算，單憑標籤無法確認 PyTorch 目標缺陷。同樣，`api_enter` 與完成的 `target_calls` 是不同證據。下列錯誤數計算**每個候選第一個回報原因**，不是所有潛在問題。例如被 `nn.Module` 別名規則拒絕的程式也可能含 `manual_seed`，兩者若同時計數會重複計入同一候選。

#### B1. 203078 — scatter_add：未形成 expanded-index 核心的觸發條件

**必要觸發條件。**第 1 階段以 dim=0 scatter、index stride `(1,0)`、目的張量寬於 index，進入專用 expanded-index 路徑。任意非連續張量不等同此組合。案例預期列加總為 `[16,0,16]`，實際為 `[16,16,0]`。

**生成結果。**106 個候選中，74 個遭靜態拒絕：61 個含 `manual_seed`、11 個頂層結構不支援、2 個 SyntaxError。32 個進入 worker，其中 18 個將整數 dim 放在 args 首位，漏掉 Tensor 方法接收者，導致 descriptor TypeError。其餘 14 個 worker 完成，但 13 個在轉換檢查即不適用；只有 attempt-0003 完成目標並取得 Pass。

[attempt-0003 原始碼](../result/pytorch-correctness/763298/phase3/203078/attempt-0003/program.py) 使用 dim=1，destination/index/source 形狀均為 `(2,3)`。取樣器將目的張量改為 stride `(6,2)`，index 仍為連續的 `(3,1)`；目的張量也未寬於 index。[執行摘要](../result/pytorch-correctness/763298/phase3/203078/attempt-0003/execution/summary.json)佐證這些條件。

**未重現原因。**唯一完成的目標測試了另一種合法跨步設定，未達到基準要求的 index 狀態。13 個目標呼叫前不適用案例，包括 4 次展開會改變邏輯值、4 次對實數套用 conjugate、3 次缺少 singleton 軸、2 次對整數套用 negative。循環不會重新設計 index 值使其可展開，也不保證選到 index 運算元。

**失敗分類。**接收者契約與生成錯誤，加上觸發條件覆蓋缺口。沒有證據顯示判定器漏掉正向執行。運算元角色對應、生成可展開值及執行契約修復，均是可一般化的後續方向，本輪尚未實作或測試。

#### B2. 204407 — slogdet：非對稱輸入、negative-bit 注入與解析判定

**首次成功鏈。**attempt-0003 因 `manual_seed` 遭拒。[attempt-0004 修復請求](../result/pytorch-correctness/763298/phase3/204407/attempt-0004/request.json) 保留 seed/attempt=3、negative、subject=0、eager，只要求修正該契約違反。LLM 提供 requires_grad=True 的固定實矩陣 `[[2,1],[3,4]]`，原始碼未建構 `conj().imag`。Worker 套用通用 negative-bit 操作並保持邏輯值；中繼資料為 shape `(2,2)`、stride `(4,2)`、offset 1、`neg=True`，再以新葉節點微分座標計算 logabsdet 梯度。

```text
A = [[2, 1], [3, 4]]，det(A)=5
正確 d(log|det(A)|)/dA = A^(-T)
expected = [[ 0.8, -0.6], [-0.2, 0.4]]
observed = [[ 0.8, -0.2], [-0.6, 0.4]]   （顯示為四捨五入後數字）
max absolute error ≈ 0.40000002
```

誤差明顯超過 `atol=0, rtol=1e-4` 門檻，實體化 eager 對照組則與參考一致。獨立 Python Fraction 消去法由保存矩陣重建反矩陣轉置，並未直接以正式判定當作真值。[原始碼](../result/pytorch-correctness/763298/phase3/204407/attempt-0004/program.py)、[目標及梯度證據](../result/pytorch-correctness/763298/phase3/204407/attempt-0004/execution/summary.json)與[可讀觀測值](../result/pytorch-correctness/763298/phase3/reviewed-observations.json)保留完整成功鏈。首次偵測為 8.78 秒。

**完整 API 預算。**122 個候選包含 93 個靜態拒絕：41 個 manual_seed、50 個 SyntaxError、2 個屬性寫入限制。其餘 29 個 worker 均完成，19 個未滿足轉換前提。10 個完成目標產生 3 個疑似失敗、6 個 Pass、1 個未評估。疑似案例為 attempts 4、24、67；後兩案在 Inductor 執行且具圖執行證據，最大梯度誤差約 2.63415。

**為何一個 negative-bit 案例通過。**[attempt-0046](../result/pytorch-correctness/763298/phase3/204407/attempt-0046/program.py) 使用對稱矩陣 `[[2,1,.5],[1,3,1],[.5,1,4]]`，雖有 negative bit 仍取得 Pass。可逆對稱矩陣的反矩陣也對稱，故反矩陣與其轉置相同，無法暴露此處觀察到的轉置型差異。這說明到達目標狀態不保證產生可觀察違反；該 Pass 不是漏報證據。

[attempt-0121](../result/pytorch-correctness/763298/phase3/204407/attempt-0121/execution/summary.json) 因對照組與參考梯度差異約 `1.043e-7`，未通過嚴格 zero-atol 比較，保持未評估。其運算元為跨步表示、`neg=False`。保守檢查減少可評估覆蓋，卻避免將參考本身不一致的比較視為已確認目標違反。

**成功歸因。**LLM 選取具辨識力的非對稱值，取樣器注入表示，執行證據確認反向傳播，數學判定器檢出差異。不能單獨歸功於 LLM 自主生成完整問題 MRE。

#### B3. 200599 — lu_solve：覆蓋一般系統，未覆蓋 singleton-batch 跨步

**必要觸發條件。**問題涉及帶大小為一批次軸的 LU 矩陣索引。第 1 階段使用 `LU[:, None]`，shape `(2,1,2,2)`、stride `(4,2,1,2)`。非連續本身不足以識別此條件，因普通 column-major LU 也未必符合 row-major 連續性。

132 個候選中，94 個遭靜態拒絕：56 個 manual_seed、12 個 randn、26 個語法錯誤。另有 2 個執行失敗，分別為 complex arange 不支援及非法 pivots。36 個完成 worker 中，20 個完成目標：18 個 Pass、2 個因 `reference_residual=False` 不適用；另 16 個在目標前即不符合轉換前提。

[attempt-0001](../result/pytorch-correctness/763298/phase3/200599/attempt-0001/program.py) 明確建構可逆 4×4 矩陣，呼叫 `lu_factor`，rhs 為 `(4,3)`。[執行中繼資料](../result/pytorch-correctness/763298/phase3/200599/attempt-0001/execution/summary.json)顯示 LU shape `(4,4)`、stride `(1,4)`，沒有批次軸。參考解與重建系統殘差均通過，但此合法結果未測試 singleton-batch 定址。

取樣器只修改**既有**大小為一的軸，不插入 `[:,None]`。四個已完成 worker 的拒絕原因為 `No singleton dimension`，最終結構比對沒有重現。attempts 38、89 完成目標，但替代解未通過殘差檢查；缺乏進一步目標缺陷證據，不能將它們計為漏報。

**失敗分類。**形狀／狀態覆蓋不足，另有生成失敗與參考檢查消耗預算。證據未顯示已確認觸發的執行被錯誤判為 Pass。

#### B4. 202706 — autocast：重入執行暴露退出狀態不變量

97 個候選中，56 個遭靜態拒絕：29 個 manual_seed、27 個語法錯誤；7 個執行失敗，包括錯誤 RNG 狀態設定、未定義變數、越界 view 及邏輯張量預算。34 個完成目標與判定的案例產生 22 個疑似失敗、12 個 Pass。

manual_seed 修復後，[attempt-0003](../result/pytorch-correctness/763298/phase3/202706/attempt-0003/program.py) 指定 CPU、BF16、enabled=True、cache_enabled=False、depth=2。建構函式也建立張量，但此判定器觀察執行緒區域上下文狀態，而非張量數值。`context_restore()` 負責建立裝飾函式遞迴及新建上下文對照組。

```text
進入前／新建上下文對照組：grad=True, inference=False, amp=False,
                        dtype=bfloat16, cache=True
重入同一裝飾函式後：      grad=True, inference=False, amp=True,
                        dtype=bfloat16, cache=False
```

[摘要](../result/pytorch-correctness/763298/phase3/202706/attempt-0003/execution/summary.json)記錄 `amp` 與 `cache` 的精確不一致，對照組則一致。首次偵測為 10.73 秒。這是離散狀態違反，不是容差選擇放大的微小浮點差異。

12 個 Pass 全部使用 depth=0。22 個疑似案例分布為：depth 1 有 2 個、depth 2 有 15 個、depth 3 有 4 個、depth 4 有 1 個。執行樣本區分了非重入與重入情境，但不證明所有 depth>0 的 autocast 參數組合都失敗。

**成功歸因。**LLM 提供合法建構選項與深度，框架提供可重用的遞迴／還原協定。相同協定也支援其他登錄的 grad 與 inference 上下文。其設計參考了基準，因此不是 LLM 無先驗的獨立發現。

**審核限制。**保守獨立審核僅裁定 22 個疑似案例中的 13 個，9 個保持未知。建構函式允許清單會拒絕未列入的呼叫，例如 [attempt-0008](../result/pytorch-correctness/763298/phase3/202706/attempt-0008/program.py) 的 `min(4, seed % 5)`。這表示審核器無法認證建構函式，**不表示已證明周遭上下文狀態遭破壞**。另有八個 Pass 未審核，因此獨立真值為 13 TP、4 TN，而非正式判定的 22 與 12。

#### B5. 181723 — CUDA graph 梯度累積：前端拒絕使判定器無法評估

基準要求對同一捕捉模組的至少兩個不同 microbatch 執行反向傳播，從 `.grad=None` 累積，以區分正確 `g1+g2` 與錯誤 `2*g2`。只生成 `nn.Module` 類別不足以建立此狀態。

161 個候選全被靜態拒絕：131 個先遇到 `Only local torch.nn.Module classes are supported`，30 個為 SyntaxError。沒有 worker 執行目標或捕捉後反向傳播，也沒有 Pass／Suspected Failure。無法由未執行程式估計判定器漏報率。

[attempt-0000](../result/pytorch-correctness/763298/phase3/181723/attempt-0000/program.py) 使用：

```python
import torch.nn as nn
# local class inside build:
class SmallNet(nn.Module):
    ...
```

此別名是合法 PyTorch 語法，但驗證器要求基底的 `ast.unparse(...)` **恰等於** `torch.nn.Module`，不解析匯入別名。原始碼也有 `manual_seed`，且 `nn.Linear` 會執行預設參數初始化。只修別名不足以建立確定性契約合規；第一個拒絕遮住其他潛在問題，因此不能主張改一行即可使 161 個候選全部可執行。

第 2 階段手寫建構函式使用完整 `torch.nn.Module` 拼寫與明確參數值，證明可檢出該正向案例。後續相同 microbatch 的負向對照取得 Pass；兩項結果都未消除第 3 階段前端限制。

**失敗分類。**框架契約與 AST 驗證限制，加上原始碼語法錯誤。結果不證明 LLM 未構想出合理累積場景，也不證明執行判定器漏掉已執行異常。

#### B6. 204033 — compiled alias：內部使用 view 未建立回傳別名的適用性

基準觀察回傳複數基底與其 conjugate imaginary view 的共享儲存效果：返回後修改 view，須影響基底。初始值一致不足以建立此性質。

98 個候選中，63 個遭靜態拒絕：37 個 manual_seed、18 個頂層結構違反、8 個語法錯誤。另有 33 個執行失敗：18 個將區域 `forward` 函式當成資料回傳而無法 pickle，9 個缺少頂層 `forward`，3 個 build 未回傳字典，其餘為形狀、鍵值或儲存錯誤。這些均未產生本指標計入的完成目標結果。

僅 [attempt-0043](../result/pytorch-correctness/763298/phase3/204033/attempt-0043/program.py) 與 [attempt-0067](../result/pytorch-correctness/763298/phase3/204033/attempt-0067/program.py) 完成編譯目標。兩份摘要均記錄：

```json
{"initial_outputs_agree": true,
 "returned_alias": false,
 "compiled_graph_executed": true}
```

- attempt 43 計算 `y=x.view(5,2)`、`z=y+1`，僅回傳 `{'output': z}`。加法配置新儲存，回傳集合僅有一個張量。
- attempt 67 計算 `z=x+y`、`view_z=z.view(2,8)`，僅回傳 `{'view': view_z}`，未回傳基底或另一個共用儲存的張量。雖然修改選到子 view，MR 的回傳集合仍只有一個可觀察成員。

因此兩案為**不適用，而非 Pass**。eager 的 `has_base=True` 與 compiled 的 `has_base=False` 本身不建立缺陷；此 MR 要求修改可觀察地傳播至另一回傳別名。中繼資料不同不直接構成語義違反。

執行證據確認 Inductor 圖確實執行。通用欄位 `backend='eager'` 不表示未編譯，因處理器明確指定 Inductor。兩案均未提供 negative/conjugate 回傳別名觸發條件，因此未重現 204033。

**失敗分類。**建構函式與序列化契約失敗，以及回傳結構不符合 MR 輸入與觀測關係。結果不證明所有編譯器別名行為正確。

#### B7. 204365 — BatchNorm：稠密輸入未測到 MKLDNN affine 跨步

108 個候選中，81 個遭靜態拒絕：75 個 manual_seed、6 個語法錯誤。27 個完成 worker 中，8 個轉換不適用，19 個完成目標且均為 Pass。沒有已記錄的目標布局是 MKLDNN。

[attempt-0005](../result/pytorch-correctness/763298/phase3/204365/attempt-0005/program.py) 建立普通稠密 `(4,8)` 輸入，running_mean=0、running_var=1、weight=1、bias=0、training=False。取樣器對輸入套用 negative 操作，得到 stride `(16,2)`、offset 1 與 negative bit；但[中繼資料](../result/pytorch-correctness/763298/phase3/204365/attempt-0005/execution/summary.json)顯示 weight/bias 仍為 stride `(1,)`，輸入布局為 `torch.strided`。

基準要求 **MKLDNN 輸入與 stride-two 仿射張量同時存在**。改變普通稠密輸入的 stride 或 negative bit，不會測試 MKLDNN 對 weight/bias 的索引。取樣器未提供 `.to_mkldnn()`，且只選直接位置張量，不搜尋任意關鍵字參數。本例雖將仿射運算元放在 args，所選設定轉換的是輸入，因此輸入 stride-two 不建立仿射跨步覆蓋。

**失敗分類。**布局／後端與運算元角色的聯合覆蓋缺口。第 2 階段轉接器與正規化判定器能處理已建構的 MKLDNN 案例，但生成未到達該路徑。19 個稠密 Pass 不表示缺陷已修復。

#### B8. 201268 — CSR add(out)：稠密廣播未測到稀疏輸出緩衝區

163 個候選中，131 個因 `manual_seed` 遭拒。32 個進入 worker，其中 2 個因越界儲存或不相容形狀失敗。其餘 30 個完成：13 個轉換不適用，17 個完成目標且均為 Pass。沒有記錄到 CSR 目標中繼資料。

[attempt-0001](../result/pytorch-correctness/763298/phase3/201268/attempt-0001/program.py) 測試稠密 `(3,4)` 加稠密 `(4,)`，`alpha=2`，沒有 `out` 關鍵字。這是合法加法測試，卻不符合非連續 CSR 輸出 values 使 copy-out 寫錯儲存位置的觸發條件。

跨步操作轉換普通位置張量，不將稠密轉為 CSR，也不建構 CSR `out` 並注入 `out.values()[::2]` 布局。中繼資料與實體化支援 CSR，不代表取樣器曾提供該設定。

**失敗分類。**稀疏布局、輸出運算元與 values 儲存的聯合生成缺口，加上靜態拒絕。17 個 Pass 不提供獨立審核的 TN：add 審核器要求相同形狀與 dtype，這些執行超出支援條件。此 API 的事後標籤全為未知，正式 Pass 不構成獨立負例真值。

#### B9. 197089 — std：已覆蓋編譯與反向傳播，未覆蓋極小尺度

第 1 階段使用 FP32 `[-2**-100, 2**-100]`，平方項可能下溢，但正確 std 非零，梯度為 `[-.5,.5]`。Inductor 執行僅是一項條件；數值尺度同樣是識別此基準觸發的要素。

62 個候選中，28 個遭靜態拒絕：8 個屬性寫入限制、19 個語法錯誤、1 個 manual_seed。34 個完成 worker 中，12 個轉換不適用；其餘 22 個完成目標反向傳播，得到 20 個 Pass、2 個疑似案例，其中 12 個有編譯圖執行證據。沒有保存結果符合極小尺度條件。

[attempt-0000](../result/pytorch-correctness/763298/phase3/197089/attempt-0000/program.py) 與數個後續候選使用 1..12、shape `(3,4)`、correction=1，變異數不在基準下溢範圍。表示操作保留邏輯值，不會將 1..12 縮放至極端量級；設定也缺乏量級或指數探索。因此編譯與反向傳播覆蓋可以同時伴隨數值觸發缺口。

**兩個額外疑似案例不計為基準重現。**FP64 的 [attempt-0038](../result/pytorch-correctness/763298/phase3/197089/attempt-0038/program.py) 與 FP32 的 [attempt-0052](../result/pytorch-correctness/763298/phase3/197089/attempt-0052/program.py) 保留 1..12，由取樣器注入 negative bit，Inductor 卻反轉梯度符號。mean=6.5、std=`sqrt(13)`，首元素導數應為 `-5.5/(11*sqrt(13))≈-0.138675`，實際為正，最大差約 0.27735。Python `math.fsum/hypot` 佐證兩例，20 個普通案例也取得獨立 TN。

這些表示／反向傳播異常不同於極小輸入問題 197089。是否與其他編譯 negative-bit 問題共用根因尚未釐清，保留為一組待分析候選，不主張新根因。同一 API 的疑似失敗不自動等於基準問題重現。

**失敗分類。**已達執行模式與反向傳播狀態，但缺乏數值尺度覆蓋。未證明判定器漏掉已確認的 197089 觸發執行。

#### B10. 200613 — copy_：CPU source 的 negative-bit 注入觸發已知變體

105 個候選中，63 個遭靜態拒絕：62 個 manual_seed、1 個語法錯誤。另有 22 個執行失敗，20 個缺少 Tensor 方法 `other` 引數，其餘為 build 型別或參考運算元錯誤。20 個完成 worker 中，15 個轉換不適用；5 個完成目標產生 4 個 Pass、1 個疑似案例。

[attempt-0011](../result/pytorch-correctness/763298/phase3/200613/attempt-0011/program.py) 的 LLM 建構函式以明確算術建立 CPU source `11..22`、shape `(3,4)` 及 CUDA 目的張量，回傳 args `(dst, src)`、`non_blocking=False`。修復保留 attempt=10，因此 `representation='negative'`、`subject=10//7=1` 選到第二個 Tensor，即 src。LLM 並未撰寫 conjugate 重現程式。

```text
注入前：CPU real source，logical values 11..22
注入後：CPU source shape (3,4), stride (8,2), offset 1, neg=True
        logical values 仍為 11..22
destination：CUDA, contiguous, neg=False
actual copy result：-11..-22
expected / materialized control：11..22
max absolute error：44
```

[執行摘要](../result/pytorch-correctness/763298/phase3/200613/attempt-0011/execution/summary.json)與[可讀觀測值](../result/pytorch-correctness/763298/phase3/reviewed-observations.json)保存結果。相同 dtype 複製的獨立 Python 邏輯索引參考無須容差論證即可辨識符號錯誤，審核標為 TP。首次自動偵測為 33.97 秒。

第 1 階段使用 conjugate 與廣播；生成案例則使用 negative、非連續來源，來源與目的形狀相同。[問題 200613](https://github.com/pytorch/pytorch/issues/200613) 明載 negative-view 與切片布局測試；保存的 [Copy.cu 摘錄](../result/pytorch-correctness/research/upstream-source-excerpts.json)亦顯示暫存複製路徑重設來源 conj/neg 旗標。符號反轉符合此機制，支持歸入同一已知問題，而非另計缺陷。歸屬依據為輸出、中繼資料、上游敘述與程式碼，未主張做過原生分支追蹤或上游修補驗證。

**原比對器漏配原因。**凍結規則要求 CPU source `conj=True`，因此判定器已檢出的 `neg=True` 案例未歸入基準，初始問題率為 2/10。後續僅在基準端納入 neg/conj 與暫存複製條件，保留[原始比對輸出](../result/pytorch-correctness/763298/phase3/frozen-matcher-evaluation.json)，經審查比例成為 3/10。這是比對器漏報，不是正確性判定器漏報；生成器輸入未變。

#### C. 跨階段原因與推論界線

| 層次 | 觀察限制 | 不支持的推論 |
| --- | --- | --- |
| 語法／靜態契約 | 嚴格 JSON 仍可含 SyntaxError；manual_seed 遭拒；Module 別名未解析。 | 所有被拒程式均為非法 PyTorch，或移除 seed 呼叫即可保證重現。 |
| 執行契約 | 缺少 self/other、回傳區域函式、缺少頂層 forward、越界 view。 | 執行例外即可確認框架缺陷，或證明已達合法基準觸發條件。 |
| 狀態取樣 | 沒有 MKLDNN／CSR 轉換、singleton 批次插入或數值尺度掃描；只轉換直接位置張量。 | 布局轉接器代表布局覆蓋，或任意非連續張量符合指定核心觸發。 |
| MR 適用性 | 編譯案例缺乏回傳別名對；LU 殘差檢查失敗。 | 目標完成即可判 Pass，或中繼資料不同即可判失敗。 |
| 判定／參考 | slogdet 對照組不一致保持未評估；成功案例有數學或狀態佐證。 | 參考不一致本身證明目標缺陷，或所有疑似失敗均已獨立確認。 |
| 問題歸屬 | 狹窄 copy 比對器漏掉已知 negative-view 變體；std 暴露不同差異。 | API 任意失敗都重現其基準問題，或比對器漏配等於判定器漏報。 |

共用提示詞禁止 RNG，卻仍有 `seed` 參數，已安裝文件字串也包含 `randn` 範例；模型多次生成 manual_seed。這值得進行提示詞消融，但**沒有對照實驗識別哪個提示詞元件造成該行為，也未量化修訂效益**。別名解析驗證、執行契約修復及更豐富的表示／尺度取樣，同樣是可測的下一步，而非本輪已完成結果。

本輪為三個已知問題各建立至少一個獨立佐證的生成案例。其餘七案由第 1／2 階段確認仍可本機重現，但第 3 階段遇到較早階段阻礙或缺乏辨識條件。未觀察到判定器漏報的結論僅限 19 個獨立正例。未執行、MR 不適用、不支援或未審核案例均保持未定。

#### D. 獨立審核的分母

| API 問題標籤 | TP | FP | TN | FN | 未知，含未執行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 203078 | 0 | 0 | 1 | 0 | 105 |
| 204407 | 3 | 0 | 5 | 0 | 114 |
| 200599 | 0 | 0 | 0 | 0 | 132 |
| 202706 | 13 | 0 | 4 | 0 | 80 |
| 181723 | 0 | 0 | 0 | 0 | 161 |
| 204033 | 0 | 0 | 0 | 0 | 98 |
| 204365 | 0 | 0 | 19 | 0 | 89 |
| 201268 | 0 | 0 | 0 | 0 | 163 |
| 197089 | 2 | 0 | 20 | 0 | 40 |
| 200613 | 1 | 0 | 1 | 0 | 103 |

上表來自既有 [independent-audit.json](../result/pytorch-correctness/763298/phase3/independent-audit.json)，並非文件修訂時新增實驗。無結果候選不作數值裁定；完成執行但超出審核器語義、形狀、dtype 或條件界限者亦保持未知。例如廣播 add 超出等形狀支援，上下文建構函式的 `min` 不在允許清單；LU、graph、compiled-alias 則缺乏對應獨立事後實作。

審核參考由初始輸入計算，不複製正式參考。其數值不確定性界限仍是保守審查政策，而非適用所有 PyTorch 核心的誤差定理；排除未知案例也使 0/50 FPR 不能外推整體。19 個審核正例中，17 個對應已知基準，2 個為上述 std 候選。三個問題身分與 19 個正向案例是不同計數單位。

### 時間、資源與清理

十個名義上每 API 300 秒的預算，在生成／執行迴圈共使用 **2,567.91 秒**（42.80 分鐘），另有 **21.29 秒**用於已安裝 API 中繼資料探查。因預先宣告 45 秒派送保留時間，各迴圈於 255.01–263.64 秒結束。整個實驗迭代耗時 **2,597 秒**。成功及失敗 worker 的執行區間合計 **864.44 秒**；這是 worker 經過時間，不是 GPU 使用率。Provider 回報 **1,642,117 個輸入 tokens**及 **325,341 個輸出 tokens**。計時使用重用中的服務與已暖機編譯快取，不屬於冷啟動基準。

模型僅啟動一次，耗時 **192 秒**，**重啟零次**。完整開發與評估配置持續 **4,208 秒**、使用兩張 GPU，合計 **8,416 GPU 配置秒（2.338 GPU 小時）**，包含啟動、閒置／協調時間及失敗開發迭代。每次迭代耗時另行保留；重疊的服務與測試步驟時間不相加為總經過時間。

測試暫存目錄已移除，互動測試終端已關閉，所有本任務所屬步驟均退出。`squeue -j 763298` 為空，`sacct` 顯示工作狀態 `COMPLETED`、退出碼 `0:0`；服務步驟於關閉時刻意取消。工作完成狀態不用作 MR 通過判定。目前無本任務所屬 vLLM 服務或配置仍在執行。

### 證據索引

下列路徑均指向工作 763298 的不可變輸入／結果或衍生報告：

- [最終第 2 階段整合](../result/pytorch-correctness/763298/phase2-frozen/summary.json)
- [凍結實驗協定](../result/pytorch-correctness/763298/phase3/protocol.json)
- [指標分母與執行覆蓋](../result/pytorch-correctness/763298/phase3/final-analysis.json)
- [原始凍結比對器結果](../result/pytorch-correctness/763298/phase3/frozen-matcher-evaluation.json)
- [經審查的問題歸屬](../result/pytorch-correctness/763298/phase3/reviewed-attribution.json)
- [獨立審核與全部保留的未知案例](../result/pytorch-correctness/763298/phase3/independent-audit.json)
- [所選觀測值的可讀版本](../result/pytorch-correctness/763298/phase3/reviewed-observations.json)
- [各迭代計時與經驗證的清理](../result/pytorch-correctness/763298/session-accounting.json)

Worker 記錄的正式引擎校驗值為 `d2c85c8d352abc278b4039685797027b03b79a357ed67160ac7564e170126708`。實驗驅動器要求凍結引擎與已驗證的第 2 階段 Python 原始碼一致。基準端審核與問題歸屬修正另行加入，從未提供給生成器。

### 獨立事後審核設計

凍結實驗期間另加一個評估器，審核已保存的觀測值，無法將資訊回饋生成流程。它以 Python 精確有理數消去法計算小型行列式的反矩陣，以 `math.fsum` 和 `math.hypot` 計算純量 std，並支援有界整數 scatter 精確加總、相同 dtype 的廣播複製、推論 batch normalization，以及有界純量／CSR 加法。上下文檢查要求建構函式不能改變周遭上下文，並以新 worker 預設狀態比較。不支援的案例保持未裁定。

審核從已保存輸入重建預期值，之後才讀取框架判定來分類 TP/FP/TN/FN。它不重跑目標 API，也不複製正式判定器的預期輸出。此子集的 FPR 與敏感度明確限於獨立裁定的生成案例，不能當作未裁定母群的估計。審核程式位於基準端，不在凍結生成快照中。

### 盲生成範圍與契約限制

判定類別的設計參考了基準。對模型隱藏重現程式，衡量的是這些通用策略下的生成能力，不代表無偏發現，也未主張模型預訓練資料不存在污染。表示注入由 sampler 完成，因此成功觸發可能是 LLM 輸入建構與執行時轉換的共同成果。

凍結契約接受寫為 `torch.nn.Module` 的區域類別，但該檢查不解析 `nn.Module` 別名。拒絕這類程式是框架契約限制，不是 Python／PyTorch 程式非法的證據。嚴格 RNG 呼叫過濾也會拒絕 `manual_seed`，即使其餘值已明確初始化。這些限制保留於實際實驗中，未在觀察結果後修改協定。靜態契約有效性另行報告，不可解讀為每個被拒或被接受 PyTorch 程式的語義合法性。
