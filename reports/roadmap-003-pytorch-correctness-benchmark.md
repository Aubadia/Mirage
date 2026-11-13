# PyTorch 正確性基準：第 1–3 階段

本文件前半保留原始計畫與執行時序。詳細技術紀錄分別見[第 1 階段](#phase-1-details)、[第 2 階段](#phase-2-details)及[第 3 階段與實際提示詞](#phase-3-details)；各問題重現與未重現的原因見[評估報告](pytorch-correctness-evaluation.md)。

## 已核准範圍

使用者於 2026-11-13 核准第 1 階段：在已準備的 PyTorch 環境中，收集十個不同、可重現且涉及張量表示或訓練狀態的正確性問題。工作包括隔離的重現程式、獨立判定依據、對照組、每案一次確定性的新程序執行，以及以執行證據支持的報告；同時評估 Mirage 現有能力，但不修改實作。當時，第 2 階段的修改與第 3 階段的實驗仍須另提具體檔案範圍計畫。

授權檔案：本路線圖；docs/pytorch-correctness-reproduction.md；benchmarks/pytorch_correctness/{README.md,candidates.json,cases.py,run.py}；tests/test_pytorch_benchmark.py；sbatch/replay/pytorch_correctness.sbatch。執行產物存於 result/pytorch-correctness/<run-id>/。

## 實施步驟

1. 檢查公開上游問題與現有實作，記錄候選案例及排除原因。
2. 以公開 API 實作最小案例、對照組、判定證據與隔離執行器。
3. 完成語法靜態檢查與排程提交預檢；所有運算均透過 Slurm，在已準備的環境執行。
4. 對符合條件的問題，以明確固定輸入重現一次；檢視替代輸入與獨立語義證據，將共同根因歸為同組。
5. 記錄實際結果及框架能力缺口，依最終基準提出第 2 階段的檔案修改範圍。

## 驗證計畫

- Python AST 檢查、bash -n 與 sbatch --test-only 僅屬靜態檢查或提交預檢。
- 在可行情況下，執行器測試涵蓋適用性不成立、判定依據不一致、非預期例外、程序逾時及結果彙整。
- 使用帳號 ACD115198、分區 8gpus、一個節點、八個 CPU、一張 GPU、四小時配置；運算命令透過 srun 執行。
- 每次執行前保存程式與設定快照、環境及原始日誌；保留失敗狀態，並預留清理時間。
- 候選案例不等於已確認缺陷。納入案例須具備合法輸入、經驗證的觸發條件、通過的對照組，以及相互佐證的判定證據；根因說明須標明證據強度。
- API 或執行環境限制與未重現結果分開報告；不得由基準重播推論 LLM 已成功生成測試或發現問題。

## 執行進度：依時間排序

- 已閱讀專案契約、生成器、條件式判定器、表示轉換介面、Slurm 輔助工具及已準備環境的中繼資料。
- 安裝版本檔案顯示 2.15.0+cu132；當時尚待計算節點驗證。
- 現有框架當時不取樣或記錄 conjugate/negative bit，僅支援已登記的稠密跨步輸入，並對具副作用或涉及編譯的 MR 採保守適用性檢查。
- 路線圖已於實作前建立；當時尚無執行結果。

- 篩選工作 763186 對 14 個候選各執行一次，得到 13 個對照組通過的判定違反，以及 1 個未重現案例（204347）。三項執行器測試於 1.971 秒內通過。執行環境為 PyTorch 2.15.0+cu132、Python 3.14.10、CUDA 13.2、NVIDIA H200。這些是暫定結果，尚非最終納入決定。
- 後來被取代的初始方案：選取十組機制，各做三次重複、兩種輸入變體，包含 203078、204407、200599、202706、181723、198118、204365、201268、197089、200613。編譯下的 conjugate clone/alias 候選保守歸為同組備選；grid-sample 低精度問題亦列為備選。

## 使用者調整：確定性單次執行

使用者明確要求排除依賴隨機性的問題，每案執行一次，取代先前三次重複、兩種變體的驗證方案。工作 763201 因 MaxGRESPerAccount 處於等待狀態時即取消，未取得配置或執行。當時規劃將帶種子的隨機 pinv 輸入改為明確常數矩陣，保留每次呼叫內的獨立數學判定與對照組，對十個固定輸入案例各執行一次。未選取依賴時序競爭的案例；autocast 使用確定性遞迴。可選的重複與變體旗標僅供診斷。

- Slurm 明確拒絕零 GPU 預檢：--gpus-per-node 僅接受 1..8。未在 Slurm 外執行 CPU 運算，也未改換分區。工作 763226 於等待時取消，未使用配置。
- 最終選取重用工作 763186 已執行的十個確定性測試單元，以固定輸入的編譯 aliasing（204033）取代帶種子輸入的 pinv。當前入選函式的 AST 與共用判定／中繼資料輔助函式，均與已執行快照完全一致。final-review.json 記錄結果雜湊、函式雜湊、已驗證條件及對照組。依使用者的單次執行政策，無須重跑。

## 第 1 階段最終狀態

- 已在 PyTorch 2.15.0+cu132 完成十個固定輸入重現：203078、204407、200599、202706、181723、204033、204365、201268、197089、200613。每案使用一個已執行程序、經驗證的述詞、獨立判定依據及通過的對照組。重現報告區分原生程式碼證據與根因假說。
- 原始上游問題與留言、固定版本的原生程式碼摘錄、已安裝 autocast/CUDA-graph 程式碼摘錄、不可變執行程式、完整張量證據及審查雜湊均保留於 result/pytorch-correctness/。
- 最終數值測試函式與已執行快照一致。三項執行器測試在 Slurm 中通過；更新後的單次結果彙整檢查另於 0.010 秒內通過，未匯入 PyTorch。
- 工作 763186 使用一張 GPU，配置 87 秒後以預期的判定違反退出碼 3 結束。工作 763201 與 763226 均於配置前取消。最終排程檢查確認無本任務所屬工作；未啟動 vLLM 服務。
- 當時未修改既有 src/mirage 實作、相依套件、設定或無關檔案。第 2 階段整合與第 3 階段 LLM 評估仍待具體檔案範圍核准，尚未執行，亦未宣稱發現指標。

- 已在重現報告提出尚待核准的第 2–3 階段實作與檔案範圍：通用生命週期／表示／判定能力、獨立基準整合、每 API 300 秒的盲生成評估、Slurm 驗證與清理。當時未修改第 2 階段檔案。

## 第 2–3 階段執行核准（2026-11-13）

使用者核准重現報告中的實作與確切檔案範圍，並明確要求第 2 階段正確性驗證完成後直接執行第 3 階段，無須再次取得階段轉換核准。

實作順序為：（1）在既有訓練契約之外，加入經審查的性質測試程式契約，以及可重用的表示／生命週期／參考判定器；（2）將十個測試案例送入與生成程式相同的 worker 與驗證器；（3）在 Slurm 中驗證負向及適用性案例與既有測試套件；（4）凍結程式與提示詞；（5）使用已準備的 provider，對每個目標 API 執行 300 秒盲生成，再獨立判定問題歸屬並報告所有分母。一般化邏輯不得包含問題編號或基準常數；基準仍採確定性單次執行政策。

僅使用報告中已核准的第 2–3 階段檔案。路線圖須持續記錄實作決策、實際驗證結果、實驗結果、資源用量及經驗證的工作階段清理；第 1 階段證據保持不可變。

### 第 2 階段實作與開發證據

新增性質測試程式策略，重用既有 provider、worker 與證據基礎設施，並加入五類經審查的性質、確定性表示操作、不透明／稀疏布局中繼資料、基準與 worker 的整合，以及不呼叫 LLM 的檢查式重播。正式程式不含入選問題編號或重現常數；通用 API 登錄表編碼的是經審查語義，而非特定失敗輸入配方。

Slurm 開發工作 763298 使用一個已準備的 vLLM 服務，啟動耗時 192 秒、重啟零次，另以一張 GPU 測試。初始整合遇到 MKLDNN 快照序列化失敗，只達 9/10；加入明確的不透明布局轉接器後達到 10/10。中間版本的完整套件通過 67 項測試。新增重播回歸測試又暴露序列化造成的版本計數器遞增，隨後修正新圖快照簽章，使其不比較絕對版本數，並修正 CLI 性質測試結果處理。所有失敗迭代均已保存。該紀錄當時仍在進行最終驗證，尚未宣稱第 3 階段結果。

### 第 2 階段驗證完成；第 3 階段開始

最終完整套件 iteration-hz5F7t 於 63.277 秒內通過 70 項測試，整次迭代為 65 秒。最終整合 phase2-frozen／iteration-7KgQnc 檢出 10/10，具備實際目標執行證據且對照組一致。兩項檢查式重播探測——204407 negative view 與 204365 MKLDNN 加 affine views——分別在迭代 aWRPx2、5353x3 完成；疑似失敗判定一致、環境未變、provider 呼叫為零。較早遭拒的重播證據仍保留；view/base 重建與重播失敗時非零 CLI 狀態現已納入驗證。真實模型的 torch.sin 冒煙測試在 property-smoke-02 產生兩個通過靜態檢查、可執行且判定通過的案例。

第 3 階段依既有核准直接開始。驅動器要求正式引擎雜湊與 phase2-frozen 一致，排除開發期冒煙測試結果，採每 API 300 秒預算及 45 秒派送保留時間，並保存原始提示詞、回應、程式與執行結果。凍結實驗期間不得依中途基準結果調整方法。

### 第 3 階段完成與最終驗證

- 凍結迭代 iteration-pwpdkQ 於 2,597 秒後以退出碼 0 結束。十個名義上每 API 300 秒的預算均已完成；實際迴圈合計 2,567.91 秒，中繼資料取得為 21.29 秒。預先宣告的 45 秒派送保留時間，使各 API 實際耗時為 255.01–263.64 秒。
- 共產生 1,154 個候選，310 個通過靜態契約，130 個完成目標執行，125 個得到適用比較，28 個為疑似失敗。靜態有效率 26.86%；有效程式中的可執行率 41.94%。其中有 661 個修復回應，以及 748 個依 API 區分的原始碼指紋。
- 獨立事後審核 iteration-3ChIy1 標記 19 個正例與 50 個負例：TP=19、FN=0、FP=0、TN=50。FPR 0/50 與偵測率 19/19 僅適用於該子集；九個疑似案例及其他不支援／未執行候選並未被假定正確。
- 經審查重現 3/10 個已知問題：204407 slogdet 首次偵測 8.78 秒、202706 autocast 10.73 秒、200613 copy_ 33.97 秒。原始比對器因遺漏上游明載的 negative-view copy 變體，只報告 2/10。原始輸出已保存；事後修正歸屬時未變更生成流程或引擎。
- std attempts 38/52 的獨立證據顯示 Inductor 對 negative-bit 輸入產生錯誤梯度符號。它們不屬於 197089 的極小輸入基準，保留為一組相關候選，根因與新穎性尚未確認，不增加問題重現率。
- 實驗後的負向對照驗證：iteration-RgKjt3 於 7.317 秒內通過 12 項性質測試，包含普通編譯 alias 與相同 graph microbatch；iteration-lnZdUc 於 2.031 秒內通過四項基準／審核測試。事後歸屬報告與所選張量觀測值匯出均在 Slurm 完成。凍結實驗期間未修改正式引擎。
- 已明確記錄生成限制：嚴格拒絕 RNG 呼叫、未解析 nn.Module 別名、缺少 Tensor 方法接收者、將函式物件當成快照資料回傳，以及 MKLDNN／CSR／極端數值尺度的觸發條件覆蓋不足。這些程式不全是非法 PyTorch，也不能自動歸為判定器漏報。
- 工作 763298 配置 4,208 秒、兩張 GPU，合計 8,416 GPU 配置秒，以退出狀態 0:0 完成。服務僅啟動一次，耗時 192 秒、重啟零次。全部 23 次迭代耗時及失敗開發日誌均保留。測試暫存已移除、終端已關閉、squeue 為空、所有步驟均退出，包括關閉時刻意取消的服務步驟；無本任務所屬服務或配置殘留。
- 最終結果、MR 條件、可重現命令、指標分母及產物連結見 docs/pytorch-correctness-evaluation.md。未新增相依套件或變更專案設定。

## 完成後要求的文件擴充

使用者在實驗完成後，要求詳細說明第 1、2、3 階段的工作、新增能力與實作方式、LLM 提示詞，以及各問題重現或未重現的原因。該次文件更新僅限本路線圖與 `docs/pytorch-correctness-evaluation.md`。

該次更新依序核對現行程式與凍結快照，檢視已保存的請求、回應、生成程式、worker 摘要及獨立審核紀錄，再補齊階段說明與逐案分析，最後檢查數值、提示詞原文、連結及編輯範圍。未修改實作、啟動 Slurm／LLM 或重跑基準；新增敘述均來自既有執行證據。

驗證使用 Python 標準函式庫，比對評估報告的 10 列執行漏斗與 `final-analysis.json`、10 列審核統計與 `independent-audit.json`，全部一致。系統訊息與完整 slogdet 使用者訊息均與已保存的 HTTP 請求逐字相符；五個類別後綴及共用前綴亦與實作一致，僅展示區塊省略前綴末尾的連接空格。兩份文件的本機連結與明示錨點皆存在，Markdown 程式區塊成對，無行尾空白。該次僅修改這兩份文件，保留先前測試時序，未宣稱新增實驗或重新通過測試。

<a id="phase-1-details"></a>
## 第 1 階段：建立十案基準

### 1.1 區分上游回報與本機重現

第 1 階段檢視 PyTorch 官方儲存庫的公開問題、最小重現範例（MRE）與討論，並對照已準備的本機環境。焦點是邏輯值相同而張量表示不同，或訓練與上下文生命週期改變時的正確性。問題仍為 `open` 不構成缺陷證明；任意例外也不直接視為錯誤結果。

保留證據包含 14 份 `issue-<ID>.json`、[固定版本的原生程式碼摘錄](../result/pytorch-correctness/research/upstream-source-excerpts.json)及[已安裝套件的程式碼證據](../result/pytorch-correctness/research/installed-source-evidence.json)。[候選目錄](../benchmarks/pytorch_correctness/candidates.json)記錄各 API、裝置、觸發條件、判定依據、根因分組及納入或排除理由。這些產物支持候選審查，但不代表已窮舉 GitHub 問題空間，也未保留完整的搜尋查詢與排序歷史，因此不能視為一次完整搜尋實驗。

### 1.2 建構固定輸入、獨立預期與對照組

[第 1 階段 cases.py](../benchmarks/pytorch_correctness/cases.py) 中的測試案例是依問題撰寫的回歸資料，**不是第 3 階段由 LLM 生成的程式**。每案回傳 `actual`、`expected`、`control`、`conditions`、容差及額外中繼資料。`check()` 透過既有的 `mirage.inspector.comparison.differences` 比較完整張量。

驗證器先檢查條件、對照組與數值可比較性，再判定 `actual` 是否違反 `expected`。對照組與預期不一致、非有限值、條件未成立、例外或逾時，都不能建立已確認的重現。JSON 預覽最多呈現 64 個張量元素；完整比較與 `evidence.pt` 仍保留全部資料。

| 問題／測試函式 | 固定觸發條件的建構 | 正確性依據與對照 |
| --- | --- | --- |
| 203078 / `scatter` | 將 index `[[0],[2]]` 展開為 `(2,16)`、stride `(1,0)`，搭配 `(3,32)` 目的張量。 | Python 純量 scatter 定義，以及將相同 index 連續化的目標對照組。預期列加總為 `[16,0,16]`，實際為 `[16,16,0]`。 |
| 204407 / `slogdet` | 對固定非對稱 2×2 complex128 葉節點套用 `conj().imag`，得到設有 negative bit 的實矩陣。 | 2×2 反矩陣轉置的解析導數、純量行列式實作的有限差分，以及 `resolve_neg()` 對照組。第 1 階段比較複數葉節點梯度的虛部；第 2 階段則直接對實數運算元微分。 |
| 200599 / `lu_solve` | 對兩個固定 2×2 系統，使用 `LU[:, None]` 加入大小為一的批次維度，stride 為 `(4,2,1,2)`。 | 2×2 封閉形式解、連續 LU 對照組及保留的殘差。第二組解應為 `[1/6,1/3]`，實際為 `[5,-1]`。 |
| 202706 / `autocast` | 在同一裝飾函式中遞迴，沒有執行緒並行或競爭。 | 退出時須還原 AMP 狀態；每層新建上下文的遞迴作為對照，觀察啟用狀態、dtype 與快取。 |
| 181723 / `cuda_graph` | 固定逐元素縮放模型與兩個不同 microbatch，捕捉圖後將參數 `.grad=None`。 | 解析梯度為兩批輸入之和，另由 eager autograd 佐證；實際累積卻為第二批梯度的兩倍。 |
| 204033 / `compiled_alias` | Inductor 回傳複數基底與 `conj().imag` view，在返回後修改 view。 | 相同 eager 程式與可直接計算的虛部變化。初始值正確不足以建立別名語義正確性。 |
| 204365 / `batch_norm` | MKLDNN 推論輸入，weight 與 bias 為 stride-two view。 | 明確的仿射正規化公式，以及邏輯仿射值相同的連續對照組。 |
| 201268 / `sparse_add` | 合法 CSR `out` 的 values 使用 stride-two 儲存，間隔位置放置哨兵值。 | 逐元素加法、連續輸出對照組，以及 values 與相鄰儲存寫入檢查。最後一項是第 1 階段額外證據；第 2 階段通用判定器未完整實作相鄰儲存檢查。 |
| 197089 / `std_gradient` | FP32 `[-2**-100, 2**-100]`、CPU Inductor、`correction=0`。 | 預期 std=`2**-100`、梯度=`[-.5,.5]`，另有 eager 對照。採尺度相符的比較，避免一般絕對容差掩蓋差異。 |
| 200613 / `broadcast_copy` | 將固定 conjugate source 同步從 CPU 複製到 CUDA，從一個元素廣播為兩個。 | 邏輯來源廣播與解析 conjugate 後的對照組。預期 `[1-2j,1-2j]`，實際虛部符號相反。 |

完整 MRE、觀測輸出及各根因說明的證據強度見[第 1 階段重現報告](../docs/pytorch-correctness-reproduction.md)。上表描述觀察症狀與輸入建構，不代表每個核心層級根因都經上游修補或原生除錯器驗證。

### 1.3 執行、去重與單次執行政策

[run.py](../benchmarks/pytorch_correctness/run.py) 為每案建立新子程序與程序群組、獨立 stdout/stderr、逾時限制、退出狀態、環境紀錄及完整張量證據，只在 Slurm 配置與步驟內執行。數值違反使用退出碼 3；不確定結果與執行失敗使用退出碼 2。因此 Slurm `FAILED` 不直接代表測試非法，退出碼 0 也不能取代判定器結果。

篩選工作 **763186** 執行 14 個候選，得到 13 個暫定違反與 1 個未重現。使用者隨後要求無隨機案例，每個固定案例執行一次，取代重複與變體計畫。工作 763201、763226 在等待時取消，未提供驗證證據。最終十案重用已執行的確定性測試單元，並以 [final-review.json](../result/pytorch-correctness/research/final-review.json) 核對結果雜湊、測試函式 AST 雜湊、條件及對照組；未以統計重複試驗宣稱穩定性。

| 排除的候選 | 理由 |
| --- | --- |
| 145093 compiled clone | 保守歸入 204033 的編譯 conjugate/negative alias 問題，不將兩個 API 計為獨立根因。 |
| 198118 pinv | 篩選使用較早的帶種子輸入版本；後續固定矩陣版本尚未執行，不能繼承舊結果。 |
| 204347 repeated cat | 原始篩選未重現；後續固定輸入備選尚未執行，因此排除。 |
| 204496 BF16 grid gradient | 篩選已重現差異，保留為備選。最終十案已涵蓋選定機制，此備選不增加成功數。 |

`candidates.json` 的頂層 `selection` 仍保留早期重複執行提案。最終政策以使用者後續指令、入選列的 `reproduction_policy`、`final-review.json` 及更新後路線圖為準。初始篩選並非全數屬於最終不使用 RNG 的案例集；14 個候選不能被描述為與最終十案具有同等證據地位。

### 1.4 交付內容與範圍

第 1 階段新增基準測試案例、目錄、隔離執行器、執行器測試、Slurm 重現腳本、重現報告及本路線圖，未修改正式生成器或進行 LLM 發現實驗。檢查指出後續能力需求：原本稠密張量與訓練迴圈路徑不足以完整表達延遲 neg/conj bit、MKLDNN／CSR 布局、上下文遞迴、CUDA 圖捕捉，以及返回後的別名修改。

<a id="phase-2-details"></a>
## 第 2 階段：在共用 worker 中驗證五類性質

### 2.1 已實作的架構

第 2 階段新增 `--strategy properties`，保留預設 `--strategy training`。LLM 提供建構函式、小型函式或模組；執行器負責轉換、狀態協定、API 呼叫、參考結果及判定。十個問題的精確重現常數保留於基準測試案例，不放入生成器。

本實驗採用以下執行流程：

```text
API + 已安裝版本的簽章／文件字串
    → API 語義類別登錄表 + 確定性設定循環
    → LLM build(device, seed)［需要時另有 forward(ctx)］
    → 靜態契約檢查／有限次修復
    → 隔離的 execute_property worker
    → 表示注入 + 快照 + 實際目標／計算圖／反向傳播證據
    → 經審查的性質處理器 + 適用性／對照組檢查
    → 通過／疑似失敗／不適用／未評估／執行失敗
    → 基準端結構比對與獨立事後審核
```

這條路徑是 AGENTS.md 三層架構中能力範圍有限的補充。第 1 層使用固定循環與預定類別對應，**未加入學習式覆蓋排序或完整的 `alpha(sigma)` 狀態轉移規劃器**。第 2 層生成建構函式，而非填滿任意訓練迴圈的全部注入點。第 3 層重用隔離 worker 與證據基礎設施，並新增性質處理器。這些能力不代表完整實作了目標架構。

### 2.2 新增能力與檔案對應

| 元件／位置 | 實作內容 |
| --- | --- |
| [contracts/programs.py](../src/mirage/contracts/programs.py)、[cli.py](../src/mirage/cli.py) | 新增 `RunOptions.strategy`、CLI 選擇及性質測試的執行／報告／重播分流；修正結果格式處理與重播失敗退出碼。 |
| [contracts/exploration.py](../src/mirage/contracts/exploration.py)，`property_configuration()` | 確定性輪替表示與運算元；梯度類別交替使用 eager／Inductor。 |
| [generator/program.py](../src/mirage/generator/program.py) | `property_request()` 組合提示詞；`analyze_property()` 驗證原始碼、AST、建構函式契約並產生 AST 指紋。 |
| [runtime/representations.py](../src/mirage/runtime/representations.py) | `equivalent_view()` 建構保持邏輯值的 view；記錄 conj/neg/layout 與 CSR values 中繼資料，避免直接讀取不透明 MKLDNN 儲存。 |
| [runtime/program.py](../src/mirage/runtime/program.py)，`run_property()` | 執行建構函式、注入表示、保存初始狀態、呼叫處理器、追蹤 API 進入／退出，附上設定、環境及中繼資料。 |
| [rules/properties.py](../src/mirage/rules/properties.py) | 實作 API 語義登錄表、五類處理器、實體化與觀測、有限值及對照組檢查、快照編解碼；不含問題編號分支或基準張量常數。 |
| [runtime/worker.py](../src/mirage/runtime/worker.py) | 新增 `execute_property` 模式，保存 `result.pt`、JSON 摘要、追蹤與程序狀態。 |
| [workflow.py](../src/mirage/workflow.py) | `run_properties()` 管理期限、生成、修復、執行及產物；`replay_property()` 驗證校驗值，不呼叫 LLM，以保存的初始資料重播。 |
| [artifacts/report.py](../src/mirage/artifacts/report.py) | `summarize_properties()` 分開統計生成、靜態有效、目標已執行及疑似失敗；不由判定結果推造 FPR 真值。 |
| [基準 evaluate.py](../benchmarks/pytorch_correctness/evaluate.py) | `validate` 將十個手寫案例接入正式 worker；`experiment` 執行凍結 LLM 評估；`report` 比對問題；後加的 `audit` 從保存輸入獨立重算參考。 |
| [sbatch/lib/development.sh](../sbatch/lib/development.sh)、[實驗工作階段腳本](../sbatch/experiments/pytorch_correctness.sbatch) | 迭代快照納入基準程式；兩張 GPU 分別供服務與測試；重用一個健康 vLLM 服務，保留失敗及耗時。 |

API 中繼資料解析、HTTP provider、程序隔離、數值比較、引擎快照及校驗值均沿用既有元件。例如 HTTP provider 的 source-only JSON schema 並非重新實作。`conditional.py`、`mr_actions.py`、`artifacts/bundle.py` 等雖列於核准範圍，也不能描述為全部重寫。

### 2.3 取樣規則與覆蓋限制

令新設定的嘗試索引為 `j`：

```python
representations = (
    'original', 'strided', 'offset', 'negative',
    'conjugate', 'singleton_stride', 'expanded',
)
representation = representations[j % 7]  # 僅 materialization / gradient_reference
subject = j // 7
backend = 'inductor' if j % 2 == 0 else 'eager'  # 僅 gradient_reference
seed = j
```

Worker 從 `args` 的**直接位置 Tensor 引數**中，以 `subject % tensor_argument_count` 選取一個，不遞迴處理關鍵字引數、稀疏 values 或模組參數。相同 `seed` 也識別 LLM 請求與嘗試次序；禁止 RNG 不代表測試必須呼叫 `manual_seed(seed)`。

| 操作 | 實作 | 前提與限制 |
| --- | --- | --- |
| original | 保留建構函式輸出。 | 不自動將張量連續化。 |
| strided | 配置尾維度大小為 2 的儲存，選取 `[...,0]` 並複製邏輯值。 | 產生 stride-two view，並非任意跨步搜尋。 |
| offset | 配置 `numel+1` 儲存，選取 `[1:]` 後重塑形狀。 | 儲存位移為 1，不搜尋位移範圍。 |
| negative | `torch.complex(zeros_like(x), -x.detach()).conj().imag` | 支援 FP32/FP64；邏輯值仍為 x，`is_neg()` 為 True，通常為 stride-two、offset-one 儲存。 |
| conjugate | `x.detach().conj_physical().conj()` | 運算元必須已為複數，不將實數 dtype 自動轉為複數。 |
| singleton_stride | 找第一個既有大小為一的軸，將 stride 設為 1，套用 `as_strided`。 | 不增加批次維度，也不保證該軸是 LU 的矩陣批次軸。 |
| expanded | 取最後一個長度大於 1 的軸的第一片，再展開至原形狀。 | 僅在該軸的值重複時合法，否則拒絕；不能為觸發缺陷而改變輸入值。 |

每個操作先以 `torch.equal(result, tensor)` 檢查邏輯值相等，再保存中繼資料。前提不成立時，可在目標呼叫前回傳 `inapplicable`。上下文、圖與編譯別名類別保留原始表示，由處理器建立協定。特別是 `compiled_alias` 處理器自行指定 Inductor，即使通用設定欄位 `backend` 顯示 eager，也不能推論未編譯。

**覆蓋缺口。**循環未包含稠密張量轉 MKLDNN／CSR、CSR `out.values()` 跨步注入、極端尺度搜尋或新增 singleton 批次維度。若 LLM 自行建構支援的結構，執行環境可處理，但轉接器存在不代表取樣器已探索對應狀態。這些缺口直接影響第 3 階段未重現結果。

### 2.4 建構函式契約與執行邊界

`analyze_property()` 要求非空原始碼且最多 40,000 字元，頂層僅有允許的 torch/math 匯入及未裝飾函式，並有唯一 `build(device, seed)`。它拒絕斷言、exec/eval/I/O、列入清單的 RNG／未初始化資料呼叫、私有屬性、global/nonlocal，以及大多數外部屬性賦值。區域類別基底以 AST 拼寫 `torch.nn.Module` 檢查，**不解析 `import torch.nn as nn` 別名**。

此驗證器執行測試契約，不證明一般 Python 合法性或完整確定性。單獨呼叫 `manual_seed` 即遭拒絕；反之，呼叫名稱黑名單無法證明 `nn.Linear` 的隱含參數初始化不使用 RNG。`x.requires_grad = True` 等合法賦值也可能被屬性寫入規則拒絕。第 3 階段靜態有效率須在這些限制下解讀。

Worker 要求建構函式回傳字典，`args` 為 tuple/list，`kwargs` 為字典。可遍歷張量的邏輯 `numel` 總和上限為 2,000,000，但不構成底層儲存位元組或模組參數的完整記憶體界限。契約違反、快照失敗或未能執行目標，均不會取得正確性 Pass。

### 2.5 性質語義與判定器實作

**A. `materialization`。**以相同 args／kwargs 建立實體化對照組。稠密張量依序 detach、解析 conj/neg、連續化、clone；CSR 重建索引與連續 values；MKLDNN 先轉為稠密邏輯資料，再轉回 MKLDNN。處理器僅適用於登錄表中經審查的 API。可寫入 `out=` 不得與位置輸入共用已觀察的儲存；BatchNorm 必須使用推論統計量。

除表示間比較外，處理器還提供 API 數學參考：scatter 使用純量迴圈，index 最多 8192 個元素；LU 以 `lu_unpack` 重建系統、`linalg.solve` 求解，並以 `atol=1e-8, rtol=1e-7` 檢查殘差；BatchNorm 使用 `(x-mean)/sqrt(var+eps)*weight+bias`；CSR 加法使用稠密算術；copy 對實體化邏輯來源作 dtype／裝置轉換、展開及 clone。這些分支編碼 API 語義，而非問題編號規則。其他純運算可能直接以實體化對照組作為參考，提供差異式證據，而非第三個獨立判定器。

**B. `gradient_reference`。**第一個運算元須為需要梯度的實數張量，最多二維、4096 個元素。detach 後建立獨立微分座標，測量對該運算元的導數，不保留建構歷史。處理器計算純量結果與 `torch.autograd.grad`；slogdet 使用 logabsdet。std/var 要求全元素純量歸約、正自由度及非零變異數；det/slogdet 要求無批次方陣、`cond<1e6`、`abs(det)>1e-12`。參考以 float64 公式計算，再轉回輸入 dtype：

```text
slogdet: d log|det(A)| / dA = A^(-T)
det:     d det(A) / dA = det(A) A^(-T)
std:     s = sqrt(sum((x-mean)^2)/(N-correction))
         ds/dx = (x-mean)/((N-correction)*s)
var:     dvar/dx = 2*(x-mean)/(N-correction)
```

處理器以 `atol=0, rtol=1e-4` 比較實際結果、實體化 eager 對照組及數學參考。Inductor 模式包裝降低後計算圖的呼叫，記錄實際執行的 FX 節點，因此圖覆蓋依據是執行證據，而非設定旗標。後端查找與 `_base`／`_version` 中繼資料使用版本相關的內部量測機制；生成觸發操作仍使用公開 API。

**C. `context_restore`。**LLM 提供上下文建構器的 args、kwargs 及 0..4 的深度。處理器記錄執行緒區域的 grad、inference、autocast enabled/dtype/cache 狀態，以每層新建上下文的遞迴作為對照，再遞迴呼叫由同一上下文物件裝飾的函式。退出檢查點須精確還原狀態。遞迴由執行協定建立，並非 LLM 自行建構完整裝飾器重現程式。

**D. `graph_accumulation`。**LLM 提供模組與 2..4 組固定簽章的 CUDA 批次。FX 檢查將運算與模組限制於經認證的確定性子集，拒絕可變緩衝區；模組的深層複本供 eager 參考使用。兩側從 `.grad=None` 開始，每批實數輸出加總成損失、逐批反向傳播，再比較最終參數梯度。目標由 `make_graphed_callables(module, first_batch)` 建立。此處 `reference` 與 `control` 是同一 eager 結果；第 1 階段輸入加總的解析計算才是該案例額外的數學佐證。

**E. `compiled_alias`。**LLM 提供 `inputs`、`output_path`、`delta` 及頂層 `forward(ctx)`。處理器建立兩份實體化上下文，執行 eager／Inductor 前向傳播，先確認初值一致，再沿 output_path 選取張量，檢查 eager 回傳集合中有超過一個張量與其共用儲存。兩側執行相同 `add_(delta)` 後，比較所有回傳值。僅在函式內使用 view、卻只回傳一個獨立張量，不符合此 MR。此處檢查回傳別名的可觀察效果，不是完整的輸入別名拓樸保留。

共用 `validate()` 依序檢查述詞、觀測值有限性、對照組／參考一致性，最後檢查實際／參考關係。預設容差為 `atol=1e-5, rtol=1e-4`；上下文採精確比較，梯度容差如上。非有限值與對照組不一致回傳 `verdict=None, correctness=unassessed`。自動結果保留 `confirmed_defect=False`，程式碼審查與獨立審核另行進行。容差是保守篩選政策，不是每種 dtype／核心都成立的誤差界限。

### 2.6 執行證據、快照與重播

`run_property()` 的目標包裝器在呼叫前記錄 `api_enter`，正常返回後才加入 `target_calls` 與 `api_exit`。因此例外追蹤可能顯示已進入目標，卻不計入完成結果的 `api_executed` 指標；`execution_failed` 不一律表示從未呼叫目標。完成結果保留 shape、stride、offset、storage group、dtype/device、requires_grad、leaf/base/version/grad_fn、neg/conj/layout、圖證據、判定、設定及環境。

| 產物 | 內容與用途 |
| --- | --- |
| `request.json` / `response.json` / `program.py` | 生成請求、含 HTTP 資料與 token 用量的 provider 回應，以及擷取的程式。 |
| `execution/request.json` | Worker 模式、原始碼、設定及引擎路徑，與 LLM 請求不同。 |
| `initial.pt` | 初始上下文與中繼資料；MKLDNN 使用邏輯資料與布局標記；標記 view 還原時保留 shape/stride/offset；模組保存 state_dict。 |
| `result.pt` / `summary.json` / trace / process logs | 完整張量觀測、精簡 JSON 中繼資料及判定、執行事件、例外與資源證據。 |
| `execution.sha256.json` / `engine-sha256.json` | 證據與程式校驗值，供重播驗證，避免修改後引擎無聲取代原實驗。 |

重播載入已保存引擎與原始碼，重建所需模組、還原初始資料及權重、檢查表示中繼資料，再執行性質，不呼叫 provider。除非明確允許，環境改變會抑制同環境判定。序列化遞增版本計數器，因此新圖重播簽章忽略絕對版本數；其他中繼資料不符則拒絕重播。此機制不還原任意保存的 autograd 圖或 Python 物件。

### 2.7 整合驗證與開發修正

`evaluate.py validate` 從 `INTEGRATION_CASES` 取出十個問題專屬建構函式，送進與生成程式共用的 `execute_property` worker。`representation=original` 表示案例已自行構造條件，不依賴取樣命中。std 設定指定 Inductor，compiled-alias 處理器也實際使用 Inductor。整合通過條件要求程序完成、目標證據、`suspected_failure` 及空的 `control_differences`，不能只看退出碼。

開發暴露多個 Mirage 整合問題：MKLDNN `torch.save` 初始失敗，使檢出僅達 9/10，加入布局標記快照後達 10/10；重播測試揭露版本計數器差異，後續再修正 view/base 重建；CLI 改讀性質結果的 `records`，而非訓練結果的 `experiments`；重播失敗也改為非成功 CLI 退出碼。這些修正處理 Mirage 整合，並非額外 PyTorch 缺陷。

最終第 2 階段 `phase2-frozen/summary.json` 檢出 10/10，完整套件通過 70 項測試。negative-view 與 MKLDNN/affine-view 重播均成功，provider_calls=0。後續 12 項性質測試與 4 項基準測試包含普通編譯 view、相同 graph microbatch、有限值／對照／未知檢查，以及獨立參考的已知答案驗證。時序保留清楚：這些後續測試並非全在第 3 階段前執行。

<a id="phase-3-details"></a>
## 第 3 階段：凍結協定下的 LLM 生成與執行

### 3.1 程式凍結、預算與模型

`evaluate.py experiment` 要求第 2 階段 `detected=total=10`，將引擎複製到 frozen-engine，並確認 Python 原始碼雜湊與已驗證版本一致。正式 `run_properties()` 對每個目標只接收 API 與裝置；問題編號留在外層基準路由與結果目錄，不進入提示詞上下文。本機模型為 Qwen/Qwen3-Coder-Next-FP8，版本 `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`，temperature 0、top-p 1、max_tokens 4096、上下文長度 16384。

每個 API 先由 worker 取得已安裝的簽章與文件字串，中繼資料時間另計；生成及執行再獲得最多 300 秒。剩餘低於 45 秒即停止派送；worker 逾時為 `min(120, remaining)`。候選上限 100000，但每次均因時間預算結束。性質生成與執行依序進行；第二張 GPU 專供 vLLM，不同時執行另一個目標 API。

### 3.2 實際送給 LLM 的提示詞

以下文字來自已保存的 HTTP 請求與 request.json，並與 `property_request()`、`description()` 核對。它記錄實驗實際提示詞，而非事後改善版本。每個使用者訊息依序串接**共用前綴＋類別後綴＋`\nContext:\n`＋JSON 上下文**。

**系統訊息，保留原文：**

```text
Follow the supplied testing contract. Return only the requested JSON object.
```

**共用使用者前綴，保留原文：**展示區塊省略末尾一個連接空格；實際串接類別後綴時保留該空格。

```text
Generate one small deterministic PyTorch property-test program from the installed API specification. Return JSON with exactly source. Use torch/math imports and define build(device, seed). build must return a dictionary whose named keys are specified below, never a tuple. Use explicitly initialized tensors (tensor, zeros, ones, arange, linspace), bounded dimensions and finite values. No RNG, external I/O, private APIs, assertions, or expected answers. The runner owns API invocation, state evidence, references, gradients and validation. Choose shapes, values, legal options and public representation operations yourself; vary them across attempt indices. Inputs may use ordinary views, strides, complex values, storage offsets or sparse layouts when the API supports them. Keep total storage below 2 million elements. Do not supply unrelated trainable parameters.
```

**五個類別後綴，保留原文：**

`materialization`:

```text
build returns args (tuple), kwargs (dict). They are the exact positional/keyword operands for the target API, including self for Tensor methods. The runner calls the API and compares legal logical materialization. Sparse CSR and MKLDNN values are supported. Avoid aliased writable operands and uninitialized data.
```

`gradient_reference`:

```text
build returns args and kwargs for the target. The first argument must be a real floating tensor requiring gradients. For std/var use a scalar reduction with dim=None; for determinant APIs use a square unbatched nonsingular matrix. The runner differentiates the scalar result (logabsdet for slogdet). Backend is controlled by configuration.
```

`context_restore`:

```text
build returns args and kwargs for constructing the target context manager, plus depth (integer 0..4). The runner checks nested decorator invocation against fresh contexts and observes entry/exit state. No threads are used.
```

`graph_accumulation`:

```text
build returns module (a deterministic torch.nn.Module on device), batches (2..4 tuples of fixed-shape input tensors). Use a small differentiable module without random layers, mutable buffers or control flow. A local class inside build is allowed. The runner sums real tensor outputs, accumulates gradients, and compares captured versus eager execution. Do not capture graphs yourself.
```

`compiled_alias`:

```text
build returns inputs (dict), output_path (list of tuple/list indices or dictionary keys), delta (finite scalar). Also define forward(ctx) returning tensors or a nested tensor structure. Choose a legal program whose output may contain views. The runner compiles forward, checks initial outputs and tests an add_ on the selected returned tensor against eager execution. Do not call compile, assert, or inspect storage/strides inside forward.
```

**完整紀錄的使用者訊息：slogdet / attempt-0000。**內容包含已安裝 API 文件與設定。`torch.randn` 範例來自公開文件字串，不是另加的隨機輸入指令。共用提示詞禁止 RNG，但文件保留隨機範例；兩者確實同時存在於模型輸入，不能以清理後的文件取代原實驗紀錄。

```text
Generate one small deterministic PyTorch property-test program from the installed API specification. Return JSON with exactly source. Use torch/math imports and define build(device, seed). build must return a dictionary whose named keys are specified below, never a tuple. Use explicitly initialized tensors (tensor, zeros, ones, arange, linspace), bounded dimensions and finite values. No RNG, external I/O, private APIs, assertions, or expected answers. The runner owns API invocation, state evidence, references, gradients and validation. Choose shapes, values, legal options and public representation operations yourself; vary them across attempt indices. Inputs may use ordinary views, strides, complex values, storage offsets or sparse layouts when the API supports them. Keep total storage below 2 million elements. Do not supply unrelated trainable parameters. build returns args and kwargs for the target. The first argument must be a real floating tensor requiring gradients. For std/var use a scalar reduction with dim=None; for determinant APIs use a square unbatched nonsingular matrix. The runner differentiates the scalar result (logabsdet for slogdet). Backend is controlled by configuration.
Context:
{"contract": "training-program", "program_kind": "property-1", "api_context": {"api": "torch.linalg.slogdet", "torch_version": "2.15.0+cu132", "signature": "linalg.slogdet(A, *, out=None) -> (Tensor, Tensor)", "signature_source": "installed docstring (unparsed)", "documentation": "linalg.slogdet(A, *, out=None) -> (Tensor, Tensor)\n\nComputes the sign and natural logarithm of the absolute value of the determinant of a square matrix.\n\nFor complex :attr:`A`, it returns the sign and the natural logarithm of the modulus of the\ndeterminant, that is, a logarithmic polar decomposition of the determinant.\n\nThe determinant can be recovered as `sign * exp(logabsdet)`.\nWhen a matrix has a determinant of zero, it returns `(0, -inf)`.\n\nSupports input of float, double, cfloat and cdouble dtypes.\nAlso supports batches of matrices, and if :attr:`A` is a batch of matrices then\nthe output has the same batch dimensions.\n\n.. seealso::\n\n        :func:`torch.linalg.det` computes the determinant of square matrices.\n\nArgs:\n    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.\n\nKeyword args:\n    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.\n\nReturns:\n    A named tuple `(sign, logabsdet)`.\n\n    `sign` will have the same dtype as :attr:`A`.\n\n    `logabsdet` will always be real-valued, even when :attr:`A` is complex.\n\nExamples::\n\n    >>> A = torch.randn(3, 3)\n    >>> A\n    tensor([[ 0.0032, -0.2239, -1.1219],\n            [-0.6690,  0.1161,  0.4053],\n            [-1.6218, -0.9273, -0.0082]])\n    >>> torch.linalg.det(A)\n    tensor(-0.7576)\n    >>> torch.logdet(A)\n    tensor(nan)\n    >>> torch.linalg.slogdet(A)\n    torch.return_types.linalg_slogdet(sign=tensor(-1.), logabsdet=tensor(-0.2776))", "documentation_truncated": false, "documentation_sha256": "6898d9f3ddd28d5031ee08773f52353307b8bf7564996cc75244e0b078176a42", "kind": "callable"}, "configuration": {"api": "torch.linalg.slogdet", "device": "cpu", "seed": 0, "attempt": 0, "family": "gradient_reference", "representation": "original", "subject": 0, "backend": "inductor"}}
```

原始 [request.json](../result/pytorch-correctness/763298/phase3/204407/attempt-0000/request.json) 與 [response.json 中完整 HTTP 資料](../result/pytorch-correctness/763298/phase3/204407/attempt-0000/response.json)提供來源紀錄。其他類別請求可見[實體化](../result/pytorch-correctness/763298/phase3/203078/attempt-0000/request.json)、[上下文](../result/pytorch-correctness/763298/phase3/202706/attempt-0000/request.json)、[計算圖](../result/pytorch-correctness/763298/phase3/181723/attempt-0000/request.json)及[編譯別名](../result/pytorch-correctness/763298/phase3/204033/attempt-0000/request.json)。

HTTP 的 `response_format` 使用嚴格 JSON 結構描述：

```json
{"type":"object","properties":{"source":{"type":"string"}},"required":["source"],"additionalProperties":false}
```

結構描述只限制 JSON 外層，不驗證 source 字串內的 Python。嚴格 JSON 仍可能包含 Python SyntaxError，因此不保證可執行。上下文欄位 `contract="training-program"` 沿用 HTTP provider 的 source-only schema 選擇器，`program_kind="property-1"` 才識別新契約；這些建構函式不是完整訓練程式。

### 3.3 修復訊息、停止條件與成功生成鏈

控制器僅對 `invalid_program` 最多修復兩次，在訊息序列中附上前一份 assistant 內容與下列診斷：

```text
Repair this diagnosed problem only:
ContractError: Unsupported or nondeterministic builder call: manual_seed
```

三個已重現問題的首次成功鏈都遇到此診斷。[slogdet attempt-0004](../result/pytorch-correctness/763298/phase3/204407/attempt-0004/request.json) 的前一回應包含 `manual_seed`；修復移除該呼叫，保留明確矩陣 `[[2,1],[3,4]]`。請求沿用 attempt/seed=3、negative、subject=0、eager。隨後由**執行環境**注入 negative-bit 表示、計算目標梯度，約 8.78 秒後回報違反。

同樣，[autocast attempt-0003](../result/pytorch-correctness/763298/phase3/202706/attempt-0003/request.json) 沿用設定 attempt=2；[copy attempt-0011](../result/pytorch-correctness/763298/phase3/200613/attempt-0011/request.json) 沿用 attempt=10、subject=1，因此轉換選到第二個位置 Tensor，即 CPU source。目錄嘗試編號與設定不同，是修復消耗下一次迴圈但保留原設定所致，並非紀錄錯置。

**修復限制。**執行時 TypeError、快照／pickle 錯誤、對照組不一致、Pass 與疑似失敗都不進入靜態修復分支。問題比對條件也不回饋 LLM。因此缺少 Tensor 方法接收者或回傳區域函式，並未獲得執行回饋自動修復。若無法確認 provider 已取消請求，控制器會寫入標記並停止後續服務重用，不將客戶端連線關閉當成取消證明。

### 3.4 從執行紀錄推導結果

本輪產生 1,154 個已解析原始碼候選，包含 661 個修復回應。其中 310 個通過靜態契約、226 個 worker 完成、130 個完成且保留目標執行證據、125 個得到適用比較，包含 97 個 Pass 與 28 個疑似失敗。生成失敗、MR 不適用、對照組不一致與目標違反各保留不同分母，不合併為單一未發現類別。

原始凍結比對器的 2/10 結果已保留。獨立審核與問題／程式碼審查隨後支持三個已知問題。原 copy 比對器只檢查 conjugate 旗標，遺漏上游 negative/sliced 變體；修正僅改變基準歸屬，不改提示詞、引擎或生成輸入。兩個 negative-bit／Inductor std 梯度案例雖獲獨立數學佐證，仍不符合 197089 的極小輸入條件，不增加十個已知案例的重現數。

獨立審核由保存的初始輸入重算參考，使用 Python Fraction 反矩陣、math.fsum/hypot std，以及純量正規化與複製；不重跑目標，也不以正式預期輸出當答案。19 TP、50 TN、0 FP、0 FN 僅適用於 69 個已審核案例；其餘 1085 個候選，包括 9 個疑似案例，保持未知。圖、LU、compiled-alias 缺乏對應審核支援，或輸入超出審核範圍時，不補造真值。

### 3.5 可重現性、成本與尚待完成的能力

工作 763298 僅啟動模型服務一次，耗時 192 秒，重啟零次。每次迭代先複製 src/tests/sbatch/benchmarks。凍結第 3 階段迭代 `iteration-pwpdkQ` 耗時 2597 秒；API 迴圈合計 2567.91 秒，中繼資料探查 21.29 秒。配置持續 4208 秒、使用 2 張 GPU，合計 8416 GPU-seconds。這些是配置資源而非 GPU 使用率；重疊的服務與測試時間不能相加為實際經過時間。清理移除測試暫存、停止服務、關閉終端，排程與計費紀錄確認所有本任務所屬步驟退出。

證據支持有限範圍的結論：五類受控性質協定驗證十個已知正向案例，盲生成與取樣器組合找到其中三案。結果不證明 LLM 自主生成十案完整 MRE、所有被拒 Python 均非法、MR 適用性可推廣至任意模型，或整體 FPR 為零。別名解析 AST、執行契約修復、巢狀／關鍵字運算元取樣，以及布局／尺度探索，都是結果指出的方向，**並非本輪已實作或經實驗驗證的能力**。

## 已核准的雙語編修

使用者授權潤飾本路線圖與 `docs/pytorch-correctness-evaluation.md` 的語言及組織方式，並於 `reports/` 建立同名繁體中文對照報告。本次採用 `ml-paper-writing` 技能，保留實驗過程、結論、數據、限制及提示詞原文。實施順序為潤飾英文文件、製作對應中文譯本，再核對章節完整性、數值與技術內容、提示詞一致性及連結。本次僅編修文件；未授權因這次編修而變更實作、實驗產物或重跑實驗。

編修驗證已完成：兩份英文文件無中文文字；每份中文報告均保留對應章節與段落內容。原有連結目標皆保留，本機連結與明示錨點均可解析。數值表格與編修前文件一致，兩個語言版本的評估報告均符合已保存產物中的 10 列執行漏斗及 10 列獨立審核統計。Python 範例的可執行 AST 保持一致，JSON 範例未變；系統訊息、完整使用者訊息範例、共用前綴、五個類別後綴與修復訊息均保留紀錄原文。另以逐段審閱核對雙語措辭及限制。僅編輯四份授權文件，未重跑實作測試或實驗。
