# Mirage MVP / POC 範圍與實作提案

本文件整理使用者在 28 題範圍訪談中確認的需求。Mirage 是 PyTorch
軟體測試框架，探索 tensor memory representation 與 training state。

需求範圍已確認，後續實作範圍與檔案已獲使用者核准。本文保留原始需求與
階段提案以利追溯；目前交付內容與驗證限制見第 13 節。
vLLM serving 與平行 worker 的實作／配置見第 14 節；A/B 模型比對依使用者指示延後。
軟體功能與研究成果分開驗收，不因實作完成而宣稱已找到新 bug。

## 1. 目標與驗收

### 研究成果

兩個方向各找到至少一個尚未通報、不同根因的 PyTorch bug：

| 方向 | 必須建立的觸發證據 |
| --- | --- |
| Tensor representation | 問題依賴 stride、contiguity、storage offset、storage identity 或 aliasing |
| Training state | 問題依賴 activation checkpointing、RNG／重算，或與 compilation、訓練生命週期的互動 |

合法用法下的錯誤數值、梯度、訓練狀態與非預期 crash 都可成為候選。
確認成果前需重播、最小化、重新驗證合法性與 MR 假設、取得獨立佐證，並查重。
查重需記錄搜尋範圍與時間，不宣稱能證明所有未公開問題都不存在。

同一根因影響多個 API 不重複計數，也不能用同一根因同時滿足兩個方向。
已知 bug 可驗證測試器，但不計入新 bug 成果。

### 軟體交付

- 可批次執行的 CLI。
- 可替換的 Python LLM provider 介面。
- coverage feedback 驅動的抽樣。
- subprocess 執行、runtime evidence 與條件式 MR 判定。
- 完整 artifacts、無 LLM 的案例重播、人工最小化後的重新驗證。

軟體交付與研究成果分開驗收。完成測試框架不保證找到兩個新 bug。

## 2. 環境與 API 池

| 項目 | 已確認範圍 |
| --- | --- |
| Framework | 只測 PyTorch |
| 主要執行環境 | Nano4，單張 NVIDIA H200 |
| CPU | 適用案例的輔助對照及複驗 |
| Framework 版本 | 固定 stable 探索，候選失敗再到 nightly 複驗 |
| Worker | 獨立 subprocess，使用 CLI 所在的 Python 環境 |
| 執行模式 | 一般 eager；compile 的 eager、aot_eager、inductor backend |
| 平行範圍 | 第一版單 GPU；後續 serving／測試 GPU 分工見第 14 節，單一測試仍使用單 GPU |

固定 API 池如下，CLI 可指定子集：

1. `torch.matmul`
2. `torch.nn.functional.linear`
3. `torch.nn.functional.layer_norm`
4. `torch.nn.functional.gelu`
5. `torch.nn.functional.softmax`
6. `torch.nn.functional.dropout`
7. `Tensor.add_`
8. `Tensor.mul_`

每個 API profile 記錄輸入限制、必要 backward path、可用轉換、MR 與 observables。
Mutation 案例需另外檢查 autograd 合法性；dropout 需專屬 RNG 前提。

Python、stable PyTorch、CUDA 與驅動版本於 Nano4 環境確認後固定。
現有專案要求 Python >= 3.14；不能未經驗證就假定選定 wheel 與環境相容。
Nightly 由另一個既有 Python 環境啟動 CLI 重播，不自動變更 stable 環境。

## 3. 探索空間

### Tensor representation

- 同 shape、同邏輯值的不同 stride／contiguity。
- 零與非零 storage offset，包括仍 contiguous 的 offset view。
- Clone／新配置所造成的 storage identity 差異。
- Shared／independent storage 與受控 alias 拓撲。
- 不同構造路徑，以及不同表示的 compile warm-up 歷史。

Runtime 必須在指定 checkpoint 取得 metadata，不能只檢查生成程式是否包含轉換。
Storage identity 以單次執行內的關係表示，不比較不同 process 的原始位址。
比較梯度時需對齊邏輯輸入與座標，避免將 leaf／view 差異誤判為梯度遺失。

### Training state

- Activation checkpointing 開關、必要的 RNG 保存與重算。
- Forward、backward、Adam update，以及第一輪與後續 iteration。
- 必要的 gradients、parameters、Adam step、first／second moments。
- Compilation 與 warm-up 歷史。
- 精度模式所需的 autocast 與 GradScaler 狀態。

| 精度模式 | 訓練設定 |
| --- | --- |
| FP32 | 一般訓練 |
| FP16 | Autocast、FP32 parameters、GradScaler |
| BF16 | Autocast、FP32 parameters |

三種模式分開計 coverage，不預設跨 dtype 等價。
FP16 必須記錄 scale 與 update 是否跳過；對齊語意 checkpoint 後才比較 Adam 狀態。
AMP 的實際 operator dtype 必須記錄，不能將配置名稱當成每個 API 的實際 dtype。

## 4. MR library

第一版包含四類 MR；每類可包含按 API／backend 特化的具體規則。

| 類別 | 核心條件與比較 |
| --- | --- |
| Layout／offset | 保留 MR 所需的 shape、值、dtype、device、初始狀態，只改表示 |
| Storage identity／aliasing | 無 mutation，且語意不依賴 identity／alias 關係 |
| Checkpoint 等價 | 保持必要 RNG 與初始狀態，排除未受控的重算副作用 |
| Mutation eager／compiled | 配對內保持相同值及 alias 拓撲，比較結果與必要 mutation 效果 |

每條具體 MR 必須包含：

- Preconditions 與 Input Relation。
- 允許差異與必須保留的初始條件。
- Runtime probes 與證據不足時的處理方式。
- Observables、語意 checkpoint、Output Relation。
- 按 API、dtype、backend 與 observable 定義的容差及 nondeterminism policy。

不要求所有 metadata 或 observable 永遠相等。Mutation 情況下不能直接把 shared
storage 換成 clone，並假設結果不變。相同 seed 也不足以建立跨 backend 的 RNG 等價。

對任意生成程式，若無法驗證副作用、輸入關係或必要語意前提，不能給予 MR Pass。
靜態限制、受控 skeleton 與 runtime probes 必須共同提供規則要求的證據。

## 5. 三層流程與 LLM

```text
選 API
  → 生成 canonical program 並驗證 target API／backward
  → 選目標狀態 predicate、checkpoint 與 MR
  → 生成合法 transition plan
  → 增量填入 training skeleton
  → 構造 source／follow-up execution cases
  → subprocess 執行與收集證據
  → 狀態及 MR 動態驗證
  → 判定、保存與 coverage feedback
```

LLM 使用可替換的 Python provider 介面。第一版不綁模型、不實作 HTTP adapter。
第 14 節在此介面上新增 vLLM HTTP adapter，保留既有 provider 的使用方式。
介面應傳入生成階段、結構化上下文與診斷，回傳生成內容及可取得的 usage 資訊。
Provider 未提供 token 使用量時記錄 unknown，不填成零。

第一版以四階段 provider 介面串接生成。第 16 節的 A＋B 更新改為由 LLM 生成
canonical 計算圖與 transition plan，framework 綁定 skeleton 並組裝 follow-up；
每次生成後檢查語法、合法性、必要路徑與靜態 MR 條件。

### 生成程式修正

- 每個生成步驟最多重試 2 次，即初次生成加最多 2 次修正。
- 只修正語法、已確認非法的用法或缺失測試路徑。
- 保存每次 prompt／response、診斷、版本與成本。
- Suspected failure 不進入自動修正流程。
- 原因未明的 exception 保存待查，不能單憑報錯就判定是生成程式錯誤。
- 修正後若失去選定 MR，可重選 MR 或保留 coverage-only；不能偷換測試目標。

## 6. 抽樣與 coverage

主要抽樣單位為 `API × 抽象狀態 × MR`。
依已驗證覆蓋次數反比加權，並保留均勻隨機探索機率。

待核准的初始公式：

```text
w(c) = 1 / (1 + verified_count(c))
p(c) = epsilon / |C| + (1 - epsilon) * w(c) / sum(w)
epsilon = 0.10
```

`C` 是當輪明確記錄的候選集合；這個公式不代表集合中每個候選都已證實可達。
抽象狀態包含相關 representation、backend、精度、checkpoint 與訓練階段。
具體分箱、state key、transition key 與各覆蓋分母需在實作時以明確資料結構定義。

只有取得必要 runtime evidence 才更新相應 coverage。
MR violation 仍可貢獻已驗證的執行覆蓋；coverage-only 的正確性維持未評估。
生成失敗、未解決計畫、未達狀態與 inapplicable 分別記錄。

輸出 API Coverage、State Coverage、Transition Coverage、API × MR Coverage、
MR Applicability、Correctness Feedback 與 Generation Cost。
每個比例列出候選集合、分母與定義；不宣稱覆蓋全部理論空間。

## 7. Runtime 與判定

每個 execution case 使用獨立 process；同一 case 的 compile warm-up 與正式執行
保留在同一 worker 內，以維持該案例指定的 cache／graph lifecycle。
CPU、GPU、backend 與 dtype 的比較條件由 MR 定義。

| MR verdict | 定義 |
| --- | --- |
| Invalid | 程式非法，必要 API／backward 未執行，或目標狀態未到達 |
| Inapplicable | Preconditions／Input Relation 不成立，或 checkpoint 無法對齊 |
| Pass | MR 適用且 Output Relation 成立 |
| Suspected Failure | MR 適用且 Output Relation 違反 |

Execution outcome 與 MR verdict 分開。Crash、timeout、OOM、exception、執行完成
皆需記錄；未完成且缺少 MR 證據時，MR correctness 可保持未評估。
合法用法下的 crash 候選可走獨立 crash oracle／人工調查流程，不偽造 MR applicability。

數值比較需處理 shape、dtype、有限值、NaN／Inf、絕對與相對誤差，以及必要的離散狀態。
不將 NaN 視為自動等價，也不為了讓失敗通過而放寬容差。
確切容差須按 API／MR／精度制定並用正反案例驗證，不能只沿用全域 allclose。

候選失敗依序重播、人工最小化、重新驗證 MR、stable／nightly 複驗及獨立佐證。
對隨機案例的有限差分或高精度對照，也必須先固定其所需隨機條件。

## 8. Artifacts、CLI 與預算

Artifacts 必須能在無 LLM 呼叫時重播，包括：

- Experiment specification、候選集合、抽樣 seed 與選擇記錄。
- 程式及生成／修正歷史。
- 初始參數、輸入、必要 optimizer／scaler／RNG 狀態。
- 可重建 storage group、view、offset、stride 與 alias 關係的資料。
- Warm-up 程式與執行歷史。
- Python／PyTorch／CUDA／硬體／backend 版本與設定。
- Runtime trace、observables、MR 證據、verdict、stdout／stderr 與資源成本。

保存 tensor 值不能取代 alias 拓撲重建。重播需驗證 reconstruction 的值與 metadata。
同環境重播及跨版本複驗分別記錄；跨版本有意改變的環境欄位必須可追溯。

CLI 擬提供 `run`、`replay`、`inspect` 與 `report`。
人工修改候選程式後可重新驗證，產生衍生 artifact，不覆寫原始證據。

下列為待核准的初始預設值，可由 CLI 配置：

| 參數 | 提案預設 |
| --- | --- |
| 一批候選 experiment 上限 | 100，包含生成未成功的候選 |
| 批次 wall-clock 上限 | 4 小時；額外受外部排程限制 |
| 單 worker 逾時 | 600 秒，包含 warm-up／compile |
| 每個生成步驟修正次數 | 2，已確認需求 |
| 均勻探索機率 | 0.10 |
| Training updates | 3 次，記錄 attempted／successful update |

預算耗盡時保存已取得證據與結束原因，不將未完成案例算 Pass。
LLM provider、worker 與批次時間都計入成本；provider 的取消／逾時契約需明確實作。
容差、warm-up 次數與失敗重播次數仍需在具體 MR 設計時確定。

## 9. 第一版排除項目

- 多 framework、多 GPU 調度、distributed training。
- Offloading、獨立 scheduler／DataLoader state 探索。
- 直接將 parameters 轉成 FP16／BF16 的訓練模式。
- 自動最小化、自動安裝 nightly、自動 upstream issue 提交。
- 綁定特定本地模型或 HTTP serving 協定。

上述為原始第一版邊界；第 14 節新增 HTTP serving 與跨 experiment 平行執行方案。
Distributed training 與單一測試跨 GPU 執行仍不在範圍內。

## 10. 實作順序與檔案授權提案

保持既有七個 package 邊界。使用獨立模組、短函式、明確型別及必要註解。
註解重點是 MR 前提、狀態／證據邊界、alias 重建與非直覺的 lifecycle 行為。
不把所有功能集中在 `__init__.py`，也不做無關重構。

### 第一批：可測試的控制與資料基礎

先實作以下明確檔案；這批不宣稱已能完整執行 PyTorch bug 搜尋。

| 檔案 | 責任 |
| --- | --- |
| `src/mirage/contracts/models.py` | Experiment、execution、evidence、MR verdict 與 outcome 資料結構 |
| `src/mirage/contracts/apis.py` | 已核准的 8 個 API 識別及選擇驗證 |
| `src/mirage/generator/provider.py` | Python provider protocol、request／response 與 usage |
| `src/mirage/generator/repair.py` | 有限修正流程、明確可修正診斷與版本記錄 |
| `src/mirage/sampler/coverage.py` | 有證據支持的 coverage 計數、明確候選分母 |
| `src/mirage/sampler/policy.py` | 可重現的反比加權及均勻探索 |
| `src/mirage/artifacts/store.py` | 版本化 JSON manifest、原子寫入及衍生 artifact |
| `src/mirage/cli.py` | CLI 參數、API 子集、artifact inspect／report |
| `src/mirage/__init__.py` | 保持現有入口，委派至 CLI |
| `src/mirage/__main__.py` | 支援 `python -m mirage` |
| `tests/test_contracts.py` | Invalid／inapplicable／未評估等狀態邊界 |
| `tests/test_generation.py` | 重試上限、unknown exception／suspected failure 不修正 |
| `tests/test_sampling.py` | seed 可重播、反比權重、證據與分母規則 |
| `tests/test_artifacts.py` | 寫入／讀回、原始證據保留、manifest 驗證 |

第一批使用標準函式庫與 `unittest`；不需要 PyTorch 或新增 runtime dependency。
尚未完成的 `run`／`replay` 能力不得回傳成功或生成假的 coverage。

### 後續批次

1. 實作 API profiles、具體 MR、受控 skeleton、state probes 與 comparator。
2. 實作 subprocess runtime、training harness、alias／RNG 重建與重播。
3. 接上 provider 與完整 CLI run，保存 generation／execution feedback。
4. 本機控制流程驗證後，在 Nano4 驗證 CPU／H200、AMP、compile 與 warm-up。
5. 以已知案例校準，開始 stable 探索與 nightly 複驗。

後續批次在編寫前提供確切檔案清單與驗證內容；依 repository 規範核准後實作。
`AGENTS.md` 的既有使用者修改不屬於本次編輯範圍。

## 11. 驗證計畫與目前限制

第一批以 `unittest` 驗證資料契約、修正邊界、抽樣、coverage 與 artifact round-trip。
再檢查 Python 語法及 CLI help／inspect／report 的真實輸出。

完整執行階段需增加：

- 合法通過案例、刻意破壞 MR 的負例、缺失證據與錯誤 state checkpoint。
- Target API 及 backward 的實際執行證據。
- Source／follow-up 初始值、RNG、alias 與語意 checkpoint 對齊。
- Subprocess timeout、非零退出、部分 artifact 與未知 exception。
- FP32／AMP、Adam／GradScaler、各 compile backend 與 warm-up。
- 同環境重播、人工修改後重新驗證及跨版本複驗。

實際驗證環境與尚未執行的檢查見第 13 節；未執行的檢查不標示通過。

## 12. 參考文件

- [既有 Mirage 架構](../docs/Architecture.md)
- [Repository agent instructions](../AGENTS.md)
- [Nano4 服務簡介](https://man.twcc.ai/@nano4-manual/S1gcaUewbe)
- [Nano4 系統架構與規格](https://man.twcc.ai/@nano4-manual/SJuKzVlwbx)
- [PyTorch checkpoint 文件](https://docs.pytorch.org/docs/stable/checkpoint.html)

線上文件可能隨版本變動；實驗契約以固定環境對應的文件與 runtime evidence 為準。

## 13. 已核准實作與交付邊界

實作保留既有七個 package，並以 `workflow.py` 連接三層流程。
使用方式與 provider 契約詳見 [README](../README.md)。

| 範圍 | 實作內容 |
| --- | --- |
| API profiles | 八個 API、必要 backward、輸入限制及受控 forward／loss grammar |
| 抽樣 | 明確候選集合、API 邊際選擇、canonical 驗證後的 state／MR 選擇、反比計數與均勻探索 |
| 狀態 | contiguous、strided、nonzero offset、clone、shared storage；checkpoint、首輪／末輪、三個語意 checkpoint、warm-up 歷史 |
| 生成 | Python provider factory、四階段生成、每階段執行驗證、最多兩次修正、完整歷史及 unknown usage |
| Runtime | 獨立 subprocess、同 worker warm-up／compile、外部逾時與 process-group 清理、部分 trace 保存 |
| 訓練 | Adam、FP32、FP16 autocast＋GradScaler、BF16 autocast、四種 backend、實際輸出 dtype |
| MR | Layout／offset、storage identity／alias、checkpoint、mutation eager／compiled，依前提限制 applicability |
| Inspector | 執行與 backward 證據、state／action 驗證、初始狀態與程式關係、checkpoint 對齊、條件式數值比較 |
| Artifacts | 初始 tensor／RNG、alias 重建 recipe、engine 原始碼快照、trace、observables、環境、hash、時間與可取得的資源用量 |
| Replay | 無 LLM 重播、跨環境差異記錄、人工縮小 shape／update／iteration、修改程式後重新驗證、衍生 artifact |
| CLI | `run`、`replay`、`inspect`、`report`；批次與逐案例進度報告 |

### 具體探索界線

第一版以 square rank-two FP32 parameter 為可訓練張量，從它建立 nonleaf API operand。
Binary API 的 weight 為該 operand 的 detached copy 或 detached alias，bias 是保存的固定值。
對齊 parameter 的邏輯座標比較梯度，避免 leaf／view 或 in-place rebasing 造成假性缺失。

此節記錄 version-one 實作界線；現行 version-two 以第 16 節為準。
第一版 LLM 生成程式須符合可驗證的單一目標運算與 loss grammar，可選擇合法的 dropout p、
softmax dim、GELU approximate 或 layer-norm eps。Skeleton／transition 保留 canonical
計算；follow-up 的合法計算差異會使 MR input relation 不成立，仍保存執行覆蓋。
任意模型、任意 Python 副作用與全新的 LLM 自創 oracle 不在此 grammar 內。

State key 明確包含 representation、checkpointing、iteration 及 warm-up；另以
checkpoint、backend、precision 區分覆蓋。Device 固定於單一 batch 並存於配置，
報告不將不同 batch 的覆蓋比例相加。候選集合不是已證實可達的全集。

容差 policy version 1 按 API、精度、backend、MR 及 observable 保存，拒絕 NaN／Inf
自動等價；Adam step 與 scaler state 使用精確比較。這些是篩查容差，實際失敗仍須
重新驗證數值條件，不能因為容差比較失敗就確認 framework defect。

### 驗證環境與研究工作

使用現有 `.venv` 的 Python 3.14.10，未另行安裝 Python。PyTorch 2.15.0+cu132
與驗證依賴放於 `/tmp/mirage-validation-deps`，未修改 `pyproject.toml` 或使用者的
`uv.lock`。可用 GPU 為 NVIDIA H100 NVL，driver 580.65.06。

此處是先前驗證的歷史紀錄，不代表目前 `.venv` 已安裝 PyTorch，亦不代表
compute node 能讀取該 `/tmp` 環境。後續環境與大型檔案依第 14 節的持久儲存規範準備；
所有運算密集驗證必須透過 Slurm。

驗證包含資料契約、抽樣、修正邊界、subprocess timeout、四類 MR、刻意注入的負例、
獨立 scalar matmul 梯度公式、checkpoint RNG、warm-up 還原、CPU／CUDA、AMP、
compile，以及完整生成／保存／無 LLM 重播／人工縮小流程。實際執行結果以交付時
測試輸出為準；跳過或失敗的項目不能宣稱已通過。

本次交付啟用 `MIRAGE_TEST_COMPILE=1`、`MIRAGE_TEST_CUDA=1` 與
`MIRAGE_TEST_E2E=1`，61 個 unittest 全部通過，沒有跳過項目；Ruff F／E9、
53 個 Python 檔案的語法解析與 `git diff --check` 亦通過。另以 CLI 實跑
CUDA／Inductor／FP16 的 shared-storage MR 與 alternate warm-up，批次及重播
均得到 Pass，重播 provider calls 為 0，環境比對一致。這些結果不代表完整
PyTorch API 或全部狀態組合已被覆蓋。

截至前一階段，仍未取得 Nano4 H200、另一個 nightly 環境或真實 LLM provider 的整合驗收證據；
本次新增的 H200 與模擬 HTTP 整合驗證見第 14 節。
`reference` provider 僅用於確定性的整合驗證，並非 LLM。尚未完成兩個新 bug 的
探索、最小化、獨立佐證及查重；這些仍是研究成果驗收，不能由軟體測試通過替代。

## 14. vLLM serving 與平行 worker

本次實作以 `Qwen/Qwen3-Coder-Next-FP8` 為預設服務模型，增加 HTTP provider、
有界 experiment 排程及 Slurm 服務編排。依使用者後續決定，**A/B 比對延後**：
不新增 Qwen3.8 載入、模型切換、比較命令或比較報告。

### 14.1 程式分工與執行語意

| 檔案／範圍 | 實作責任 |
| --- | --- |
| `generator/http_provider.py` | 標準函式庫 HTTP client、JSON schema、取樣設定、原始請求／回應及 usage 保存 |
| `generator/provider.py`、`repair.py` | `vllm` provider 選擇、跨 subprocess 回應 metadata／服務錯誤分類，服務錯誤不進入程式修正 |
| `scheduler.py` | 有界 experiment 數、獨立生成配額、集中抽樣與 coverage 提交、暫停／取消及順序紀錄 |
| `workflow.py` | 每案例隔離、四階段生成與驗證、完成後提交 coverage、排空工作後獨占重播 |
| `runtime/process.py`、`worker.py` | 取消 token、逾時終止 process group、保留程序與 provider 錯誤證據 |
| `contracts/programs.py`、`cli.py` | worker／生成並行上限與參數驗證、服務故障的非零退出狀態 |
| `sbatch/lib/serving.sh`、`environment.sh` | 服務啟停、環境／就緒檢查、GPU 隔離、時間預算與輸出配置 |
| `sbatch/experiments/parallel.sbatch` | 同一 allocation 內啟動 serving 與 Mirage 測試 |
| `sbatch/validation/` | 控制流程測試、真模型 smoke、完整 PyTorch 整合測試 |

HTTP adapter 透過 `/v1/chat/completions` 要求包含 `source` 與 `actions` 的 JSON。
格式約束不取代既有 AST、target API／backward、state 與 MR 條件檢查。
Canonical、skeleton、transition、follow-up 依既有順序執行，每個 accepted stage 都驗證。
同一組 source／follow-up 依序執行；不同 experiment 才會平行推進。

每案例有自己的 subprocess、初始值／RNG、engine snapshot 與輸出目錄。
Controller 在完成時統一提交 coverage，仍由 Coverage 拒絕重複 experiment；
在途計畫不能計入已驗證覆蓋。Report 保存 dispatch、selection 的 coverage snapshot
與 completion 順序；同 seed 不保證非同步抽樣順序一致。

疑似失敗會暫停新派發，等所有在途案例結束後獨占測試 GPU 重播；重播不呼叫 LLM。
OOM／timeout 保留為執行結果，不直接宣稱 framework defect。成本分別記錄
wall-clock、各程序時間與生成時間，重疊的時間類別不得相加當作總耗時。

### 14.2 配置與資源

| 配置 | 預設／行為 |
| --- | --- |
| Provider | CLI 明確指定 `--provider vllm`；保留 `module:factory` 與 `reference` |
| CLI workers | `--workers 1`，可調整；整合 Slurm job 預設 2 |
| 生成上限 | `--generation-concurrency 2`；與 worker 數獨立配置，無無界排隊 |
| Slurm | account `ACD115198`、partition `8gpus`、1 node、8 CPUs、4 小時 |
| 整合 job GPU | 明確由原 template 的 1 張改成 2 張；serving／testing 各 1 張 |
| Step 分工 | 兩個 `srun --exclusive --exact` step，各 4 CPUs、1 GPU 與半數 allocation 主記憶體；保留 Slurm GPU visibility |
| 模型 | `Qwen/Qwen3-Coder-Next-FP8`；只使用已準備的本機權重 |
| Context | `MIRAGE_MAX_MODEL_LEN=16384`；輸入與輸出總長需在此上限內 |
| 輸出／取樣 | `MIRAGE_MAX_TOKENS=2048`、temperature 0、top-p 1；每請求帶案例 seed |
| HTTP deadline | `MIRAGE_HTTP_TIMEOUT=120` 秒，另受 worker 與 batch 剩餘時間約束 |
| 就緒期限 | `MIRAGE_SERVE_START_TIMEOUT=900` 秒，受 allocation 剩餘時間約束 |

完整環境變數與提交方式見 [sbatch/README.md](../sbatch/README.md)。
兩個 Mirage worker 共享測試 GPU，可能增加記憶體壓力，不保證提高吞吐量。
單張 H200 是否能載入指定模型並以此 context／並行度運作，仍待真模型驗證；
不能由參數量或腳本存在推定已成功，失敗時不自行改變量化或增加 GPU。

每個 step 明確指定 `--gpus-per-node=1`、`--gpus-per-task=1` 與 `--mem`。
實測僅設定 `--gpus-per-task=1` 仍會繼承 job 的 GPU／記憶體配置，阻塞另一個 step；
`mirage_split_resources` 依 Slurm 的 allocation 記憶體環境變數平分，不額外申請資源。

### 14.3 環境與儲存

不預設 vLLM 已安裝。使用者已補上 PyTorch：目前專案宣告 `torch>=2.15.0`，
`.venv` 可找到 PyTorch 2.15.0，實際 compute-node 版本以 job 的 `environment.json` 為準。
該階段尚未在測試環境找到 `vllm` 與 `transformers`，因此先完成程式與模擬服務測試。
後續經使用者核准的獨立 serving 環境與模型準備見第 15 節；被測環境及使用者的
依賴／lockfile 不因 serving 安裝而變更。

| 位置／變數 | 用途 |
| --- | --- |
| `MIRAGE_PYTHON` | compute node 可讀的 Python >=3.14 與被測 PyTorch 環境 |
| `MIRAGE_SERVE_PYTHON` | 獨立且相容的既有 serving Python；同目錄需有 `vllm` executable |
| `MIRAGE_MODEL_PATH` | `/work/asciibase64/Mirage/models/<model>/<revision>/` 的已準備權重 |
| `MIRAGE_MODEL_REVISION` | 明確記錄使用者準備的 snapshot revision；不是權重來源的自動驗證 |
| Serving 環境若需另建 | `/work/asciibase64/Mirage/environments/vllm/<environment-id>/`，沿用既有 Python |
| 持久 cache | `/work/asciibase64/Mirage/cache/huggingface/` 與 `cache/vllm/` |
| 測試／探索／重播結果 | `./result/<category>/<job-id>/` |
| Slurm 日誌 | `./result/<category>/logs/<job-name>-<job-id>.out`／`.err`；提交前建立父目錄 |
| 一次性編譯 cache | job-specific local scratch；必要證據保存到結果目錄 |

vLLM 套件與模型權重需分別準備。腳本不自動安裝 Python／套件，並以 offline mode
載入本機模型，避免正式工作觸發隱含下載。Serving 與被測 PyTorch 分開，避免 vLLM
依賴替換測試版本。`uv sync` 會依現有專案依賴安裝 PyTorch，但不會安裝 vLLM 或權重。

### 14.4 服務生命週期與故障語意

「常駐」只指本次 allocation 內載入一次、持續接受請求；job 結束即停止。
短命的 provider subprocess 只送 HTTP，不載入模型。

1. 檢查 interpreter、本機模型路徑、context／output 限制與可用 port。
2. 啟動 serving step，保存套件版本、宣告 revision、config hash、host、job／step ID
   與 GPU UUID；等待 health 與 model ID 就緒。
3. 在 testing step 確認 PyTorch、單張可見 GPU，並用 UUID 驗證與 serving GPU 不同。
4. 啟動受剩餘時間約束的 Mirage batch；保留保存證據與關閉服務所需時間。
5. 完成、取消或服務故障時停止子程序與 serving，保留非零錯誤退出狀態及日誌。

HTTP status error、連線失敗、逾時、錯誤回應格式與截斷回應各自分類，
不自動解讀成生成程式非法。原始請求／回應與可取得的 usage 保留；缺失 usage 是 unknown。
HTTP timeout 不等於 server 已取消：無法確認取消時停止派發，整合 job 隨後關閉服務。
若使用外部 endpoint，Mirage 不關閉他人管理的服務，報告取消未確認並停止自己的工作。

### 14.5 驗證與尚未驗收項目

- `test_http_provider.py`：真實 HTTP stub 的請求內容、修正上下文、usage、錯誤模型、
  截斷回應、格式／HTTP 錯誤、逾時與跨 subprocess metadata／故障分類。
- `test_scheduler.py`：experiment 並行上限、獨立生成上限、排空後獨占工作、取消、
  規劃不算覆蓋，以及 coverage 單次提交。
- `test_runtime.py`、`test_cli.py`：取消後程序退出與證據保留、並行參數與服務故障退出碼。
- `test_sbatch.py`：服務清理、測試 step 非零退出碼傳遞、受管參數拒絕與主記憶體分配。
- `test_parallel_workflow.py`：模擬 HTTP 服務搭配真實 PyTorch，以單／雙 worker 執行、
  檢查不同案例初始狀態隔離、相同案例初始值一致、artifact 完整性、零 LLM 呼叫重播，
  並驗證 HTTP timeout 後不再派發案例。
- `provider_scheduler.sbatch` 執行不需模型的控制流程測試；`full_suite.sbatch` 執行
  CUDA、compile、端到端測試；`vllm_smoke.sbatch` 留待 serving 環境與權重就緒後執行。

語法／靜態與 `sbatch --test-only` 可直接檢查；運算密集工作全部透過 Slurm。
模擬 HTTP 服務不代表真實 LLM。雙 step GPU 隔離已另以 PyTorch 驗證；
後續真模型載入與 smoke 結果見第 15 節。全面吞吐量評估與單一 HTTP 請求的
服務端取消確認仍未驗收；無法確認取消時仍停止派發並關閉整合服務。

### 14.6 本次執行證據

所有運算測試都由 Slurm 派發，未下載模型或安裝套件。被測環境為既有 Python
3.14.10、使用者安裝的 PyTorch 2.15.0+cu132、CUDA runtime 13.2 與 NVIDIA H200。

- Job `761134`：完整 77 項測試通過，零跳過，耗時 338.105 秒；包含 CUDA、compile、
  模擬 HTTP＋真實 PyTorch 單／雙 worker，以及無 LLM 重播。
  [完整測試日誌](../result/validation/761134/tests.log)
- Job `761162`：兩個同時執行的 step 各取得一張不同 UUID 的 GPU，退出碼 0；
  各 4 CPUs、204800 MiB 主記憶體。這是 step 隔離驗證，沒有載入 vLLM。
  [隔離驗證日誌](../result/validation/logs/mirage-step-isolation-761162.out)
- Job `761171`：最終 HTTP／排程／取消／CLI／shell 控制流程 26 項測試全部通過，
  包含服務清理、非零退出碼傳遞與主記憶體平分。
  [控制流程日誌](../result/validation/761171/tests.log)
- 59 個 Python 檔案 AST 語法檢查、所有 batch／helper 的 `bash -n` 與
  `git diff --check` 通過；整合 serving 與真模型 smoke 的 scheduler preflight 通過。

先前 job `761112` 發現新測試錯誤要求 coverage-only 案例也必須得到 MR Pass；
修正測試後才得到上述通過結果。Coverage-only 正確性仍為 unassessed。
初始 step 配置阻塞第二個 step 的驗證已中止並保留日誌；上述 `761162` 使用修正配置。
不把失敗／中止的驗證記為通過，也不以測試通過宣稱找到新的 framework bug。

官方配置參考：

- [Qwen3-Coder-Next-FP8 model card](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8)
- [vLLM online serving](https://docs.vllm.ai/en/latest/serving/online_serving/)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)

## 15. 獨立 serving 環境與模型準備

使用者核准後，以 `sbatch/serving/prepare.sbatch` 派發準備工作；沿用既有 Python，
不另行安裝 Python，不向 Mirage `.venv` 加入 vLLM 依賴。

Job `761235` 在 H200 compute node 成功完成安裝、kernels 準備、固定 revision
下載及模型 index 所列分片存在性檢查，耗時 478 秒、退出碼 0。
這些檢查不等同於模型載入或 MR 測試通過。

| 項目 | 實際準備結果 |
| --- | --- |
| Serving Python | 既有 Python 3.14.10 |
| vLLM | 0.33.0 |
| Serving PyTorch | 2.14.0+cu132；與被測 PyTorch 2.15.0+cu132 分開 |
| Transformers | 5.20.2 |
| FlashInfer | 0.7.2.post1，另安裝匹配的 cubin 與 cu130 JIT cache wheels |
| 模型 | `Qwen/Qwen3-Coder-Next-FP8` |
| Revision | `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c` |
| 權重 | 40 個 safetensors 分片，約 74.9 GiB |
| Serving interpreter | `/work/asciibase64/Mirage/environments/vllm/qwen3-coder-next/bin/python` |
| 模型目錄 | `/work/asciibase64/Mirage/models/Qwen3-Coder-Next-FP8/da6e2ed27304dd39abadd9c82ef50e8de67bdd4c/` |

安裝與 kernels 的下載 cache 都在 `/work/asciibase64/Mirage/cache/` 下分類保存。
服務執行時也明確設定 `FLASHINFER_WORKSPACE_BASE`，避免新建的 kernel cache 落到 home。
重跑準備 job 可指定 `MIRAGE_MODEL_REVISION`，沿用固定 snapshot；正式模型服務採 offline mode。

可追溯紀錄：

- [準備 metadata](../result/serving/761235/preparation.json)
- [完整 serving 套件版本](../result/serving/761235/serving-requirements.txt)
- [可直接載入的環境配置](../result/serving/761235/serving.env)

使用方式：

```bash
source result/serving/761235/serving.env
export MIRAGE_PYTHON="$PWD/.venv/bin/python"
sbatch sbatch/validation/vllm_smoke.sbatch
```

真模型 smoke 以 `afterok:761235` 相依條件提交為 job `761273`，已完成：

- Qwen3-Coder-Next-FP8 以 TP=1、16K context、最大 2 個生成 sequence 載入單張 H200。
  首次啟動包含 torch.compile、CUDA graph capture 與 DeepGEMM 暖機，快取保存在 `/work`。
- Serving 與 Mirage 使用不同 GPU，保留各自的 Python／PyTorch 環境。
- Canonical、skeleton、transition、follow-up 四次生成全部 accepted，沒有修正重試；
  此案例記錄 2021 input tokens、345 output tokens，MR 判定 Pass。
- 關閉模型服務後執行重播：兩次 execution completed、MR Pass、provider calls 0，
  environment changes 為空。
- Job 整體退出碼 0、耗時 660 秒；serving step 的 cancelled 狀態來自測試完成後
  明確停止常駐服務，不代表 job 失敗。

[真模型案例報告](../result/validation/761273/run.json)、
[無 LLM 重播報告](../result/validation/761273/replay.json)、
[服務環境紀錄](../result/validation/761273/serving/environment.json)。

平行驗證 job `761321` 在另一節點啟動較慢，900 秒 readiness 期限到達時
仍在載入權重（85%），因此退出碼為 1，尚未執行案例。此結果記為服務啟動逾時，
不記為 MR 違反。保留原始日誌後，明確設定 `MIRAGE_SERVE_START_TIMEOUT=1800`，
以 job `761383` 重試相同的 2 workers、2 個生成配額、10 個 `torch.matmul` 案例；
GPU 數量、分區與模型設定不變。重試已完成，結果如下：

- Job 退出碼 0，總耗時 509 秒；案例批次本身約 306.86 秒。
- 10 個案例全部 executed：8 個 MR Pass，2 個 coverage-only／unassessed。
  8 組 MR 均通過動態 applicability 檢查，沒有 suspected failure。
- 排程事件中的同時在途案例峰值為 2，最後所有案例均完成；服務與測試的 GPU
  隔離檢查通過。這是平行流程驗證，並非單／雙 worker 吞吐量對照實驗。
- 40 次生成，19978 input tokens、3454 output tokens，沒有修正重試或未知用量；
  100 筆 provider／初始化／驗證／執行子程序紀錄均為 completed。
- 覆蓋 API 1/1、state 9/30、transition 5/5、API × MR 3/3、組合 10/42。
  分母為本次宣告的候選集合，並非完整理論空間，也未證明所有候選均可達。
- 工作結束時停止 serving step；不保留常駐 GPU 工作。

[平行批次報告](../result/experiments/761383/run.json)、
[平行服務環境](../result/experiments/761383/serving/environment.json)。

後續從專案根目錄提交相同配置：

```bash
source result/serving/761235/serving.env
export MIRAGE_PYTHON="$PWD/.venv/bin/python"
export MIRAGE_WORKERS=2
export MIRAGE_GENERATION_CONCURRENCY=2
export MIRAGE_SERVE_START_TIMEOUT=1800
mkdir -p result/experiments/logs
sbatch sbatch/experiments/parallel.sbatch --apis torch.matmul --cases 10
```

這個 job 會啟動模型服務、執行案例並在結束時停止服務。無需重新下載模型；
啟動等待時間仍受 allocation 剩餘時間約束。

## 16. A + B：計算圖生成與狀態轉換計畫

本節是現行 generation contract version 2，取代第 13～15 節的四次模板生成流程。
LLM 每個正常案例呼叫兩次：canonical 計算圖生成及 transition 計畫生成；每次最多
兩次修正。Framework 綁定訓練迴圈、組裝 source／follow-up、執行並判定 MR。
不進行模型 A/B 對比，也不新增 optimizer、scheduler、offloading 或 distributed 維度。

### 16.1 A：受控計算圖

`graph-v2` 請求回傳 `{"source": "..."}`。只允許
`forward(x, weight, bias)` 與 `training_loss(output)`，使用單次賦值的局部變數及 return。
目標 API 必須是 forward 的第一個運算，直接使用既有 API profile 指定的 operands；
這保證 harness 建立的 stride／offset／alias 表示直接送入目標呼叫。

目標結果之後可由模型組合分支、合併、常數縮放及非線性運算。白名單為：

- Tensor 加、減、乘；有限數值常數範圍 `[-2, 2]`。
- `torch.tanh`、`torch.sigmoid`、`torch.relu`、單參數的 `torch.nn.functional.gelu`。
- Tensor `float()`、`square()`；loss 另允許 `mean()`。
- 目標之後連同 loss 最多 12 個運算，source 最多 12000 字元；forward 保持 rank two，
  loss 必須是 scalar。運算必須依賴目標／output，局部變數不得重複賦值或閒置。

額外 mutation、隨機操作、detach、迴圈、外部狀態與任意 Python 不在此語言內。
目標 API 本身既有的 mutation／stochastic 性質仍保留，限制對應 MR 的 applicability。
Prompt 提供契約及探索偏好，不提供完整答案；允許模型選擇拓撲、係數與 loss。
Reference provider 仍是明確標示的確定性測試 fixture，不計為 LLM 能力證據。

Framework 將驗證過的程式拆成 target call、graph tail 與 loss。`api_enter/api_exit`
包住實際 target call，before_api snapshot 取得其真正 operands 的 metadata，並在
目標結果上另掛 gradient hook。新版 backward 證據要求 input、target、output 三條
hook 證據；沒有目標梯度證據即不可視為有效的 backward 測試。

Compiled 模式將 target 與 tail 分別編譯，保留明確觀測邊界；這一版不測跨兩段邊界
的整圖 fusion。編譯圖節點、實際 compiled call 完成事件、目標呼叫事件及梯度證據
共同驗證路徑，不能僅憑設定旗標或圖中節點存在給予有效覆蓋。

### 16.2 B：可執行計畫

`plan-v2` 請求回傳 `{"actions": [...]}`。框架提供已驗證的計算圖、目標 predicate、
配置、MR 與 action 契約，不提供完整 action 答案。此版計畫語言為：

1. `restore_initial_state`。
2. 零個或多個 `warmup:REP`，每個 action 真正執行一次 forward／backward／Adam。
   REP 可為 contiguous、strided、offset、clone、shared，模型選擇次數、表示及順序。
3. `restore_training_state`，恢復 parameter、gradient、optimizer、AMP scaler 與 RNG。
4. `checkpoint:0` 或 `checkpoint:1`，符合 target 的 checkpointing 條件。
5. 恰好 `updates` 個 `train:REP`，REP 必須等於 target.representation。

`target.warmup=none` 要求零次 warm-up；`same` 要求所有 warm-up 都使用目標表示；
`alternate` 要求至少一次使用不同表示。有 warm-up 時，模型可選 1 至
`2 * warmup_steps` 次。正式訓練的基本依賴順序保持固定，並非任意訓練程式規劃器。

執行器逐一解讀計畫，完成 action 後才發出帶 index 的 `plan_action` 事件。
Inspector 同時檢查計畫合法性、完整的 action 完成序列，以及指定 checkpoint 的
實際 metadata、warm-up 歷史類別與 gradient 證據。計畫存在不代表狀態已到達。

Source／follow-up 使用相同計算圖。Layout／storage source 使用沒有 warm-up 的
contiguous 配置；checkpoint 配對保留相同 warm-up 計畫，只切換 checkpointing。
Mutation source 沿用既有 eager 配置；若 warm-up 等條件不滿足既有 input relation，
保留 coverage 並標為 inapplicable，不放寬 MR 前提以追求 Pass。

### 16.3 錯誤、版本與統計

靜態語法錯誤與明確 graph／plan contract violation 可回傳模型修正；此分類不宣稱
原程式是非法 PyTorch。原因未明的 runtime exception、runtime unresolved plan、
未達狀態與 MR 違反不進行猜測式修正。Suspected failure 仍需重播及獨立證據。
已通過結構檢查的新版圖若缺少 runtime API／backward 證據，也保留為待查的
unknown exception，不要求模型改寫；可能的執行器／framework 異常不得被生成修正掩蓋。
原有數值容差繼續作為 screening threshold；擴大計算圖後的條件數與數值誤差需要
逐案例複核，不能僅由閾值超出確認 framework defect，也不自動放寬閾值。

- Program 及 execution record 以 `version=2` 識別；舊版省略 version 時視為 1。
- Bundle 保留生成 source、實際 actions、原始請求／回覆、修正歷史、執行證據及 engine。
- 新 controller 重播舊 artifact 時仍使用其保存的 engine，送入舊版支援的欄位。
- Eager 候選也納入 none／same／alternate，coverage 使用 `candidate_schema_version=2`。
  `transition` 仍計算宣告候選中的抽象轉換模板，不代表所有具體路徑。
- `diversity` 另列 accepted／verified graph、unique graph、verified plan instance、
  unique verified plan 與修正次數。具體路徑沒有已證明的全集分母，只報計數。
- Graph fingerprint 正規化局部變數名稱並保留常數；改名不算新圖，改係數可能算新圖。
  這不是通用語意等價判斷。Plan fingerprint 基於實際 action 序列。
- 不同版本／不同候選集合的覆蓋比例不相加；跨 target 的不同 plan 也不等同於
  在相同 predicate 下找到不同路徑，分析時應保留配置與目標條件。

### 16.4 驗證與執行

```bash
export MIRAGE_PYTHON="$PWD/.venv/bin/python"
mkdir -p result/validation/logs
sbatch sbatch/validation/generation_planning.sbatch \
  test_graph_generation test_transition_planning
sbatch sbatch/validation/full_suite.sbatch

source result/serving/761235/serving.env
sbatch sbatch/validation/llm_graph_plans.sbatch
```

真模型 job 使用兩張 GPU、兩個 worker、12 個 matmul 案例；要求有效案例、實際圖與
計畫多樣性，並在關閉模型服務後重播一個適用 MR 的案例。另記錄正式訓練各步
梯度都為零的案例作為生成品質觀察；零梯度合法，不自動視為程式錯誤或 MR 違反。
測試失敗會保留日誌及
非零退出碼，不改用 reference provider。所有運算由 Slurm 派發。

已取得的驗證證據：

- `761553`：A 初版 7 項測試通過，涵蓋八個目標 API、CUDA、三個 compile backend、
  MR 與缺少 target gradient 的否定案例，36.406 秒。
- `761564`：圖、計畫、生成修正與 HTTP 相關共 30 項測試通過，34.886 秒。
  包含獨立 scalar chain-rule 梯度檢查、不同 warm-up 路徑與缺少恢復步驟的否定案例。
- `761586`：使用新 controller 重播舊版真模型案例，MR Pass，兩個 execution completed、
  provider calls 0、environment changes 為空。
- `761568`：完整 92 項測試通過，零跳過，246.549 秒；包含單／雙 worker、
  模擬 HTTP、CUDA、compile、生成計畫與無 LLM 重播／縮小。
- `761638`：補充的 11 項圖／計畫測試通過，38.721 秒，包含新版圖的 FP16／BF16、
  三個 compile backend 下的 checkpoint／mutation，以及不同 warm-up 歷史。
- `761642`：最後的 grammar／provider／pipeline 11 項檢查通過，0.524 秒，
  包含重複目標呼叫的拒絕案例。
- `761681`：缺少 runtime API／backward 證據的故障注入測試通過，確認只呼叫一次
  provider，保存 unknown 診斷而不進行改寫，9.399 秒。
- `761576`：真模型生成暴露 prompt／diagnostic 問題：模型在目標之後引用 `bias`，
  原本泛用的錯誤訊息未能協助修正。4 個案例 repair_exhausted 後主動中止，另 2 個
  initialization_failed 來自中止中的初始化；stop_reason=interrupted，未得到 MR 結果。
  這輪不計為驗收通過，完整紀錄保留。
- 修正為包含名稱、行號與可用 locals 的具體診斷，並在 prompt 明確禁止目標後的
  bias 存取；未放寬 grammar。
- 重試 `761776` 明確使用 `--nodelist=25a-hgpn057`（scheduler preflight 通過），
  保持 8gpus 分區、兩張 GPU 與其餘資源。12 案例中 11 個 executed（8 Pass、3 unassessed），
  1 個超出圖運算上限且 repair_exhausted；嚴格驗收退出碼 1，未記為通過。
- 再將運算上限診斷改為實際 forward／loss 計數與需刪減的數量，prompt 建議較短的
  輸入相依計算圖；grammar 上限仍為 12。`761813` 的 12 項測試通過，41.366 秒。
- 最後重試 `761814` 使用相同明確節點與資源，COMPLETED、退出碼 0，總耗時 429 秒。
  12 個案例全部 executed，10 個 MR Pass、2 個 coverage-only／unassessed；24 個配對
  execution 均 completed，API、backward、actions 與 target-state 證據均為 true。
- 得到 11 種 graph fingerprint、12 種已驗證 plan；graph 重複比例約 8.33%。此批次
  的 target predicates 均不同，不能宣稱找到同一 predicate 的 12 條不同路徑；
  同一 predicate 下的不同路徑執行能力另由 `test_transition_planning` 驗證。
- 實際 32 次模型呼叫（正常 24 次＋8 次修正），16896 input tokens、2375 output tokens，
  無 unknown usage；80 筆初始化／provider／驗證／執行子程序紀錄均 completed。
  案例批次 wall time 約 235.43 秒，provider 子程序累計約 25.47 秒；重疊時間不可相加。
- 已驗證 state 12/90、抽象 transition 12/15、API × MR 3/3、組合 12/126。
  這些分母屬於新版宣告候選，不代表所有理論可行狀態或路徑。
- `0004` 案例的正式訓練梯度各步皆為零，列入 `zero_gradient_cases`；不把結構多樣性
  當成每個程式都有同等測試價值，也不把合法零梯度直接判為錯誤。
- 關閉模型服務後重播得到 MR Pass，兩個 execution completed、provider calls 0、
  environment changes 為空。驗收結束後不保留模型常駐工作。

[最終驗收摘要](../result/validation/761814/validation.json)、
[完整批次報告](../result/validation/761814/run.json)、
[無 LLM 重播](../result/validation/761814/replay.json)。

實際模型產物可直接檢查：

- [計算圖與 loss 範例](../result/validation/761814/cases/batch-2717b824c0b74e4e-0000/canonical/program.py)：
  matmul 後接 tanh、縮放、sigmoid、平移及 ReLU，loss 為平移後的平方平均。
- [兩步 warm-up 計畫](../result/validation/761814/cases/batch-2717b824c0b74e4e-0001/transition/program.json)：
  contiguous → offset warm-up，恢復訓練狀態，再使用 offset 正式訓練。這份 action
  序列由模型選擇並實際執行，並非 prompt 提供的完整答案。
