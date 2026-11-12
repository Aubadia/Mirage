# Mirage MVP / POC 範圍與實作提案

本文件整理使用者在 28 題範圍訪談中確認的需求。Mirage 是 PyTorch
軟體測試框架，探索 tensor memory representation 與 training state。

需求範圍已確認，後續實作範圍與檔案已獲使用者核准。本文保留原始需求與
階段提案以利追溯；目前交付內容與驗證限制見第 13 節。
後續 vLLM serving、平行 worker 與模型比對方案見第 14 節，尚未實作。
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
第 14 節規劃在此介面上新增 vLLM HTTP adapter，保留既有 provider 的使用方式。
介面應傳入生成階段、結構化上下文與診斷，回傳生成內容及可取得的 usage 資訊。
Provider 未提供 token 使用量時記錄 unknown，不填成零。

LLM 參與 canonical program、training skeleton、transition plan 及 follow-up program
生成。增量注入後檢查語法、合法性、必要路徑與靜態 MR 條件。

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

LLM 生成程式須符合可驗證的單一目標運算與 loss grammar，可選擇合法的 dropout p、
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

目前仍未取得 Nano4 H200、另一個 nightly 環境或真實 LLM provider 的整合驗收證據。
`reference` provider 僅用於確定性的整合驗證，並非 LLM。尚未完成兩個新 bug 的
探索、最小化、獨立佐證及查重；這些仍是研究成果驗收，不能由軟體測試通過替代。

## 14. 待實作：vLLM serving、平行 worker 與模型比對

本節記錄後續實作／配置方案。本次僅更新文件，未安裝套件、下載模型、啟動服務、
提交 Slurm job 或修改程式。以下新增配置名稱與檔案均為提案，不能當作現有功能使用。

### 14.1 已選方向與初始配置

使用者選擇以 Qwen3-Coder-Next 為主並進行模型比對、多個 Mirage worker 平行執行，
以及同一個 Slurm job 配置兩張 GPU。主模型沿用前述 FP8 版本；對照模型採
Qwen3.8-27B。具體 revision、相容套件版本及資源限制需在準備環境時確認並固定。

| 項目 | 配置方案 |
| --- | --- |
| 主模型 | `Qwen/Qwen3-Coder-Next-FP8` |
| 對照模型 | `Qwen/Qwen3.8-27B`，先以 BF16 作為對照配置 |
| 模型比對 | 依序載入 A、B；單一 serving GPU 同時只保留一個模型 |
| Slurm allocation | 同一節點、兩張 GPU；一張 serving、一張執行 Mirage 測試 |
| 其餘 job 預設 | account `ACD115198`、partition `8gpus`、8 CPUs、4 小時，沿用使用者 template |
| Mirage worker 數 | 建議初始值 2，共享測試 GPU；保留降為 1 的配置 |
| 同時生成請求數 | 建議初始上限 2，與測試 worker 數分開配置 |
| Model context | 建議從 16,384 tokens 起驗證；輸入與最大輸出合計不得超過此值 |
| Generation settings | 保存 temperature、top-p、seed、有限輸出 token 上限與 chat template 設定；具體值於整合測試固定 |
| Thinking | 對照模型先使用關閉 thinking 的配置，實際支援方式以固定版本驗證 |
| 時間預算 | 模型載入、兩輪比較、生成、測試與清理共用 allocation 的 4 小時 |

兩張 GPU 是針對此方案明確調整原本單 GPU 的 template。模型能否在單張 H200
以指定 context／並行度運作，仍須實際驗證，不能僅由參數量推定。
若不相容或記憶體不足，保存原因並回報；不自行增加 GPU 或更換量化版本。

### 14.2 環境準備與安裝責任

不預設 vLLM 已安裝。2026-11-12 的唯讀檢查結果：目前 shell 的 `PATH` 找不到
`vllm`，專案 `.venv/bin/python` 為 Python 3.14.10，其套件 metadata 中沒有
`vllm`、`torch` 或 `transformers`。這不排除其他環境已安裝，也不是 compute-node 驗證。
`uv sync` 只同步專案宣告的依賴；目前 `pyproject.toml` 未宣告這些 runtime 套件。

vLLM 套件與模型權重是兩件事：前者提供 serving 程式，後者必須另行取得或由已存在的
模型目錄載入。使用模型 repository ID 啟動時可能觸發下載，因此準備階段應明確解析
revision、完成權重取得並驗證路徑，不把下載時間隱藏在正式 A/B 測量中。

| 環境 | 選擇方式與責任 |
| --- | --- |
| Mirage／被測 PyTorch | 沿用 `MIRAGE_PYTHON` 指定已準備且 compute node 可讀的環境；固定被測 stable 或 nightly 版本 |
| vLLM serving | 擬新增 `MIRAGE_SERVE_PYTHON`，指定獨立且相容的 serving 環境；不讓其依賴改動被測 PyTorch |
| 現有安裝 | 先檢查使用者提供或環境中可發現的 interpreter、版本與路徑，再於 Slurm allocation 驗證 GPU serving |
| 缺少安裝 | 列出相容版本與安裝位置後，在獲准的環境準備範圍內安裝套件／取得權重；不要求使用者必須親自下載 |

不自動安裝 Python，也不假定 vLLM 支援目前 Mirage 的 Python 版本。
Serving 可使用另一個既有、相容的 Python interpreter；若找不到，需先解決環境選擇。
正式 job 缺少必要環境或權重時明確失敗，不在執行途中隱式安裝或替換依賴。

### 14.3 Slurm 配置與服務生命週期

「常駐」指模型在本次 job 的服務階段載入一次，持續接受多個請求；不是永久服務。
Job 結束、取消、逾時或發生不可恢復錯誤時，服務與其子程序都必須結束。

1. 在同一 allocation 內，以兩個可同時執行的 `srun` step 分別啟動 serving 與
   Mirage controller。每個 step 只分配一張 GPU，CPU 也須分配在 allocation 額度內。
   驗證 step isolation 與同時啟動行為，避免其中一個 step 佔滿資源而讓另一個等待。
2. 保留 Slurm 管理的 GPU visibility，不硬編碼實體 GPU 編號。Mirage 子程序繼承
   測試 step 的 GPU；serving GPU 不可出現在測試 worker 的可見裝置中。
3. 等待有期限的 health check，確認服務可用及 model ID 正確，再發送生成請求。
   保存 job ID、host、port、模型 revision、啟動參數與環境版本。
4. HTTP provider 使用 job 內部 endpoint。每次 provider 子程序只送出 HTTP 請求，
   不重新載入模型。請求逾時受整體剩餘預算約束，排隊時間也計入。
5. 驗證 HTTP 斷線／取消後 serving 是否真的停止相應工作；不能把 client timeout
   當成 server 已取消的證據。追蹤未完成請求，避免超時工作持續佔用生成配額。
6. 接近 allocation 結束時停止派發，預留保存證據、停止 worker 與關閉服務的時間。
   使用 trap 保留失敗 exit status，輸出開始／結束／耗時與退出狀態。

整合 job 放在 `sbatch/experiments/`，驗證 job 放在 `sbatch/validation/`，
重播放在 `sbatch/replay/`；共用服務啟停邏輯放在 `sbatch/lib/`。
若提供獨立 serving 入口，新增 `sbatch/serving/` 分類並同步文件；整合方案直接呼叫
共用 helper，不透過巢狀 `sbatch` 取得另一個 allocation。

### 14.4 HTTP provider 與平行執行

在既有 Python provider protocol 上新增 OpenAI-compatible HTTP adapter，傳入
四階段 prompt、上下文及修正診斷，將回應轉為現有 `Response`／`Usage`。
保存模型識別、完整請求與回應、生成參數、延遲及可取得的 token usage；未知值仍為 unknown。
JSON schema 約束只處理回應格式，不能取代程式 AST、runtime、state 與 MR 驗證。
連線失敗、服務崩潰、格式錯誤、生成程式不合法分開記錄，不把服務故障算作 framework bug。

現有 `workflow.py` 逐案例執行；只啟動 vLLM 不會讓 Mirage 自動平行。
新增有界排程，讓不同 experiment 同時推進，並遵守以下限制：

- 每個 experiment 的 canonical、skeleton、transition、follow-up 與各階段驗證
  保留依賴順序。每組 source／follow-up 初期仍依序執行。
- 每個案例使用獨立程序、RNG 狀態、初始資料及 artifact 目錄，避免共享可變訓練狀態。
- 生成請求與測試工作各有並行上限及有界等待佇列，避免大量排隊耗盡記憶體或時間。
- Controller 統一派發並提交 coverage 更新；in-flight 計畫與已驗證 coverage 分開。
  保存抽樣時的 coverage snapshot、派發／完成順序；相同 seed 不保證非同步完成順序相同。
- 兩個 worker 共享測試 GPU，可能產生 OOM 或延遲，不能保證吞吐量提高。
  記錄資源失敗；suspected failure 在其他 worker 停止佔用後獨占重播，再判斷是否穩定。
- 重播繼續直接使用已保存程式及初始狀態，不依賴 vLLM，也不新增 LLM 呼叫。

### 14.5 模型 A/B 比對

先凍結一組涵蓋既有八個 API、四類 MR 候選的任務清單，保存可比較的初始條件、
配置、seed、生成／修正限制及每模型預算。兩模型使用相同任務定義與 prompt 模板；
各自的生成結果與後續修正診斷允許不同。MR applicability 仍須個別動態驗證。
此階段不讓兩模型的自適應 coverage 抽樣改變任務集合，避免比較不同難度的工作。

執行 A 後先排空請求、停止服務並確認釋放資源，再載入 B；保留 B 的預算，
不能讓 A 用完全部 allocation。未完成的配對任務標記 incomplete，不補算成功。
兩模型結果與 coverage 分開保存，不相加為一個覆蓋率。

比較指標包含合法生成率、runtime 有效案例率、修正次數與 token 成本、MR applicability、
單位時間的已驗證覆蓋及資源失敗率。模型載入、生成與測試成本分開記錄；
同時報告總 wall-clock，不把重疊的 worker 時間相加當作總耗時。
不以 suspected failure 數量直接判定模型較好，也不把不同量化／thinking 配置下的差異
歸因為模型本身的純粹差異。

### 14.6 儲存與配置紀錄

| 資料 | 目標位置 |
| --- | --- |
| 固定 revision 的模型權重 | `/work/asciibase64/Mirage/models/<model>/<revision>/` |
| 資料集 | `/work/asciibase64/Mirage/datasets/<dataset>/` |
| Hugging Face／vLLM 等大型持久 cache | `/work/asciibase64/Mirage/cache/<purpose>/` |
| 如需新增的 serving 套件環境 | `/work/asciibase64/Mirage/environments/vllm/<environment-id>/`，沿用既有 Python |
| 驗證／探索／重播輸出 | `./result/<test-category>/<run-id>/` |
| A/B 輸出 | `./result/comparison/<run-id>/<model-label>/` |
| Slurm stdout／stderr 與服務日誌 | 對應 run 的結果目錄；提交前先建立 Slurm 日誌父目錄 |
| 一次性編譯 cache | allocation 內的 job-specific local scratch；重播所需證據在清理前保存至結果目錄 |

將模型與 cache 路徑明確傳入服務環境，避免大型下載落到預設 home cache。
既有 sbatch helper 仍預設寫入 `artifacts/slurm/`；後續實作需同步調整 helper、
提交命令與文件到上述分類路徑。本次未移動既有 artifacts，也未修改現有 scripts。

### 14.7 預計程式範圍與驗收順序

| 範圍 | 預計變更位置 |
| --- | --- |
| HTTP adapter 與生成紀錄 | 新增 `src/mirage/generator/http_provider.py`；按需要擴充 `generator/provider.py` 與生成 artifact |
| 有界排程與集中 coverage 更新 | `src/mirage/workflow.py`、新增 `src/mirage/scheduler.py`、既有 coverage／runtime 程序管理 |
| A/B 任務與報告 | 新增 `src/mirage/comparison.py`；更新 `cli.py` 與 `artifacts/report.py` |
| Slurm 編排與路徑 | `sbatch/lib/`、`sbatch/experiments/`、`sbatch/validation/`、必要的 `sbatch/serving/` |
| 測試與操作說明 | HTTP、排程及比較測試；`README.md`、`sbatch/README.md`，必要時更新 `AGENTS.md` 的分類說明 |

此表是後續影響範圍提案，不代表本次已授權修改以上檔案。正式實作前依 repository
流程列出確切檔案與對應驗證；程式使用英文註解、適當換行與既有模組邊界。

1. 先驗證環境、模型載入、服務就緒與關閉，再接通 HTTP provider 的單 worker 流程。
2. 用可控的 HTTP stub 驗證請求／回應、usage 缺失、格式錯誤、連線錯誤及逾時；
   驗證故障分類與取消，不以真模型回覆取代控制流程測試。
3. 驗證兩 worker 的並行上限、案例隔離、coverage 單次提交、取消清理及 evidence 保存。
   同一固定案例集比較單／雙 worker 結果；排程改變不能改變 MR 判定條件。
4. 在 Slurm allocation 執行真實模型 smoke、GPU 分工／資源競爭測試、完整整合與
   無 LLM 重播。保存實際結果，不把服務 health 成功視為 Mirage 測試通過。
5. 最後執行固定任務的模型 A/B，比對品質、吞吐量與成本，再決定並行度與 context 配置。

輕量語法／靜態檢查可直接執行；`.sbatch` 使用 `bash -n` 與可用的
`sbatch --test-only` 預檢。推論、編譯、完整整合及大量測試全部經 Slurm；
預檢或提交成功均不代表 job 已執行，更不代表 MR Pass。

模型與服務配置參考官方文件，實作時對固定版本再次確認：

- [Qwen3-Coder-Next-FP8 model card](https://huggingface.co/Qwen/Qwen3-Coder-Next-FP8)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [vLLM Qwen3.8-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
