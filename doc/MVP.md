# Mirage MVP / POC 範圍與實作提案

本文件整理使用者在 28 題範圍訪談中確認的需求。Mirage 是 PyTorch
軟體測試框架，探索 tensor memory representation 與 training state。

需求範圍已確認，後續實作範圍與檔案已獲使用者核准。本文保留原始需求與
階段提案以利追溯；目前交付內容與驗證限制見第 13 節。
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
| 平行範圍 | 第一版單 GPU；多 GPU 調度延後 |

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
