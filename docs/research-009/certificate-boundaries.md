# Source certificate 的證明範圍與後續缺口

本附錄記錄第009輪最終 certificate 修正的實作邊界。這是實作作者的原始碼審查與已保存結果核對；跨作者審查另見[獨立程式審查](independent-review.md)。本次沒有執行 Torch、重播或新增實驗，也沒有修改 production 程式。

核對的 `src/mirage/backends/pytorch.py` SHA-256 為 `f0a09c2781c482a07085f92b1d682d239a03924a6ee553e88d048735daeb4163`。舊相容標籤 `explicit_straight_line_values_v1` 保留，但認證規則已改變；跨版本比較應以凍結程式與檔案雜湊識別，不能只比較標籤。

## 靜態 certificate 實際證明什麼

[`explain_certificate`](../../src/mirage/backends/pytorch.py) 追蹤 build/run 中已審查的 namespace、Tensor、primitive、dtype、device、shape、容器與返回值種類，證明的是**目標呼叫周圍程式屬於目前審查過的綁定／效果子集**。回報 `verified` 不等於整個目標 API 正確、任意 Python 程式純淨或所有環境下確定性。`unknown` 是缺少這份證明，不代表程式非法或狀態不可達。

| 已處理的通用能力 | 保留的拒絕邊界 |
|---|---|
| build/run 的已審查純數值操作、native Tensor receiver 與返回值種類 | 方法同名不足以認證；未知物件、隱藏 helper/class、overrides 與外部效果不自動取得 Tensor 身分 |
| 已追蹤 tuple/list/result-tuple 的 `*args`、字串鍵 owned dict 的 `**kwargs` | 未知容器、callback-bearing 引數、非字串 keyword、重複 keyword 仍拒絕；展開後仍檢查 helper 的 `out`/`inplace` |
| 原生複數常值、[官方 dtype 別名](https://docs.pytorch.org/docs/2.15/tensor_attributes.html#torch-dtype)及 import alias | imported/interface 名稱不可重綁；framework attribute 不可覆寫；任意相似名稱不當成 dtype |
| owned build dictionary 的已知更新、容器別名追蹤 | 直接／間接容器循環、未審查的 tensor mutation、metadata-sensitive 條件與觀察仍不認證 |
| 不透明 target call 可保留執行機會 | 未審查 target 返回值不能只靠 `.sum`、`.clone` 等拼字取得 Tensor 方法證明 |

例如 metadata target 可以執行，周圍程式也可能取得 source certificate，但其 `is_contiguous` 等結果不受值等價 MR 保證。已知 dict 展開的 helper `out=None` 仍因缺乏 literal provenance 而保守拒絕；直接寫出的 `out=None` 保留既有支援。成功重播、固定 seed 或 unchanged RNG 都不能替代來源綁定／效果證明。

## Runtime 條件仍獨立成立

[`representation_evidence`](../../src/mirage/backends/pytorch.py) 與 [MR catalog](../../src/mirage/rules/catalog.py) 還要求執行有效、目標屬於已審查 logical-value 語意類別、初始邏輯狀態與 alias topology 一致、目標邏輯輸入一致、checkpoint 對齊，以及表示轉換確實保留所需座標。有限觀察值與數值域也是獨立義務。任何一項缺失都不能因 certificate 存在而補成 Pass。

read-only representation MR 與 schema-derived write-role relation 分開：有寫入角色時，現有 materialized-value relation 比較返回的邏輯值，沒有聲稱驗證全部 alias 傳播、外部效果或完整 optimizer 狀態。helper 寫入限制與 target schema/effect 檢查也不互相取代。context restoration、compiler/capture 邊界等路徑使用各自的 runtime 合約；不要求把所有狀態化程式硬塞入純數值 source subset。

[數值 auditor](../../src/mirage/inspector/numerical_gradients.py) 是另一個有限範圍的正確性檢查：需可歸因的完成 backward 事件、local eager read-only 邊界、支援的實數 dense float32/float64 座標、真實 cotangent、重建 baseline output/VJP 相符，以及 directional finite difference 與單側收斂條件。它檢查目前 target call 的 physical-storage chart，沒有重播任意 pre-target 訓練歷史，也不證明 forward 語意正確。

[路由器](../../src/mirage/inspector/oracles.py) 保留 `mr_verdict` 與數值狀態分別紀錄。數值 agreement 不能合成 MR Pass；baseline-verified violation 只能提高為 suspected failure。`unknown` 也可能只是未排程、沒有 backward 事件、預算不足或座標不可重建，不能一律稱為不穩定。

## 已有證據與可宣稱範圍

[fixtures-03](../../result/general-state-research/778971/fixtures-03/summary.json) 已保存兩組均完成且通過的結果；[reference](../../result/general-state-research/778971/fixtures-03/reference/fixed/summary.json) 與 [candidate](../../result/general-state-research/778971/fixtures-03/candidate/fixed/summary.json) 各為 10/10。這證明 supplied known-trigger 回歸案例在共享修正後仍可被既定 oracle 檢出，沒有證明自主發現能力提高、沒有提供十個新缺陷，也不使所有原始候選成為 confirmed defect。

raw evaluator 的 `confirmed` 或 `controls_independently_audited` 名稱本身不是獨立性證據。重用原 finite-difference 結果或同 oracle 重播只能支持該證據的重現性；獨立佐證要明列不同的推理／檢查來源、重新核對假設，並附對應 artifact。第008輪 slogdet 的另外解析檢查是既有本地機制的佐證，不能歸功於本輪 planner，亦不能稱為新 upstream bug。

[development-01](../../result/general-state-research/778971/development-01/summary.json) 使用仍缺 dtype alias 的較早凍結版本，不能回填成最終版本成績。其 candidate `std` 四個有效案例中，三個缺 `numerical_domain`，另一個缺 build tensor indexed-write certificate；前三個已有 source certificate。這區分了數值政策的保守範圍與來源認證缺口，不支持藉降低數值門檻換取更多 MR。

## 尚待拓展的泛用能力

1. **有證據的初始化與狀態效果。** build tensor indexed initialization 仍被拒絕；可另行研究只允許來源／所有權可證的 native Tensor 寫入，保留 RHS/index 綁定、alias 與初始狀態檢查。一般 inplace、module state、optimizer/scheduler/AMP/RNG/分散式效果需要各自的合約，不能直接擴大方法白名單。
2. **狀態化程式的專屬證明。** checkpoint、offload、compiled/captured callable、梯度累積與訓練歷史需要可歸因事件、必要觀察值與語意 checkpoint 對齊。增加純 build 操作只降低其中一個門檻，不能證明這些機制已完整涵蓋。
3. **可驗證的目標與程式位置連結。** AST 提示對 helper、動態呼叫、compiler/capture factory 保持 unresolved；未來應以 runtime call-site linkage 改善修補定位，避免把靜態 ordinal 當成 target/backward 身分。
4. **更完整且有界的 oracle 與證據整理。** 目前數值檢查對 complex、sparse、autocast、寫入目標、任意訓練歷史及模糊重複 operand 有限制。擴充需先建立對應座標／效果／數值證明；目前應明列 abstention 原因，並將重現、最小化、假設重審與獨立佐證分階段保存。

Mirage 的研究方向仍是 **API × tensor 記憶體表示 × 訓練狀態轉移 × 適用 MR** 的組合探索，並以 runtime path/state 證據確認。不能為取得更多 certificate 刪除困難的訓練狀態，或把研究縮成易通過的 layout 測試。有限的 operand/axis catalog、單 seed 與固定時間預算不構成完整狀態空間或無缺陷證明；文獻定位亦不支持 memory-layout testing 的首次性主張，詳見[定位與限制](literature/synthesis.md)。

## 正式實驗之後：局部 VJP 的獨立來源效果憑證

本節是 `final-01` 凍結實驗完成後另行授權的修正，**不更新上文的原始審查雜湊、正式實驗引擎或發現數**。事後檢查發現 candidate slogdet attempt 8 已產生負視圖並完成真實 backward，卻因 `run` 中的梯度啟用操作而無法進入數值 oracle；保存證據、獨立 Fraction 參考與精簡控制實驗見[漏檢分析](final-slogdet-analysis.md)。該解析佐證不依賴有限差分，但也不是新的自主發現或 upstream 新缺陷。

新增 `numerical_source_certificate`，其標籤為 `explicit_straight_line_local_vjp_v1`。它仍分析整份 build/run 的已審查綁定／效果，並只增加一項 `run` 語法的適用範圍：**已證明為 native Tensor receiver 的 `requires_grad_`，且旗標在原始 AST 中明確為 literal `True`**。支援 bound method 與帶已證 Tensor 第一引數的 unbound `torch.Tensor.requires_grad_`，以及直接 `requires_grad=True` keyword；省略旗標、`False`、數字 `1`、動態變數、`*args`／`**kwargs` 展開及重複參數均不由此分支放行。原本已有的 build 規則不變。

這是局部重建 target VJP 所需的來源效果證明，不是一般訓練狀態等價證明。native receiver／namespace 綁定、不可重綁 framework、拒絕未審查 hook／override／custom class、helper 寫入限制、owned-container 與 metadata-sensitive 表達式限制均繼續成立。沒有以 unchanged RNG、執行成功或重播成功代替來源證明，也沒有刪除整體來源安全檢查。

[`runtime/stateful.py`](../../src/mirage/runtime/stateful.py) 將新憑證另存於 record；只有[局部數值路由](../../src/mirage/inspector/oracles.py) 接受這個精確標籤。原 `program_certificate` 與 representation MR 的 `deterministic` 要求不變：對於僅因 `run` 梯度啟用而被新憑證接受的程式，舊憑證仍為 `None`，representation MR 仍 inapplicable。完整／有效執行、實際完成的直接 backward 事件、RNG、eager/read-only schema、支援座標、baseline output/VJP 重建一致、差分穩定與單側條件仍各自必要；未修改數值 auditor 的門檻或公式。agreement 仍不能轉成 MR Pass，violation 仍只是 suspected failure。

跨作者原始碼審查未發現新增的綁定／效果放行缺口；後續獨立凍結版本的 Slurm 驗證已完成：130項相關測試通過、固定案例10/10、同一漏報程式重播由局部數值oracle標記suspected_failure，而MR仍inapplicable。原始正式結果與source保存不變；完整版本、失敗重試及執行證據見[驗證紀錄](validation.md)。
