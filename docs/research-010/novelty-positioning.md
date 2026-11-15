# 本輪研究定位與限制

本輪可主張的具體能力是：**將 target backward 的實際貢獻、註冊 leaf 的累積梯度、保持數值的梯度表示轉換、真實 optimizer 更新，以及對齊的更新後狀態比較，接成有執行證據的條件式 MR。** 這是 Mirage 記憶體表示 × 訓練狀態方向的一個有限切片；不是首次 layout testing、gradient fuzzing 或訓練程式測試的主張，也不是已證明較強 bug discovery 的結論。

本筆記於 2026-11-15 核對既有 literature 筆記、[本輪分析](analysis.md)及當前程式，初次補讀下列兩篇 primary paper，文末另列三項有限交叉核對。未執行程式、未讀取未完成正式實驗結果；這不是完整 related-work survey。

## 文獻事實與本輪差異

| 對照 | 可由來源支持的事實 | 本輪的有限差異／推論 |
| --- | --- | --- |
| GPU-Fuzz | 論文明確把 shape、dtype、stride／memory layout 列入動機；§3 以 operator constraints、solver 產生測試，執行時使用 compute-sanitizer。[primary paper v3，§1、§3](https://arxiv.org/html/2602.10478v3) | 因此不能把「探索 layout」本身當新穎性。本輪著重已有 backward 來源的 gradient 在真實更新邊界被消費，並以條件式 post-state equivalence 判讀。這是從兩者測試單位與 oracle 作出的定位推論；不表示 GPU-Fuzz 的全部實作必然沒有相似機制。 |
| NablaFuzz | §IV 使用不同執行情境比較 API 輸出、forward/reverse AD 與 numerical differentiation，也涵蓋高階梯度與數值不穩定過濾。[primary paper v1，§IV](https://arxiv.org/html/2302.04351v1) | 本輪 optimizer MR 檢查相同梯度值在不同表示下的更新一致性，不取代 AD 正確性 oracle。target 歸因建立「哪個執行貢獻被消費」的證據，沒有證明該貢獻的數學導數正確。 |

「gradient injection」需要先限定含義。改變 VJP 的 cotangent、指定任意 `.grad` 數值，以及在 optimizer 前更換等值 `.grad` 的儲存表示，是不同測試操作；本次有限文獻核對不能用這個籠統名稱建立優先權。Mirage 實作確實更換 `.grad` 物件，適合稱為**具來源證據與數值保持約束的梯度表示替換**，而不是宣稱完全沒有 injection。[`OptimizerObserver._provenance`、`_mutate`](../../src/mirage/runtime/optimizer.py#L266)

## 可審查的技術主張

1. **探索座標連到實際階段。** `parameter_key`、target call/path、保留的 pre/post update checkpoint 與 completed companion 把梯度表示目標綁到同一參數及更新。模型提出 plan 不等於達成；`inspect_optimizer` 以實際事件驗證，新 `optimizer-chain-1` 覆蓋與舊 operand units 分開。[contracts](../../src/mirage/contracts/capabilities.py#L74)、[catalog companions](../../src/mirage/generator/state_plan.py#L129)、[inspector](../../src/mirage/inspector/stateful.py#L104)
2. **來源鏈有明確邊界。** direct target Node 的觀察梯度須與同一 backward invocation 的 leaf accumulation 相符；到 step 前須保持 identity、version 和邏輯值。harness 之後才執行等值表示轉換，source 本身仍執行 backward 和 step。因此測到的是 optimizer 如何消費轉換後梯度，沒有證明 autograd 曾寫入該轉換後表示。[provenance／mutation](../../src/mirage/runtime/optimizer.py#L266)
3. **MR 的意義來自前提。** 兩邊須有有效執行、已審查 native optimizer／source effects、相同初始狀態與環境、相同 optimizer registry、相同 pre-update 邏輯 parameters／gradients／slots，且只有允許的表示差異；之後才比較 post-update parameters／slots。單純存在 step、hook 或兩份快照不夠。[`optimizer_gradient_pair`](../../src/mirage/rules/catalog.py#L65)

以上是當前實作可檢查的機制，不是對文獻「首次組合」的證明。配置重用讓同一 source 在不同 execution configuration 下接受測試，符合測試案例 `e=(P,c)`；若增加直接使用既有轉換庫的候選，它也是已知轉換的供應／成本改進，不能單獨當作新 fuzzing 原理。

## 現有邊界與需要的實驗證據

目前已審查域是 exact native SGD、CPU float64 dense tensors、cold slots、單一步驟與直接綁定的 trainable leaves；正 momentum 可產生 cold-start slot，但尚未涵蓋 warm-state 演化。多步累積、Adam、AMP、compile、分散式、共享儲存與其他 dtype／device 都不能由此推廣。[domain checks](../../src/mirage/runtime/optimizer.py#L97)、[roadmap010](../roadmap-010-optimizer-gradient-chain.md)

兩個 member 若共享同一錯誤 VJP，仍可能得到相同更新並通過這個 MR。新 optimizer certificate 亦沒有自動擴大舊 local numerical audit 的 admission；需分別報告 optimizer MR 與 numerical audit 的適用性、abstention 和結果。[numerical-audit boundary](numerical-audit-boundary.md)

實驗應分別呈現：可選目標 → 實際執行 → 完整來源鏈 → 表示目標達成 → MR 適用 → suspected failure；另列每階段成本、未測／未知與版本化覆蓋。著作控制案例及 known fixtures 證明實作行為，不能算自主發現。公平配對預算下是否產生更多不同的可驗證更新情境、更多適用比較或可獨立確認的缺陷，仍須由完整結果回答；單 seed 短時觀察不能支持跨 API／框架的廣泛效果。


## 有限延伸核對：直接先例與訓練測試（2026-11-15）

本次僅增加以下三個 primary source；原生測試的程式碼也屬直接證據，不因未以論文發表而排除。PyTorch 固定為 **v2.5.1，commit `a8d6afb511a69687bbb2b7e88a3cf67917e1697e`**，由[官方 release 的 commit 連結](https://github.com/pytorch/pytorch/releases/tag/v2.5.1)核對。下列行號由該 commit 的原始檔計算，永久連結固定完整 SHA；不是網頁文字擷取器重排後的行號。

1. **PyTorch 原生 optimizer 測試是最直接的先例。** `test_rosenbrock_sparse` 建立對應的 uncoalesced sparse／dense 座標梯度，覆寫 `.grad`，兩邊執行真實 step，再以容差比較更新後參數。[梯度建構、替換與比較，L437–493](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/test/test_optim.py#L437-L493) 這已具備「邏輯對應的梯度表示 → 真實更新 → post-state 比較」的基本關係。它使用解析座標梯度，且允許 sparse／dense 算術的浮點差異；不可描述成與本輪 target VJP 歸因及 dense stride／offset 的逐項相同協定。同檔另有 `_compare_between` 的多步 freeze／unfreeze、None-grad、參數與 optimizer state 比較，以及 `test_state_dict_deterministic` 的 warmup／restore 後並行更新；這些是不同測試，不能合併宣稱為同一 layout MR。[多步狀態比較，L666–718](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/test/test_optim.py#L666-L718)、[warmup／restore，L1374–1408](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/test/test_optim.py#L1374-L1408)
2. **Muffin（ICSE 2022）已測試訓練階段的動態 traces。** §3.3 收集各層 forward、loss、backward gradients，跨函式庫逐階段比較，並考慮正常數值差異的傳播。因此「觀測訓練中的梯度／中間階段」亦不是本輪新穎性。該節足以支持這個限制，不能據此斷言其全部實作沒有 optimizer 表示測試。[作者論文 v1，§3.3](https://arxiv.org/html/2204.08734v1)
3. **ModelMeta（ISSTA 2025）已結合模型層級 MR、回饋式生成與訓練 observables。** §3.3 以 reward 引導模型生成，§3.4 包含 outputs、training loss、gradients 等檢查。因此不能把「MR 加回饋探索」或「MR 比較訓練梯度」本身當本輪創新。這份核對沒有建立其與 Mirage 全部狀態域的一對一比較。[作者論文 v1，§3.3–3.4](https://arxiv.org/html/2507.04354v1)

**修正後的研究假說：** 可研究的是跨 API 的 training-state predicates、target backward 歸因、runtime reachability、conditional MR 與 coverage-guided exploration 的整合，是否能以合理成本補到既有測試未涵蓋的情境。這是待驗證的整合與增量價值假說，**不是「梯度表示 → 更新 → post-state MR」的發明主張**。Warm／多步／slot／restore 是 Mirage 目前的能力缺口，不能寫成已證明的文獻空白。

本輪有限 SGD 切片與 single-seed 實驗不足以建立優先權、超越原生測試或廣泛提升 bug finding。後續比較需納入上述原生測試覆蓋，拆開 predicate／provenance／候選供應與成本策略的貢獻，並以多 API、獨立種子及獨立確認的缺陷或可驗證增量情境支持結論。本次只核對來源與程式文字，沒有執行其測試，也沒有讀取正式實驗的 partial 結果。
