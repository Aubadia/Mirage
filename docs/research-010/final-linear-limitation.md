# Final seed0 的 linear 限制：歸因能力、生成錯誤與證書邊界

本文件只讀審閱已完成的 `result/optimizer-gradient-chain/779954/final-seed0/{reference,candidate}/linear` JSON、trace、stderr 與保存來源；未載入 `.pt`、執行 Torch、重播、修改正式程式或重算正式指標。兩組各 3 attempts、3 provider calls、0 valid；這不等於兩組都沒有呼叫 linear 或執行 backward。

## 保存證據

| Arm / attempt | 保存狀態 | 實際阻礙 |
| --- | --- | --- |
| reference / 0000 | worker completed；API 已執行；verdict invalid | output-gradient hook 與 backward action 已發生，但 direct target Node/operand evidence 缺失；`first_blocking_stage=backward` |
| reference / 0001 | `repair_unchanged`；沒有 worker execution | 回覆與被拒來源 payload 相同；`duplicate_repair_kind=response` 在 generation 階段攔下 |
| reference / 0002 | worker exception；`unresolved_runtime`、quarantined | build 第 5 行 `torch.arange(1,5).reshape(4,4)`；保存例外為 `shape '[4, 4]' is invalid for input of size 4` |
| candidate / 0000 | worker completed；API 已執行；verdict invalid | 與 reference/0000 相同的 direct-backward 歸因限制；另有 optimizer source-certificate blocker |
| candidate / 0001 | `repair_unchanged`；沒有 worker execution | 相同 payload 修復被攔下，不是 provider service failure |
| candidate / 0002 | worker completed；API 已執行；verdict invalid | 改用零輸入仍遇到同一 direct-backward 歸因限制；另有相同 certificate blocker |

三個完成執行的 source 都保存 `api_executed=True`、`required_backward_executed=False`。`semantic_coverage.events.outputs` 含 `(call=0, invocation=0)`；`nodes`、`operands` 為空；`backward_bindings` 統一記錄 `composite or unobserved input edge`。trace 有 `api_enter → api_exit → target_output_gradient → training_action(backward) → final`。所以缺的是要求的 target 歸因證據，不是 Python 沒呼叫 backward；不能把 output hook 升格成目標導數或 optimizer chain 的證明。

Reference/0002 的 traceback 明確定位在 `build`，尚未到 target。既有 bounded rerun 再次得到同一 RuntimeError，耗時約 28.99 秒，`provider_calls=0`；紀錄仍為 `initial_state_equivalence=unassessed`、`correctness=unassessed`。四個元素無法填入 4×4 shape 是本次來源與 traceback 支持的生成錯誤診斷；正式 framework 分類未事後改成 harness ContractError，同例外重現也不代表框架缺陷。

## 兩個獨立的框架能力缺口

1. **直接 Node 歸因目前排除內部複合路徑。** 凍結 backend 的 `backward_binding`（54–76 行）只接受 fresh output Node，且每個非空 `next_functions` edge 都須直接匹配呼叫前保存的 operand gradient edge；任一無法匹配即回傳上述 unknown 原因。因此 JSON 證據足以證明這三次觸發了該限制，但沒有保存完整內部 Node 拓樸，不能從本審閱斷言是哪一個內部轉置、融合或 view 節點造成。某些 linear 實作經內部矩陣／轉置路徑是合理機制推測，不是本次已觀測的節點證據。

2. **target 回傳型別的靜態證書仍過度保守。** Candidate/0000、0002 明確生成 CPU float64、原生 SGD、來源擁有的 `backward()` 與 `step()`；但保存的 `optimizer_certificate_diagnostic` 在 `out.sum` 分別第 19、23 行回報 `unproved_receiver`。凍結 backend 598–603 行對不在已審閱 Tensor-return 集合中的 target 保留 `target_result`，而 `linear` 未得到 Tensor binding。因此 optimizer evidence 記錄 `source effects lack the optimizer-specific certificate`。這與 Node 歸因失敗是獨立障礙；即使將 loss 寫法改到既有證書可接受的形式，也不能修復 runtime 歸因，更不能宣稱已建立更新鏈。Reference/0000 沒有 optimizer，且採 float32，本來就不在本輪 native SGD CPU-float64 更新域內。

## 為何本輪沒有後續改善

兩組 corpus 都沒有 active seed，沒有已驗證 parent 可供 configuration library 或 state-goal planning 使用。修復收到的診斷只說 required backward evidence missing，未具體區分 adapter 不支援複合路徑與作者真的漏接 loss/backward；模型各用一次原樣回覆消耗 provider call，duplicate guard 正確避免重複執行。

Reference 約 102.47 秒、candidate 約 78.26 秒停止，皆為 `stop_reason=time_limit`；下一次派送 reserve 分別約 99.51、59.40 秒。這是既有 deadline reserve 所致，不是每組保證用滿 120 秒或候選全數不可達的證明。不能由單 seed、零 valid 的 linear 結果推導整體生成器優劣。

Numerical audit 也沒有救回這些案例：兩組 attempt0 的保存理由是 `original execution/state/path is not verified`；candidate/0002 為 `not selected by bounded audit schedule`。沒有執行出 numerical agreement/violation。即使日後解除 schedule，未建立原始 execution/path 的適用條件仍不能略過；含 `optimizer.step()` 的舊 numerical certificate 限制也仍獨立存在。

## 泛用後續方案，未在正式實驗後實作

- 將「有 output backward、但目前 adapter 無法建立 target 路徑」作為明確診斷與未支援能力記錄，避免反覆要求 LLM 修補一個已執行 backward 的來源。保留已量測的 forward／output-hook 事實，但不回填本輪 backward-valid coverage 或 correctness Pass。
- 設計有界的 call-boundary／複合 backward 證據：保存呼叫前輸入 edge、呼叫內建立的有限子圖、輸出與 operand 路徑，觀察同一次完成 backward 的內部傳播。必須排除繞過目標的額外分支、重複 operand 的歧義、外來 hook、alias／mutation、跨 invocation 借證；若經 view/transpose，須證明對應的座標 pullback，不能只比最終 leaf.grad。使用另一種證據範圍與版本，不把 composite 邊界冒稱單一 direct target Node。這也是未來 call-local VJP admission 的前置工作之一。
- 分離 native callable 的回傳型別證據與表示不變性／effect 證據。對可證明 native binding、schema 或實際回傳型別的有限域，建立受限 Tensor-result binding，讓安全的 loss 操作可分析；保留未知型別、subclass／override、metadata-sensitive 回傳與未審閱 effects 的 unknown。不能僅將所有 `probe.call` 回傳視為 Tensor，或因 forward 成功便發證。
- 增加有界的純靜態 shape/numel 約束診斷，例如已證 native factory 加 literal reshape 的元素數矛盾。只在確定條件下於 worker 前阻止生成錯誤；動態 shape 或外部 RuntimeError 仍保留 unknown，不把一般框架例外自動當成可修復的作者錯誤。

以上需要獨立正反控制及新的 Slurm 驗證；本文件沒有提供新執行、缺陷確認或 post-final 指標改善。

## 證據固定

下列相對路徑均以 `result/optimizer-gradient-chain/779954/final-seed0/` 為根。

| 保存檔案 | SHA-256 |
| --- | --- |
| `reference/linear/summary.json` | `fdd003d7d3cbf843fc4de854d04c5e6448c7d6f82ba970252f86dec5206c6101` |
| `reference/linear/attempt-0000/program.py` | `ac4aeb9d8d2191318f321a86cdd532ba6cf1b20ace88d92b15f5172a49b2460a` |
| `reference/linear/attempt-0002/program.py` | `287111e1b4970d4cd6296c7fd75064cb09a20f099228a5be2aff490c1b4f7834` |
| `candidate/linear/summary.json` | `4d3aa0469f8cc72a7c874869977cd395bb0adbc6dfd47a80b35a6381d5947677` |
| `candidate/linear/attempt-0000/program.py` | `7a284891a2d139b3549e727a2332776a3a026a1792ddb9d94c87ca9474d97e48` |
| `candidate/linear/attempt-0002/program.py` | `9ff9346bf24e2f6d51d3ec183f6aeca2fcbbee14b78f9563ef7a348568c728c4` |
| `candidate/linear/engine/mirage/backends/pytorch.py` | `15af18243fd562c33ef43e8d2362d942e523226fdddfd4d1a4f0a50ba088c8ac` |
| `candidate/linear/engine/mirage/runtime/worker.py` | `3ee089495789d8da4579c194f24afa1faee602b8b3381717b95b3f9e8dd644be` |

主要索引：[reference summary](../../result/optimizer-gradient-chain/779954/final-seed0/reference/linear/summary.json)、[candidate summary](../../result/optimizer-gradient-chain/779954/final-seed0/candidate/linear/summary.json)、[reference 初始化例外](../../result/optimizer-gradient-chain/779954/final-seed0/reference/linear/attempt-0002/execution/stderr.txt)。
