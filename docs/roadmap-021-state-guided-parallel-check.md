# Roadmap 021　狀態導向的多 API 平行測試

## 交接

- 使用者已核准本 roadmap、實作與最終 Slurm 評測。分支 feat/POC-Training-State-Not-Loop。起點為 roadmap-020 完成之後，P6 採用四次嘗試與整體 wall time，不能當作 180 秒模型預算對照。
- 狀態：已實作，最終版回歸 804881 通過。正式評測 804883 的模型已載入、十個 API pool 已啟動，且已確認多 API 同時測試。已達使用者指定的交接點。之後若被要求查看結果，讀本文末尾的同一 job 產物。
- 最終停止條件：十個 benchmark API 並行，各自 180 秒實際 LLM 時間。確認 vLLM 就緒且服務過真實請求、框架已執行測試後，可交還使用者。最終結果由作業自行彙整與關機。job ID、凍結版本、結果及記錄路徑須交接，提交和啟動均不等於測試成功。

## 已核准的行為

API 名稱為唯一必要語義輸入。依安裝版本取得 API spec、overload 與能力。在證據中分開記錄輸入合法性、指定 checkpoint 的目標狀態和 oracle 可比條件。ACETest 式的實作 guard 需要能對應同版本，DeepConstr 式以具體接受、拒絕與合法補集反例修訂假設，避免將裝置限制或數值條件升格為全域限制。RESTler 式資源依賴與執行回饋導引最短合法 prefix 和 suffix。已知 benchmark 機制只供獨立事後歸因，舊 property 已棄用、僅供參考。

狀態家族包括 autograd 模式、梯度緩衝與累積、graph 生命週期、module/buffer、activation checkpoint/recompute、saved tensor/offload、autocast/reentry、scaler 與跳步、optimizer 持久狀態、parameter group/scheduler、RNG、資料進度、訓練 checkpoint/restore、compile、capture/replay、分散式同步。家族依實際能力驗證，不支援、未知、未達、已達須分開，家族名稱與規劃本身不能充當 evidence。先涵蓋可行家族，再選有因果關係的組合。LLM 產生或修復語義程式，框架建構可區分的合法輸入。

每個合法來源均獨立展開可觀測表示法並派 CPU/CUDA；合法表示法變體也要派差分。已有相同初態、程式、變換和路徑的 initial followup 可去重並引用證據。單側 target 合法和指定 state 達成分開。主 MR/數值評估與裝置差分並列，主判定不抑制差分。兩側有能力就嘗試，對齊初態、表示、歷史及觀測後才可判定；能力不支援、不可比、穩定差異、未完成、環境缺席均分列。差異需重播與獨立佐證。模型用完預算後框架繼續 drain，硬時限保留可續跑工作。

## 生成、執行與回饋

- 每個 API 有自己的 corpus、輸入與狀態回饋、表示法 backlog、獨立模型計時與 artifacts。可同時 admission 全部 API。共享排程器依等待的 API 輪流發出可用的 generation / CPU / CUDA 名額，空閒容量可被還有工作的 API 使用。實際 GPU 並行度另受測試 GPU 的裝置名額約束。
- API spec 來自安裝版本的 metadata、overload、能力和可讀 guard。有限語法能處理維度與 shape 關係。用一般輸入和 guard 兩側的有限樣本探測，依 overload、裝置和實測接受／拒絕逐步修訂；無法解析時保留未知。採樣器可向生成器提供相符的具名 operand 配方。原始樣本與程式一起封存。
- 生成器可產生最小語義片段，在需要時使用真實 training 歷史。由 runtime 驗證實際目標呼叫、必要 backward 和具名 checkpoint。需累積梯度且父程式可安全追加時，保存父 build 和 run 前綴，從父初態、RNG 驗證相同的呼叫、梯度及 backward 歷史。其他歷史需要包住原 forward 或更改 graph 保留時，父程式只作生成參考。
- 每個執行成功的來源都展開該來源有證據支持的 tensor representation。來源與合法變體都進 CPU/CUDA 佇列。即使目標 training state 未達、主 MR 已通過或沒有主 MR，也保留裝置差分義務。比較以相同語義輸入、寫入、歷史、兩側能力、可比觀測與容差為前提。有穩定差異時使用獨立 benchmark audit 核對機制。缺原始檔或未完成歸因一律明列。
- 跨框架時，保留共享排程、回饋與證據分母，替換 spec 擷取、執行／裝置 adapter、tensor 轉換、事件觀測及框架特有的 MR 前提。這次只驗 PyTorch 和單個測試 GPU；分散式目標須有真實 multi-rank 證據。驗證泛化需用事前保留、未參與設計的 API。

最小流程（各 API 的生成、表示法與差分消費者可重疊執行）：

```text
parallel for api in input_names:
    spec, pool = inspect_installed_api(api), open_private_pool(api, llm_seconds=180)
    while pool.can_generate():
        goal, prefix, recipe = plan(spec, pool.verified_feedback)
        source = shared_workers.generate_or_repair(goal, prefix, recipe, charge=pool.llm_clock)
        evidence = shared_workers.execute_and_inspect(source, goal)
        if evidence.legal_target_return:
            pool.enqueue_device_pair(source)
            pool.enqueue_eligible_representations(source, evidence)
        pool.feedback(evidence)
    drain(pool)   # inspect each realized representation, enqueue legal device pairs
    persist(pool.coverage, pool.oracle_verdicts, pool.costs)
audit_frozen_benchmark_and_aggregate(all_pools)
```

## 實作與驗收順序

| 階段 | 工作和受影響範圍 | 驗收與交接 |
|---|---|---|
| 1 | 差分、worker、既有封存 replay、來源/變體佇列。src/mirage/artifacts/replay.py、inspector/comparison.py、runtime/worker.py、sampler/representations.py 與對應 tests | 合法來源與變體有唯一差分記錄，含 previously_reached；主 oracle pass 不抑制差分。可跑的另一側不因先執行的一側失敗而消失。保留兩側實跑、不可比、中斷與續跑證據。 |
| 2 | runtime/api_metadata、generator/state_plan 與生成提示、workflow、inspector/report、對應 tests | Spec 約束有版本和來源；正負與補集結果可追溯。選擇家族、具體目標、prefix 和決策使用的回饋可追蹤。僅 runtime 驗證的 target/state/transition 計入覆蓋。 |
| 3 | 正式 check、CLI、runtime/{multi_api,pipeline,scheduler}、benchmark driver 與實驗 sbatch | 沿用 stateful 的每 API pool/獨立 LLM 計時、公平共享 generation 與執行容量，多 API 實際重疊。模型等待、worker、重播、差分不扣 LLM 額度；報告各自耗時與 campaign wall。 |
| 4 | 驗證並在 8gpus 提交最終 job，遠端結果放 /work/asciibase64/Mirage/results/ 下 | 凍結程式、候選集、模型、配置、seed 與 Slurm job；先完成輕量檢查和針對性回歸，用 srun 做 compute 測試。區分正式執行、preflight 與未驗證的驗收項目。 |

最終 metrics 包含每個 API 與整體 E2E、排隊、LLM 請求與 latency、tokens、worker/重播/差分時間、實際目標回傳、合法來源、家族/狀態/轉移/組合、表示法分母與達成、CPU/CUDA 派送／實跑／可比、MR 與數值判定、穩定差異、疑似違反、重播與獨立 benchmark 歸因。凍結的十題全數保留在分母。補充研究驗證包括同預算的 feedback/constraints 消融與保留 API；最終指定十題的單輪量測無法取代這些對照。

## 實作紀錄

- 已核對既有 check 使用 whole-wall+fallback、原 stateful 能提供 per-API pipeline、SharedScheduler 與 LLM-only 計時。下一步依上述驗收分項實作與測試。

- 2026-11-18 實作。已將 name-only check 接入獨立模型預算與 worker drain；多名稱 check 使用既有共享排程器。逐 API 旁路消費 source 與表示法子案例，寫入 device-differential-2、強制雙側與單側原因，保留舊版重播格式。已加入 16 家族的有限目標與動態 witness、具名建構器、實作 guard/近域探測、補集反例與學習回饋。候選家族或缺少 multi-rank 的環境絕不算 reached。
- 首輪 Slurm 804779，/work/asciibase64/Mirage/results/stateful-check-e2e/validation/804779/tests.log。凍結 iteration-01，58 項針對性測試通過，417.825 秒，exit 0。覆蓋真實 torch/CUDA、同一來源的初始表示、差分重播、中斷對帳、狀態正負控制、共享容量與多 API pool。最終版另包含 input recipe、在線約束與進一步的 prefix/報表檢查，須用最終快照再驗。
- 最終腳本 sbatch/experiments/stateful_check_e2e.sbatch。2 GPU 分別給模型與測試，16 CPU、192 GiB、8gpus。8 個共享 execution slots、2 個 generation slots、10 個 API pool 同時開放，單測試 GPU 序列化實際 CUDA 區段。12 小時為 allocation 上限；舊 4,398 例若每個都補一個 CUDA 程序即可能超過 4 小時，保留 30 分鐘供歸因與收尾。每 API 的模型額度仍是 180 秒。

- 最終凍結於 /work/asciibase64/Mirage/results/stateful-check-e2e/protocols/20261118-stateful-check-180s-001。protocol SHA-256 a1b0c32eb8f88c86699f59fb18be4aee29768b1669f8ab37d3d324cde005c6e0。verify 通過；最終回歸快照 iteration-02 與正式快照交集內全部 73 個檔案的 SHA-256 相同。兩個 batch 腳本的 Slurm test-only 均接受，正式送件 804883，依賴凍結回歸 804881 正常結束。正式 job 產物根 /work/asciibase64/Mirage/results/stateful-check-e2e/804883/。
- 最終版回歸 804881 在 2026-11-19 00:06:40（台北）前完成，69 項通過，用時 451.596 秒；Slurm COMPLETED、exit 0:0、wall 00:07:33。包含 CPU/CUDA、補集／在線約束反例、真實 prefix replay、強制雙側、重啟、資源分配、benchmark audit 負向控制。詳見 /work/asciibase64/Mirage/results/stateful-check-e2e/validation/804881/tests.log。804883 於 00:06:40（台北）開跑。

- 啟動驗收已通過。模型 Qwen/Qwen3.8-27B 經健康檢查 HTTP 200、模型清單與凍結 revision 核對，啟動耗時 192 秒，ready 時間 2026-11-19T00:09:53+08:00。十個不同 API 均有 check-start 檔案。凍結 driver 的 started 檢查已回傳 started=true、overlapping_target_sessions=true；多個 session 同時取得真實模型 HTTP 200 及 target api_exit，包括 torch.Tensor.scatter_add、torch.linalg.slogdet、torch.std 和 torch.Tensor.copy_。例證為 campaign/api-000/execution/provider-001/http-d95184962fa34f3294f63fe4309217e5.json 以及相同 session 的 attempt-0001/execution/source-trace.jsonl。此時正式測試仍在執行，依使用者要求停止監控。
- 下一個 session 若要看最終結果，從 804883 的 benchmark-results.json 及 campaign/api-*/benchmark-metrics.json 讀聚合與逐 API 指標，並對照 Slurm 終態和 cleanup.json。缺總表時讀 campaign/summary.json、各 API 的 summary.json 與 differential-ledger.json，先區分尚在執行、收尾失敗與歸因未完成。列出實際 target return、可比差分、穩定差異與獨立歸因重現的不同分子，維持原始 10 個 API 分母；目前未宣告任何正式 benchmark 重現率或整體 E2E。
