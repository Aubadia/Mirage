# Roadmap 022　獨立生成、裝置差異觀測與具名輸入證據

使用者已核准實作與 Slurm 驗證。起點是 roadmap 021 的 804883，結果在 /work/asciibase64/Mirage/results/stateful-check-e2e/804883。先通過封存回歸，再以 021 同 roster、模型 revision、seed、2 GPU、8 workers、2 generation slots、每 API 180 秒 LLM 重跑。確認真實模型回應及多 API 的 target 返回即可交接。

| 項目 | 實作範圍 | 驗收 |
|---|---|---|
| 請求隔離 | generator/provider、http_provider、runtime/scheduler、workflow/pipeline。隔離不明請求的生成名額，分離測試 drain。 | 以 804883 provider-051 的逾時重建十個 pool，兩個生成名額。其他 pool 零連帶取消、至少一個新回覆、既有測試繼續、未確認名額不超賣。檢查重複確認、服務暫時失效、預算計費。 |
| 裝置觀測 | inspector/comparison、artifacts/replay、報表與 campaign scorer。逐呼叫欄位保存結構、值差、時點及可比條件。候選與 oracle、正式重現分別計數。 | 804883 固定 182 筆雙側返回全部有具名結果或具體缺口；保留 12 筆既有一致，至少 37 筆有可對齊觀測、至少兩個語義家族。無 oracle 時穩定與單次注入、不同輸入/alias/cotangent 及額外 API 作正負控制。 |
| 輸入建構器 | runtime/input_constraints、runtime/stateful、backends/pytorch、inspector/oracles。靜態綁定與執行 recipe 證據，維持局部 VJP 前提。 | 29 個封存變體（15 份程式）全數消除框架建構器 unbound_name；記錄每例真正執行的有限差分或剩餘前提。核對 shadow、竄改、錯誤座標及額外 API 正負控制。生成與 oracle 不新增 benchmark 名稱分支。 |
| 調度交接 | benchmarks 的封存驗證與正式 driver、sbatch/validation、針對性測試 | 凍結所有輸入，Slurm 跑驗收；失敗則修正重試。通過後才提交正式 job，保存 hash、job ID、產物與啟動證據。 |

## 交接進度

- **目前狀態：三項實作與封存驗收完成；正式 job 805400 已啟動並達交接點，已停止監控。** 若後續要求最終 metrics，先讀下方同一作業的 benchmark-results.json、逐 API benchmark-metrics.json、cleanup.json 及 Slurm 終態。

- 已實作請求名額保留與 vLLM 單筆終態檔、獨立的逐欄原始差異與重播、recipe 認證及局部梯度檢查。v1/v2 的封存 manifest 仍使用原凍結引擎。新 manifest 為 device-differential-3。原始差異、穩定候選、oracle 與 benchmark 歸因各有獨立欄位。
- 驗收快照 roadmap022-02，Slurm 805362，產物 /work/asciibase64/Mirage/results/stateful-check-e2e/validation/805362/。182 個成員及 29 個梯度成員的相對路徑集合均經 SHA-256 驗證。169/182 量到具名欄位；12/12 舊 oracle 判定保留。29/29 接受框架注入的建構器，13 個真正完成擾動並報數值梯度違反，9 個未取得路徑證據，7 個仍缺程式效果證明。真實 CPU/CUDA 的其他 API 結構化輸出及寫入控制亦通過。較早的 805346 是第一輪快照。
- 待最終回歸通過後才可送件。凍結目錄 /work/asciibase64/Mirage/results/stateful-check-e2e/protocols/20261119-observed-check-180s-001，protocol SHA-256 3c769ccb00e0c9706ba37e88aa8abf11e1b14b06b3c1db884e0f9a5b70df4d53。76 個正式檔案與 805362 快照雜湊完全相同。除程式內容 hash，所有 protocol 欄位均已逐欄與 804883 比對相等。

- 805362 Slurm COMPLETED、exit 0:0、耗時 7:20；119 tests 全數通過，424.998 秒，tests.log 已保存。805346 較早版本亦完成（118 tests）。正式 job 805400 已在驗收完畢後送到 8gpus；請勿重複送件。結果根目錄 /work/asciibase64/Mirage/results/stateful-check-e2e/805400/。

- 2026-11-19 台北：805400 於 02:31:28 開跑，Qwen/Qwen3.8-27B 載入 214 秒，健康檢查 HTTP 200、revision 已核對。凍結 driver 的 started=true、overlapping_target_sessions=true。10/10 個 pool 已啟動、21 個請求終態檔，9 個 pool 同時段已有真實模型回應及目標 api_exit。例證 campaign/api-002/execution/provider-001/http-47a85f6eb4df47748b46ef2af4933f47.json 及 campaign/api-002/execution/attempt-0000/execution/source-trace.jsonl。依使用者要求在啟動後交接。封存結果中 13 個局部數值違反（其中 10 個未達整體狀態）不列為新盲測重現數。
