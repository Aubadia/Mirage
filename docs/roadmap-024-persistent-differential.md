# Roadmap 024　正式差分 worker 整合

使用者核准實作與送件。沿用 roadmap 023 與 805400 的設定。估計 campaign 約 2 小時 50 分，待正式數據驗證。

1. 在 runtime/scheduler、runtime/multi_api、workflow 與 artifacts/replay 接入共用的常駐執行器。每個 Slurm 測試 step 使用一張可見 GPU，CUDA 上限 4、CPU 執行 2、CPU 比較 1；每 API 獨立佇列並共享容量。範圍包含單 API check、封存驗收與正式 campaign。保留舊版封存策略。
2. 每個合法生成成員與確實執行的表示法變體都派到 CPU/GPU；即使無適用 oracle 或要求的 training state 未達到也保留原始差分。沿用凍結引擎與原始配置。案例隔離、compiler reset、獨立重播、worker timeout 與回收不得影響其他 session。
3. 在 benchmark driver 報告每 API 及全 campaign 的 E2E、LLM 與差分 latency、完成對數、差分第一個入列到最後完成的 wall 與吞吐量。
4. Slurm 重跑 805400 的既有 96 筆固定驗收 cohort（10 API、118 對）；逐筆對照兩側、strict 狀態與原因、差異及重播欄位，核對獨立與共用程序身分，覆蓋逾時替換。核對其餘封存案例的合法成員枚舉。
5. 驗收通過後凍結正式程式，對照 805400 roster、seed、模型 revision、資源與預算，10 API 各 180 秒 LLM；確認模型載入、實際回應及目標 API 返回後停止監控。大型檔案在 /work/asciibase64/Mirage/results/。不等待正式 job 結束。

進度：開始實作。接手時讀取此檔及其後的驗收與送件紀錄。

已接上 scheduler、process、multi_api、workflow、artifacts/replay、runtime/differential_pool 與正式 benchmark 指標。相同凍結引擎清單的 API 可借用同一 worker；原一般 CUDA probe 仍可取得獨占執行期。獨立重播的兩側必須用首輪比較後才建立的程序。驗收程式為 benchmarks/archive_pool_acceptance.py，批次腳本為 sbatch/validation/persistent_differential.sbatch。
808408 回歸未通過：修正一般 CPU 與 generation 名額、凍結測試位置及 804883 timeout fixture。新凍結 protocol 在 /work/asciibase64/Mirage/results/stateful-check-e2e/protocols/20261119-resident-differential-180s-002/。808494 進行回歸及 805400 封存驗收。正式 job 尚未提交。

808494 COMPLETED 0:0。56 項回歸中 48 通過、8 條件式整合測試略過。acceptance.json 在 /work/asciibase64/Mirage/results/stateful-check-e2e/validation/808494/。96/96 雙側與比較簽章吻合、含重播 118 對、CUDA 同時執行峰值 4 且 PID 有重用；強制逾時替換後下一案成功。全封存枚舉保留 3515 個合法成員，另納入 106 個合法 follow-up（只做枚舉，未包含在 96 筆實際重播數）。批次入列至最後更新 97.84 秒、1.206 對/s；成員 latency P50 23.79、P95 84.55 秒。正式驗收口徑包括排隊、程序啟停及獨立重播，與 023 的就緒後吞吐量分列。
正式 job 808947 已提交 8gpus；使用上述 002 凍結快照。結果根目錄 /work/asciibase64/Mirage/results/stateful-check-e2e/808947/。接下來讀 service-ready.json 並用 benchmark driver 的 started 命令確認兩個 session 各有真實 HTTP 回應與目標返回；再檢查正式差分程序證據後交接。

已達使用者交接條件，停止監控。808947 啟動驗證檔為 /work/asciibase64/Mirage/results/stateful-check-e2e/808947/startup-verification.json。模型 Qwen/Qwen3.8-27B 啟動 187 秒，健康與模型身分驗證通過。取樣當下 9 個並行 API session 均有真實 HTTP 200 及 target 返回。正式執行已看到 CPU、CUDA、comparison 程序各自重用，7 對已完成比較。最終數據待 job 完成後讀取同目錄的 benchmark-results.json 與每個 API 的 benchmark-metrics.json；無需重送。
