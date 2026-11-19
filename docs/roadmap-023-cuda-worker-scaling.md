# Roadmap 023　單卡常駐 CUDA worker 比較

已核准比較 1、4、7、10 個 worker，依實測更新方案。沿用 805400 的凍結程式、原始 tensor、觀測與 oracle。僅增加本實驗的 benchmark driver 與 sbatch/experiments 腳本。

- 在同一 Slurm 節點上配置一張完整 GPU、12 個 CPU（等同原 testing step）與 96 GiB RAM，不載入 LLM。
- 固定分層抽取 805400 合法成員，涵蓋十個 API、來源與變體、原先有／無差異、需重播的成員。每個選中成員都重新做 CPU、CUDA 及比較；重播用新的一組程序。固定兩次反向順序，控制同一批輸入及資源。
- 報告含啟動的 elapsed、worker 就緒後的測試時間、吞吐量、顯存取樣、實際執行程序數、錯誤及 oracle/重播一致性。先排除覆蓋缺失、OOM 或結論失真的配置，再以同工作量的執行 wall 選擇；若結果接近，標示不確定性。只變更實驗 driver，不直接變更正式預設。
- 已完成：Slurm 806654；凍結 driver SHA-256 4bbab3dce1964d542ff1ae8e1d3f00cb9a8961ab08646c7dcb54a0014cef20cd。結果根目錄 /work/asciibase64/Mirage/results/cuda-worker-scaling/806654/。

每輪 96 個成員（23 source、73 followup），其中 22 個需獨立重播，共 118 對。另配置 2 個 CPU 執行程序與 1 個 CPU 比較程序；原案例在 CUDA 上建構時，CPU 投影也派給具 CUDA 能力的程序。順序 1/7/10/4，第二輪反向。已使用 summary.json、各 trial 的 cases.json 與 cohort.json 核對每筆兩側及比較的 signature。

首組實測發現即使只有 1 個 worker，compiled_execute 次數也可能偏離封存並改變不可比原因。追加相同 cohort、順序與資源的隔離對照，唯一的執行變更是在每個成員開始前呼叫 torch.compiler.reset()。兩組分開計時與分析。

隔離對照已完成 Slurm 806666（afterany:806654）。凍結 driver 在 protocols/a40a6419b8dc9569/。兩個 job 均為 COMPLETED、exit 0:0；本項實驗已結案。

## 實測與選擇

兩個單卡 job 均已完成。806654 保留跨案例 compiler cache，806666 每次進入案例前重設 compiler。兩者在不同 H200 節點；worker 數的排序使用同一 job、同一張卡的數據。每組固定執行兩輪，每輪有 118 對 CPU/GPU 與比較。wall 包含兩批程序的啟動與關閉、執行、比較和重播，不包含 LLM 或正式 campaign 的其餘階段。warm 為程序就緒後兩批處理時間之和。P50/P95 為 member 入列到比較完成的延遲，採兩輪各自分位數的中位數。顯存為取樣的整卡已用量。

|Job|CUDA workers|兩輪 wall 秒|warm 秒|對/暖機秒|P50/P95 秒|最大取樣 GiB|相對封存 signature 不一致 / 192|
|---|---:|---|---:|---:|---:|---:|---:|
|806654|1|84.14 / 82.06|68.91|1.712|12.02 / 43.29|1.91|4|
|806654|4|77.06 / 76.61|63.29|1.864|4.40 / 41.95|4.13|1|
|806654|7|78.23 / 78.01|63.01|1.873|3.21 / 41.88|6.24|0|
|806654|10|79.85 / 78.92|62.77|1.880|3.25 / 41.86|8.52|0|
|806666|1|96.73 / 91.20|77.31|1.526|16.66 / 51.81|1.91|0|
|806666|4|78.47 / 78.23|62.78|1.879|6.43 / 46.65|4.21|0|
|806666|7|80.60 / 78.57|62.35|1.893|4.90 / 46.30|6.18|0|
|806666|10|81.79 / 81.32|62.78|1.879|5.03 / 46.80|8.45|0|

每輪 96 個最終成員中，4 筆 strict agreement、92 筆 strict incomparable；22 筆出現原始差異，其中 21 筆有穩定差異欄位。各輪 236 次 CPU/CUDA side 執行中，234 次 target 返回，2 次為與封存一致的 target 例外。這些是差分判讀，本實驗沒有重新執行 primary MR、benchmark 歸因或 LLM。未 reset 時，兩個編譯案例的 compiled_execute 次數偏移，strict 不可比的原因從 requested_state_not_attained 變成 semantic_history_changed；改用 reset 後逐筆驗證兩側及比較 signature。

## 建議納入正式執行器

每張測試 GPU 初始採用 4 個常駐 CUDA 程序，整張卡共享。API 各自排隊，worker 透過公平調度領用工作，CPU 執行與比較沿用獨立容量。每個合法 source/followup 以相同初態與語意輸入分派 CPU/GPU，兩側完成即比較；無已匹配 MR 亦執行。原始差異、適用性、oracle 違反及獨立重播分開記錄。每個 case 重新構造 training state；後端需有 compiler、RNG、grad/autocast、hooks 與裝置資源的隔離或退回新程序。程序污染、致命 CUDA 錯誤或逾時時只替換該 worker，另行保留原因。重播使用新程序。其他設備、其他框架或工作負載以相同固定 cohort 流程重選寬度。

本次交付為受控實驗與配置建議。正式 scheduler 整合後，以相同 10 APIs、每 API 180 秒 LLM 預算驗收完整 campaign；統計全部合法成員、兩側與重播終態、每 API latency、差分 drain、整體 E2E、結果一致性及顯存。大型 campaign 的總加速幅度仍待測量。
