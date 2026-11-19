# Roadmap 025　生成終態與 state 能力探索

已核准 P1–P5。保留 805400、808947 封存。正式評測採 10 API、每 API 180 s LLM、4 個常駐 CUDA worker。依既有交接條件，正式 job 確認真實模型回應與框架執行後可停止監控。

|階段|實作範圍|驗收|
|---|---|---|
|P1|provider、workflow、resume、session 與 benchmark 報告分開記錄截斷、未確認請求、執行與歸因。限額內至多一次短回答重試。|以 808947 真實兩筆截斷重算；對照 timeout、未完成差分、缺失與僅驗存在的 artifact。|
|P2|以合法來源的實際見證建立前綴，與原先申請的目標分開保存。|核對 13 個 compile 合法 source 與 0 個原目標達成；保留舊結果。|
|P3|前置登記最小產物使用目標，以成功程式作為後續單一變更的起點。|封存 source 新執行、額外 state 及非 benchmark factory 控制；新 proposal 需有自己的證據。|
|P4|22 個 state 的能力探測、前綴選擇、後端事件驗證；優先 scope、RNG、compile/capture factory 與複合目標。|逐項正反例、缺環境或證據、不同 target 及文件變體；保留 MR 前提。|
|P5|延用正式 sbatch，凍結同設定重跑，保存 E2E、LLM/case latency、差分工作量及吞吐量。|先通過封存與控制組，確認正式模型載入、至少兩個 API 同時真實測試及差分啟動。|

目前狀態：已閱讀現有程式，開始 P1。檔案範圍為相關的 generator/provider、state_plan、runtime 與 inspector、workflow、artifact 報告、benchmark、對應 tests 與 validation sbatch。范例與固定 workload 放在 validation，通用路徑不使用 benchmark 名稱。結果在 /work/asciibase64/Mirage/results/stateful-check-e2e/。接手時先讀本頁末尾的 job、凍結版本及尚待驗收項目。

## 從 API 名稱到下一個 state

1. 以當前安裝版本解析公開 API、Python callable、類別／回傳協定、operator schema 與環境。文件、signature 中的 callback、async、callable 詞彙只幫助首次選目標。環境不足或沒有對應 observer 時保留具名原因。
2. LLM 生成合法 target 使用方式。每個 API 的 pool 分別記錄選了什麼、合法返回、實際使用的產物、已得到的 state witness、最近的執行診斷、已花的模型時間。截斷時保存 HTTP 證據，用同一預算重試一次。框架執行和表示法展開沿用既有佇列與差分排程。
3. 一次執行有三層資訊：target 返回、某個 state 的具體見證、本次事先指定的目標是否全部達成。對合法但未達標的來源，保存可驗證 checksum 的程式、初始參考與見證索引；此索引與可變異 corpus 分開。疑似違反與不滿足變異義務者排除。
4. 仍無有效來源時，從見證選一個單軸能力重新提案，驗證相同來源。成功後，從已登記的複合目標選一次單軸擴展，讓 LLM 看到原程式和缺的條件。新提案有自己的 trace、goal 與 verdict。這些步驟不要求每個 API 都有可用的複合軸。
5. 另外回報實際執行、state 達成、差分覆蓋、oracle 結論和 benchmark 歸因。僅驗 artifact 存在者可完成清點；無法提供獨立歸因證據。

## 22 個目標的證據與資格

下表是 PyTorch adapter 可檢查的有限目標。首次選擇種子可以用文件提示；表中判定依據是執行期觀察。跨框架可沿用選取、排程、前綴及分母，需為框架實作相應事件、產物身分、裝置及梯度證據。

|State|資格與實際檢查|
|---|---|
|inference_target|target 入口的 inference mode|
|target_backward|完成的 backward invocation 連到 target node 或已登記的產物|
|leaf_accumulation|兩次相鄰 backward 對同一 leaf 的 target 路徑有貢獻|
|same_graph_twice|同一 target call 的同一 node 在兩次完成的 backward 出現|
|module_buffer_operand|邊界的實際 tensor 與已登記模組 buffer 同一物件|
|target_recomputed|同一 checkpoint frame 內的重算並連到完成的 backward|
|saved_target_unpacked|target 儲存張量卸到 CPU，於相應 backward 恢復原裝置|
|autocast_target|target 進入時其裝置的 autocast 確實啟用|
|scope_reentry|同一 target 產物在外層未退出前再次進入、隨後平衡退出；可觀察的 Python context 協定。是否承諾恢復特定狀態由獨立 oracle 決定|
|overflow_step_skipped|真實 scaler、梯度、綁定參數與跳過的更新|
|target_update|已連結的 target 梯度與實際 optimizer update；現有此 observer 限 CPU|
|produced_optimizer_state|目標返回 optimizer，實際 step 產生持久狀態|
|produced_scheduler_step|目標返回 scheduler，實際 step 改變學習率|
|target_rng_consumed|target 呼叫邊界內 CPU／CUDA default 或顯式 generator 的 state 前後變化；探測首次呼叫及 RNG 目標|
|distinct_batches|相鄰呼叫同一 operand 位置的已登記輸入不同，且同 shape/dtype 的有限值確有差異|
|restored_update|綁定參數、已填充 optimizer state、還原後更新實際使用還原狀態；現有 adapter 支援有限 native optimizers|
|compiled_target|觀察受支援 backend 的 FX graph 執行；factory 產物另核對註冊與實際呼叫身分|
|captured_target_replay|需要 CUDA、支援的 capture observer、已登記 factory 產物的真實 replay|
|synchronized_gradient|已初始化至少兩 rank，在完成的 backward 取得綁定參數梯度並完成 collective|
|distinct_batch_accumulation|兩批不同輸入與同一 leaf 的兩次貢獻依呼叫與 invocation 對齊|
|autocast_compiled_target|同一 compiled target 呼叫內觀察到 autocast|
|autocast_recomputed_target|同一已還原的 checkpoint frame 在 backward 重算時觀察到 autocast|

## 交接紀錄

- P1–P4 已實作。第一輪封存與控制組 Slurm 809717，COMPLETED 0:0。82 個測試中 74 通過、8 個依環境設定略過。另以兩個 Gloo rank 驗證同步梯度。
- 808947 重新讀取證據：lu_solve attempt 1 截斷，158 runtime-valid、346 個合法差分成員、662 個已審 artifact；copy_ attempt 47 截斷，155 runtime-valid、293 個合法差分成員、542 個已審 artifact，另兩個僅驗存在（attempt 22 與 32 的 source）。兩個 session 的差分 ledger 均完整。
- compile 13 個合法前綴與原有 0 runtime-valid 均保留。809717 對 attempt 3 新提案確認 compiled_target，autocast_compiled_target 的基準／單一變更控制分別為未達成／達成。證據在 /work/asciibase64/Mirage/results/stateful-check-e2e/validation/809717/archive/archive-state-acceptance.json。
- 805400 固定 96 個成員、含獨立重播 118 對，全部配對簽章吻合。CUDA 並行峰值 4，PID 重用與 timeout 替換已驗證。固定 cohort 吞吐 1.256 對/s，與新生成 campaign 的吞吐分開報告。
- 第二輪凍結版本 /work/asciibase64/Mirage/results/stateful-check-e2e/protocols/20261119-evidence-guided-180s-002；驗收 job 809736。增加正式 workflow 的連續截斷控制、由 planner 自動選出一次單軸擴展、目標 catalog 按已觀測能力更新分母。分區 8gpus 現拒絕 normal QoS；明示使用帳號與分區同允許的 h200_256gpu，資源數量維持 808947 設定。
- 接手先讀本頁最後一筆狀態，再查 sacct -Xj 809736。成功才提交同一凍結快照的正式 sbatch。在模型真實回應、多 API 返回及實際 CPU/CUDA 比較均有證據後交接。正式結果完成時讀 benchmark-results.json 和各 API 的 benchmark-metrics.json。

- 最終驗收 809736 已 COMPLETED 0:0。83 個測試、75 通過、8 個略過，兩 rank 與封存控制另行通過。正式 selector 成功選擇原始前綴與後續複合目標。805400 固定 cohort 96 個成員對照一致，118 對含重播，1.224 對/s。
- 正式 job 809797 已提交至 8gpus。正式提交前分區恢復 AllowQos=ALL，原始 normal QoS 通過 preflight，因此正式使用與 808947 相同的 normal。逐欄對照協定，除 snapshot_sha256 外完全相同。輸出根目錄 /work/asciibase64/Mirage/results/stateful-check-e2e/809797/。目前等待模型與差分啟動證據，請勿重複送件。

- **P5 啟動驗收已達成，已停止監控。** 809797 服務就緒花 188 秒。啟動取樣中 10/10 API 各有真實模型回覆與 target 返回，session 時段重疊。差分 ledger 當時已有 7 次完成比較，程序紀錄顯示 CPU、CUDA 與 comparison 都已有重用。此為啟動截面，各檔案更新時間可能不同。
- 啟動證據在 /work/asciibase64/Mirage/results/stateful-check-e2e/809797/startup-verification.json。下一個 session 若被要求看最終數據，先查 sacct -Xj 809797，再讀同目錄 benchmark-results.json、campaign/api-*/benchmark-metrics.json。如仍在跑，只報告當時已發布的資料。target 返回、原規劃 state 達成、oracle 違反與成功重現應分欄呈現。請勿從啟動快照推算 E2E 或重現率。
