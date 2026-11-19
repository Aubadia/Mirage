# Roadmap 026　由執行證據補足 oracle 前提

已核准。正式評測對照 809797，每 API 180 秒、10 個獨立 API pool、4 個常駐 CUDA worker。此次優先實作 context 產物的通用協定。舊框架已定義的恢復語義決定 oracle 資格，benchmark 名稱、issue 與離線稽核均不進入規劃。

|階段|實作與驗收|
|---|---|
|P1|在 backend、runtime 與 inspector 分別記錄 state、產物使用方式及缺少的契約條件。通用 decorator 能力小測須真的執行受控函式且區分同一／不同 context。能力探測獨立於正確性與 coverage。普通 callable 只安排有界微分探測。|
|P2|planner、workflow 與 feedback 根據缺失前提提出使用原始前綴的單一變更；每個使用方式一次生成，最多一次基於新證據的修復。保留每個合法來源及變體的 CPU/GPU 差分。|
|P3|在 Slurm 重跑 808947/10 正例與 808947/110、111、809797/14、55 負例；用只截至 14 的前綴、遮蔽識別名稱、真實 provider 傳輸驗證接下來兩次全新 state 規劃內確實送出。驗收 decorator 同實例／新實例、僅 with 與普通 callable。保存 request、回應、parent、選擇次序及稽核證據。|
|P4|驗收成功才使用相同凍結版本送正式 sbatch。記錄候選、請求、合法、方式見證、契約適用、違反、重播與成本。見到實際模型回覆、多 API 的 target 返回與雙裝置比較後停止監控。|

交接：目前在 P1。修改範圍為對應 backend/runtime、inspector、planner、workflow、報告及 benchmark 驗收與 Slurm 腳本。驗收 fixture 可識別歷史 run；共用演算法只消費能力、觀測及契約。其他框架要提供相應的產物身分、scope 邊界及有根據的恢復契約。後續進度與 job 追加於本頁。

P1/P2 已接入正式 state planner、stateful request、runtime/inspector、pipeline feedback、prefix corpus 與 benchmark funnel。Scope 恢復沿用既有契約資格；decorator 入口以實際 factory 產物返回的 wrapper code 與閉包身分連結，能力探測使用獨立副本，另存於記錄。P3 驗收已送 810609，凍結 protocol 為 /work/asciibase64/Mirage/results/stateful-check-e2e/protocols/20261119-oracle-preconditions-180s-001。先查該 job 的 tests.log、controls.log、dispatch.log、acceptance.json 與 HTTP 請求。正式 job 尚未提交。

810609 通過控制組：84 項回歸含 8 項條件略過，舊正例在兩個獨立程序均為 positive，4 個直接重入負例均為 unknown。Slurm 在服務載入後建立測試 step 時回報 socket timeout，驗收 job 以 1 結束，沒有 HTTP 派送證據。保持相同快照重試驗收，未送正式測量。

排程器已回覆驗收重送 job ID 810621，沿用 001 快照。改查 validation/810621/，避免重複提交。

P3 通過。810621 COMPLETED 0:0；測試共 84 項，其中 76 通過、8 條件略過。validation/810621/ 的五個封存案例與額外獨立正例 replay 均滿足門檻。遮蔽名稱的規劃只使用截至 attempt 14 的資料；第 1 次新規劃即送出帶原始 parent 與 required_use 的請求，HTTP 200。接受紀錄在 acceptance.json。實際生成程式只進入一次 decorated scope，一筆基於新使用方式的修復重複相同輸出，runtime-valid=0，未新增重現。P4 正式 job 810666 在 8gpus，protocol 001，除快照外其餘參數與 809797 相同。等待多 API 返回與真實雙裝置比較後交接。

P4 已達交接門檻，停止監控。810666 的模型載入 186 秒，取樣時 10/10 API session 均有真實 HTTP 回覆與目標返回，執行時段重疊；已有 4 個合法成員具 CPU/CUDA 兩側目標返回與比較結果。證據 /work/asciibase64/Mirage/results/stateful-check-e2e/810666/startup-verification.json。後續只在使用者要求時查 sacct 及 benchmark-results.json，逐 API 的 oracle_precondition_funnel 與 generation.oracle_precondition_request_seconds 在 benchmark-metrics.json。正式啟動截面不構成最終重現數。
