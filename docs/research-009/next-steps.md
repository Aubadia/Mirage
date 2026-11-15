# 後續泛用能力優先順序

[完整架構差距審查](architecture-gap.md)指出，應先將既有訓練能力接成「梯度表示→target backward→真實optimizer update→parameter／slot MR」的完整流程；下列生成與證據機制支援這個優先目標。

這些是完成本輪凍結版本後的設計建議，尚未實作；正式搜尋結果不拿來反覆調整本輪模型或替換測試API。

| 優先 | 缺口 | 泛用方案 | 驗收條件 |
| --- | --- | --- | --- |
| 1 | 描述目標不等於能把正確tensor送進正確呼叫 | 為可追蹤邊界建立穩定call-site識別，連結生成注入點、實際operand路徑、state checkpoint及最後生效動作；compiled/captured內部呼叫採獨立trace邊界，無法對應時保留unknown | helper、bound receiver、keyword、重複呼叫及compile/capture都有正反控制；不得用原始碼順序推論內部邊界 |
| 2 | 複合狀態需要多個合法且有順序的動作，單次patch容易只達成其中一半 | 在既有三層架構內，將已支持的狀態動作與前後置條件做成小型可檢查計畫；先驗證局部依賴與順序，再由LLM補入訓練程式；讓memory與training的合取成為可追蹤的計畫義務 | 報告所有合取目標中合法、部分達成、全部達成、MR適用的比例及成本；不同API轉移驗證，不能只增加容易的target-count predicate |
| 3 | build中的正常tensor初始化寫入仍缺少證明 | 以allocation ownership與alias/escape追蹤證明寫入只發生於新配置的私有初始化tensor；在公開或建立view前後分別檢查，未知呼叫和外部alias維持拒絕 | 私有初始化正向控制、共享alias/semantic-input寫入/未知callee負向控制；保留MR input relation檢查，不做全面in-place白名單 |
| 4 | 有效coverage-only案例無法得到適用MR；數值audit受穩定性與時間限制 | 每種oracle公布適用域、未知原因與估計成本，按已驗證的backward/path證據排程；另行建立獨立佐證流程，儲存minimized source、精確環境和假設 | unknown不轉成Pass；nonsmooth/ill-conditioned負向控制與錯誤梯度正向控制；獨立oracle失敗不能被同一checker重播取代 |
| 5 | family層級回饋仍混合提案、修補與執行成本 | 分開記錄proposal成本、完整repair episode成本、執行成本和新獲得的verified狀態；跨family轉移時保留不確定性及探索最低機會 | 不以相依repair當IID樣本；保留coverage-only及昂貴訓練狀態，不以容易MR數最大化取代研究目標 |

成功標準應同時包括：單位實際時間的有效程式與適用MR、具明確backward及operand-gradient證據的狀態、真正memory×training合取的到達、不可判定原因、及獨立確認的缺陷。增加狀態名稱或總joint數本身不代表改善找bug能力。

這些方案沿用Mirage的state predicates → legal transitions → runtime evidence → conditional MR流程。跨DL框架需額外後端契約與驗證；本輪PyTorch結果不能外推為已完成跨框架泛用性。

開發02額外暴露規劃協定冗餘：examined IDs與proposals已命名IDs必須跨欄位一致。後續可保留每個planned／unresolved proposal的明確目標與理由，由程式推導可計算的集合，減少LLM重複輸出自洽資訊；仍不得把空回覆推定為已檢查全部catalog。required regions同樣可按選定goal推導或再做有界協定修復，不能模糊接受錯注入位置。
