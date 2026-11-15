# 後續能力擴充（future，未納入本輪實作）

優先補齊可驗證的測試能力，再增加 API 數量。本輪已把梯度來源與真實更新連結，但目前是單步、cold-state SGD 的有限域；下列順序依[本輪分析](analysis.md)、[數值檢查邊界](numerical-audit-boundary.md)及[研究定位](novelty-positioning.md)提出，不代表已實作或保證研究首次。

1. **P0：call-local effect 證書補足 numerical admission。** 讓含合法後續更新的程式，也能獨立檢查已捕獲的 target VJP。前提須證明 native target 身分，以及呼叫前、forward/backward 間與快照後的 effect 邊界；輸入保留參數、輸出、實際 cotangent、RNG／context 與未受後續更新污染的快照。仍須通過 baseline replay，並維持既有 FD 穩定性及單側檢查門檻。陽性使用正確導數加合法 step；陰性涵蓋 target 替換、隱藏 hook、快照污染與真正錯誤 VJP。前幾者應拒絕適用，後者在前提成立時仍應報 suspected failure；不能直接放行整個 optimizer certificate。

2. **P0：複合 target 的有限 backward 歸因與型別證據。** 正式 `linear` 顯示 target／backward 可以已執行，直接 Node 歸因仍無法適用。應分開「沒有 backward」與「有 backward、歸因未知」的診斷，對受限 native call 建立呼叫內 subgraph、輸入座標變換與 cotangent pullback 證據；只接受唯一且可驗證的 target 貢獻，無法分離的額外分支保留 unknown。來源分析須將受證明的 native return-type 與值／效果證明分開，避免把所有 target 回傳一律當 Tensor。陽性可涵蓋 transpose／reshape 包裝的合法複合路徑；陰性包含兩次 target 呼叫、取消／額外梯度分支、不同 invocation 及隱藏 hook。另用受限 literal shape 分析攔截 arange／reshape 元素數矛盾；不能將一般 RuntimeError 自動歸為可修復。詳見 [正式 linear 限制](final-linear-limitation.md)。

3. **P0：分開覆蓋目標與 oracle 的預算配置。** 正式更新鏈由0增至15，但舊語義單位40降至17、舊transition10降至5，數值 agreement8降至0；這是有限預算的測試組合取捨。保留普通 operand／training-state、optimizer-chain 與獨立導數檢查各自的量測與候選分母，對候選預測 oracle applicability、實測成本及各維度新增覆蓋，配置可審查的最低探索份額後再做適應性選擇。Fresh quota 不能替代 oracle 多樣性，異版本的單位也不能直接相加。驗證需同預算比較每條測試路徑的實際達成／適用／拒絕原因，並檢查廉價 library 不會長期排擠其他合法候選；不可只優化新鏈總數。這是下輪設計，未用正式結果回頭調整本輪策略。

4. **P1：warm-state、多步與 optimizer slot 的語義 checkpoint。** 測試記憶體表示如何影響歷程中的更新。本輪已具 cold momentum slot 比較，但正式15個有效 source／15個follow-up都未產生非空slot；先定義 step／microstep／phase、parameter／slot 的穩定選擇器，保存 warm slots、群組選項、輸入與 RNG；只允許 MR 明列的表示差異，在對齊階段比較參數與 slots，不能靠相同呼叫次數對齊。陽性是相同 momentum 歷程搭配不同等值表示；陰性是錯步號、缺 slot、錯群組及漏更新。證據不足應 inapplicable；前提完整而 slot 數值偏離才是 suspected failure。另需評估更新量相對參數尺度過小時的篩檢敏感度：目前主判定對 post-state 使用既定 dtype-scaled 容差，exact comparison 僅為診斷；若增加 update-sensitive oracle，須先建立減法消去與捨入誤差界線及錯誤更新控制，不能單憑降低容差把微小差異當 bug。

5. **P2：backward 寫入既有 grad buffer。** 這與目前 backward 完成後替換 `.grad`、測 optimizer consumption 不同。需記錄初始梯度值及 layout、實際累積事件、identity／version，區分寫入原 buffer 與更換物件；歸因應連到本次增量，不能把既有梯度算成 target 貢獻。配對保持初始梯度及語義輸入相同，再比較累積梯度與適用的更新狀態。陽性包含等值異 layout buffer 的合法累積；陰性包含 backward 後才換 buffer、額外梯度分支、清零及跨 invocation 借用證據。

6. **P3：擴 API／device／dtype 前建立域別證據。** 每個新域先界定合法 target path、可歸因梯度、同步／非決定性條件及容差；輸入關係與輸出比較須各自成立，不能沿用 CPU float64 結論。陽性用有獨立預期的原生控制；陰性涵蓋不支援 dispatch、缺量測、錯 dtype／device 與非決定性。分列規劃、worker、MR、重播成本及 unknown 原因，再以固定預算、隔離 corpus、多 seed 評估。已知配置可降低生成成本，仍不證明 child 達成或 MR 適用。

以上皆為 future 建議，不擴大本輪核准範圍或完成條件。研究主張仍限於 memory representation × training transition 的可驗證探索；覆蓋增加、控制案例通過與自主確認缺陷須分開報告。
