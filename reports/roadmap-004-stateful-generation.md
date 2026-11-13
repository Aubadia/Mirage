# 路線圖 004：通用表示變異與 LLM 訓練狀態生成

## 狀態與已核准方向

使用者已核准實作，並授權每個階段通過驗收後自動接續。G1–G7 已在記錄的實作邊界內完成驗收；凍結實驗、獨立審閱、audit 修正與資源清理皆已完成。下方歷史規劃段落描述當時範圍，執行記錄及狀態表則描述目前進度。[英文對照版](../docs/roadmap-004-stateful-generation.md)記錄相同範圍與證據。

設計有三項約束：表示變異針對實際張量輸入，不使用 API 專屬配方；訓練狀態操作及順序由 LLM 生成；正確性判定器實作可重用的語義關係，不逐 API 撰寫。核心必須接受新 API 而無須新增專用判定分支，並將框架無關的調度與 PyTorch adapter 分離。未來 TensorFlow adapter 是擴充方向，不是本次實作交付項目。

先前[路線圖](roadmap-003-pytorch-correctness-benchmark.md)與[評估報告](pytorch-correctness-evaluation.md)保留為歷史證據，其中 10/10 整合及 3/10 引導式生成結果，不是新路徑的驗收結果。既有工作目錄中的無關修改保持不動。

## 責任與不可越過的邊界

| 元件 | 責任 | 邊界 |
| --- | --- | --- |
| LLM | 生成初始化、完整狀態轉移程式、語義檢查點及 MR 相容程式對。 | 不提供預期答案，不認證適用性，不決定判定結果。 |
| Mutation engine | 探索實際運算元結構，列出合法表示轉換、套用並驗證效果。 | 接收張量描述與契約，不接收 API 名稱、問題編號或 API 專屬變異優先序。 |
| Inspector | 以證據確認目標執行、所需反向連接、狀態到達、變異到達及檢查點對齊。 | 生成宣告與設定旗標不是證明，不自動補做缺少的訓練操作。 |
| Oracle library | 驗證經審查語義類別的前提、輸入關係及輸出關係。 | 不假設所有輸出一律相等，也不要求每個目標 API 各寫一個判定器。 |
| Backend adapter | 提供框架專屬 API 中繼資料、運算元檢查、變異實作、觀測與重播支援。 | 明示不支援能力；核心不假定所有後端都有 PyTorch 的表示特徵。 |

開放 API 輸入不等於普遍的正確性評估。合法執行若沒有已支援且已成立的關係，仍記錄經驗證覆蓋，但正確性保持未評估。API 規格與可取得的 schema 可支持合法性及副作用判斷，不得編碼失敗輸入配方。LLM 的語義分類僅是提案，不是認證。

## 階段順序與驗收政策

| 階段 | 交付物 | 驗收關卡 | 記錄狀態 |
| --- | --- | --- | --- |
| 1 | 契約、後端邊界與固定驗收定義 | G1：責任及證據需求明確 | 通過（工作 769979） |
| 2 | 通用運算元探索與表示變異 | G2：跨輸入結構重用變異，保留必要關係 | 通過（工作 770003） |
| 3 | LLM 擁有的訓練程式生成與執行 | G3：可執行狀態轉移來自生成原始碼 | 通過（工作 770038） |
| 4 | 執行檢查與狀態／轉移證據 | G4：無觀測證據不能取得覆蓋 | 通過（工作 770050） |
| 5 | 語義類別 MR 與判定器整合 | G5：已驗證關係能區分違反與不適用 | 已實作子集通過（工作 770087） |
| 6 | 端到端驗證、生成多樣性與凍結 | G6：新路徑已獨立驗證，可開始量測 | 通過（frozen-02，工作 771760） |
| 7 | 凍結 LLM 實驗與獨立歸屬審查 | G7：結果完整、可審查且推論範圍正確 | 通過（工作 771760；reviewed-v2） |

依序執行。驗收失敗時返回負責的先前階段，保留失敗證據，不暗中放寬條件。檢查通過後在核准範圍內續行，不為一般階段轉換重複請求核准；重大設計或檔案範圍變更仍依 AGENTS.md。驗收須檢視實際結果，Slurm 提交成功或工作退出不能單獨代表通過。

## 階段 1：契約與後端邊界

### 工作

定義版本化契約，涵蓋運算元樹、別名群組、變異計畫與義務、抽象狀態述詞、生成程式、語義檢查點、執行紀錄及 MR 規格。區分計畫述詞、已觀察事實及不支援／未知證據。追蹤操作來源：LLM 原始碼、表示變異、量測機制或判定計算。

新增後端介面及初始 PyTorch adapter。PyTorch 匯入與操作集中在 adapter 或既有明確屬於後端的模組；新核心調度與判定邏輯只使用框架中立紀錄。以小型假 adapter 測試邊界，但不宣稱 TensorFlow 相容。

新增 `stateful` 策略與獨立契約版本，保留既有策略。生成介面為 `build(device, seed)` 加 `run(ctx, probe)`。Probe 只觀測或標記邊界，不替生成程式實作 backward、optimizer 更新、上下文遞迴、編譯、capture 或 replay。

初始狀態維度為梯度狀態、上下文狀態、計算圖生命週期及執行模式；記錄每類支援的可觀測述詞，未知圖事實保持未知。量測前固定驗收案例、數值政策、資源界限及明確有限的覆蓋分母。先以穩定、確定性的順序列舉合格變異候選並保存設定與順序；除非核准契約明列，適應式排序另行處理。

### 依評估補強的工作（E1、E3、E4、E5、E6、E8）

在狀態述詞之外，定義不依賴 API 的輸入覆蓋設定，包含有限數值量級、值分布、合法形狀邊界、dtype／布局能力、運算元角色及可觀察關係。類別是生成目標，不是預寫重現程式。分開記錄要求類別、實際性質及判定關係資格。形狀與數值在建立 MR 程式對前生成，表示變異不能為符合自身前提而暗改語義輸入。

協調確定性輸入政策與文件內容。保留原始安裝文件字串及雜湊，將 RNG 範例標註為文件而非生成指令，保存實際送出的完整標註提示詞。區分初始化種子設定與隨機取樣：單獨設定種子不證明案例依賴隨機，種子也不認證明確初始化。輸入及登記參數／緩衝區須有明確值，檢查隱含模組初始化及保留的 RNG 相依狀態；無法確認確定性時保持限制，不計為成功的確定性案例。

定義可呼叫物件引數綁定、檢查點資料序列化及必要程式入口契約。區分綁定方法與未綁定 Tensor 方法，在簽章或 schema 可提供時明示接收者需求；未知簽章不補猜。可執行函式留在保存原始碼中，檢查點只含支援值或已登記的重建紀錄。

### G1 驗收

- 契約能區分不支援能力、缺少證據、非法執行、關係不適用、Pass、Suspected Failure，以及僅覆蓋／未評估。
- 變異選擇不接收 API／問題編號；語義判定器選擇不限制 API 的進入資格。
- 假 adapter 可走通核心紀錄流程；PyTorch adapter 僅宣告已實作能力。
- 初始狀態述詞、語義類別認證、變異範圍、預算及驗收資料已成文；未指定的語義性質不能預設為真。
- E1/E3/E4：契約測試區分種子設定與隨機輸入、綁定與未綁定呼叫、可執行原始碼與檢查點資料；缺失資訊不以 API 專屬配方填補。
- E5/E6/E8：固定輸入覆蓋類別與「要求、實際觀察、關係適用」三層證據；類別不含基準常數、問題編號或 API 專屬觸發組合。

證據：契約檢查、相依邊界檢視、adapter 能力清單及固定驗收規格。此關卡不宣稱真實模型發現能力。

## 階段 2：通用輸入探索與表示變異

### 工作

探索實際位置／關鍵字引數，以及巢狀字典、串列、tuple、namedtuple 中的張量。以 `kwargs["out"]` 等結構路徑識別，不固定選某個位置引數。保留重複物件參照並記錄儲存別名群組。稀疏元件由布局 adapter 處理，不將結構索引視為一般數值變異。

重用已支援的 stride、offset、clone／實體化、negative bit、conjugate bit、singleton stride 及保值展開操作。資格由張量中繼資料與關係義務決定。第一版認證稠密跨步轉換；其他布局操作須先通過自己的合法性、別名、梯度及重播檢查。能辨識布局，不等於支援其變異。

明確處理讀寫角色、別名保留、autograd 連接及參數／optimizer 身分。不對既存計算圖無條件 detach；無法維持必要關係即拒絕轉換。未知寫入或別名行為不能取得等價性判定。

Eager 在實際目標呼叫邊界檢查運算元。編譯／capture 路徑需在 tracing／capture 外建立受支援的準備邊界，並驗證實際執行目標消耗了變異輸入。量測若改變圖執行、造成 graph break，或只看到 tracing，不能建立執行模式覆蓋。

### 依評估補強的工作（E3、E6、E7）

將盲目輪替改為對所有已發現運算元路徑及支援別名群組，先過濾資格再確定性列舉。實數不套 conjugate、整數不套 negative，缺 singleton 軸不算已到達目標的變異。保存被拒候選與原因，但對已知前提不成立者不浪費目標 worker 啟動；套用時仍須再次驗證，因靜態資格只是暫定。

需要時向輸入生成器要求缺少的通用前提，例如合法 singleton 維度、可展開的重複值或支援布局。由 LLM 建構新的 canonical 輸入，MR 兩側皆從此輸入開始；mutator 不能只替其中一側新增批次軸或改值。此覆蓋請求與正確性修復回饋分開。

宣稱完整十案 G6 通過前，加入並驗證可重用的 PyTorch MKLDNN 邏輯實體化及 CSR values 儲存變換 adapter。以相同路徑機制探索巢狀／關鍵字可寫輸出與相關唯讀運算元，保留 CSR 索引結構及別名義務。布局資格來自後端能力與已驗證語義契約，不使用 API 名稱對布局表。跨輸入組合依合格候選在宣告組合預算內列舉，不排入基準專屬布局／運算元組合。

### G2 驗收

- 找出所有宣告支援容器中的張量，包括關鍵字與巢狀輸入；重建保留容器結構及重複參照。
- 相同運算元描述與義務，無論 API 身分皆產生相同候選；代表 API 重用同一變異程式，不新增名稱分支。
- 每個支援操作都有關係保留成功案例與前提失敗拒絕案例；驗證邏輯值、shape/dtype/device 義務、別名與所需梯度連接。
- 不安全參數替換、別名遺失、非法重疊寫入及不支援布局，須拒絕或標記未解決，不能計為有效變異。
- 分開保存計畫、合格、已套用與已觀察狀態；只有已驗證到達才計入覆蓋。
- E3/E7：資格矩陣涵蓋實數／整數／複數、缺 singleton 軸、合法／非法展開及位置／關鍵字／巢狀運算元；不可能操作先過濾並保存拒絕證據。
- E6：adapter 測試涵蓋 MKLDNN 邏輯值與含可寫輸出的 CSR values，包含未驗證別名／副作用的拒絕案例。看到或成功序列化布局，不等於目標已消耗該布局。
- E6/E7：在宣告預算內展示通用多運算元探索；新生成輸入須一致供給 MR 兩側。

證據：運算元探索紀錄、變異計畫、前後中繼資料與數值、關係檢查及拒絕理由。缺可靠證據的 compile／capture 注入明示不支援，不退回 eager 冒充成功。

## 階段 3：LLM 生成訓練轉移

### 工作

先生成並驗證 canonical program，再要求 LLM 生成完整合法轉移程式，必要時產生 MR 相容程式對。輸入包含公開 API 規格、程式契約、抽象狀態目標、關係需求、預算及既有程式上下文，不提供問題編號、MRE、已知觸發序列或預期答案。

Forward／backward、梯度清除與累積、optimizer 操作、上下文、checkpoint，以及候選使用的 compile／capture／replay，其位置和順序均由 LLM 決定。Runner 不因 `depth` 或 `microbatches` 欄位自動建立隱藏生命週期。共同語法與 probe 介面不指定單一轉移序列。

驗證支援的匯入別名，盡可能沿用 source-only JSON 外層。以有界回饋修復語法、契約、已確認非法操作及已驗證缺少的路徑／狀態。未知執行例外與疑似正確性失敗，不進入「修到 Pass」迴圈。保存每次嘗試及操作來源。

### 依評估補強的工作（E1、E2、E3、E4、E5、E8）

將下列版本化 edge-case 補充加入 canonical／輸入生成，由通用覆蓋設定選擇，而非 API 身分。平衡普通案例與有限量級極端值、重複／非均勻值、具意義的對稱／非對稱結構，以及合法 singleton／batch／broadcast 形狀。透過能力資訊要求支援的稀疏／不透明布局及可寫輸出。不能只產生平凡輸入：對稱矩陣、相同 microbatch 或恆等計算可能合法地掩蓋差異。不得要求案例必須失敗，也不因合法 Pass 缺乏辨識力而拒絕它。

解析一般匯入別名及模組基底類別，同時驗證初始化與副作用。解析 JSON 外層中的 Python 原始碼，在昂貴目標工作前檢查必要入口與檢查點資料結構；區域輔助函式可留在程式，但不能任意序列化為資料。中繼資料足夠時，呼叫前檢查方法引數綁定。新契約不強制舊版頂層 forward，除非該程式介面確實需要。

建構關係時說明必要觀測值，不給操作序列。例如別名關係需要兩個可觀察的相關張量；未暴露的區域 view 不滿足條件。如何建構與使用由 LLM 決定。僅在上限內修復已診斷契約／合法性／路徑問題；數值參考失敗或未解釋目標例外，不用來要求模型簡化掉疑似缺陷。

**擬議通用 edge-case 提示詞補充（未用於歷史實驗）：**

```text
Consider edge cases from the assigned coverage category as well as ordinary
inputs across the run. Choose legal, explicitly initialized inputs using the
supplied API specification and backend capabilities.
For numerical coverage, consider exact zeros, repeated and nonuniform values,
mixed signs where legal, and small nonzero or large finite magnitudes relative
to the dtype's representable range. Derive scales from dtype metadata rather
than assuming one fixed floating-point format. Record whether conversion or
execution changes the intended magnitude category.
For structural coverage, consider legal singleton dimensions, batch dimensions,
broadcast-compatible unequal shapes, and alternative supported layouts. Use
empty dimensions only when permitted by both the API and the selected relation.
When relationships between operands or returned values matter, expose the
observables required by that relation. Include nontrivial, distinguishable
values where legal; do not assume that every edge case should fail.
Select a bounded subset for this program rather than combining every edge case.
Preserve required domains, initial conditions, differentiation coordinates and
source/follow-up input relations. If a category is not applicable, report that
limitation instead of forcing an invalid input or inventing an expected answer.
You still own the training-state operations and their ordering. The framework
may mutate tensor representations but will not construct your training history.
```

此補充是擬議設計，不是生成改善證據。不提供問題編號、精確失敗數值或已知生命週期序列。請求採固定類別排程；模型對類別到達的說明，在 Inspector 驗證前不具證明力。

### G3 驗收

- 相同契約能執行不同合法轉移結構，不為每種結構新增執行模板。
- 從程式移除 backward、上下文進入、optimizer 更新等必要操作，執行也缺少該操作；runner 不補做。
- 假 provider 與手寫契約測試驗證別名處理、有界修復、錯誤回傳、缺少目標路徑及未知例外保留。
- 保存確切提示詞、原始碼、設定、修復歷史及操作來源；生成有效性與狀態到達分開。
- E1/E2/E3/E4：回歸資料涵蓋嚴格 JSON 內非法 Python、合法匯入別名、缺方法接收者／運算元、隱含隨機初始化、不支援檢查點資料與真正缺少的入口。修正第一個診斷不能自動認證剩餘程式。
- E5/E8：提示詞測試核對通用補充、類別排程及關係義務，不含基準專屬內容。實際 edge-case 到達與提示效益在 G4/G6 驗證，不從文字推定。
- E4/E8：缺可觀察伙伴的別名關係仍不適用；runner 不製造伙伴，也不插入狀態轉移序列。

證據：原始碼與追蹤對應、契約／修復測試。真實 LLM 轉移多樣性待 inspector 完成後於 G6 量測；假 provider 不算 LLM 生成結果。

## 階段 4：Inspector 與經驗證狀態覆蓋

### 工作

在語義檢查點及實際 API 事件收集後端觀測，區分目標進入與正常完成，記錄梯度路徑、相關參數與 optimizer 狀態、上下文進出狀態，以及支援的圖執行證據。登記待觀測資源，但不接受生成宣告作為狀態證明。

驗證要求的狀態述詞並摘要實際轉移。分開保存正規化原始碼指紋與執行轉移簽章；只改常數或名稱不建立不同訓練狀態路徑。比較檢查點語義，不比較原始操作次數。

固定轉移簽章的抽象欄位與有限候選集合分母。只計有證據的轉移，不支援觀測及部分執行另列。

### 依評估補強的工作（E5、E6、E7、E8、E12）

在目標檢查點量測實際輸入覆蓋：dtype／布局、運算元路徑與角色證據、形狀／singleton／broadcast 事實、支援的別名關係，以及零／非零數、重複性、符號分布與依 dtype 定義的有限量級區間。辨識原本要求的極小非零值是否已捨入為零，或大值是否成為非有限值。保留判定所需確切資料，不只保存摘要區間。

記錄候選經過的解析、靜態檢查、初始化／快照、變異資格／套用、目標進入／返回、必要反向傳播、狀態到達、MR 適用性及獨立審核。第一個阻擋診斷作互斥漏斗類別，補充診斷另存，避免把多個錯誤當多個候選，也避免把所有 worker 失敗都說成未進入目標。

要求的輸入多樣性、觀察到的輸入多樣性與已驗證訓練轉移多樣性分開。對稱或相同輸入可能合法 Pass；到達 negative-bit、編譯或累積狀態，不表示一定能觀察到差異。

### G4 驗收

- 假檢查點標籤、缺目標呼叫、不相關梯度或缺狀態轉移，不能取得完整覆蓋。
- 設定為 compiled／captured 不足以證明實際執行，只有 tracing 不能通過。
- 名稱／常數不同但抽象執行路徑相同的程式，不增加轉移多樣性；真正不同且已驗證的路徑能區分。
- 證據不足、目標述詞未達、非法程式與判定違反各有不同診斷。
- E5：案例區分要求的 tiny 值與實際可表示的非零輸入；另記錄輸出下溢，不自動判為缺陷。
- E6/E8：布局與別名覆蓋須有實際目標觀測；單一回傳 view 不滿足兩個觀測值的別名關係。
- E12：可由事件重建完整階段漏斗，不重複計入修復候選、API 進入、完成執行或次要診斷。

證據：追蹤、狀態快照、述詞評估、轉移簽章及刻意缺少證據的對照案例。

## 階段 5：可重用語義類別判定器

### 工作

依語義能力與明確認證選擇關係，不按目標名稱分支。初始關係類別涵蓋邏輯值表示等價、可微分計算、經認證的分解／累積、持久狀態還原、上下文還原、執行模式等價及可觀察別名／副作用。代數參考只對有一致定義關係的語義類別加入，不逐 API 增加公式。

某能力組合尚未實作認證或觀測時，對應類別保持不可用。所有宣告支援關係須有經審查的前提、輸入關係、允許差異、語義檢查點、觀測值、容差及非確定性政策。動態觀測可確認可量測條件，但無法單獨證明任意程式的純度、可微分性或等價性。

在條件成立時重用既有可靠比較與有限範圍公式。僅調整必要接點：新判定器消費 LLM 轉移的執行證據，不暗中建構受測流程。梯度比較保留宣告微分座標，不 detach 後換成另一問題。方向有限差分須檢查定義域、平滑性、步長及數值穩定性，模糊結果保持未評估。

### 依評估補強的工作（E8、E9、E10）

凍結前，以普通、近零、極端尺度及條件不良輸入驗證數值政策。各語義關係採尺度相符比較及有依據的參考精度；不單純加大絕對容差壓掉對照不一致，也不讓固定絕對容差抹去極小預期值。參考不一致、非有限觀測及殘差／條件數認證失敗，與目標違反分開。輸入覆蓋若超出關係定義域仍可保留，不強套梯度或殘差判定。

獨立審核依支援語義契約擴充，不依問題身分。例如合法的廣播逐元素關係、對已認證基本資料使用的有界純量 min/max，以及能分析副作用的上下文建構函式。任意多載呼叫與未知副作用仍不支援。預期值須獨立於正式預期輸出重建；共用輸入描述不代表可以複製判定答案。

加入無辨識力的對照，例如關係允許的對稱輸入與相同 microbatch；觀測一致時須維持 Pass。多個相關觀測值是適用性要求，不是製造缺少觀測值的許可。

### G5 驗收

- 每個宣告支援關係均通過已知有效案例、拒絕前提／輸入關係不成立案例，並檢出刻意破壞的觀測值。人工破壞只驗證判定器，不報為 PyTorch 缺陷。
- 對適用多個 API 的關係展示跨 API 重用，不增加 API 專屬判定分支；證據不足的關係明示不可用。
- 非有限或不穩定參考、檢查點不對齊、副作用不確定及對照組不一致，不能成為錯誤的已確認缺陷。
- 沒有 oracle 時仍保留定義清楚的僅覆蓋執行；所有自動失敗在獨立佐證前都只是疑似失敗。
- E8/E9：普通與無辨識力對照保持有效；近門檻對照不一致、殘差／條件數失敗依情況保持未評估或不適用。人工符號／尺度錯誤僅在數值關係有依據時檢出。
- E10：審核測試涵蓋支援的廣播形狀、經認證純輔助呼叫與刻意不支援變體；報告支援度及未知理由，未知數下降本身不證明可靠。
- E9/E10：獨立標籤不來自正式 Pass／Suspected Failure；容差修改不能只以提高基準檢出或減少拒絕為理由。

證據：關係／能力矩陣、正負對照、人工錯誤檢查、適用性理由及數值政策驗證。

## 階段 6：整合、真實模型多樣性與凍結

### 工作

透過新流程接通生成、變異、執行、檢查、判定選擇、證據保存及重播。重播由支援的初始狀態執行保存原始碼及變異計畫，不虛構任意活躍 autograd 圖的序列化，也不呼叫 LLM。不支援的狀態還原明確報告。

十個已納入基準均透過新 worker 重播，轉移操作明載於來源程式；舊 handler 不得暗中補做。逐案記錄新路徑檢出、失敗、不支援或未評估。目標仍為全部十案檢出；舊 10/10 不證明新實作，不能以放寬關係掩蓋剩餘缺口。

進行真實模型開發試驗，確認 LLM 產生多種合法轉移。四個初始狀態維度各選一個已宣告的非平凡述詞，要求至少兩個不同、經執行驗證的轉移簽章到達該述詞。這是工程驗收目標，不是母群多樣性估計。若能力不足或模型未達標，關卡保持未完成並分析原因。

預先宣告開發 API 集與另外的保留 API 集，兩者取自支援語義能力。凍結後測試保留 API，不增加名稱分支。若依其結果修改程式或提示詞，該集合改列開發證據，重新凍結時另定新的保留集。

### 依評估補強的工作（E1–E12）

試驗前建立 E1–E12 解方驗收矩陣，以通用合成輸入及保留 API 範例驗證；精確基準重現常數仍限於基準案例。涵蓋數值極端、singleton／broadcast 形狀、支援布局與可寫輸出、可觀察別名及非平凡狀態序列。量測要求類別轉為實際類別的比例與關係適用性，不只量測原始碼有效率。

另做預先宣告的開發比較：基礎提示詞，對照相同提示詞加通用 edge-case 補充。引擎、模型、API 集、類別要求與各組名義預算相同；保存提示詞版本、順序、實際耗時、token 用量及快取／服務條件。不同生成程式是不同案例，不是重複基準重現。單次比較只作描述，無信賴區間或改善保證。若 edge 類別仍缺失，明示缺口，不插入已知問題的精確輸入。

正式凍結前，在開發階段驗證獨立審核及歸屬：已記載的同機制變體不被過窄比對器排除，同 API 的不同異常也不算原問題。此邏輯不進入生成器。最終凍結協定明列採納的提示詞修改、輸入類別預算、數值政策、審核支援與歸屬版本。

### G6 驗收

- 整合執行與相關既有回歸檢查通過；確定性保存程式重播的 provider 呼叫為零，且必要證據一致。
- 十案結果矩陣完整。宣稱全部支援須有新路徑 10/10 檢出與一致對照；不完整矩陣不能當作完整整合。預定完整十案實驗前須解決剩餘能力缺口。
- 真實模型試驗達到各維度轉移多樣性目標，不依賴 runner 隱藏生成的操作；記錄生成、修復、狀態到達及適用性分母。
- 保留集展示無新增 API 專屬 mutation／oracle 分支的重用；不支援案例明示，不計為已評估成功。
- 凍結原始碼雜湊、提示詞、能力認證、政策、模型／環境設定、API 集、預算及歸屬規則。開發失敗保留，但不納入凍結實驗樣本。
- E1–E12：每項解方都有證據或明確未解決紀錄。宣告能力須通過其測試；完成清單不代表所有舊缺點已解決。
- E5/E6/E8：真實模型試驗對驗收集中每個已宣告適用的 edge 類別族，記錄至少一個合法且目標實際觀察到的案例。缺可靠數值判定者可算觀察覆蓋，不算正確性 Pass。缺少承諾類別時，其生成驗收保持未完成，不用手寫基準輸入補指標。
- E5：基礎與 edge 提示比較報告所有分母，包括失敗、修復與未評估；保留未改善結果。其他條件改變時，不將差異歸因提示詞。
- E10/E11：審核與歸屬對照包含已支援同機制變體及無關異常，使用獨立標籤及版本化規則。

證據：新路徑整合矩陣、試驗提示詞／程式／追蹤、多樣性紀錄、保留集結果、重播校驗值、回歸日誌及凍結協定。失敗不能成為加入基準專屬後備流程的理由。

## 階段 7：凍結實驗與獨立評估

### 工作

新 LLM 實驗對每個目標 API 配置名義 300 秒的生成／執行預算（依使用者授權更換 partition 後，在正式生成前恢復，見預算修訂歷程），包含生成、修復、探索執行、變異衍生案例、MR 兩側及迴圈內驗證。中繼資料取得、服務啟動與後續獨立審核另行計時；報告派送保留及實際耗時。不把相同的名義 API 預算描述成實際生成工作量完全相等。

使用明確初始化值與確定性觸發。每個設定案例執行一次；來源／後續程式及不同變異是不同執行案例，不是統計重複。生成程式數與衍生執行數分開計算，不為增加表面信心而重跑確定性案例。

問題歸屬在執行後判定，不將特定問題資訊回饋生成。以獨立證據區分基準重現與同 API 的其他異常，明示未裁定案例與誤報分母。舊引導式實驗只作歷史對照，除非證明條件匹配，不能視為受控因果基準。

### 依評估補強的工作（E5、E6、E10、E11、E12）

除狀態與轉移覆蓋外，公布輸入類別覆蓋及逐階段失敗漏斗。分開計算生成原始碼候選、修復回應、唯一原始碼指紋、canonical 輸入、mutation 衍生執行、完成目標／反向路徑、適用比較與獨立裁定案例。重複原始碼不增加轉移多樣性，但不同合法輸入仍可增加輸入覆蓋。

保留凍結自動歸屬結果。後續獨立證據若改變問題歸屬，另出版本化審查結果與理由，不覆寫原始結果，也不回饋已完成生成。新異常與已知問題重現分列，所有審核未知保留；FPR 僅適用於獨立裁定負例。

評估 edge 引導是否產生要求輸入，以及輸入是否進入可用判定器。區分生成遺漏、變異不可行、不支援布局、狀態未達、關係不適用、參考不確定、獨立證明的判定器漏報及比對器漏配。普通稠密值上的 API Pass，不表示稀疏、不透明布局或極端尺度問題已修復。

### G7 驗收

- 每個目標都有完整結果，或明確的基礎設施／預算失敗紀錄。低發現率是有效實驗結果，不是修改凍結協定的理由。
- 以明確分母報告有效／可執行率、狀態到達、已驗證轉移多樣性、變異合格／套用／到達、MR 適用性、已知問題重現、審核偵測率／FPR、首次偵測時間、生成數與成本。
- 區分合法程式未生成、狀態未達、變異限制、MR 不適用、經獨立正例證明的判定器漏報，以及問題歸屬漏配。
- 保留原始請求／回應、程式、支援範圍內的確切初始資料、變異計畫、追蹤、觀測、環境、雜湊及審核結果，並確認服務／程序／配置清理。
- E5/E6/E12：結果包含要求／實際／關係適用的 edge 覆蓋與完整執行漏斗；類別及分母符合凍結協定。
- E10/E11：自動與審查歸屬可分別重現，未知不視為負例，同 API 的無關異常不增加基準重現率。

證據：新的英文評估報告與繁體中文對照版，以及完整不可變執行產物。凍結實驗不設定最低問題重現百分比。

## 評估問題與解方追溯

下列問題來自[既有逐案評估](pytorch-correctness-evaluation.md)。E1–E12 是規劃需求編號，不是新發現缺陷。精確問題輸入仍屬基準證據；解方採通用契約與語義類別。歷史觀察支持改善動機，不證明新提示詞或實作必然改善結果。

| 編號 | 已觀察限制與證據 | 負責階段及擬議解方 | 驗收證據 |
| --- | --- | --- | --- |
| E1 | 大量 manual_seed 拒絕、已安裝文件含 RNG 範例，模組隱含初始化也需審查（B1–B5、B7–B10、C）。 | 1/3/6：明確確定性政策、保留原文件加標註、區分種子／取樣診斷、初始化檢查。 | 契約測試與真實模型拒絕／有效率；未做對照不宣稱文件範例造成該行為。 |
| E2 | CUDA graph 的 161 個候選全遭靜態拒絕，131 個先被模組基底拼寫檢查擋下，30 個語法錯誤（B5）。 | 3/6：解析匯入別名、JSON 內 Python 語法，以及後續初始化／副作用檢查。 | 接受合法別名，仍診斷語法及確定性問題，不假定修一處就全部有效。 |
| E3 | 18 個 scatter 失敗缺接收者，20 個 copy 失敗缺 other（B1/B10）。 | 1/2/3：簽章／schema 綁定，探索接收者、位置、關鍵字及巢狀運算元路徑。 | 綁定／未綁定方法與缺引數測試；無中繼資料時保持未解決。 |
| E4 | Compiled-alias 候選回傳不可 pickle 區域函式、缺必要入口或 build 型別錯誤（B6）。 | 1/3/6：區分執行原始碼與檢查點資料，提早驗證新契約。 | 快照往返、錯誤資料對照及有界修復，不強制舊生命週期模板。 |
| E5 | 普通量級未達極小尺度；一般 LU 形狀缺 singleton 批次軸（B3/B9）。 | 1/3/4/6/7：依 dtype 的量級及合法形狀 edge 引導、實際類別檢查與預先宣告提示比較。 | 實際合法目標輸入涵蓋宣告類別，分開要求、建構、目標觀察及判定適用。 |
| E6 | 稠密 BatchNorm/add 未提供 MKLDNN affine-stride 或 CSR 可寫 values 設定，直接位置取樣過窄（B7/B8）。 | 1/2/3/4/6：能力驅動布局／輸入生成、布局 adapter、關鍵字／巢狀探索與有界跨運算元組合。 | 跨 API adapter／運算元測試與目標布局覆蓋，不排 API 專屬組合。 |
| E7 | 96 個完成 worker 在記錄目標完成前就因轉換不適用（A）。 | 2/4/6：資格過濾、拒絕證據及獨立的通用輸入前提請求。 | 設計上避免已知不可能派送，實際結果比例量測後才報告。 |
| E8 | 對稱 negative-bit 矩陣合法 Pass；兩個編譯輸出缺可觀察別名伙伴；相同批次可掩蓋累積差異（B2/B5/B6）。 | 1/3/4/5/6：辨識值多樣性、明確觀測需求及無辨識力負向對照。 | 合法 Pass 保持 Pass，缺伙伴保持不適用；LLM 建構狀態序列與相關輸出。 |
| E9 | 一個 slogdet 因對照不一致未評估，兩個 LU 未過參考殘差檢查（B2/B3）。 | 5/6：尺度／條件數相符參考驗證與預定不確定性政策。 | 保留未評估／不適用，不放寬容差追求檢出，能辨識有依據的人工錯誤。 |
| E10 | 1,154 候選僅審核 69 個，1,085 未知含 9 個疑似案例；廣播形狀及純輔助呼叫屬部分支援缺口（B4/B8/D）。 | 5/6/7：語義類別審核、認證輔助函式、獨立預期及明示支援分母。 | 已知答案與不支援對照，不由正式判定造標籤，也不假設未知為負例。 |
| E11 | 過窄 copy 歸屬原為 2/10，審查後為 3/10；兩個 std 異常非其基準重現（B9/B10）。 | 6/7：機制層級歸屬測試、自動／審查版本分開及獨立佐證。 | 同機制變體與無關異常分列，歸屬不回饋生成。 |
| E12 | 1,154 候選含 661 修復回應與 748 個依 API 區分指紋，失敗分布於不同階段（A 與結果）。 | 4/6/7：互斥階段漏斗、次要診斷另存、程式／輸入／執行數及執行多樣性。 | 由產物重建全部分母，重複文字不增加狀態轉移覆蓋。 |

### 提示比較與範圍約束

擬議提示比較衡量通用補充是否改變實際輸入類別覆蓋與可用測試產出，不以重現特定問題為目標。兩組標記為 `base` 與 `base_plus_edges`，使用相同類別要求，處理差異僅為額外引導文字。開始前宣告相同名義預算、執行順序及資源政策，報告實際耗用、提示 token 額外成本、修復數及未評估結果；不可看結果後暗換較容易類別。

承認設計受到舊基準知識啟發。跨 API 與保留集檢查評估重用，不證明沒有基準影響或模型預訓練污染。指定檢查及必要的真實模型實驗未執行前，不宣稱解方成功。本次規劃更新不改歷史評估資料或已記錄提示詞。

## 預定實作檔案

以下是後續實作的具體規劃範圍，不表示檔案已修改。優先重用既有程式，不重構無關路徑；額外檔案依 AGENTS.md 更新範圍。

| 範圍 | 檔案 |
| --- | --- |
| 契約與目標取樣 | 修改 `src/mirage/contracts/programs.py`、`src/mirage/contracts/exploration.py`；新增 `src/mirage/contracts/capabilities.py`。 |
| 後端邊界 | 新增 `src/mirage/backends/__init__.py`、`src/mirage/backends/base.py`、`src/mirage/backends/pytorch.py`。 |
| 運算元探索與變異 | 新增 `src/mirage/runtime/operands.py`；修改 `src/mirage/runtime/representations.py`。 |
| 生成與修復 | 修改 `src/mirage/generator/program.py`、`src/mirage/generator/repair.py`。 |
| 執行與檢查 | 新增 `src/mirage/runtime/stateful.py`、`src/mirage/inspector/stateful.py`；修改 `src/mirage/runtime/probes.py`、`src/mirage/runtime/worker.py`。 |
| 語義關係 | 新增 `src/mirage/rules/catalog.py`；修改 `src/mirage/rules/conditional.py`、`src/mirage/rules/properties.py` 的必要整合接點。 |
| 流程與產物 | 修改 `src/mirage/workflow.py`、`src/mirage/cli.py`、`src/mirage/artifacts/report.py`、`src/mirage/artifacts/bundle.py`。 |
| 測試 | 新增 `tests/test_capabilities.py`、`tests/test_operands.py`、`tests/test_stateful.py`、`tests/test_relation_catalog.py`；修改相關 `tests/test_program.py`、`tests/test_exploration.py`、`tests/test_properties.py`、`tests/test_cli.py`、`tests/test_artifacts.py`、`tests/test_pytorch_benchmark.py`。 |
| 驗證與實驗整合 | 修改 `benchmarks/pytorch_correctness/evaluate.py`、`sbatch/experiments/pytorch_correctness.sbatch`；新增 `sbatch/validation/stateful_generation.sbatch`。 |
| 文件 | 本路線圖及英文版；未來的 `docs/stateful-generation-evaluation.md`、`reports/stateful-generation-evaluation.md`。 |

不規劃新增相依套件、安裝 Python 或修改專案設定。TensorFlow 實作、分散式執行、任意活躍圖序列化及普遍數值認證不屬第一版範圍；不能由後端介面或開放 API 集暗示已支援。

## 執行、證據與完成政策

使用已準備且計算節點可存取的環境，所有 PyTorch 運算、編譯、真實模型驗證及整合套件均交由 Slurm。遵守 [sbatch/README.md](../sbatch/README.md) 的帳號、分區、腳本位置、資源檢查及日誌規則。靜態檢查與提交預檢不算執行驗證。開發需要重複模型驗證時，在本任務內重用健康 vLLM 服務，以明確資源分開推論與測試步驟，任務結束確認清理。

證據存於 `result/stateful-generation/<run-id>/stage-<number>/`，各次嘗試保留原始碼／設定快照。每個關卡紀錄包含實作雜湊、命令、環境、實際通過／失敗／不支援數、失敗案例及產物路徑。工作進行時，以實際結果更新雙語路線圖；不為符合時程而將失敗關卡標為完成。

## 本次規劃更新紀錄

- 建立分階段英文路線圖與對應繁體中文報告。
- 實作階段 1–7 均為規劃中；本次未實作原始碼、執行新 PyTorch 測試、提交 Slurm 工作或進行 LLM 實驗。
- 文件驗證通過：雙語版本包含相同七個關卡及標題數，實作／產物路徑一致，本機連結有效且無行尾空白；英文文件無中文文字。此結果僅驗證規劃文件，不表示任何實作關卡已通過。

## 依評估結果補強的規劃更新

使用者要求將先前 PyTorch 評估中已觀察限制的解方納入階段，包括通用邊界案例提示。本次僅更新本路線圖及英文對照版：檢視失敗原因、對應階段工作與驗收、加入擬議提示詞補充，再核對雙語完整性及連結。歷史結果與提示詞不變；解方仍是規劃，不表示已實作或量測到改善。

本次規劃修訂已完成：將十二項評估問題對應至七個階段的工作與驗收，加入完整的擬議 edge-case 提示詞及開發期比較。輕量檢查通過：階段與問題數一致、兩版本提示詞完全相同、檔案路徑清單一致、十個本地連結及適用錨點有效、英文文件無中文、程式碼圍欄成對且無空白格式錯誤。SHA-256 檢查確認兩份歷史評估均未變更。本次未實作、執行 PyTorch、提交 Slurm 工作或執行模型實驗；G1–G7 仍為規劃狀態。

## 執行授權與進度

使用者已授權依預定檔案範圍實作並依序執行階段 1–7；每階段通過驗收後自動接續下一階段。保留失敗紀錄並修復後再推進，不以程式修改推定驗收通過。目前階段 1 進行中：檢查既有契約、實作框架中立的能力／證據紀錄與後端邊界，驗證契約行為後再接續。核准的驗證順序仍為輕量靜態檢查、Slurm 執行期／整合驗證、真實模型開發試驗及凍結後正式實驗。

### G1 驗收紀錄

階段 1 基礎契約驗收通過：Slurm 工作 `769979` 的 9/9 測試通過，涵蓋假後端證據流程、未知／僅 LLM 宣告不算證據、關係檢查點對齊、方法綁定、檢查點資料拒絕、種子／取樣區分、契約版本與確定性類別列舉。原始碼雜湊與不可變測試快照位於 `result/validation/769979/`。靜態編譯與批次腳本語法亦通過。目前僅宣告 tensor 描述與稠密儲存身分能力；變異、執行與數值認證尚未宣告。初始候選集為四個狀態維度、十個輸入類別；變異先從單項開始，G2 驗證最多兩運算元組合。數值比較在 G5 驗證具名政策前保持不可用。自動接續階段 2。

### G2 驗收紀錄

階段 2 在工作 `770003` 通過：23 項契約／變異測試全數通過。先前工作 `770001` 的 22 項通過後，審查補上 CSR values 別名偵測與重疊寫入拒絕，再以最後工作驗證。涵蓋巢狀／namedtuple／關鍵字運算元、重複參照、不同共享 view 拒絕、空配置、實數／整數／複數資格、套用時重驗、兩運算元列舉、兩個不同 API 的梯度連線、CSR 可寫 values 與 MKLDNN 邏輯實體化。這是變異元件驗證，尚不是生成程式的目標執行或 G6 基準整合證據。不同共享 view、可微 CSR 重建、未知寫入副作用及 capture／compile 內變異仍不支援。自動接續階段 3。

### G3 驗收紀錄

階段 3 在工作 `770038` 通過：累計 32 項測試全數通過。工作 `770018` 先發現證據 enum 無法以既有 weights-only 模式重載；最終實作改存基本型別紀錄並通過 artifact 重載。人工／假 provider 測例驗證保留圖與重新計算兩種歷史得到相同累積梯度、移除 backward 後 runner 不補操作、方法綁定、實際呼叫邊界變異、合法匯入別名、JSON／程式契約、基礎／edge 提示差異，以及未知例外不自動修復。這些驗證僅建立生成／執行介面正確性，不代表 LLM 發現能力或真實模型多樣性。編譯／capture 執行認證與數值判定尚不可用。自動接續階段 4。

### G4 驗收紀錄

階段 4 元件驗收在工作 `770050` 通過（39/39）。工作 `770048` 先通過 38 項，但審查發現觀測器的 detach 操作混入轉移指紋，因此最終增加測試確認額外觀測不能增加訓練轉移覆蓋。檢查點事實由 inspector 擷取，生成字典不能宣稱狀態已達成；重複檢查點名稱視為歧義，未觀測編譯執行保持未知，轉型為零的極小值不算極小非零覆蓋。目標輸入中繼資料及輸出別名群組與數值快照分開記錄。這些驗證確認目前 eager 證據路徑；G6 前仍須實作編譯／capture 狀態觀測。自動接續階段 5。

### G5 驗收紀錄

階段 5 目前支援的 oracle 元件驗收在工作 `770087` 通過（53/53）。目錄宣告七個語義類別及明確前提；宣告不表示認證已實作。表示等價比較僅對已審查的原始碼／副作用／數值子集啟用，其餘類別組合在證據 adapter 完成前仍不適用。對照涵蓋實際跨 API 配對、人工符號錯誤、極小／極大有限值、非有限值未知、病態矩陣拒絕、基本 min/max 與多載 helper 區分，以及獨立廣播算術審核；整數溢位審核範圍保持未知。工作 `770084` 發現執行紀錄共用可變設定，`770086` 驗證深複製修正（52/52），`770087` 包含最終 helper 對照。人工錯誤不是 PyTorch 缺陷。自動開始階段 6 整合與能力補全；十案例與真實模型驗收尚未通過。

### G6 開發試驗協定（模型執行前宣告）

新路徑固定案例在工作 `770759` 偵測 10/10，工作 `770811` 獨立審核全部參考與控制一致，未重跑目標程式。工作 `770911` 另驗證零 provider 重播及整合案例／審核路徑。這些是人工固定案例結果，不是 LLM 生成結果；後續執行期修改須在最終凍結前驗證。

第一輪真實模型開發試驗使用既有 Qwen/Qwen3-Coder-Next-FP8 服務、temperature 0 與既有服務設定。開發狀態目標：`torch.sin` 觀測兩次 backward、`torch.no_grad` 完成兩個 restoration scope（僅前向目標）、`torch.square` 兩次連到目標的反向事件、`torch.cos` 實際執行編譯圖。各配置最多 12 個程式／修復回應及 180 秒生成與執行預算。第一個合法程式作 canonical，後續 transition 請求提供其原始碼、狀態謂詞與已觀測轉移指紋，不提供指定操作序列；各目標須取得兩種不同且經驗證的指紋。

另以 `torch.sin` 與 `torch.add` 做描述性的提示比較，各 API 固定先 base、再 base-plus-edges，每組 120 秒、最多 12 候選。順序固定並記錄，單次比較受快取／服務條件影響，不能作因果推論。類別要求與模型／引擎設定相同。初始模型生成保留集為 `torch.sigmoid` 與 `torch.Tensor.mul`，各 120 秒、最多 8 候選，待開發驗收通過後才執行；若依結果修改，將該集合轉為開發資料並宣告新保留集。輸入覆蓋分開記錄生成 source 與變異 follow-up；布局見證可由 mutation 引擎產生。要求、觀測與 oracle 適用類別仍分開。

最終十 API 實驗各保留 300 秒，須待 G6 的狀態多樣性、edge 類別見證、保留集重用、獨立固定案例控制、回歸檢查與引擎／提示凍結通過後才開始。缺少見證時記錄並分析，不將固定問題輸入塞入生成器補覆蓋。

### G6 整合與第一輪開發測試

目前使用兩張 GPU 的共用 Slurm allocation `770948`：一張提供既有模型服務，另一張分配給附加的測試 step。第一次完整回歸 `iteration-sQhMpu` 執行 141 項測試，其中一項因測試用 provider 的子程序無法在 unittest discovery 下匯入 `test_stateful` 而失敗。在該測試中明確提供模組路徑後，`iteration-yVd22Q` 執行 142 項測試：141 項通過，一項因未提供保存案例的 audit 路徑而跳過。此跳過項目不能取代另行執行的十案例審核；模型生成也尚未被當作回歸證據。

Pilot `stage-6/pilot-01` 使用不可變更的程式快照。早期觀察到的問題包含使用不可用的 NumPy、以 `probe.call` 呼叫非目標操作、目標輸出未連接 backward、Python 語法錯誤、使用保留 checkpoint 名稱，以及返回無法序列化的 optimizer。這些均保留為開發失敗。下一輪通用 prompt 說明會釐清 probe 僅供目標呼叫、目標結果的可微連接、直接返回 tensor 觀察值，以及 optimizer 並非必填；不提供具體訓練序列。通用輸入類別定義也明列既有的 dtype 相對量級門檻。上述變更必須以新一輪 pilot 驗證，不能修改正在執行的快照或其結果。

State-goal 排程現在根據 canonical 的執行能力證據，而非 API 名稱選擇：觀察到目標 backward 後，可要求梯度、目標連接圖生命週期及編譯執行狀態；觀察到 context exit 後，可要求 restoration 狀態。LLM 自行規劃操作，inspector 驗證是否達成。Context-manager callable 的能力資訊可以移除不適用的 backward 要求，但不會代為建立 scope。Context instance identity 與巢狀深度也會保存，供之後獨立歸因使用。

### G6 第二輪診斷與多樣性指標修正

`pilot-01` 在 sin、context、square 三項沒有 runtime-valid 程式；compiled 有一項合法 canonical，但四項皆沒有達成指定狀態的轉移。`pilot-02` 每項取得 12 份回覆：sin 有 6 項 runtime-valid，其中 5 項達成狀態；context 與 square 仍為 0；compiled 有 2 項 runtime-valid，其中 1 項達成狀態。保存來源的重播比對成功，provider 呼叫數為 0。這些結果尚不足以通過 G6。

審查發現 `pilot-02` 的 sin 原始 signature 計數為 4，但排除 observation detach 拷貝後只剩一種核心流程，因此不能據此宣告多樣性。新的 signature 保留原始事件供查核，但在多樣性計數中排除所有 detach/detach_，並透過 optimizer 的公開 step hook 與實際 zero_grad 返回事件觀察更新及清除。這些 observer 不會執行訓練操作。

下一輪 `pilot-03` 改用較精簡的 `stateful-2` prompt：開頭明示目標 API、backward 要求與精確 state predicate；要求 minimal canonical；釐清 module parameter 物件 identity、僅目標使用 probe.call，以及直接返回 tensor 觀察值。仍由 LLM 自行決定完整 training sequence，沒有加入案例輸入或操作模板。通用正式實驗將交替探索 forward 與 backward 路徑，避免因某輸入的 backward 不適用而排除合法 forward 測試；此排程對所有 API 相同。held-out 與正式實驗仍未啟動。

### G6 下一輪開發協定：增加回覆機會預算

Pilot 04 雖有成功達成狀態的例子，仍未通過多樣性門檻。在 pilot 05 開始前，保留相同四個 API 及核心 predicates，但每個 API 允許 300 秒、最多 24 份回覆。這是增加開發機會預算，並未降低至少兩種已驗證訓練轉移 signature 的要求。held-out API 尚未使用。

語法處理現在只會在 Python 明確報告最後一行單獨的 `}` 不匹配，且移除該行後整份程式可解析時，移除這一個括號；不新增或改寫任何訓練操作。原始回覆、原始程式、正規化程式及處理記錄均保存，原始語法有效數與正規化後計數分別報告。無效的 optimizer/scheduler/scaler 字典與不可攜的 checkpoint helper 會得到明確契約診斷，不再僅記為未定位的 attribute error。Context 比較會拒絕未驗證的同一 instance 直接重入；未使用的 optimizer 建構憑證也不代表其更新已獲認證。

預算派工保留時間改為 10 秒與「目前最長已完成 candidate 時間的 1.5 倍加 5 秒」之較大值。這是依成本派工的策略，不保證基礎設施不會延遲；實際保留時間及已用時間均記錄。來源生成、修復、執行與 MR pair 成本仍納入每 API 的名目預算。上述開發修改將在 prompt 比較、held-out 測試及最後凍結前固定。

### G6 聯合狀態條件的多樣性解方

Pilot 05 的 sin 已達兩種 signature，但 no_grad 的 21 個達成狀態案例仍只有一種序列；單純增加回覆機會未解決此多樣性缺口。Pilot 06 開始前，保留相同核心 predicates、300 秒及最多 24 份回覆，加入確定性且 API 無關的輔助條件：final checkpoint 具有一個或兩個已觀察目標邊界。這是聯合抽象狀態要求，不是操作順序或程式模板；由 LLM 決定如何達成。Sampler 只接收 predicates 及 attempt index，不接收 API 或 issue 識別資訊。

邊界單位明確定義為一次完成的 eager target probe，或一次實際執行且包含目標的 compiler graph 邊界。對 context factory，建構與返回 context 的進出分開計數；graph 邊界不宣稱等同融合後 backend kernel 數。原始事件、達成的 predicates 與 action signatures 均保留，原先每維度至少兩種 signature 的門檻不變。本次是開發協定修改，不會與先前 pilot 作因果比較。

### G6 pilot 驗收、oracle 審查及輸入回饋協定

Pilot 06 在其快照上通過四項核心多樣性檢查：sin／context／square／compiled 分別產生 3／5／2／2 種已驗證 signature，但 G6 尚未完成。最後審查加入針對失敗的 compiled invocation、任意 custom backend、無關 eager 路徑的 backward，以及 implementation identity 觀察值的控制。Compiled-state 證據現在要求支援的目標 graph 成功執行，必要 backward 也必須連到該邊界。Execution-mode 等價還須具有保守的 source certificate；任意 callback、實作 metadata 與資料相依 Python 分支仍不適用。上述變更須在最終 engine 重新驗證。

第一次 base／edge 比較每 arm 有 12 份回覆，未建立 small_finite 或 large_finite witness。例如生成來源把 float32 的 1e-8 稱作 small，但仍高於宣告的極小量級門檻；不能據此宣稱 prompt 改善。通用類別回饋現在把缺少的類別及實測 dtype 門檻提供給 LLM。Refinement 要求重新生成 MR 兩成員共用的合法初始輸入，不改變原 verdict、不提供答案、不由 runner 變動數值，也不 refine Suspected Failure。覆蓋 refinement 有獨立來源註記，沿用最多兩次回覆的界限。類別排程只在全新 request 前進，不因 repair response 前進；各 arm 的實際類別計數分別報告。

在下一次模型呼叫前宣告：pilot 07 保留四項核心／聯合狀態目標，每 API 300 秒、最多 24 份回覆。新描述性比較 `edges-02` 依序對 sin 與 add 執行 base、base-plus-edges，每 arm 300 秒、最多 24 份回覆，採相同 engine 與 coverage-feedback 政策。`edges-01` 保留為先前未通過驗收的比較，不能與 `edges-02` 混算；原 held-out 集合仍未使用。完整回歸與十案例／control 驗證必須通過，才能接受本輪開發結果。

### G6 最終驗收協議：分開檢查狀態與輸入目標

Pilot 07 未通過原有多樣性門檻（sin/context/square/compiled 的已驗證簽章數為 3/2/1/1）。數值覆蓋修訂消耗了狀態多樣性檢查的回應額度。在 pilot 08 之前，保留四組核心與聯合述詞、每 API 300 秒與 24 次回應；僅在此專注於狀態的檢查中停用選用的輸入覆蓋修訂，並以新提案而非修復回應推進聯合狀態排程。獨立的 edges-03 保留覆蓋修訂、相同 sin/add 基礎與 edge 順序，以及每組 300 秒／24 次回應。兩項驗收仍皆為必要；正式評估同時使用狀態抽樣與輸入覆蓋修訂。修復或覆蓋修訂本身不構成多樣性，必須有執行證據。

重播驗收現在除判決外，也要求保存的來源與後續執行之觀察值、狀態／檢查點證據、目標路徑證據與簽章相等；僅判決相同並不足夠。凍結前，須以相同來源快照重新驗證完整回歸套件、十個案例及獨立控制、開發檢查與尚未使用的保留集。先前 pilot/edge 結果保留為開發紀錄，不可滿足最終來源雜湊驗收。

`edges-02` 完成四組比較（各 24 次回應）：sin-base/sin-edges/add-base/add-edges 的執行證據有效數為 24/24/22/23，聯集涵蓋全部十個輸入類別。此結果通過較早快照的 edge 驗收，但僅比較判決的重播不能滿足最終的證據重播門檻。這是描述性開發結果，不能作為 edge prompt 具有因果改善效果的證據。下一輪完整回歸為 `iteration-bNraBG`。

最終引擎的回歸 `iteration-bNraBG` 通過 154 項測試，另有一項依賴案例資料路徑的稽核略過（共 155 項，測試時間 81.172 秒）。隨後 `fixtures-03`／`iteration-oMtHkk` 偵測全部十個人工案例，十個控制也獲獨立佐證；案例驗收通過，牆鐘時間為 58 秒。這是整合與 oracle 證據，不是 LLM 發現率。Pilot 08 與 edges-03 現以相同正式程式來源執行。凍結前將指定保存的案例路徑，單獨執行先前略過的稽核。

Pilot 08 在 `iteration-AdOIV1` 完成並通過：sin/context/square/compiled 分別取得 24/24/24/21 次回應，其中 9/15/7/7 個案例的執行證據有效，已驗證轉移簽章數為 4/4/2/3。Compiled 因時間准入限制停止，並非完成 24 次回應。保存的來源與後續執行之證據重播通過，provider 呼叫數為零。最終引擎的多樣性子項通過驗收。同來源的 edges-03 比較執行中；保留集評估與 G7 尚未開始。

`edges-03`／`iteration-mmlxBP` 在最終來源版本通過：四組皆從 24 次回應取得 24 個執行證據有效的案例，聯集建立全部十個輸入類別的證據，且語義證據重播通過、provider 呼叫數為零。這是達成覆蓋的描述性證據，不是 prompt 因果效果估計。先前略過的保存案例稽核已於 `iteration-pBSKv7` 執行，七項 benchmark 測試全部通過（2.337 秒）。開發檢查完成後，預先指定且尚未使用的保留 API `torch.sigmoid` 與 `torch.Tensor.mul` 現於 `heldout-01` 執行，每組 120 秒、最多八次回應。此保留集檢查後不計畫更改正式程式來源。

### G6 保留集失敗與圖生命週期說明

Heldout-01 未通過重用性驗收：sigmoid 的八次回應只有一個執行證據有效的 canonical，沒有達成目標的轉移；Tensor.mul 則有達成轉移。Sigmoid 產生的 loss 重用了已被消耗的上游計算圖，即使重新計算下游目標輸出也是如此。這是生成程式合法性失敗，不是 oracle 漏報，也不是已確認的框架缺陷。此保留集現在成為開發證據，不會再被視為未見資料。

後續模型呼叫前，加入 stateful-3 的通用 autograd 圖所有權說明：共享上游中間值也共享圖生命週期；由 LLM 選擇合法的保留／重算方式；backward 呼叫數與連到目標的梯度事件不同。不提供可執行序列或 API 專用範例。判決規則與門檻皆保持不變。重新驗證完整回歸、fixtures-04、pilot-09（原四組目標、300 秒／24 次回應）與 edges-04（原四組順序、300 秒／24 次回應）。Sigmoid 僅作為開發檢查重新執行，使用 300 秒／24 次回應。之後使用新指定且尚未使用的保留集 torch.tanh 與 torch.Tensor.sub，同樣各 300 秒／24 次回應。Prompt 與預算調整一起評估，不單獨宣稱任一項的因果效益。最終報告同時保留原保留集失敗與新結果。

Pilot 09 的 context 組未通過原多樣性門檻，只達成一種轉移簽章。早期 canonical 回應包裝了計算，而非呼叫指定的 context factory。通用 backward 說明與此 forward-only 請求無關。在排程中的 edges-04 開始之前，改以既有 requires_backward 旗標決定是否加入圖生命週期段落，不檢查 API 名稱。Forward-only 請求保留先前的完整 prompt／版本；backward 請求保留 stateful-3 說明。Pilot 09 保留為失敗開發紀錄。使用任何新保留 API 前，將以 fixtures-05、完整回歸、pilot-10 與已宣告的 edges-04 協議驗證此來源。改動的是 prompt 適用性，不是狀態述詞、數值規則或多樣性門檻。

條件式說明版本通過 edges-04（`iteration-NUea40`，586 秒）：執行證據有效數為 21/23/24/24，觀察到全部十個類別，且證據重播通過。完整回歸 `iteration-rgSvHY` 的 155 項測試全數通過，沒有略過（測試時間 87.953 秒）。Fixtures-05 的十個偵測與獨立控制皆通過。Pilot-10、已退役 sigmoid 的開發複查與 heldout-02 依序執行，並以機器檢查前置門檻；前置條件失敗時，不會向模型查詢新的保留 API。正式實驗仍未開始。

### G6 使用完整的既定生成時間預算

Pilot 10 的 square 組在 24 次回應上限前只達成一種已驗證簽章，因此即使其他組已達成多樣性，整體門檻仍未通過。在 pilot 11 前，保留目前來源、四組核心／聯合目標、每 API 相同的 300 秒預算與 oracle 條件；將回應上限提高至 100000，使每組都由時間准入而非較早的回應數上限限制。這是統一套用於四個 API 的開發生成機會調整。不合併或改標 pilot 10 為通過。正式程式來源未改動，因此相同來源的 edge、案例與回歸證據仍有效。Sigmoid 複查與新保留集呼叫仍須先通過新 pilot 的實測結果。

### 正式生成前的預算修訂

Slurm 未接受將 job 770948 從四小時延長至四個半小時的嘗試（回傳 `Unspecified error`；實際時限經確認仍為四小時）。新 allocation 的提交預檢估計要到隔日上午才開始；預檢不代表已執行工作。依使用者最初允許必要時調整建議五分鐘預算的授權，在任何正式模型呼叫之前，預先宣告 G7 統一採用**每 API 180 秒**。十個 API、模型設定、狀態／mutation 策略與 oracle 門檻皆不變。Freeze 指令現在記錄明確的 --budget 值（預設 300，必須為正），實驗值必須完全一致。開發檢查保留各自記錄的預算。不可將這批 180 秒結果與歷史 300 秒實驗作因果比較。保留事後證據整理與服務關閉時間；若剩餘時間不足以完成全部協議，則遵循正常 Slurm 排程，不繞過 allocation 限制。

Pilot 11 在 iteration-XKxYJw 以最終引擎通過：sin/context/square/compiled 分別產生 52/56/46/24 個可解析來源，16/23/18/8 個執行證據有效的案例，以及 2/4/6/3 種已驗證轉移簽章。配對兩側的語義重播通過，provider 呼叫為零。相同來源已通過全部 155 項回歸、十個案例／獨立控制與 edges-04。退役 sigmoid 檢查現在執行中，新保留集仍須等待其結果。提高回應上限所建立的是開發能力證據，不是生成率改善的受控估計。

### G6 最終驗收與 G7 啟動

Heldout-02／iteration-ceILJE 在未更改的正式引擎通過：tanh 收到 24 次回應、產生 23 個可解析來源，其中三個案例的執行證據有效，包含兩個達成轉移的案例；Tensor.sub 收到 24 次回應，九個案例的執行證據有效，包含八個達成轉移的案例。證據重播通過。退役 sigmoid 的複查另有 24 次回應、12 個執行證據有效案例及 11 個達成轉移案例；它是開發證據，不是保留集證據。

Freeze iteration-sdzBp9 成功。協議位於 result/stateful-generation/770948/stage-6/frozen-01/protocol.json。回歸 iteration-rgSvHY、fixtures-05、pilot-11、edges-04 與 heldout-02 的全部 43 個正式 Python 來源雜湊一致。凍結 evaluator 雜湊為 84c740ad69bbf9f75dd930475de276e32cf7d865161c7662628e207486b56a0d；正式預算統一為每 API 180 秒。G6 在已記載的能力範圍內通過驗收。G7 隨即於 stage-7/formal-01 啟動。實驗期間不更動凍結來源、prompt、門檻或歸因程式；低重現率也是可接受的實驗結果。

### G6 驗收完成；依使用者授權在 G7 前更換 partition

使用者明確允許更換 partition 以縮短排隊。Heldout-02 通過：tanh 的 24 次回應（23 個可解析來源）包含 3 個執行證據有效案例與 2 個達成轉移案例；Tensor.sub 的 24 次回應包含 9 個有效案例與 8 個達成轉移案例。嚴格重播通過。退役 sigmoid 的開發複查也通過，24 次回應中有 12 個有效案例與 11 個達成轉移案例。Freeze-01 以 180 秒預算通過全部來源雜湊驗收，但**沒有在此協議下執行任何正式生成**。

Dev partition 提供 H200 GPU、最長四小時與較高排程優先層級，提交預檢估計等待時間明顯短於 8gpus。依使用者授權，正式實驗明確改用 dev、兩張 GPU、八個 CPU、90 分鐘 allocation。正常關閉已完成的開發 allocation，驗證新環境／案例後，以**每 API 300 秒**建立新凍結協議，在任何正式模型呼叫前恢復原建議預算。保留 freeze-01 為已被取代且未使用的協議。此資源調整不更改正式 generator、mutation 或 oracle 來源。兩次服務啟動、配置資源時間與清理證據分別記錄。預檢估計不代表工作已提交或執行。

替換工作 771760 以 afterany:770948 相依條件提交至 dev，在清理完成後於 H200 節點 25a-hgpn015 啟動；實際排程沒有等到預檢所估計的時間。開發工作 770948 配置 12,177 秒後以 exit 0 完成。關閉前，服務 metrics 確認 running／waiting 請求皆為零，且已移除測試快取。中斷後的 unified-exec terminal 已無法連線，其閒置且屬於本任務的 step 忽略 TERM，因此被終止。服務／terminal step 的結束狀態為 cancelled（signal 9），各測試 iteration 的退出狀態則另外保留。舊工作所有 step 均已退出。清理證據保存於 cleanup-before-stop.json 與 cleanup-final.json。替換服務正在啟動，尚無正式模型呼叫。

### G6 最終驗收與 G7 啟動

替換服務啟動耗時 185 秒。測試軟體／環境欄位與開發工作相同，僅 hostname、partition 與 job ID 不同；服務欄位僅 host、job ID 與 GPU UUID 不同。Fixtures-06／iteration-t0VrEz 在替換節點偵測全部十個案例，十個控制也獲獨立佐證（74 秒）。Freeze-02／iteration-a0JvIk 通過完整來源雜湊驗收，記錄每 API 300 秒。協議位於 result/stateful-generation/771760/stage-6/frozen-02/protocol.json；舊 180 秒版本未被使用。

G6 驗收完成。G7 現於 result/stateful-generation/771760/stage-7/formal-01 執行十個凍結目標。生成器只取得安裝版本 API 規格、通用設定及自己生成／觀察的歷程。此輪不更改正式來源或 evaluator。獨立歸因仍在執行後進行；低發現率是合法結果，不會因此改變協議。

### G7 報告來源與事後審閱

正式執行位於 allocation 771760 的 `stage-7/formal-01`；生成與 production 評估持續維持凍結。評估報告已補上正式 prompt 原文、僅 backward 使用的圖生命週期說明、context 欄位、檔案／請求雜湊及最終開發驗收數據。歷史 roadmap 003 與其評估沒有修改。

檢查自動歸因程式時發現範圍不一致：編譯目標紀錄保存整個 FX graph 的輸出，但單一目標參考預期的是 API 輸出。Tuple 與 tensor 的差異不足以證明框架缺陷。因此保留 `evaluation-automatic-v1.json`，待十個 API 預算全部完成後，在 Slurm 執行另行保存的 `posthoc_review.py`，產生 `evaluation-reviewed-v2.json` 與 `review-evidence-v2.json`。審閱會將未認證的整張圖輸出排除於單一目標 audit，並對排除範圍保留不確定性；不改動生成程式、production 判決、預算或後續請求。最終指標須使用審閱標籤，並揭露被撤回的判定。

正式生成與版本化審閱完成後，將在 `benchmarks/pytorch_correctness/evaluate.py` 修正相同的輸出範圍檢查，並在已授權的 `tests/test_pytorch_benchmark.py` 加入針對性的回歸案例。驗證未認證的整圖輸出、部分 negative 證據與獨立 positive eager 證據。這是實驗後的 audit 修正；保留凍結 evaluator 快照，並區分其雜湊與修正後檔案。沒有計畫修改生成器或 engine。

## 最終 G7 驗收與交付

正式執行位於 `result/stateful-generation/771760/stage-7/formal-01`，使用已通過驗收的 `stage-6/frozen-02/protocol.json`。Driver 呼叫為 `experiment-stateful --output <formal-01> --gate <frozen-02/protocol.json> --budget 300`；`iteration-UYFdmZ` 保存精確指令與不可變來源，2,825 秒後以 exit 0 結束。十個 API 預算全部完成；180 秒的 frozen-01 協議從未用於正式生成。

### Phase 1：沿用 benchmark 證據

沿用先前收集的十個確定性 upstream issues 與人工最小程式，不宣稱新的 issue 搜尋或新缺陷發現。最終 production engine 在 fixtures-05，以及移至 `dev` 後的 fixtures-06，均通過十個案例／控制檢查。每個 fixture 都產生 production Suspected Failure，且控制得到獨立佐證。這些驗證框架能力，不是生成發現率。既有 roadmap 003、重現報告與歷史評估均保持不變。

### Phase 2：通用實作與驗證

G1–G5 實作來源／狀態契約、後端介面、實際 operand mutation、完整來源 LLM 路徑、runtime 狀態證據與已支援的語義類別 relations。G6 驗證 runtime 多樣性、有效證據中的十種輸入類別、新保留 API 重用，以及配對兩側的精確語義重播。正式執行前的驗收來源通過全部 155 項測試，沒有略過，使用 87.953 秒。最終開發表格與限制記錄於[評估報告](stateful-generation-evaluation.md#最終開發驗收證據)。結果驗證的是已實作子集，不代表實作全部 training-state 維度、所有 PyTorch APIs 或 TensorFlow。

生成器接收已安裝 API specification、完整來源契約、依 runtime 產生的狀態目標、輸入類別與 relation 義務。每個訓練轉移由模型撰寫，mutation 則操作已發現的 tensors。最終 [forward prompt 與 backward 專用說明](stateful-generation-evaluation.md#凍結請求原文範例)、[完整請求](../result/stateful-generation/771760/stage-7/formal-01/203078/attempt-0000/request.json)及凍結雜湊讓此界線可被審查。Issue ID 與 benchmark 程式只用於 benchmark 執行／事後歸因，不進入 production requests。

### Phase 3：實測結果與原因

正式實驗記錄 434 次生成嘗試、433 份取出的來源、432 個靜態有效候選、255 次完成來源執行、177 次完成 follow-up、176 個執行證據有效候選、60 個達成轉移述詞的候選，以及 95 次適用比較。適用比較全部 Pass，沒有 production Suspected Failure。獨立確認的已知問題重現為 **0/10**；Phase 2 已提供案例的結果為 **10/10**，兩者量測不同任務。

審閱後的獨立計數為 TP=0、FP=0、TN=39、FN=0。FPR 的 0/39 只適用於已裁定負例子集；由於沒有符合條件的正例，violation-detection rate 未定義。另有 30 個負例不符合必要狀態／路徑範圍，365 次嘗試保持 audit-unknown。所有 first-detection times 都是 censored。報告逐 API 說明觸發條件、實際證據、生成／狀態／mutation／適用性失敗原因及原始記錄連結，特別區分 std 已生成極端輸入但缺少數值／執行憑證，以及普通 dense Pass 與缺少 sparse、MKLDNN、跨裝置觸發條件。

### 審閱修正、驗證與證據保存

`posthoc_review.py` 在 `iteration-v85oR2` 使用凍結 evaluator 執行（exit 0、五秒），保留 automatic-v1，另建立 reviewed-v2 與成員證據。八個根據整圖輸出產生的自動 positive 標籤撤回為 unknown，其中包含三個符合指標範圍的 FN。生成程式、production verdict 與預算都沒有改變。

正式生成結束後，`benchmarks/pytorch_correctness/evaluate.py` 已修正為將未認證的 FX 整圖輸出排除於單一目標數學 audit 及歸因，並在 `tests/test_pytorch_benchmark.py` 加入兩項針對性測試。第一版 audit guard 在 `iteration-YBxIUt` 通過九項測試；加入對應 attribution guard 後，最終九項測試在 `iteration-4ur2Pq` 通過（2.351 秒，無略過）。`iteration-0AVDgP` 隨後對每個保存候選與 reviewed-v2 交叉核對，標籤／分類與指標均無差異。此實驗後修正改變 evaluator 雜湊；凍結 evaluator 快照仍保留，43 個 production engine 雜湊皆不變。155 項完整測試結果指正式實驗前的實作，不代表 audit 修正後曾重跑 157 項完整測試。

證據：[automatic-v1](../result/stateful-generation/771760/stage-7/formal-01/evaluation-automatic-v1.json)、[reviewed-v2](../result/stateful-generation/771760/stage-7/formal-01/evaluation-reviewed-v2.json)、[成員審閱證據](../result/stateful-generation/771760/stage-7/formal-01/review-evidence-v2.json)、[修正 evaluator 交叉核對](../result/stateful-generation/771760/stage-7/formal-01/corrected-evaluator-crosscheck.json)。[英文評估](../docs/stateful-generation-evaluation.md)與[繁體中文評估](stateful-generation-evaluation.md)包含對應結果與 prompt 原文。

### 資源關閉與驗收決定

Job 771760 使用使用者明確授權的 `dev` partition、兩張 GPU 與 90 分鐘時限，配置 3,526 秒後結束；job 770948 則使用 12,177 秒。服務啟動時間分別為 185 與 217 秒。有一次 allocation 更換，沒有 allocation 內失敗重啟，正式生成開始後也沒有改動預算。兩個本工作 jobs 及所有 steps 均已退出，testing terminal 已關閉，待處理推論數為零，工作專用測試快取亦已移除。兩個 service steps 在本工作的關閉流程中以 signal 9 結束，記錄明示此事，沒有將其描述成服務正常 exit。

[資源紀錄](../result/stateful-generation/771760/resource-ledger.json)保存 46 個 iteration 的時間／狀態，並區分啟動、正式工作、audit 與 GPU 配置時間。[清理證據](../result/stateful-generation/771760/cleanup-final.json)確認沒有仍執行的本工作 jobs。G7 通過，因為所有目標、指標、不確定性、原始證據與清理皆已交代；驗收從未要求最低發現率。本任務沒有留下仍執行的運算或推論服務。
