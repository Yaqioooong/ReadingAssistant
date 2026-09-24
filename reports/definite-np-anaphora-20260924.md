# 定指名词短语回指：「路旁闪出的女子是谁？」的失败链路与修复

日期：2026-09-24 · 触发：用户在 app 里当场报的一轮对话

## 0. 现场

用户先问：

> 西游记中师徒四人到达女儿国时发生了什么事？国王对唐僧做了什么？唐僧如何处理的？

回答正确，引用第五十四回。紧接着追问：

> 路旁闪出的女子是谁？

系统返回「信息不足，请补充章节、人物或具体情节后继续提问」。**正确答案（蝎子精，毒敌山琵琶洞，第五十五回）就在语料里。**

日志（`logs/qa_20260924.log` / `logs/agent_20260924.log`）：

```
问答[gate] q=路旁闪出的女子是谁？ intent=book raw=book channel=llm
问答[检索] q=路旁闪出的女子是谁？ doc=None hit=0          ← 首检 0 命中
检索agent[step 1] action=grep   args={'term': '路旁闪出'}  new_chunks=2
检索agent[step 2] action=grep   args={'term': '女子'}      new_chunks=8   ← 漂移入口
检索agent[step 3] action=read_chapter args={'chapter_index': 32}
检索agent[step 6] action=grep   args={'term': '白骨夫人,…青罐,炒面筋'}
检索agent[预算耗尽] steps=6
问答[agent] steps=6 stopped=budget finished=False chunks=35
问答[agent] 合并后 16 条（其中原有 0 条）
问答[answer] 工具判定信息不足 → HITL  missing=…现有《西游记》片段只有白骨精（尸魔）三戏唐…
```

## 1. 四环链路（每环都有本地实测）

### A. 闸门只认代词，定指名词短语回指整类漏过

问句的指代是**定指名词短语**「路旁闪出的女子」，不是代词：

```
jieba: 路旁/s 闪出/v 的/uj 女子/n 是/v 谁/r ？
detect_pronoun('路旁闪出的女子是谁？')  -> []      # 「谁」词性 r 但不在 PRONOUNS
has_own_anchor(...)                     -> False
```

`resolve_query` 第一步 `if not detect_pronoun(question)` 就返回，**上文一个字都没进 query**。代价：

```
retrieve('路旁闪出的女子是谁？')                                ->  0 hits
retrieve('路旁闪出一个女子 旋风 摄走')                          ->  0 hits
retrieve('女儿国 路旁闪出的女子是谁')                           -> 16 hits，正解排第 2
retrieve('…（上文相关实体：助手、hello、唐僧、王后）')            -> 11 hits，正解 rank1/rank2
```

**一个平行的漏网**：`那段/这本/这部/那种` 都被 jieba 标成 `r`，但不在 `PRONOUNS` 词表里 →
`detect_pronoun('你刚才说的那段再解释一下')` 也返回 `[]`（gold `mt-11` 正是这个形态）。

### B. agent 的两个漂移路径

- **step 1 其实已拿到正解场景**：`search_text('路旁闪出')` 命中 2 条 —— idx 40（瞽者）与 **idx 60**（「女王不必远送…长老慢下龙车」），后者就是上一轮引用的那段。
- **正解就在紧邻的下一回**：idx 61 = 第五十五回开头，「沙僧道：'是一个女子，弄阵旋风，把师父摄了去也。'」名字随后点明。离答案只差一步「读下一章」。
- **step 2 `grep '女子'` 是漂移入口**。真实词频（全书 1520 块）：
  `'人' 1315 / '妖怪' 170 / '女子' 59`，而 `'山坡下' 21 / '路旁闪出' 2 / '白骨夫人' 2`。
- **step 5 是「用碎片里读到的细节拼成一句书里并不存在的原句」**：`'山坡下闪出一个女子'` 字面命中 **0**；
  `'青砂罐'/'绿磁瓶'/'炒面筋'` **只在 idx 32**（白骨精章）出现 —— 正是 step 3 读到的。
  而**语义检索对任何查询都会返回结果**，这条没有依据的线索被"确认"，烧掉 step 4/5/6。

### C. 合并按「越晚越优先」，最早发现的证据第一个死

`ordered_agent = reversed(read_chunks) + reversed(other_chunks)`；本次 `originals` 为空 → `reserve=0`。
用**真实的 `merge_agent_chunks`** 复刻本次形状（cap=16、chunks=35）：

```
converged=True  → 保留区间 c13..c25，step1 的 c1/c2 全部落选
```

`c2` 就是 idx 60。判官随后只能看到白骨精 —— 与日志吻合。

⚠️ **这不是普通 bug**：`tests/test_agent.py` 另有一条测试明确要求「agent 名额应全部给最新一批」，
注释写着小雷音寺那次「后出的被丢掉」是要防的回归。两条证据都真，
区别在于**小雷音寺那次的 agent 收敛了**（结论来自它读完的章），本次 `stopped=budget finished=False` **没收敛**。

## 2. 修复

### A. 闸门扩到定指名词短语回指（`graph/rewrite.py`）

- `detect_pronoun` 补**形态档**：「指示词 + 量词」（`那段/这部/那种`…），用**显式量词集合**
  卡第二个字 —— `那么/这么/那样` 也是 `r`，是副词性的，必须天然挡掉。
- 新增 `detect_descriptive_reference`：问句里的**非通用普通名词**字面出现在最近 4 条上文中即命中。
  **不用最长公共子串**（先试过、实测不成立）：上文「路旁**又**闪出**一个**女子」vs
  问句「路旁闪**出**的女**子**是谁」，最长公共子串只有 2 字。判据写进 docstring，
  并有一条测试用 LCS 把这件事实钉住，谁把判据改回去就会红。
- 两条路径互斥、汇合后共用同一套 `collect_candidates` 与**同一个追加格式**（既有 20 条测试全绿）。
- 配置 `cqr_anaphora_enabled`（默认 True），可对两条路径做单变量 A/B。

### B. agent 反漂移护栏（`graph/agent.py`）

- **过宽 grep 告警**：按**单个 term** 的命中数，超 `_BROAD_GREP_MAX_HITS = 40` 时在 observation 最前面加警告。
  取 40 的依据就是上面那两组实测数字（把 59/170/1315 与 2/2/21 干净分开）。
  **只提示、不拒绝、不改片段** —— 拒绝执行会白吃掉一步本就紧张的预算。
  按单词判、不按合计，否则「一次给几个候选名让工具比较先后」的正常用法会被误伤。
- **邻接提示**：`read_chapter` 返回里附上**确定的**下一回 `chapter_index` 与回目
  （实测 60 → `chapter_index=61《第五十五回 色邪淫戏唐三藏 性正修持不坏身》`）；取不到章节表时静默降级。
  grep 的提示里给原则。
- **prompt**：新增「证据必须来自本书语料」一节，写明「不要把碎片里读到的细节拼成一句书里没有的原句再去搜」
  （措辞刻意区分于「凭空编造」—— 那些词是它真从 idx 32 读到的）。

### C. 未收敛时保留「锚」片段

`merge_agent_chunks` 加 `converged: bool = True`，由 `qa.py` 传 `run.finished`：
- `converged=True` → **行为一位不变**（既有两条 merge 测试原样通过）。
- `converged=False` → 每章留**首次**发现的片段作锚，`agent_budget // 4` 个名额。

复刻验证：

```
converged=True  → c1/c2 全落选（旧行为）
converged=False → c1、c2 都在（保留区间 c13..c12,c11,c37..c29, c1,c2,c3,c4）
```

### 接线：定指回指要显式放行 agent（**这一步是被 A/B 逼出来的**）

A/B（各 3 次，只跑这一题）第一阶段：

| | agent | 答对 | clarify |
|---|---|---|---|
| A 开（仅闸门） | **否**（0 步） | **0/3** | 否 |
| A 关（旧行为） | 是（3–4 步） | 2/3 | 1/3 |

修好闸门后 retrieve 从 0 → 11 命中，**但答案改错了**：3/3 输出「原文片段没有交代那女子的身份」。
查清原因：那 11 条命中的确有**正确的第 61 章**，但命中的那一块正文里**没有「蝎子精」三个字**
（名字在 61 章的另一块）—— **检索命中了正确的章，却落到错误的块**，问句与它之间没有字面桥梁。
而 `judge` 的规则是「无检索结果才算信息不足」，于是**有片段反而把 agent 挡在门外**，
而 agent 的 `read_chapter` 正是唯一能读到那一块的手段。

故新增 `rewrite.is_descriptive_anaphora` + `QAState.cqr_anaphora`，
`route_after_judge` 对这条路**无条件放行 agent**（仍受 `agent_used` 每题至多一次的闸约束）。

接线后 A/B：

| | 答对 | clarify |
|---|---|---|
| A 开 | **3/3** | 0/3 |
| A 关（旧行为） | 1/3 | 2/3 |

**修前**：首检 0 命中、白烧 6 步 agent、2/3 要用户补充线索。
**修后**：首检即命中、agent 只走 3–5 步、3/3 答对、0/3 追问。

## 3. 顺带的评测集改动

- `eval/agent_gold.json` 新增 `ag-anaphora-01`（class=`anaphora`，`expect_contains=蝎子精`）。
  从 18 → 19 条。**不设 `must_use_any_tool`** —— 修好之后它不该再依赖某个工具。
- `cqr_anaphora_enabled=False` 时该题仍会失败（1/3），所以这条 A/B 就是它的回归。

## 4. 如实记录的边界

- **样本极小**：A/B 各 3 次。按既有纪律（噪声底约 1–2 题），3 次的绝对数字不可当基线；
  这里可作证据的是**分离与方向**：开→关从 3/3 掉到 1/3、澄清从 0/3 升到 2/3，
  且失败形态（「原文没交代」）与「要补充线索」是**两种不同**的坏结果。补跑 ≥5 次可加固。
- **`ag-anaphora-01` 的 gold 未验证语料在场**：`expect_contains='蝎子精'` 我确认过第 61 章正文里有，
  但没像 09-23 那样跑「真值字符串在语料里字面命中 >0」的校验。下次扩集时补上。
- **jieba 分词是这一档的天花板**：`那部书` → `那/r`+`部书/n`、`那本书` → `nr`，
  形态档认不出。要再往上收就得引入裸 `这`/`那` 规则，会放宽一批没有实测依据的形态 —— 留作后续，
  并已用 `test_known_ceiling_is_jieba_segmentation` 如实钉住。
- **未做的**：`ag-pronoun-02`（多候选指称选择）与本次无关，仍停在 09-23 的状态（卡在「谁来在多候选间选」，
  需要模型参与，与「规则优先零 LLM」冲突）。
- **过宽 grep 阈值是语料相关的**：40 是对《西游记》1520 块量的。换一个规模不同的语料库要重新量。

## 5. 改动清单

| 文件 | 内容 |
|---|---|
| `src/reading_assistant/graph/rewrite.py` | 指示词量词形态档、`detect_descriptive_reference`、`is_descriptive_anaphora`、闸门放宽 |
| `src/reading_assistant/config.py` | `cqr_anaphora_enabled` |
| `src/reading_assistant/graph/qa.py` | state 字段 `cqr_anaphora`、节点写入、`route_after_judge` 放行、merge 传 `converged` |
| `src/reading_assistant/graph/agent.py` | 过宽 grep 告警、邻接提示、prompt 反漂移、`merge_agent_chunks(converged=)` |
| `tests/test_cqr.py` | 26 → 39 条 |
| `tests/test_agent.py` | 36 → 44 条 |
| `eval/agent_gold.json` | +`ag-anaphora-01` |

`uv run python -m pytest -q` → **593 passed**（基线 572）；`uv run ruff check src tests` 干净。
