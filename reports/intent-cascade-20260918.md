# 检索门意图识别：从「每轮必调 LLM」到三级级联

> 2026-09-18 · 目标：**减少检索门的 LLM 调用次数，同时不降低意图识别准确率**
> 实现：`src/reading_assistant/graph/intent.py`（新）+ `graph/qa.py::gate`（改造）
> 评测：`eval/run_intent_eval.py --mode fake|real`，gold 扩到 33 轮（tune 20 / holdout 13）

## 1. 问题

改造前 `gate` 节点对**每个会话内轮次**都调一次 LLM 做三分类（book / history / chat）。
而正常读书会话里绝大多数轮次的答案就是 `book` —— 这是一笔可以省掉的钱。
线上日志里 gate 与滚动摘要在关键路径上串行，每轮固定付两次 LLM 调用。

## 2. 设计：代价不对称是主轴

分层的依据不是「哪层准」，而是**判错的代价不对称**：

| 判错方向 | 后果 | 代价 |
|---|---|---|
| 把书问题判成 `book` | 最多多检索一次 | **廉价** |
| 把书问题判成 `chat`/`history` | 问题被直接吞掉，用户对着闲聊式回答发懵 | **昂贵** |

`docs/design-intent-recognition.md` 里 mt-07 那个事故（首轮发 `hello` 掉进书问答链路、
检索为空、被判「信息不足」并弹澄清任务）就是昂贵方向的真实代价。

于是三条铁律，全部落在代码结构里：

1. **fallback 方向恒为 `book`** —— 拿不准一律进检索；
2. **L2 向量原型层只敢自动判 `book`，绝不敢自动判 `chat`/`history`** ——
   「确认安全默认值」廉价，「押注偏离默认值」昂贵；
3. `chat`/`history` 的自动判定**只**交给 L1 规则层，且规则一律高精度构造：
   宁可漏判（漏了掉到 L2/L3 兜底），绝不误判。

```
L1 规则快通道   高精度正则/词表     命中即出          0 次 LLM
L2 向量原型     3 类原型句最近邻    高置信确认 book   0 次 LLM
L3 LLM 兜底     改造前的 gate prompt 只处理余量        1 次 LLM
```

### 2.1 铁律 2 是实测出来的，不是拍脑袋

用真实 embedding（DashScope `text-embedding-v4`）测 gold 问题对三类原型句的余弦分：

| 问题 | 期望 | book | history | chat | 判 book 的领先 |
|---|---|---|---|---|---|
| 张三喜欢谁？ | book | **1.000** | 0.375 | 0.388 | +0.612 |
| 刚才说的那个计划，执行者是谁？ | book | **0.900** | 0.446 | 0.340 | +0.455 |
| 你好，请问面壁计划是什么？ | book | **0.803** | 0.383 | 0.315 | +0.420 |
| 我上一个问题里问的那个人是谁？ | book | 0.447 | **0.747** | 0.469 | **−0.300** |
| 你刚才说的那段再解释一下 | book | 0.627 | **0.677** | 0.497 | **−0.050** |
| 我上一轮问的罗辑是谁？ | book | 0.761 | **0.875** | 0.398 | **−0.114** |

第 4 行就是铁律的判决书：「我上一个问题里问的那个人是谁」是**书问题**，
但 history 得分 0.747 远高于 book 的 0.447。
**如果 L2 敢输出 history，这里就会吞掉用户的问题。** 所以它不许输出 history。

### 2.2 阈值标定

阈值扫描（L2 只判 book，其余走 LLM）：

| threshold | margin | 判 book | 误判 |
|---|---|---|---|
| 0.75 | 0.06 | 4/11 | 0 |
| 0.80 | 0.06 | 4/11 | 0 |
| 0.80 | 0.10 | 4/11 | 0 |
| 0.82 | 0.06 | 3/11 | 0 |
| 0.85 | 0.06 | 3/11 | 0 |

取 **threshold=0.75 / margin=0.08**：整个 `[0.75, 0.85]` 区间误判恒为 0，
默认值贴区间下沿以多省调用，靠 margin 挡住「book 与 history 一样高」的情形。
两个参数都在 `Settings` 里可配，`INTENT_PROTOTYPE_THRESHOLD<=0` 直接停用 L2。

## 3. 实现

| 文件 | 改动 |
|---|---|
| `graph/intent.py` | **新增**：L1 规则（19 条模式）+ L2 `PrototypeRouter` + 级联编排 `route_fast` |
| `graph/qa.py` | `gate` 改为级联、新增 `intent_channel/intent_score/intent_embedding` 状态；`cache_check` 复用 L2 向量；**收紧 gate prompt 边界** |
| `config.py` | 新增 `intent_rules_enabled` / `intent_prototype_threshold` / `intent_prototype_margin` |
| `api/schemas.py` + `routes/sessions.py` | `AskResponse` 增加 `intent_channel` / `intent_score` 观测字段 |
| `tests/test_intent_cascade.py` | **新增** 48 条单测 |
| `tests/test_multi_turn.py` | 旧断言 `gate_calls == 2` 钉的是「每轮必调 LLM」的旧行为，改写为钉新契约 |
| `eval/multi_turn_gold.json` | 20 轮 → 33 轮，新增 13 轮对抗集并标记 `split=holdout` |
| `eval/run_intent_eval.py` | 报告新增 LLM 兜底轮次、通道分布、分 split 统计 |

### 3.1 顺带白嫖：复用 L2 算出的 embedding

`cache_check` 原本独立 `retriever.embed()` 一次。L2 也要嵌同一个问题 —— 于是把向量
经 `intent_embedding` 传下去复用（`retriever.embed` 本身有 L1 缓存，同一轮次不会重复付费）。

⚠️ 坑：`cache_check` 嵌的是 `question + 补充说明`，带 `clarification` 时与 L2 嵌的裸
`question` **不是一个向量**，所以复用加了 `not clarification` 条件；规则命中的轮次
L2 根本不跑，`embed` 是惰性回调，一次 embedding 都不花。

## 4. 评测结果（real 模式，DeepSeek + Postgres + 真实语料）

基线 = 把级联关掉（`INTENT_RULES_ENABLED=false INTENT_PROTOTYPE_THRESHOLD=0`），
即改造前的「每轮必调 LLM」行为。同一份 gold，同一进程环境，只差这两个开关。

| 指标 | LLM-only 基线 | 三级级联 | 变化 |
|---|---|---|---|
| 路由准确率（33 轮） | 100.00% | **100.00%** | 持平 |
| **检索门 LLM 调用次数** | **33** | **6** | **−81.8%** |
| ├ tune 20 轮 | 100% / 20 次 | 100% / 1 次 | −95% |
| └ holdout 13 轮 | 100% / 13 次 | 100% / 5 次 | −61.5% |

通道分布：`rule 23 / prototype 4 / llm 6`。

**分通道准确率（关键证据）**：

| 通道 | 准确率 | 说明 |
|---|---|---|
| `rule` | **23/23 = 100%** | 零 LLM |
| `prototype` | **4/4 = 100%** | 零 LLM |
| `llm` | 6/6 = 100% | 剩下的难例 |

即：**27 轮（81.8%）完全不花 LLM，且这 27 轮全部判对**。收益主要来自 L1 规则（23 轮），
L2 原型层只贡献 4 轮（12%）—— 但它是「规则够不着、又明显是书问题」那批的安全网。

fake 模式（关键词桩 LLM）：78.79% / 33 次 → **87.88% / 10 次**（准确率反升，见 §5）。

## 5. 两个值得记的发现

### 5.1 既有 LLM gate 的缺陷被留出集抓出来，并已修复

`我上一个问题里问的那个人是谁？`（书问题）在**基线里也判错**成 history，
级联同样判错 —— 因为它走的是 `channel=llm`，是 LLM 的锅，不是新层引入的。
根因：prompt 的边界例子只覆盖了「刚才/上一条」，没覆盖「我上一个问题」这种
**元问题外壳 + 书内容内核**的句式，模型锚在「我上一个问题」上。

修复（`_build_gate_prompt`）：把判别口径写成对立面 ——
> 出现「我上一个问题／上一条／刚才」字样**不等于** history：
> 问「记录本身」（我上一个问题是什么）才是 history，
> 问「记录里提到的书内容」（我上一个问题里问的那个人是谁）仍是 book

修完基线 33/33、级联 33/33。这与 `design-intent-recognition.md` §5.5 记录的
2026-09-09 那次「真机抓到 gate 误判→收紧 prompt 边界」是同一类修复。

### 5.2 fake 模式的「LLM」是个关键词桩，别拿它的数字说事

`eval/run_intent_eval.py --mode fake` 的 `FakeRouterLLM` 按 `('上一个','问过哪些',…)`
这类关键词路由，**没有** design doc 那条边界修正。所以：

- 它在对抗集上只有 46%（基线）—— 是桩的缺陷，不是真实模型的水平；
- 级联在 fake 模式下反而**更准**（87.88% vs 78.79%），因为规则层把桩缺的边界
  知识补上了。

结论：**对比级联收益要看 real 模式**；fake 模式的价值是链路自检与快速回归。

## 6. 诚实声明（哪些结论不硬）

1. **阈值是在同一批数据上标定的**（gold + 对抗集），所以「准确率 100%」有乐观偏差。
   但真正的安全属性是**结构性**的、与标定无关：L2 的返回类型被限制为
   `book | None`，`test_prototype_never_returns_chat_or_history` 用 8 组向量
   钉住「它不可能输出 chat/history」。风险最高的那类错误不是靠调参避开的，是靠类型避开的。
2. **`split=holdout` 不是严格的留出集**：13 轮对抗集是在冻结阈值之后写的，
   但 §5.1 的 prompt 修复确实是**看了 holdout 的失败才做的**（已披露）。
   修复后 holdout 从 84.62% → 100%，而这个提升包含了对该场景的针对性修补。
3. **33 轮样本量小**，`llm` 通道只有 6 轮。改造中复跑过 4 次 real：
   未修 prompt 时 96.97% / 93.94%（mt-09 t3 在 LLM 路径上翻转）—— LLM 路径本身有波动，
   而 rule/prototype 通道 4 次全对。
4. **L2 的边际贡献不大**（4/33）。若只保留 L1 规则 + LLM 兜底，能拿到约 70% 的调用削减
   且实现更简单。L2 的价值在「规则够不着的明显书问题」，属于锦上添花而非必需。
5. **`mt-01 t3` 的 `in_answer` 断言本身是脆的**：4 次 real 复跑里有一次
   `content_accuracy` 掉到 50%，唯一未命中项是 `mt-01 t3`（递归元问题
   「我上一个问题是什么？」要求回答里出现同一句话）。该轮 `channel=rule`，
   而 `_build_context_prompt(question, history, summary)` **不接收 intent** ——
   即 gate 走哪条通道在结构上不可能影响回答内容，这是回答模型的措辞波动
   （是否带引号／问号）。`design-intent-recognition.md` §5.5 已对「再上一个呢」
   记过同类问题（字面回显断言过苛），本项未改。
6. `intent_channel` 是**按轮次**统计的：澄清续答轮不重跑 gate，其 channel 是 checkpoint
   继承值。报告里的「LLM 兜底轮次」按轮次定义计数，未做去重。

## 7. 复现

```bash
# 基线（等价于改造前：每轮必调 LLM）
INTENT_RULES_ENABLED=false INTENT_PROTOTYPE_THRESHOLD=0 \
  uv run python eval/run_intent_eval.py --mode real

# 三级级联
uv run python eval/run_intent_eval.py --mode real

# 单测
uv run python -m pytest tests/ -q
```

报告落 `eval/reports/intent_report_{fake,real}.json`，含 `llm_gate_turns`、
`intent_channel` 分布与分 `split` 准确率。本次 before/after 的原始证据另存为
`eval/reports/intent_report_real_{baseline,cascade}.json` 与
`eval/reports/intent_report_fake_{baseline,cascade}.json`（`eval/reports/` 被 gitignore，属本地证据）。

## 8. 后续可做

- **L2 边际贡献低** → 若想再压调用数，方向不在原型层，而在把 `book` 的判定前移到
  更便宜的信号（书名号已做；可扩到「含疑问词 + 无元语言第一人称」这类结构特征）。
- **`summarize` 的开销**是同类问题且更大（每轮必触发、O(n²)、在关键路径上等 ~3s），
  见 `docs/` 内相关分析，本次未动。
