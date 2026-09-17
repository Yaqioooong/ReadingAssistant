# RAG 离线评测机制

一套能**回归、能归因、能对比**的离线评测体系。核心原则：按失效点分层，指标要确定性，判分要能重复跑出同一个数。

---

## 一、三层架构

```
                        ┌─────────────────────────────┐
   用户提问 ──────────▶ │  路由层  intent              │  该不该走 RAG？多轮能否记住上文？
                        │  run_intent_eval.py          │  → route_accuracy / content_accuracy
                        └──────────────┬──────────────┘
                                       │ 走 RAG
                        ┌──────────────▼──────────────┐
                        │  检索层  retrieval           │  该召回的 chunk 进 top-k 了吗？
                        │  run_retrieval_eval.py       │  → Recall / Precision / NDCG / MRR / MAP
                        │  hard_negative.py            │  → 硬负样本（语义相近的干扰项）
                        └──────────────┬──────────────┘
                                       │ 喂给 LLM
                        ┌──────────────▼──────────────┐
                        │  生成层  e2e                 │  拿到料会不会答错、会不会瞎编？
                        │  run_eval.py                 │  → pass_rate / 拒答率 / 幻觉率 / 相关性
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  回归闭环                     │  这次改动是涨了还是跌了？
                        │  compare_reports.py          │  → 指标 delta + PASS→FAIL 迁移
                        │  sample_review.py            │  → 人工抽检，纠正 gold 偏差
                        └─────────────────────────────┘
```

**为什么必须分层**：指标掉了要能立刻知道该修检索还是修生成。混在一起测，失败原因归因不了。

---

## 二、指标定义

### 检索层（binary relevance，k 默认 5/10/20）

| 指标 | 公式 | 回答什么问题 |
| --- | --- | --- |
| `Recall@k` | \|top-k ∩ gold\| / \|gold\| | 该召回的都召回了吗 |
| `Precision@k` | \|top-k ∩ gold\| / k | 召回里有多少是噪声 |
| `Hit@k` | 1 if top-k ∩ gold ≠ ∅ | 至少命中一个了吗 |
| `MRR@k` | 1 / 第一个相关的排名 | 最相关的排得够靠前吗 |
| `NDCG@k` | DCG@k / IDCG@k | 整体排序质量（位置加权） |
| `MAP@k` | Σ Precision@i / min(\|gold\|,k) | 多相关项的排序平均质量 |

> 只看 Recall 会掩盖排序问题：全部召回但都排在第 20 位，Recall@20=1 而 MRR=0.05。
> 所以 **NDCG + MRR 必须一起看**。

### 生成层

| 指标 | 定义 | 越高越好 |
| --- | --- | --- |
| `context_recall` | gold chunk 被检索覆盖的比例 | ✅ |
| `context_precision` | top-k 中属于 gold 的占比 | ✅ |
| `faithfulness` | 答案句子能在上下文中找到依据的比例（规则代理） | ✅ |
| `answer_relevancy` | 疑问类型与答案形态是否匹配（规则代理） | ✅ |
| `citation_presence` | 是否带引用标记 | ✅ |
| `citation_validity` | 引用的 chunk_id 真实存在的比例（防幻觉引用） | ✅ |
| `abstention_rate` | **不可答题**中正确拒答的比例 | ✅ |
| `false_refusal_rate` | **可答题**中被误拒答的比例 | ✅ |
| `hallucination_rate` | **不可答题**中凭空作答的比例 | ❌ 越低越好 |

> `faithfulness` / `answer_relevancy` 是**规则代理**，不是 LLM 判定。
> 它们衡量「能不能找到依据」，无法识别语义等价和细微事实矛盾 —— 这正是 `judge.py` 的补充位。
> 代理指标负责**回归门禁**（确定性、可重复），LLM judge 负责**语义深挖**（需校准才可信）。

### 判官校准

| 指标 | 含义 |
| --- | --- |
| Cohen's kappa | judge 与人工标注的一致性（已扣除随机巧合） |

**kappa < 0.6 判定该 judge 不可信**，只能以规则指标为准。这是 LLM-judge 类方案的标配护栏。

---

## 三、跑评测

### 一键全套（推荐）

```bash
uv run python -m eval.run_all --mode real                  # 跑三层，产出统一报告
uv run python -m eval.run_all --mode real --only e2e       # 只跑生成层
uv run python -m eval.run_all --mode fake --only intent    # 冒烟
uv run python -m eval.run_all --mode real --judge llm      # 附加 LLM 判官 + 校准
```

产出 `eval/reports/suite_<ts>.json`（机器读）+ `suite_<ts>.md`（人读）。

### 单独跑某一层

```bash
uv run python -m eval.run_retrieval_eval --mode real       # 检索层
uv run python -m eval.run_eval --mode real                 # 生成层
uv run python -m eval.run_intent_eval --mode real          # 路由层 + 多轮
```

> **多轮评测已并入 intent**：`run_intent_eval.py` 读的就是 `multi_turn_gold.json`，
> 同时覆盖「该不该走 RAG」和「多轮能否记住上文」两件事。

### fake vs real

| 模式 | 用途 | 注意 |
| --- | --- | --- |
| `fake` | 链路冒烟、CI 快速门禁 | 向量退化，**指标数值无意义** |
| `real` | 真实评测 | 需要 API key，慢 |

**永远不要拿 fake 的指标跟 real 对比，也不要拿去发版判断。**

---

## 四、回归流程（核心）

```bash
# 1. 第一次跑：没有基线时自动钉住本次结果（也可显式 --pin）
uv run python -m eval.run_all --mode real

# 2. 改代码...（改检索参数 / 换 embedding / 调 prompt）

# 3. 再跑一次，自动跟基线对比
uv run python -m eval.run_all --mode real

# 4. 看 delta 与回归用例；如需单独对比
uv run python -m eval.compare_reports --list
uv run python -m eval.compare_reports --out report.md
uv run python -m eval.compare_reports --fail-on-regression   # CI: 有回归则 exit 1
```

对比输出两块：

1. **汇总指标 delta 表** —— 绝对值 / 相对变化 / ↑↓ / 是否越过阈值
2. **per-case 状态迁移** —— 🔴 PASS→FAIL（回归）/ 🟢 FAIL→PASS（修复）/ 新增 / 移除

> ⚠️ 基线是有模式属性的：**不要用 fake 模式的基线去对比 real 的结果**。
> 换模式前先删掉 `eval/reports/baseline_suite.json` 重新钉。

### 阈值怎么定

`--threshold`（默认 0.02）是**绝对**阈值，只适配 **0~1 的比率型指标**：

- 检索类指标（Recall/NDCG）：建议 **0.02**，样本少时波动大
- 拒答/幻觉率：建议 **0.05**，这类判定本身有离散性

#### ⚠️ 量纲不同的指标走单独的显著性策略

对 `avg_latency_ms`（毫秒）这类指标，绝对阈值 0.02 会让 `abs(delta) >= 0.02`
**恒成立** —— 即「任何变化都算显著」。

实测（2026-09-17）历史 6 次 real 运行：

| run | avg_latency_ms | pass_rate | 相邻 Δ | 0.02 绝对阈值下判定 |
| --- | --- | --- | --- | --- |
| 09-07 | 1421 | 1.0 | — | 基线 |
| 09-11a | 774 | 1.0 | -647 | 🟢 提升 |
| 09-11b | 932 | 1.0 | +158 | 🔴 回归 |
| 09-16a | 687 | 1.0 | -245 | 🟢 提升 |
| 09-16b | 581 | 1.0 | -106 | 🟢 提升 |
| 09-17 | 710 | 1.0 | +129 | 🔴 回归 |

**5/5 次相邻对比被标记，而质量（pass_rate）全程未变** —— 假警报率 100%。
这种「狼来了」会训练人忽略回归信号，把门禁废掉。

**修法**：噪声指标族走 `_THRESHOLD_OVERRIDES`（`eval/compare_reports.py`），
要求**相对变化与绝对变化同时超阈**：

```python
_THRESHOLD_OVERRIDES = {
    'avg_latency_ms': {'rel': 0.5, 'min_abs': 200.0},
}
```

| 判据 | 结果 |
| --- | --- |
| 历史 5 次自然波动（最大 ±45%） | 全部不显著 ✅ 正确放过 |
| 真实劣化（2 倍以上） | 全部显著 ✅ 正确拦截 |
| 比率型指标 | 行为不变，仍用全局绝对阈值 ✅ |

> 加新的噪声指标（如内存、磁盘、token 数）时，往 `_THRESHOLD_OVERRIDES` 加一条即可，
> 键用指标名末段（`avg_latency_ms` 同时匹配 `e2e.avg_latency_ms`）。
> 覆盖测试见 `tests/test_eval_compare_thresholds.py`。

---

## 五、加题

### ⚠️ 先看这条：语料文件名 = 文档主键

`_upload_books()` 用**语料目录里的真实文件名**（`path.name`）作为 `doc_map` 的键，
gold 里的 `document` 靠它换成 `document_id`。所以文件名对不上时**不会报错，只会静默失效**：

| 情况 | 后果 |
| --- | --- |
| 语料改名，gold 没跟着改 | 该题 `document` 解析不到 → 生成层退化成全库检索、检索层 gold 为空被 `[skip]`（连分母都不算） |
| gold 写了语料里不存在的名字 | 同上 |
| 只改 gold 不改语料 | 同上 |

契约：**`eval/uploads/` 里的文件名必须与 gold 的 `document` 字段逐字符一致**，
`tests/test_eval_golden_files.py::test_epub_fixture_filename_matches_the_referenced_document`
把两个方向都钉住了（改名会直接 FAIL，而不是安静地少统计几道题）。

当前大语料 fixture 用的是与生产入库完全相同的文件名：

```
eval/uploads/西游记 (吴承恩) (z-library.sk, 1lib.sk, z-lib.sk).epub
```

它与应用 `uploads/` 下的同名书**字节一致**（sha256 相同），因此 `DocumentService`
的分层去重会把它映射到同一个 `document_id`（当前为 7），不会重复入库。

### 加题前的自检

命里 `tests/test_eval_golden_files.py` 会在几秒内告诉你题有没有写歪（不连库、不调模型）：

```bash
uv run pytest tests/test_eval_golden_files.py -q
```

它检查：id/题干在本文件内唯一、两层共用 id 的题干一致、`document` 在语料里存在、
可答题必须有 `expect_keywords`（否则判定退化成「回答非空即通过」）、不可答题不得带关键词、
五类覆盖齐全、跨文档用例的 `documents` 顺序被打乱、`contains`/`chunk_ids` 恰好二选一且限定了文档。

### 加端到端题（生成层）

编辑 `golden_set.json`：

```json
{
  "id": "love-16",
  "question": "王五为什么讨厌朱六？",
  "document": "张三爱情故事.txt",
  "expect_keywords": ["因为", "抢"],
  "expect_unanswerable": false,
  "note": "因果推理"
}
```

**分类是硬要求**，每类都得有，否则评测集只会告诉你「它能答对」，不会告诉你「它什么时候答错」：

| 类别 | 例子 | 作用 |
| --- | --- | --- |
| 正向基础 | 张三喜欢谁？ | 基线能力 |
| 陷阱 | 王五喜欢李四吗？（反向关系误导） | 引用准确性 |
| 多跳推断 | 朱六想让张三喜欢谁？ | 推理链路 |
| 信息不足 | 张三的生日是哪天？ | 拒答能力 |
| 跨文档 | 谁在说服张三放弃李四？ | 多文档召回 |

### 加检索题（检索层）

```bash
uv run python -m eval.build_retrieval_gold --mode real --limit 20
```

`build_retrieval_gold.py` 会跑端到端，把 PASS case 的 `citations` 沉成 gold chunk。
**这是弱标注，必须配人工抽检**（见下节）。

或手工指定 gold，更可靠：

```json
{
  "id": "love-16",
  "question": "王五为什么讨厌朱六？",
  "document": "张三爱情故事.txt",
  "document_id": 2,
  "chunk_ids": ["doc2-3"],
  "note": "因果推理"
}
```

### 加硬负样本

```bash
uv run python -m eval.hard_negative --mode real              # 默认：按相对相似度取最近的非 gold 邻居
uv run python -m eval.hard_negative --mode real --min-sim 0.5 --top-n 30
```

挖两类干扰项，输出 `retrieval_hard_neg.json`：

- `hard_negatives` —— 语义负样本：用 gold chunk 自己的 embedding 检索，相似度高但不是 gold 的
- `neighbor_negatives` —— 邻接负样本：同文档里位置相邻（±2）的 chunk

这类负样本**合成不出来**，必须从真实语料挖。相似度越高越「硬」，最能暴露混合检索的短板。

### ⚠️ 不要随便设 `--min-sim`

硬负样本是**相对概念**：真正的干扰项是「离 gold 最近的非 gold chunk」。
设一个绝对阈值看似严谨，实际很容易失效。

实测数据（本项目 1520 个 chunk / 14 个 case / 24 个 gold chunk）：

| 口径 | 相似度范围 | ≥0.6 的条数 |
| --- | --- | --- |
| 所有「非自身」pair（456 条） | 0.4225 ~ 0.7441 | 8 条（1.75%） |
| 排除整个 gold 集后的真实候选 | 0.4225 ~ **0.5792** | **0 条** |

注意第二行：那 8 条高分对**全部是同一 case 内部 gold 之间的相似**
（shaobai 的两个块互为 0.7441）—— 它们本来就会被 `gold_set` 过滤掉。
真实候选池的天花板只有 **0.5792**，所以 `--min-sim 0.6`
在本题库上**挖不到任何东西**。

更糟的是**它的失败是静默的**：脚本正常退出、报告正常落盘，只是 `hard_negatives` 为空。
工具没崩，但等于没用。

所以 `--min-sim` 默认是 `None`，按相对相似度取最近的非 gold 邻居。
确需绝对阈值时，先看脚本打印的分布统计，照着真实区间定。
脚本在挖不到时会明确告警并打印观测区间，不再假装成功。

**另一条限制**：邻接负样本要求文档有富余段落。
如果 gold chunk 已覆盖所在文档的全部内容（本项目的 doc3/doc4 就是），
就挖不出邻居。脚本会告警说明，而不是让你误以为代码有问题。

## 六、人工抽检闭环

**问题**：`build_retrieval_gold.py` 是从「模型自己答对的 case」里捞 gold —— 等于自己出题自己判。
检索一退步，这些 case 直接跌出 PASS，gold 集反而缩小，误差被掩盖而不是被发现。

**解法**：分层抽样 + 人工判定 + 回写。

```bash
# 1. 分层抽样（按 note 分组），固定 seed 保证可复现
uv run python -m eval.sample_review --gold eval/retrieval_gold.json --ratio 0.2

# 2. 人读 eval/reports/review_<ts>.md，填写同名的 csv 的 verdict 列
#    verdict: 对 / 错 / 不确定

# 3. 回写 gold：标错的 gold chunk 被剔除，同时写审计记录
uv run python -m eval.sample_review --apply eval/reports/review_<ts>.csv

# 4. 看人工与自动判定的一致率（分歧越大说明 gold 越不可信）
uv run python -m eval.sample_review --agreement eval/reports/review_<ts>.csv
```

审计记录落在 `eval/reports/review_audit.jsonl`，可追溯「谁在什么时候改了哪条 gold」。

---

## 七、文件清单

| 文件 | 作用 |
| --- | --- |
| `metrics.py` | 统一指标库（纯函数，无 IO，可独立单测） |
| `judge.py` | LLM 判官（opt-in）+ Cohen's kappa 校准 |
| `compare_reports.py` | 基线对比、per-case 回归、CI 门禁 |
| `hard_negative.py` | 硬负样本挖掘（语义 + 邻接） |
| `sample_review.py` | 分层抽样、人工判定回写、审计 |
| `run_all.py` | 一键跑全套，统一报告 |
| `run_retrieval_eval.py` | 检索层评测（vector vs hybrid） |
| `run_eval.py` | 生成层端到端评测 |
| `run_intent_eval.py` | 路由层 + 多轮评测 |
| `build_retrieval_gold.py` | 从端到端 PASS case 沉淀弱标注 gold |
| `tests/test_eval_metrics.py` | 指标单测（含边界与防回归用例） |
| `tests/test_eval_run_all.py` | 汇总编排单测（runner 返回形态、落盘顺序） |
| `tests/test_eval_hard_negative.py` | 负样本挖掘辅助函数单测 |
| `tests/test_eval_golden_files.py` | **黄金集文件自身的结构守卫**（唯一性/文档可解析/判定策略/类别覆盖/文件名契约） |

### 数据集

| 文件 | 层 | 内容 |
| --- | --- | --- |
| `golden_set.json` | 生成 | 端到端黄金集（分类标注，38 例） |
| `golden_multi_doc.json` | 生成 | 跨文档用例（8 例） |
| `retrieval_gold.json` | 检索 | 检索集（34 例；历史 14 例为 chunk_id 弱标注，xjy-r01~r20 为 `contains` 短语定位） |
| `retrieval_adversarial_gold.json` | 检索 | 对抗集（短语定位，30 例） |
| `retrieval_hard_neg.json` | 检索 | 硬负样本（挖掘产物） |
| `multi_turn_gold.json` | 路由 | 多轮对话集 |
| `human_labels.json` | — | 人工标注（用于 judge 校准，需手工维护） |

---

## 七点五、2026-09-17 扩容与重构记录

引入《西游记》（1332 chunk / 全库 1520）把黄金集从 50 例扩到 114 例，
指标终于脱离天花板（原先检索层 vector/hybrid 双双 1.0、打平，没有判别力）。
详细报告见 `reports/retrieval-golden-expansion-20260917.md`。

扩容顺手逼出 4 个**静默**缺陷（前三个在生产路径上）：

| # | 缺陷 | 症状 | 修复 |
| --- | --- | --- | --- |
| 1 | 重复上传把 `index_status` 打回 `indexing`，而检索白名单只认 `indexed` | 整本书的问答 `hit=0` → 判「信息不足」转 HITL，日志无异常 | `graph/ingest.py`：只有真会走 `chunk_and_index` 时才占租约 |
| 2 | 租约时间戳写 aware UTC，落库被按会话时区换算，`_as_utc()` 又按 UTC 解读 | 卡死的 `indexing` 记录永不过期，对账修不动 | 写 naive UTC |
| 3 | Chroma 的 distance / embedding 是 numpy 标量，透传进 QA state 的 `score` | LangGraph checkpoint 序列化抛 `TypeError` → 接口偶发 500 | 产出端收敛为内建 `float`（3 处） |
| 4 | 判官词表漏词（「没有提到」不在表内） | 模型诚实拒答被判 FAIL（假阴性） | 补词表，并把边界写进 §八 |

另外移除了 `qa._looks_like_no_info`（按 `answer[:80]` 匹配词表的启发式）：
它要判的是「这次回答可不可信」这个语义谓词，词面匹配双向都错，
且拿它做「删缓存 / 丢答案」的依据代价过大。改为两条结构化机制：

- **写入门禁**：检索环境退化（目标文档不在 `list_indexed_document_ids()` 内）时不写拒答缓存 ——
  那类判定说的是「系统当时状态不对」，不是「书里没有」；
- **读时失效**：拒答行 + 本轮带澄清 → 判定已不适用（精确通道安全，语义通道会误命中）。
  ⚠️ 拒答行的 `answer` 是 None，旧启发式**结构上够不到**这个真正需要它的场景。

改动后生成层通过数不变（31/38），但 `citation_coverage` 0.939→1.0、
`false_refusal_rate` 0.0606→0、`answer_relevancy` 0.784→0.837 ——
正是「不再把正常回答误判成拒答」的直接体现。

---

## 八、已知边界

诚实说明这套机制的局限：

1. **`faithfulness` / `answer_relevancy` 是规则代理**，基于 token 重叠和疑问类型匹配。
   识别不了语义等价（「李四被张三喜欢」vs「张三喜欢李四」）和细微事实矛盾。
2. **检索 gold 仍是弱标注为主**，人工抽检是按比例抽样，不是全量核对。
   抽检比例越高越可信，但成本线性上升。
3. **`hard_negative.py` 依赖 embedding 质量**，fake 模式下挖出的负样本没有语义意义。
4. **`judge.py` 需要 `human_labels.json`** 才能校准；没有人工标注时，
   judge 结果只能当参考，不能当验收依据。
5. **没有做显著性检验**，回归判断基于固定阈值。
   样本量小时（<30 题）单题翻转就能造成几个百分点的波动，建议先看 per-case 迁移再下结论。
6. **信息不足题的判定是纯词表匹配**（`run_eval.HONEST_UNANSWERABLE`），
   **漏词即假阴性**：模型明明诚实拒答，措辞不在表里就被判 FAIL。2026-09-17 实测踩过
   「没有提到 / 没有交代」缺失导致两道题误判。加题时若发现此类 FAIL，
   先人工读一遍答案再决定是补词表还是改用例。
   ⚠️ **这份词表只属于评测层，产品侧已经没有对应物了**（2026-09-17 重构）：
   产品对「信息不足」的判定全部改为结构化 —— 「检索为空」由 `judge` 节点看
   `chunks`，「有片段但答不了」由模型用 `request_clarification` 工具表态，
   缓存失效看拒答行 + 澄清标记。所以评测词表与产品行为之间存在**刻意的**不对称：
   它反映的是「什么样的措辞算诚实拒答」，不是「产品会怎么做路由」。
   改这份词表只影响评测判定，不会改变产品行为 —— 反之亦然，别把两者当成一回事。
7. **`contains` 短语定位题的 gold 规模取决于短语本身的稀有度**。
   短语越常见，命中的 chunk 越多，recall 越容易虚高。守则会拦住「未限定文档」的短语题，
   但拦不住「限定文档内仍然常见」的短语——出题时仍需人工确认命中数在 1~3 之间（见 §五）。
8. **端到端结果受缓存影响**。`CACHE_ENABLED=true` 时重复跑同一集会命中上一次的回答缓存，
   得到的是「已被判定过的答案」而非重新生成。要评估生成质量的真实变化，
   先清 `qa_cache`（或设 `CACHE_ENABLED=false`）再跑。
