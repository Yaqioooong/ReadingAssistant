# 黄金评测集扩容（2026-09-17）：引入《西游记》语料

## 一、为什么扩

`docs/CHANGES-20260917.md` 的「未完成项」里，P2-1（`chunk_size` 标定）被卡住的理由写得很直白：

> golden 仅 14 例，天花板效应明显，标定结果不可信 —— 前置条件：先把 golden 扩到 50~100 例

天花板效应已实测确认：扩集前检索层 14 例 **vector 与 hybrid 的 Recall@5 全是 1.0**，
两个系统打平，指标丧失了判别力。原因不难理解 —— 语料总共 4 个小文档、188 个 chunk
（`张三爱情故事.txt` 182 字节、`名词大动词.txt` 3.6 KB），任何像样的 embedding 都能全中。

于是引入《西游记》（人民文学出版社彩皮版）作为**唯一的大部头语料**：

| 项 | 值 |
| --- | --- |
| 解析结果 | 107 章、811,797 字 |
| 入库分块 | **1,332 chunk**（`chunk_size=800` / `overlap=100`） |
| 占全库比例 | 1332 / 1520 = **87.6%** |

语料规模从 188 chunk 涨到 1,520 chunk，且新增部分集中在一本书内部 ——
这正是自注意力式稠密检索最容易翻车的场景（同书内大量语义近似段落互相竞争）。

## 二、加了什么题

三层共新增 **64 例**（原 50 例 → 114 例）：

| 文件 | 层 | 原 | 现 | 新增 |
| --- | --- | ---: | ---: | ---: |
| `golden_set.json` | 生成 | 15 | 38 | +23 |
| `golden_multi_doc.json` | 生成 | 4 | 8 | +4 |
| `retrieval_gold.json` | 检索 | 14 | 34 | +20 |
| `retrieval_adversarial_gold.json` | 检索 | 17 | 30 | +13 |

生成层新增题按要求覆盖五类（正向基础 7 / 陷阱 5 / 多跳推断 5 / 信息不足 4 / 全库检索 2，
另有 4 例跨文档对照进 `golden_multi_doc.json`）。检索层按 `专名 / 引语 / 时间 / 人物 / 结构 / 数字`
分类，并刻意为三类难点各留了题：

- **前前言/前言/附录等非正文**（`chapter_index` 1~3、13）—— 这些块极易被正文淹没；
- **数字/数量事实**（法宝重量、卷数、年号）—— 稠密检索最弱的一类；
- **跨文档配对** —— 小文档在 1332 chunk 的大书旁边，验证「大书不淹没小书」。

新增检索题一律用 `contains` **短语原位定位** gold，而不是写死 `chunk_id`：
文档一旦重新入库（内容哈希变化 → chunk id 全变），写死的 id 会集体失效；
短语定位则自动跟随，且每道题的 gold 严格控制在 1~3 个 chunk，避免 gold 集膨胀导致 recall 恒为 1。

## 三、扩集逼出来的 4 个缺陷

这是本次扩容最大的收获 —— 指标从「一片 1.0」变成「29/38」的同时，暴露了 4 个此前被
小语料掩盖的问题。前三者在**生产路径**上，都是静默失效。

### 3.1 【生产】重复上传把文档打回 `indexing`，整本书在问答里消失

**现象**：23 道西游可答题**全部**返回空答案、零引用；老 15 题全过。

**根因**（`graph/ingest.py`）：
`add_book` 节点无条件执行 `index_status = 'indexing'`，但 `route_after_add` 对
「重复文件且未 `force`」直接 `END`，**没有任何节点会把它写回 `indexed`**。
而 QA 的检索白名单 `list_indexed_document_ids()` 只认 `indexed`：

```python
hits = [hit for hit in hits if hit.document_id in indexed_ids]   # qa.py
```

于是命中全部被过滤 → `hit=0` → 判为「信息不足」→ 转 HITL，日志里只有一行 HITL，**没有任何异常**。
用户视角就是「文件明明在库里，却一问三不知」。

**为什么以前没暴露**：只有重复上传才会触发，而 `eval/run_eval.py` 每轮都重传全部语料 ——
所以每个文档每轮都被踩一次。原本老 15 题靠回答缓存（`channel=exact`）返回旧答案，
缓存把故障盖住了；新题没有缓存，故障立刻现形。

**修复**：只有真的会走 `chunk_and_index` 时才占用 `indexing` 租约。
**回归测试**：`tests/test_graph.py::TestIngestGraph::test_duplicate_upload_keeps_document_indexed`
（已验证：还原旧代码即 FAIL `assert 'indexing' == 'indexed'`）。

### 3.2 【生产】租约时区错位，让对账永远无法自愈

`index_started_at` 是 `timestamp WITHOUT time zone`，写入时传的是 aware UTC datetime，
psycopg 会**按会话时区（Asia/Shanghai）换算后再落库**；而 `reconcile._as_utc()` 在列值不带
tzinfo 时直接按 UTC 解读。结果：租约被看成 **8 小时之后**才开始，卡死的 `indexing`
记录永远不会过期，`reconcile_index_status.py` 也就永远修不动它们 ——
实测 `--lease-timeout 0` 仍报「无卡死记录」。

**修复**：写 naive UTC。**注**：该缺陷在 sqlite 上无法复现（PG 特有的时区换算），
故未加单测，靠本文档记录。

### 3.3 【生产】numpy 标量混进 QA state，checkpoint 序列化直接崩

**现象**：`TypeError: Type is not msgpack serializable: numpy.float64`
把整轮端到端评测**打死在第 16 道题**，报告都写不出来。

**根因**：两条路径把 numpy 标量泄漏成了「分数」：

1. `ChromaVectorStore._distance_to_cosine()`：Chroma 的 distance 是 numpy 标量，
   经 `1 - dist/2` 与 `min/max` 运算后仍是 numpy 标量；
   `n_results` 较小时 Chroma 又可能返回内建 float —— 所以此坑是**间歇性**的。
2. `all_chunks()` 取回的 embedding 是 numpy 标量序列，稀疏路用其现算余弦时，
   `sum(x*y ...)` 把 numpy 类型一路透传。

分数进了 state 的 `chunks[*]['score']`，LangGraph checkpoint 用 ormsgpack 序列化 state 时就炸。
生产表现：**同一条问题偶尔 500**。

**修复**：在三处产出端收敛为内建 `float`（`_distance_to_cosine`、`all_chunks`、
`retriever/_qa 的 _cosine_similarity`）。
**回归测试**：`tests/test_vector_store.py::TestScoreTypeContract`（复现出与线上完全一致的
`TypeError: ... numpy.float64`）、`tests/test_rag.py::TestCosineSimilarityTypeContract`。

### 3.4 【评测】判官词表漏词，把诚实拒答判成 FAIL

信息不足题的判定是纯词表匹配（`run_eval.HONEST_UNANSWERABLE`）。
两例信息不足题模型明确答了「原文片段中**没有提到**…」，但词表里只有「未提及」没有「没有提到」，
于是被判成「应诚实说明无法回答，实际给出了内容」—— **假阴性**。

**修复**：补入模型实际高频使用的措辞（没有提到 / 没有交代 / 未交代 / 无从得知 …）。
因为只有 `expect_unanswerable` 的题走这份词表，放宽不会让可答题蒙混过关
（已验证：编造答案仍判 FAIL）。

### 附：新增黄金集文件守卫测试

`eval/README.md` 一直把「加题」写成纯手工编辑 JSON，而 `tests/` 里只有「代码」测试，
**没有任何测试盯数据集文件**。踩过的静默坑（`document` 拼错 → 静默降级；
`contains` 写错 → gold 为空被 `[skip]`；可答题漏写关键词 → 判定恒真）现在由
`tests/test_eval_golden_files.py`（34 条）拦住，不连库、不调模型、秒级完成。

其中还钉住了本次踩到的**文件名契约**：`doc_map` 的键是语料目录里的真实文件名，
所以 `eval/uploads/` 里的文件名必须与 gold 的 `document` 字段逐字符一致。
当前 fixture 与生产入库使用的文件名完全相同：

```
eval/uploads/西游记 (吴承恩) (z-library.sk, 1lib.sk, z-lib.sk).epub
```

（与 `uploads/` 下的原书 sha256 一致 → 命中分层去重 → 映射到同一 `document_id=7`，
不会重复入库。实测把 fixture 改名会直接 FAIL 5 条，而不是安静地少统计 20 道题。）

## 四、修复前后指标

### 检索层（`--mode real`，filtered 口径）

| 数据集 | 例数 | 系统 | Recall@5 | 扩集前 |
| --- | ---: | --- | ---: | --- |
| `retrieval_gold` | 34 | vector | 0.824 | **1.000**（14 例） |
| | | hybrid | 0.838 | **1.000**（14 例） |
| `retrieval_adversarial_gold` | 30 | vector | 0.844 | — |
| | | hybrid | 0.883 | — |

两个系统不再打平（hybrid 领先 vector 1.4~3.9 个点），指标恢复了判别力。
修复后两次全量检索评测 **零 `[skip]`**（每道题都定位到了 gold）。

### 生成层（`--mode real`，38 例）

| 指标 | 修复判官词表前 | 修复后 |
| --- | ---: | ---: |
| passed | 29/38 | **31/38** |
| pass_rate | 0.763 | **0.816** |
| unanswerable_recognition | 0.600 | **1.000** |
| answerable_accuracy | 0.788 | 0.788 |
| citation_coverage | 0.939 | 0.939 |
| hallucination_rate | **0.000** | **0.000** |

老 15 题全部保持通过（无回归）。
`unanswerable_recognition` 从 0.6 直接升到 1.0 就是 3.4 那一处词表漏词造成的 ——
**产品侧一行没改**，只是判官不再误读模型的诚实拒答。

## 五、遗留：7 道西游题的检索未召回

38 例中 7 道 FAIL（其余 2 道原 FAIL 已确认是 3.4 的判官漏词，修词表后转为 PASS），
全部是**真实检索未召回**。值得强调的是模型的反应是正确的 —— 它没有编：

> 原文片段中未出现孙悟空漂洋过海、寻仙访道、最终拜师这一情节，
> 所给内容均为他保唐僧取经途中向神仙求助的片段，无法据此回答他拜谁为师。

`hallucination_rate = 0` 佐证了这一点：答不出来时它老实说答不出来。

已定位到的失败模式（供 P2-1 `chunk_size` 标定参考）：

| 题 | 目标事实 | 现象 |
| --- | --- | --- |
| xjy-01 | 灵台方寸山·须菩提祖师 | 答案 chunk 不在 top-20（全书仅出现 1 次，被取经途中的「祖师/神仙」段落压过） |
| xjy-13 | 斜月三星洞 | 同上（词面极稀有，语义上却被大量「师父/传授」段落竞争） |
| xjy-14 | 金角银角=看金炉/看银炉的童子 | 检索为空（`hit=0`，直接转 HITL） |
| xjy-10 / 12 / 16 | 文殊菩萨 / 六耳猕猴 / 金鱼·莲花池 | 召回到相关段落，但目标事实所在 chunk 落在 top-6 之外 |

这 7 例恰恰说明了为什么原来那套 14 例的集子标定不出 `chunk_size`：
小语料里所有 chunk 都在 top-5，`chunk_size` 怎么调都是满分。

## 六、复现方式

```bash
uv run pytest tests/test_eval_golden_files.py -q          # 黄金集结构守卫（秒级）
uv run pytest tests/test_vector_store.py tests/test_graph.py -q   # 本次新增的 3 组回归

uv run python -m eval.run_retrieval_eval --mode real
uv run python -m eval.run_retrieval_eval --mode real --gold-file eval/retrieval_adversarial_gold.json
uv run python -m eval.run_eval --mode real
```

⚠️ 跑生成层前建议清 `qa_cache`：命中缓存时拿到的是「上一次已被判定过的答案」，
会掩盖生成质量的变化（见 `eval/README.md` §八）。
