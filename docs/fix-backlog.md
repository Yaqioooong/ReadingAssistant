# 待修复清单（Fix Backlog）

> 📄 **改动汇总见 [`docs/CHANGES-20260917.md`](CHANGES-20260917.md)** —— 含完整文件清单、数据侧操作、
> 设计决策取舍、过程额外发现的 5 个问题。本文档专注「待修项 + 完成记录」。

> 来源：2026-09-14 ~ 09-17 代码答疑与深挖会话
> 核查时间：2026-09-17 11:11（每项状态均经脚本实测）
> 状态：**12 项已完成 11 项**；仅 P2-1（chunk_size 标定）待做（需先扩 golden 集）
> 完成时间：2026-09-17 11:30

## 🔍 一句话病根

**写进去的机制，读的一侧没有实现。**

| 机制 | 写入方 | 读取方 |
|---|---|---|
| `index_status` 状态机 | ✅ 有（4 个赋值点） | ❌ **无任何消费方** |
| `vector_store.delete()` | ✅ 有（2 处实现） | ❌ **无调用点** |
| 表格原子化 | ✅ 有 | ❌ 无上限校验 |
| `chunk_id` 位置语义 | ✅ 有 | ❌ 无人校验内容一致性 |

> 这四件事是同一种失败模式：**生产者建好了，消费者没接**。
> 后果统一表现为「静默」——不报错、不告警、功能表面正常。

---

## P0 · 数据正确性（静默错误，会污染答案）

### ✅ P0-1 · HNSW 向量索引已退化，dense 静默漏召真 top-1

| | |
|---|---|
| **状态** | ❌ 未修（实测仍复现） |
| **位置** | `rag/chroma_db/`（索引数据） |
| **工作量** | 小（半天，含验证） |

**现象**
```
查询「王五喜欢李四吗？」 gold=doc2-0
  精确 L2 排名 = 1        ← 全库 1520 chunk 中就是最近邻
  chroma 实际排名: k≤100 找不到，k=200 才出现
```

**证据**：同向量、**同默认参数**重建干净索引 → `k=6` 就正确召回。
→ 不是参数问题，是 HNSW 图结构退化。

**影响**：14 例 golden 中 **2 例（14%）** 的 gold 被静默漏召，而它们的精确排名是第 1、第 2 名。非近似检索的正常误差。

**推测成因**：反复 reindex —— Chroma 的 HNSW 删除是**标记**而非物理摘除，反复 upsert/delete 渐进破坏图连通性。

**修法**
```python
# 无需重新 embedding —— all_chunks() 里向量都在，直接搬进新集合
old = vs.all_chunks()                    # 拿全量向量
new_col.upsert(ids=..., embeddings=...)  # 写入新集合
# 验证四情景对比后再切换
```

**验收**：14 例 golden 全部命中（当前 13/14），且 `doc2-0` / `doc3-0` 在 top-6 命中。

---

### ✅ P0-2 · 大表格无上限 → 撞 embed 窗口 → 整本书入库失败

| | |
|---|---|
| **状态** | ❌ 未修 |
| **位置** | `rag/chunking.py:81-94`（表格原子通道） |
| **工作量** | 小（半天，含测试） |

**现象**
```
行数    表长       chunk数   最大块      倍数
5000    499013    1         499013     623.8x   ⚠ 无界
```

**影响链**
```
大表格 499,013 字符
  → embed_documents() 批量调用 DashScope
  → ❌ 400: Range of input length should be [1, 8192]
  → 被 chunk_and_index 的 except 捕获
  → 整篇文档标记 'failed'，re-raise
```
**一本书含附录表/数据表 → 整本书入库失败**（批量调用，一个超限输入炸掉整批）。
且 `'failed'` 状态同样无人消费（见 P0-3）。

**修法**：原子性 + 上限，两个都要
```python
_MAX_TABLE_CHARS = 3000   # 或按 token 折算

if kind == 'table':
    if len(segment) <= _MAX_TABLE_CHARS:
        chunks.append(TextChunk(text=segment, block_type='table', ...))
    else:
        # 退化路径：按行切，但每块重复表头
        header, _, body = segment.partition('\n| --- |')
        for batch in _batch_rows(body, _MAX_TABLE_CHARS - len(header)):
            chunks.append(TextChunk(text=f'{header}\n| --- |{batch}', block_type='table'))
```
> **核心原则：可以切，但表头必须每块都带**（行组 + 表头复制）。

**验收**：5000 行表 → 多块，每块含表头，单块 ≤ 上限，`embed_documents` 不报 400。

---

### ✅ P0-3 · 3/6 文档卡死在 `index_status='indexing'`，无人察觉

| | |
|---|---|
| **状态** | ❌ 未修（实测仍卡死） |
| **位置** | `graph/ingest.py:60`（写入）、检索层（应读未读） |
| **工作量** | 小（清洗）+ 中（接通消费方） |

**现象**
```
doc2   indexing   chunk_count=1     张三爱情故事.txt
doc3   indexing   chunk_count=2     名词大动词.txt
doc4   indexing   chunk_count=3     sample_duplicate.txt
```
向量完整可检索，功能表面正常 → 一直没人发现。

**成因**：同一本书在 09-07 与 09-09 各上传一次 → 第二次 file_hash 命中 → 设 `'indexing'` → 中途中断 → 状态再未被翻转。

**影响**
- `index_status` 作为崩溃恢复锚点**已被污染（50% 假阳性）** → 启动对账方案直接跑会误报一半
- 崩溃后卡死的文档对用户完全不可见

**修法（两步，顺序不能反）**
1. **先清洗历史假阳性** —— 对 `indexing` 记录用向量库实际存在性校验后修正为 `indexed`
2. **再接消费方** —— 启动对账扫 `indexing` + 超时阈值 → 标 `stale`；检索层过滤非 `indexed`

**验收**：启动无假阳性告警；人为 kill 一次 reindex 后能被检出。

---

### ✅ P0-4 · 检索层不过滤 `index_status`，`indexing` 中的文档照常被检索

| | |
|---|---|
| **状态** | ❌ 未修 |
| **位置** | `rag/retriever.py`、`graph/qa.py`（零引用） |
| **工作量** | 小（1~2 小时） |

**证据**
```bash
$ grep -rn "index_status" src/reading_assistant/rag/ src/reading_assistant/graph/qa.py
（空）
```

**影响**：入库中断/失败的文档，其**部分向量**仍参与检索并作为引用来源返回。

**修法**：检索入口或 rerank 层按 `document_id` 批量查一次状态，过滤非 `indexed`（可加缓存，避免每问一次 DB）。
> 注意：与 P0-3 配套 —— 状态未清洗前直接过滤会**误杀**那 3 个文档。

---

## P1 · 可靠性（有损但不静默）

### ✅ P1-1 · reindex 崩溃无自动恢复机制

| | |
|---|---|
| **状态** | ❌ 未修（设计缺口） |
| **位置** | `api/app.py:34`（lifespan）、`graph/ingest.py:88-114` |
| **工作量** | 中（1~2 天） |

**现状**：崩溃窗口 = `vector_store.add()` 完成后、PG commit `indexed` 前。
- 进程崩溃不走 `except` → 连 `'failed'` 都写不上 → 状态**永久卡在 `indexing`**
- `lifespan` 只调 `init_db()` 建表，**不扫**残留记录
- 恢复全靠手动重跑 reindex + chunk_id 幂等覆盖

**前置条件**：源文件必须还在，否则路由返回 `409 sourcefile not found`。
该 409 对用户毫无解释力——他不知道自己遇到的是崩溃残留。

**修法**：见 P0-3 两步法 + 友好的 409 文案。

---

### ✅ P1-2 · `vector_store.delete()` 是死代码；chunk 数变少会留孤儿

| | |
|---|---|
| **状态** | ❌ 未修（**当前库里 0 孤儿**，雷已上膛未击发） |
| **位置** | `storage/vector_store.py:48,107,171` |
| **工作量** | 中（取决于方案） |

**证据**：全仓库 `.delete(` 命中的都是 SQLAlchemy `session.delete()` 或 FastAPI 装饰器。

**实测复现**（doc5，临时集合）
```
原分块 chunk_size=800  → 181 个
重切   chunk_size=2000 →  66 个
「reindex」后 count = 181   ← PG 只记录 66，孤儿 115

用孤儿原文检索 top-6 → 6 条里 5 条是幽灵
元数据（document_id/chapter）完全正常，无法区分
```

**为什么"幽灵"比"全丢"更危险**

| | 先 delete → 崩 | 只 upsert → 幽灵 |
|---|---|---|
| 检索 | 返回空 | 返回**旧边界的错误文本** |
| 模型 | 诚实说查不到 | 自信地用错误语料作答 |
| 性质 | 响亮的缺失 | **静默的谎言** |

> 与项目自身价值观一致（`interview-qa-answers.md:176`）：

**修法：两个都不选，走版本化 id**
```
id = doc{content_hash[:8]}-{index}
写完新版本 → 再删旧版本 → 原子切换
```
一次解决：孤儿残留 / PG 重建后 id 碰撞 / chunk 数变少。

**验收**：改 chunk_size 重切后，向量库 count == PG chunk_count == 新 chunk 数。

---

### ✅ P1-3 · `chunk_and_index` 中 `get_document` 无判空

| | |
|---|---|
| **状态** | ❌ 未修 |
| **位置** | `graph/ingest.py:104` |
| **工作量** | 极小（5 分钟） |

```python
document = get_document(session, document_id)
document.chunk_count = len(chunks)     # ← 记录不存在时 AttributeError
```

**影响**：文档记录缺失时抛 `AttributeError` → 被外层 `except` 捕获 → 在一条**不存在的记录上**标记 `failed`（静默失败）。

**修法**：加判空 + 明确日志。

---

## P2 · 检索质量

### ☐ P2-1 · `chunk_size=800` / `chunk_overlap=100` 从未标定

| | |
|---|---|
| **状态** | ❌ 未标定 |
| **位置** | `config.py:66-67`、`config/chroma.yml` |
| **工作量** | 中（1~2 天，需扩 golden 集） |

**证据**：`grep -rn "chunk_size" eval/` → **空**。`eval/` 只对比检索模式，分块参数全程固定。

> ⚠️ **`interview-qa-answers.md:160` 的既有说法不成立**
> 原文："对齐 embedding token 窗口（中文约 700~900 字）"
> 实测：`text-embedding-v4` 上限 **8192 tokens**；20000 字符才报 400。
> **800 字符只用窗口 ~10%，余量 10 倍。**
> 真实约束是检索粒度（精度 trade-off），非硬件限制。

**修法**
```infographic
infographic sequence-steps-simple
data
  title chunk_size 标定方案
  items
    - label 锁定变量
      desc 只变 chunk_size，检索模式/模型/k 全锁死
    - label 扫网格
      desc 400/600/800/1200/1600 × overlap 0/10%/20%
    - label 双层指标
      desc Recall@6 + 块内答案完整率
    - label 计成本
      desc 块数×embedding 成本、top-k 被巨块占满
```

**前置**：golden 集需从 14 例扩到 50~100 例（当前天花板效应明显）。

---

### ✅ P2-2 · `rrf_k` 与 `hybrid_pool_size` 隐式耦合

| | |
|---|---|
| **状态** | ❌ 未修 |
| **位置** | `config/chroma.yml`（`rrf_k: 60` / `hybrid_pool_size: 50`） |
| **工作量** | 极小（加注释 + 断言） |

**结构性定理**
```
双路最差分 = 2/(k+pool)      单路最优分 = 1/(k+1)
双路必然碾压单路 ⟺ k > pool - 2
```
当前 `k=60 > 48`，余量仅 12 → RRF 输出被严格分区，**BM25 独有条目永远进不了 top-6**。

⚠️ **地雷**：`hybrid_pool_size` 调到 70 → 阈值变 68 → `k=60 < 68` → **融合语义静默翻转**。

**修法**：注释说明 + 启动断言 `rrf_k > hybrid_pool_size - 2`，或改成不依赖 pool 的融合方式。

---

### ✅ P2-3 · 同一个 `min_score` 作用在两种量纲上

| | |
|---|---|
| **状态** | ❌ 未修 |
| **位置** | `rag/retriever.py:216-228` |
| **工作量** | 小（1~2 小时） |

```python
score = hit.score        # dense: 1/(1+L2) 换算分
score = cosine           # bm25: 原始余弦
if score < min_score:    # ← 同一个 0.45 常量
```

| 支路 | 公式 | `0.45` 实际含义 |
|---|---|---|
| dense | `1/(3-2cos)` | **cos ≥ 0.389** |
| bm25 | `cos` | **cos ≥ 0.450** |

> 换算成立的前提是 Chroma 用 **L2 空间**（当前 `metadata: None` → 默认 L2，实测换算精确匹配）。
> **若集合改成 `hnsw:space=cosine`**：`converted = 1/(2-cos)` → `0.45 ⟺ cos ≥ -0.222` → **dense 闸门变空操作**。

**修法**：统一转余弦后比较，或拆成两个独立配置项。

---

## P3 · 安全

### ✅ P3-1 · `upload_book` 无路径白名单（任意文件可入库）

| | |
|---|---|
| **状态** | ❌ 未修（刀一遗留） |
| **位置** | `mcp/server.py:68-100` |
| **工作量** | 小（需先定策略） |

**攻击链（全部使用本项目合法工具）**
```
① upload_book('/Users/aasing/.ssh/id_rsa.txt')
   → 通过（.txt 在白名单）→ 复制进 uploads/ → 解析 → 分块 → embedding 入 Chroma
② ask_book('我上传的那个 txt 文件内容是什么？')
   → 检索命中，原文返回
```
**无一处 description 被篡改。**

**修法**：限制在 `uploads/` 或用户书籍目录内；或引入显式可配置白名单。
> ⚠️ 行为变更 —— 会限制合法上传路径（如从 `~/Downloads` 传书），需先定策略。
> 相关：RAG 语料本身即注入载体（Indirect Prompt Injection），建议入库前扫 prompt injection 特征。

---

## P4 · 文档维护

### ✅ P4-1 · 面试稿题 10 表格描述已过时

| | |
|---|---|
| **状态** | ❌ 未更新 |
| **位置** | `docs/interview-qa-answers.md` 题 10 |
| **工作量** | 极小（10 分钟） |

**原文**：「表格：现在被压平成文本流、结构全丢」

**实际**：表格结构化（哨兵 + Markdown 原子块）**早已完成**，且有测试覆盖。

> ⚠️ 按旧稿背 = 主动交代一个已修好的缺陷，还错过展示机会。
> 应改为：哨兵协议 + 原子块设计 + **无界漏洞**（P0-2）的诚实交代。

---

## ✅ 已修复

### ✅ P0-0 · MCP 层依赖 HTTP 层（分层倒挂）

| | |
|---|---|
| **状态** | ✅ 已修复（2026-09-16） |
| **验证** | `pytest tests/test_layering.py` → 2 passed；全量 170 passed |

**问题**：`mcp/server.py` 从 `api.deps` 导入 5 个依赖，把整个 FastAPI 拖进 stdio 进程。

**修复**
| 文件 | 动作 |
|---|---|
| `src/reading_assistant/runtime.py` | 🆕 5 个纯工厂下沉 |
| `src/reading_assistant/api/deps.py` | 精简为「请求级会话 + re-export」 |
| `src/reading_assistant/mcp/server.py` | 导入源 `api.deps` → `runtime` |
| `tests/test_layering.py` | 🆕 子进程 import 检测，钉住分层约束 |

**关键手法**：re-export 同一个函数对象 → `dependency_overrides` / `cache_clear` / `monkeypatch` 全部零改动。

---

## 建议执行顺序

```infographic
infographic sequence-steps-simple
data
  title 修复优先级
  items
    - label P0-1 重建索引
      desc 唯一真正改变检索结果的修复
    - label P0-3 清洗状态
      desc 必须先做，否则阻塞 P0-4
    - label P0-2 表格上限
      desc 防整本书入库失败
    - label P0-4 检索过滤
      desc 依赖 P0-3 的状态干净
    - label P1 版本化 id
      desc 一次解决孤儿/碰撞/重切
    - label P2 标定与解耦
      desc 需先扩 golden 集
```

**最短见效路径**：P0-1（重建索引）→ 立刻见到 14/14。
**阻塞关系**：P0-4 必须等 P0-3 状态清洗完，否则误杀 3 个文档。
**需先决策**：P1-2 方案选型（版本化 id vs 显式 delete）、P3-1 白名单策略、P2-1 是否值得扩 golden 集。

---

## 附：核查脚本输出（2026-09-17 11:11）

```
[1] MCP 依赖 fastapi        : ✅ 已修复
[2] 卡在 indexing 的文档    : ❌ [(2, '张三爱情故事.txt'), (3, '名词大动词.txt'), (4, 'sample_duplicate.txt')]
[3] 孤儿/账实不符           : ✅ 无
[4] vector_store.delete 调用: ✅ 仍无调用点（死代码）
[5] HNSW 索引退化           : ❌ doc2-0(真top1)未进 top50
[6] 表格 chunk 上限         : ❌ 无上限
[7] 检索过滤 index_status   : ❌ 不过滤
[8] upload_book 路径白名单  : ❌ 无
[9] chunk_size 标定         : ❌ 未标定
[10] 题10 表格过时描述      : ❌ 仍存在
[11] ingest get_document 判空: ❌ 无判空
```

---

# ✅ 修复完成记录（2026-09-17）

**验证**：`PYTHONPATH=. pytest -q` → **308 passed**（改前 255）；改动文件 `ruff check` 全过。

## 代码改动

| # | 项 | 改动 | 位置 |
|---|---|---|---|
| P0-1 | HNSW 退化 | 新增蓝绿重建脚本，带**上线前验证** | `scripts/rebuild_vector_index.py` 🆕 |
| P0-2 | 表格无上限 | 表头感知切分，`_MAX_TABLE_CHARS=3000` | `rag/chunking.py` |
| P0-3 | 状态卡死 | `index_started_at` 租约 + 启动对账 | `storage/models.py`、`reconcile.py` 🆕、`api/app.py` |
| P0-4 | 检索不过滤 | retrieve 节点按 `indexed` 白名单过滤 | `graph/qa.py` |
| P1-1 | 无崩溃恢复 | 同上（租约对账 + 手动脚本） | `scripts/reconcile_index_status.py` 🆕 |
| P1-2 | 孤儿向量 | 内容寻址 id + `delete_stale` | `rag/chunking.py`、`storage/vector_store.py`、`graph/ingest.py` |
| P1-3 | 缺判空 | 明确异常 `DocumentNotFoundError` | `graph/ingest.py` |
| P2-2 | rrf 耦合 | 不变量文档化 + 构建时校验告警 | `rag/retriever.py`、`config/chroma.yml` |
| P2-3 | 分数尺度 | 统一到余弦（按集合 space 换算） | `storage/vector_store.py`、`rag/retriever.py` |
| P3-1 | 路径穿越 | 三层防御（规范化 + 白名单 + 敏感黑名单） | `mcp/policy.py` 🆕、`mcp/server.py` |
| P4-1 | 文档过时 | 题 10 表格段订正 | `docs/interview-qa-answers.md` |

## 测试新增（本次会话补齐）

第一路实现 agent 未写新测试，由我补齐 **57 个用例**：

| 文件 | 用例 | 覆盖 |
|---|---|---|
| `tests/test_index_integrity.py` 🆕 | 33 | 版本化 id、`delete_stale`（双后端）、分数换算、RRF 不变量 |
| `tests/test_reconcile_index_status.py` 🆕 | 11 | 租约过期、完整/不完整、dry-run、幂等 |
| `tests/test_table_chunking.py` | +10 | 表头复制、不腰斩、不丢行、5000 行回归 |
| `tests/test_layering.py` 🆕 | 2 | MCP 不得依赖 fastapi |
| `tests/test_mcp_policy.py` 🆕 | 25 | 符号链接穿越、敏感路径、env 覆盖 |

## 数据侧操作（真实库，已备份）

备份：`/tmp/readingassistant-backup-<ts>/`（pg_dump 1187 行 + chroma 39M）

| 步骤 | 结果 |
|---|---|
| 1. chunk id 迁移 | 1520 块 → 新 id（`doc5-18` → `doc5-b2f1a90c-18`），零残留 |
| 2. 缓存引用迁移 | 52 条记录 / **90 个引用**（迁移脚本原先遗漏的派生数据） |
| 3. 状态对账 | 3 条卡死 `indexing` → `indexed`（向量完整），复跑幂等 |
| 4. 索引蓝绿重建 | golden 命中率 **0.857 → 1.000** |

## 关键实测证据

```
# P0-1 HNSW 退化修复前后
修复前：查询「王五喜欢李四吗？」gold doc2-0 精确 L2 排名=1，
        chroma top_k≤100 召不到，k=200 才出现
修复后：top_k=6 即为 top1

# P0-2 表格上限（5000 行表）
修复前：单 chunk 499,013 字符 = 623.8x chunk_size → 撞 8192 token 上限
修复后：多块，每块 ≤3000 且都带表头

# P2-3 分数尺度
修复前：dense 用 1/(1+L2dist)，bm25 用原始余弦，共用 0.45
        实测换算 0.45 ⟺ dense 的 cos≥0.389 / bm25 的 cos≥0.450
修复后：统一余弦（L2+归一化 → cos = 1 - dist/2），实测 dist=0.7134 → cos=0.6433
```

## 过程中发现并修掉的两个额外问题

1. **`reconcile_index_status.py` 缺 schema 兜底** —— 旧库上因 `index_started_at` 列不存在直接报错。已补 `init_db(engine)`（幂等）。
2. **id 迁移遗漏派生引用** —— 问答缓存 `citations[].chunk_id` 嵌的是旧格式 id，迁移后成悬空引用，会让答案的「跳转到出处」失效。已把缓存引用迁移并入 `migrate_chunk_ids.py`（同为 dry-run 默认）。

## 「先修复后启用」的依赖顺序（重要）

`P0-4 检索过滤` 依赖 `P0-3 状态清洗`：若先启用过滤，那 3 个「向量完整但状态卡死」的
正常文档会被检索层**误杀**。本次已按 `迁移 → 清洗 → 重建 → 过滤` 顺序执行。

## 评测验证（2026-09-17 13:32）

`PYTHONPATH=. .venv/bin/python -m eval.run_retrieval_eval --mode real` → **vector / hybrid 全指标 1.000**

| 版本 | vector MRR | hybrid MRR |
|---|---|---|
| 修复前（09-16 15:28） | 1.000 | 1.000 |
| 修复后（09-17 13:32） | 1.000 | 1.000 |

MRR 无回归（recall/mrr/map 均已饱和 1.0）。

**逐指标复核发现 precision 实际提升**（初版只比 MRR，属漏报，已订正）：

| 指标 | 修复前 | 修复后 | 归因 |
|---|---|---|---|
| precision@5/10/20（vector & hybrid） | 0.9524 | **1.0000** | P2-3 分数尺度统一 |

机制：旧 dense 用 `1/(1+L2dist)`，`min_score=0.45` 等效 `cos ≥ 0.3889`；统一到余弦后等效 `cos ≥ 0.4500`，阈值收紧 → 剔除边缘非 gold 块。
验证实验：同索引只变阈值语义，precision@5 `0.9464 → 1.0000`。

官方评测口径（带 `document_id` 过滤）**测不出 HNSW 退化** —— 已补 cross 口径。

同口径 A/B（跨书库，无过滤）才见真实收益：

| 索引 | 命中率 |
|---|---|
| 旧索引（退化） | 11/14 = 0.786 |
| 当前索引（重建） | **14/14 = 1.000** |

详见 `docs/retrieval-eval-report-20260917.md`；复检工具 `scripts/compare_index_health.py`。

### 新增待办（本次评测暴露）

- ✅ **回归门禁假警报**（2026-09-17 修复）—— 全局绝对阈值 0.02 对 `avg_latency_ms`
  （毫秒）恒为真，历史 5/5 次相邻对比被误判，而质量全程未变。改为按指标族分策略
  （`_THRESHOLD_OVERRIDES`：相对 + 绝对同时超阈）。测试 `tests/test_eval_compare_thresholds.py`（22 例），
  README「阈值怎么定」段已订正。

- ✅ **已给 `run_retrieval_eval.py` 增加跨书库口径**（2026-09-17）—— 现在同时跑 `filtered` / `cross`；报告新增顶层键 `cross_summary`，per_case 新增 `*_cross_mrr`；CLI 加 `--no-cross`。配套测试 `tests/test_eval_retrieval_scope.py`（8 例）
- ☐ **索引健康检查纳入常规运维** —— 用 `scripts/compare_index_health.py` 或定期跑重建脚本的 golden 校验

### id 迁移的派生引用点（补记）

| # | 派生引用 | 数量 | 状态 |
|---|---|---|---|
| 1 | 向量库 chunk 本体 | 1520 | ✅ |
| 2 | 问答缓存 `citations[].chunk_id` | 90 | ✅ |
| 3 | 评测集 JSON（gold + hard_neg） | 192 | ✅ |

`eval/reports/*` **刻意不改**：历史报告是过去结果的快照。

## 仍未完成

- **P2-1 `chunk_size` 标定** —— 需先把 golden 集从 14 例扩到 50~100 例（当前天花板效应明显），否则标定结果不可信
- **P3-1 白名单默认值** —— 为兼容既有测试，默认根目录含 `<project_root>`；如需收紧到严格 4 条，需放宽 `tests/test_mcp_server.py` 的修改禁令（或允许其注入 `READINGASSISTANT_MCP_UPLOAD_ROOTS`）
- **既有 lint 债务** —— `scripts/deploy_trace.py` 4 个错误、`experiments/multi_agent.py` 5 个错误，均为改动前既存
