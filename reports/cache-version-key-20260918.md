# 缓存键的「文档版本」维度：从装饰品到真正执行

> 2026-09-18 · 起因：审阅 `qa.py:439` 时问「查缓存为什么要用 content_hash？」
> 结论：它**看起来在防一件事，实际一件都没防住**，而且约束与代码矛盾。
> 本次把它做成真东西，并修掉过程中自己踩出的一个迁移缺陷。

## 1. 设计意图是合理的

缓存键概念上是 **（问题，文档版本，文档范围）**。动机很正当：书的内容换了版本，
旧答案是从另一版原文推出来的，不能继续发。所以读写都按
`(question_hash, content_hash, document_id)` 匹配。

## 2. 取证：三本账

### 账一 单书路径 —— 冗余

`question_hash` 里已经拼了 `str(document_id)`（`qa.py` 构造处），而
`Document.content_hash` 只在 `add_book` 插入时赋值、**且自身 UNIQUE**
（`models.py`；`service.py` 是唯一写入点，全仓无 UPDATE 路径）
→ 同一个 `document_id` 的 `content_hash` 恒定
→ 精确键里多加的这一维**永远命中同一行，筛不掉任何东西**。

### 账二 约束与代码矛盾 —— 地雷（已拆）

`QaCacheEntry.question_hash` 是**单列 UNIQUE**（`ix_qa_cache_question_hash`），
`content_hash` 只有普通索引。即：**代码声明 content_hash 参与身份，DB 却禁止两行只差 content_hash。**

实测（探针）：把同一 `document_id` 的 `content_hash` 从 v1 改成 v2 后再问同一问题

```
读路径按 (qhash, v2) 判 miss   ← 正确
写路径插新行                   ← 撞 UNIQUE constraint failed: qa_cache.question_hash
```

该调用点无 try/except → **整轮问答失败，且永久复发**（旧行还在，下次同样撞）。

### 账三 全库路径 —— 维度直接不存在（已补）

```python
def _document_content_hash(session, document_id):
    if document_id is None:
        return ''          # 全库问答：版本维度 = 空串
```

于是**上传 / 删除 / 重索引任何一本书，全库缓存照样命中**。实测：

```
全库提问 → 上传一本新书 → 再问同一问题 → cache_hit=True，仍返回旧答案（TTL 30 天）
```

最坏形态是**全库拒答被长期缓存**：某本书还没入库时问它 → 「原文没提到」写进缓存 →
书入库完成后重问，**仍然答「没提到」**。

而全库是主力用法（当日 api 日志 `docs=[]` 614 次 vs 指定书 343 次），
生产库 122 条缓存里全库占 42 条、其中拒答 19 条。语料变化在本项目还很频繁
（重复上传、重索引都会改状态）。

### 旁证：没有任何测试覆盖「改版 → 失效」

`tests/` 里 `content_hash` 全是被当静态 fixture 值写入的，没有一条断言
「文档版本变化后缓存应失效」。所以雷没被踩到、洞没被暴露 —— 不是运气，
是这个维度**从来没被真正执行过**。

## 3. 修法

| # | 改动 | 文件 |
|---|---|---|
| 1 | 键改成 **`(question_hash, content_hash)` 复合唯一索引** `uq_qa_cache_question_version`；`question_hash` 退为普通索引 | `storage/models.py` |
| 2 | 迁移 `_ensure_qa_cache_version_key`：改约束必须 DROP INDEX（不是 DROP CONSTRAINT），且**收敛式** | `storage/database.py` |
| 3 | 全库路径改为**语料指纹** `_corpus_fingerprint`：已入库文档 `(id, content_hash)` 集合的哈希 | `graph/qa.py` |
| 4 | 回归测试 12 条 | `tests/test_cache_version_key.py` |

### 为什么选「复合唯一」而不是「upsert 覆盖」

读路径本来就是按 `(question_hash, content_hash)` 一起查的 —— 精确层
（`get_qa_cache_entry`）如此，语义层（`list_qa_cache_entries`）也按 `content_hash` 筛候选。
**读的写法已经假定复合语义，只有 DB 约束与之矛盾。** 改约束是让代码说的成真；
改成 upsert 则要反过来删掉读里的版本判断，且语义层那个 `content_hash` 筛选会变成空操作。
本项目的取向是「让声明的维度真正被执行」，故选前者。

代价：改版后旧版本行会留下（由既有 `prune_qa_cache` 淘汰），换来「旧版本永不误命中」。

### 语料指纹只统计 `index_status == 'indexed'`

与检索层共用同一信号（`list_indexed_document_ids` 也只认这批），
所以「库里有没有这本书」与「这次回答可不可信」不会各自漂移。
尚未索引完的书不改变指纹 —— 与它在检索里不可见保持一致。

## 4. 迁移：从「条件式」改成「收敛式」（这是本次最有价值的一课）

我第一版迁移写成「发现旧唯一索引就删、发现缺复合索引才建」——**只在首次迁移时正确**。

真实后果：用户**开着 `--reload` 的开发服务器**，我每存一次 `.py` 它就重启并跑一遍 `init_db`。
那一版实现把真实库的 `ix_qa_cache_question_hash` 删掉后就再没补回来，留下缺索引的中间态。
（证据：api 日志里 16:18:26 一秒内十次 `ReadingAssistant API 启动`；事后查真实库，
`ix_qa_cache_question_hash` 完全消失。）

修成**收敛式**：无论中间历史如何，每次都把目标 schema 补齐。
迁移在每次启动都执行，它应当把库**收敛到**目标形态，而不是依赖「这是第一次」。

对应测试两条：`test_migration_repairs_partial_migration_state`（中间态自愈）与
`test_migrated_schema_matches_fresh_schema`（迁移库与全新库索引集合必须一致 ——
正是它咬出了上面那个 bug）。

## 5. 验证

| 验证 | 结果 |
|---|---|
| 全量测试 | **463 passed**（新增 12 条） |
| ruff | All checks passed |
| 真实 Postgres 迁移 | 幂等；122 行无损；schema 与全新库**完全一致** |
| 真实 PG 端到端 | 同一 `question_hash` 写入两个版本**成功**（旧约束下必炸）；事务内验证后回滚，生产库未改动 |
| 语料指纹开销 | **0.165 ms/次**（6 本已入库文档），可忽略 |
| 评测链路回归 | intent eval fake 87.88% / LLM 兜底 10 次，与改造前一致 |

修复后真实库索引：

```
ix_qa_cache_question_hash        (question_hash)                     unique=False
uq_qa_cache_question_version     (question_hash, content_hash)       unique=True
ix_qa_cache_content_hash         (content_hash)                      unique=False
ix_qa_cache_document_id          (document_id)                       unique=False
```

## 6. 覆盖的回归（`tests/test_cache_version_key.py`，12 条）

单书：改版失效 / 两版本各占一行 / 同版本仍命中。
全库：语料变化失效 / **陈旧拒答失效** / 同语料仍命中 / 未索引文档不影响指纹。
语义层：候选按版本筛（异版本不进入候选）。
迁移：替换旧唯一索引 / 中间态自愈 / 与全新库 schema 一致 / 幂等。

## 7. 仍然存在、本次未动的

1. **换版靠 `Document.content_hash` 变更来表达，而目前没有任何「就地替换书内容」的入口**。
   本次把路修通了（不再炸、且版本真的参与键），但触发路径仍缺 —— 现阶段实际会改变
   版本维度的主要是**全库语料指纹**（增删书、重索引）。
2. 旧版本行会累积，靠 `prune_qa_cache`（上限 10000）兜底，未做按版本主动清理。
3. `_corpus_fingerprint` 会在一次问答里被调用多次（cache_check / create_hitl / answer），
   每次 0.165ms，未做请求内缓存 —— 刻意如此：请求中途语料变了也不会写出错误的一行，
   复合键让这种竞态变成「两行不同版本」而不是崩。
