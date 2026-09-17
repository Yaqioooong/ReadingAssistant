# 代码答疑记录

> 本文件记录项目中代码问题的问答。
> 记录格式：**问题 → 代码位置 → 结论 → 代码细节 → 备注/坑**

---

## 2026-09-14 · EPUB 解析器（`src/reading_assistant/parsers/epub_parser.py`）

### Q1. `_first_metadata` 里 `get_metadata('DC', name)` 的 `DC` 是什么含义？为什么要这样写？

**代码位置**

```python
@staticmethod
def _first_metadata(book: epub.EpubBook, name: str) -> str | None:
    values = book.get_metadata('DC', name)
    return values[0][0] if values else None
```

#### DC = Dublin Core（都柏林核心元数据集）

一个图书元数据的国际标准命名空间。EPUB 内部的 `content.opf` 文件用 XML 描述书籍元信息，长这样：

```xml
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>三体</dc:title>
  <dc:creator>刘慈欣</dc:creator>
  <dc:language>zh</dc:language>
  <dc:identifier id="pub-id">urn:isbn:9787536692930</dc:identifier>
</metadata>
```

`dc:` 前缀就是 Dublin Core 命名空间。EPUB 规范强制要求 `title` / `language` / `identifier` 必须属于 DC 命名空间，作者通常也是 `dc:creator`。

`get_metadata('DC', 'title')` 的语义 = **给我所有 DC 命名空间下名为 title 的条目**。

#### 为什么返回值要写成 `values[0][0]`？

`ebooklib` 的 `get_metadata(namespace, name)` 返回的不是字符串，而是**列表**，每个元素是 `(值, 属性字典)` 元组：

```python
book.get_metadata('DC', 'title')
# → [('三体', {}), ('三体（全集）', {})]
```

原因：XML 里同一个标签可以出现多次（多语言标题、多作者、多 ISBN 都很常见），所以必须用列表返回。

```python
return values[0][0] if values else None
```

| 片段 | 含义 |
|---|---|
| `values[0]` | 第一条元组 `('三体', {})` |
| `values[0][0]` | 元组第一个元素，才是真正的字符串 `'三体'` |
| `if values else None` | 一条都没找到时返回 `None`，而不是抛 `IndexError` |

#### 为什么只取第一条？

这是函数名里 `_first_` 的含义——设计上只要一个主标题、一个主作者，不做多语言 / 多作者的复杂处理。对 ReadingAssistant 的场景够用。

> ⚠️ **坑**：如果以后遇到多语言 EPUB，这里可能拿到非预期的那个标题（顺序取决于文件里的排列）。届时需要改成优先挑 `xml:lang` 匹配的条目。

> ⚠️ **坑**：作者字段是 `creator` 而不是 `author` —— Dublin Core 标准里没有 `author` 这个词，作者叫 `creator`。`get_metadata('DC', 'author')` 会**静默返回空列表**，不报错。

**调用处**

```python
title = self._first_metadata(book, 'title') or path.stem
author = self._first_metadata(book, 'creator')
```

---

### Q2. `book.spine` 是什么？

**代码位置**

```python
chapters = []
for idref, _linear in book.spine:
    item = book.get_item_with_id(idref)
    ...
```

#### spine = 书的"脊柱"，即「阅读顺序表」

一本 EPUB 由一堆独立的 XHTML 文件组成（封面、各章节、附录、版权页……）。这些文件本身**没有先后顺序**，谁先谁后靠 `spine` 定义。

`spine` 在 `content.opf` 里长这样：

```xml
<spine toc="ncx">
  <itemref idref="cover" linear="yes"/>
  <itemref idref="chapter1" linear="yes"/>
  <itemref idref="chapter2" linear="yes"/>
  <itemref idref="appendix" linear="no"/>
</spine>
```

每个 `<itemref>` 的 `idref` 指向 `manifest` 里某个资源文件的 id。

#### ebooklib 把它加载成了什么？

```python
# ebooklib/epub.py:1651
def _load_spine(self):
    spine = self.container.find('{%s}%s' % (NAMESPACES['OPF'], 'spine'))
    self.book.spine = [(t.get('idref'), t.get('linear', 'yes')) for t in spine]
```

所以 `book.spine` 是一个 **`(idref, linear)` 二元组的列表**，这也是代码里写成 `for idref, _linear in book.spine` 的原因。

| 字段 | 含义 |
|---|---|
| `idref` | 资源 id，配合 `book.get_item_with_id(idref)` 取出真正的文档对象 |
| `linear` | `'yes'` = 属于正文线性阅读流；`'no'` = 跳转内容（如脚注、弹窗页），正常阅读时可跳过 |

> 注意：这里代码把 `linear` 用 `_linear` 接住后**直接丢弃了**——也就是说目前 `linear="no"` 的页面也会被当作正文解析。如果发现解析出来的章节里混进了脚注页 / 附录，就是这里的原因。

#### 为什么必须走 spine，不能直接遍历所有文档？

EPUB 的 `manifest` 里包含**所有**资源：正文、封面图、CSS、字体、导航页（nav / toc.ncx）。直接遍历会混入大量非正文内容。`spine` 是唯一能告诉解析器"哪些是正文、按什么顺序"的信息源。

#### spine 与代码里过滤逻辑的配合

```python
for idref, _linear in book.spine:
    item = book.get_item_with_id(idref)
    if item is None:
        continue
    # ebooklib 将 nav/toc 也标记为 ITEM_DOCUMENT，按文件名排除导航页
    name = (item.get_name() or '').lower()
    if item.get_type() != ebooklib.ITEM_DOCUMENT or name.startswith(('nav', 'toc', 'ncx')):
        continue
```

三层过滤，各管一件事：

1. **`book.spine`** —— 拿到阅读顺序（第一道筛：相对 manifest 已经收窄很多）
2. **`item.get_type() != ITEM_DOCUMENT`** —— 排除 CSS、图片、字体等非文档资源
3. **`name.startswith(('nav', 'toc', 'ncx'))`** —— 排除导航页。因为 ebooklib 会把 `nav.xhtml` 这类导航文件也标记成 `ITEM_DOCUMENT`，只看类型会漏掉，所以再按文件名兜底过滤

> 💡 更严谨的做法是结合 `linear` 字段过滤，或者用 `book.toc` 交叉验证章节结构。但按文件名黑名单在绝大多数 EPUB 上够用。

---


---

## 2026-09-14 · 入库流水线（`src/reading_assistant/graph/ingest.py`）

### Q3. 第 103 行的 `session_scope` 是什么作用？

**代码位置**（`ingest.py` 的 `chunk_and_index` 节点内）

```python
chunks = chunk_book(book)                              # 1. 切块
vectors = embeddings.embed_documents(...)              # 2. 调 embedding 模型（慢、走网络）
vector_store.add([...])                                # 3. 写向量库
with session_scope(session_factory) as session:        # 4. 最后才碰 SQL  ← 第 103 行
    document = get_document(session, document_id)
    document.chunk_count = len(chunks)
    document.index_status = 'indexed'
```

#### `session_scope` 的定义（`storage/database.py:54`）

```python
@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """事务性 session 上下文：正常提交，异常回滚。"""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

一句话：**"自动提交 / 自动回滚"的数据库会话管理器。**

| 情况 | 行为 |
|---|---|
| `with` 块正常走完 | `commit()` 落库 |
| 块里抛异常 | `rollback()` 撤销 + `raise` 继续上抛 |
| 无论如何 | `close()` 释放连接 |

所以 `with session_scope(f) as session:` 的语义 = **这段改动要么全成，要么全不算**。

#### 为什么要单独开一个作用域？

关键在于它前面的步骤 2、3 都是慢操作，还可能直接失败：

- `embeddings.embed_documents()` 会调 embedding 模型，耗时且可能走网络
- `vector_store.add()` 写向量库

DB 事务只在最后一步开，**持有时间尽可能短**。如果从函数开头就 `session = factory()` 一路握到最后，那 embedding 调用的几秒会全程占着一个连接和一个事务。**SQLite 上尤其要命**——写事务拿着锁不放，其他请求全部阻塞。

#### 异常分支为什么重新开了一个 session？

```python
except Exception:
    logger.exception('入库[chunk_and_index] 失败 doc=%s', document_id)
    with session_scope(session_factory) as session:   # ← 新的一个
        update_document_index_status(session, document_id, 'failed')
    raise
```

不能复用原来的 session：它已经在 `session_scope` 的 `except` 里被 rollback + close 掉了。必须开新的才能把 `failed` 状态写进去，否则异常一抛，文档状态会永远卡在 `indexing`。

> ⚠️ **潜在问题 1**：`document = get_document(session, document_id)` **没有 None 检查**。若文档记录不存在，下一行 `document.chunk_count` 抛 `AttributeError`，被外层 `except` 捕获后，会在一条不存在的记录上标记 `failed`（静默失败）。建议补判空。

> ⚠️ **潜在问题 2**：**跨存储一致性**。`vector_store.add()` 不在这个事务内，若其后 SQL 更新失败并 rollback，向量库已留下数据、SQL 里状态是 `failed`。两者没有共同事务。
> 缓解：重试时 chunk id 为 `doc{document_id}-{chunk.index}`，是幂等的，会覆盖。所以问题不大，但清理逻辑要留意。


---

## 2026-09-15 · 双写一致性（reindex 崩溃恢复）

### Q4. reindex 时进程崩在「向量写入成功、PG 尚未更新」这一瞬间，靠什么恢复？

**结论：没有任何自动恢复机制。靠的是「手动重跑 reindex + chunk id 幂等覆盖」。**

#### 崩溃瞬间留下的状态

reindex 的路径：`add_book`（把 `index_status` 置为 `'indexing'`）→ `chunk_and_index`。
崩在向量写完、PG 未更新时：

| 存储 | 状态 |
|---|---|
| Chroma | 新向量已落盘（`upsert` 成功） |
| PG `index_status` | `'indexing'`（add_book 已写过这个值） |
| PG `chunk_count` | **旧值** |

> ⚠️ **关键**：进程崩溃（SIGKILL / OOM / 断电）**不会走 `except` 分支**，所以连 `'failed'` 都写不上。
> 状态是「卡死的 `indexing`」，而不是 `failed`。这两者区别很大：`failed` 是明确终态，`indexing` 是"看起来还在进行中"的假活状态。

#### 逐项排查：没人来兜底

| 检查项 | 位置 | 结论 |
|---|---|---|
| 启动对账 | `api/app.py` 的 `lifespan` | ❌ 只调 `init_db()` 建表，**不扫** `index_status='indexing'` |
| 检索层校验 | `rag/`、`graph/qa.py` | ❌ 搜不到 `index_status`，向量**立刻可被检索**，无人察觉台账不一致 |
| 看门狗 / 超时回收 | 全局搜 `reindex\|repair\|recover\|reconcile\|stale` | ❌ 只有路由和注释，无实现 |

#### 实际靠什么活下来：确定性 id + upsert

```python
# ingest.py chunk_and_index
id=f'doc{document_id}-{chunk.index}'
```

`chunk.index` 在 `rag/chunking.py:85,99` 按内容顺序分配，同样文本必然产出同样 index。
配合 Chroma 的 `upsert`，重跑 reindex 会**原地覆盖同一批 id**，把 `index_status` 重新推到 `indexed`。

**恢复动作** = 手动 `POST /documents/{id}/reindex`（或 MCP `reindex_book`）。

> ⚠️ **前置条件：源文件必须还在**
> ```python
> if not document.file_path or not Path(document.file_path).exists():
>     raise HTTPException(status_code=409, detail='sourcefile not found, cannot reindex')
> ```
> 文件被删/移走则此路不通，且 409 对用户毫无解释力——他不知道自己遇到的是崩溃残留。

#### 幂等覆盖救不了的那个洞：chunk 数变少

**场景**：原 150 个 chunk，改分块参数后变 120 个。重跑 reindex 只覆盖 `doc5-0` ~ `doc5-119`，
`doc5-120` ~ `doc5-149` 这 30 个**孤儿向量留在库里，仍可被检索命中**。
而 PG 的 `chunk_count` 已是 120 —— 台账上看不出来，但模型会拿到不属于任何章节的文本。

**根因**：`vector_store.delete()` 是**全仓库死代码**。两处实现都定义了，但**没有任何调用点**
（文档删除路由压根不存在）。reindex 前不会先清空旧向量。

**BM25 索引的情况（反而自愈）**：`_content_version` 是挂在实例上的内存变量，进程重启归零，
下次查询基于 `all_chunks()` 全量重建。
但 `rag/bm25_index.py` docstring 自认：同进程内 reindex 且 chunk 数不变时**不会自动失效**，需显式 `refresh()`。

#### 修复路线（按性价比排序）

```infographic
infographic sequence-steps-simple
data
  title 双写一致性修复路线
  items
    - label 启动对账
      desc lifespan 扫 indexing 超时记录，标记 stale
    - label 版本化 id
      desc doc{id}-v{hash8}-{index}，写完再删旧版本
    - label 显式清理
      desc reindex 真正调用 delete，去掉死代码
    - label 一致性校验
      desc 暴露 chunk_count vs 向量库 count 的对账接口
```

**① 启动对账（最划算，约 20 行）**
`lifespan` 中扫 `index_status='indexing'` 且更新时间超过 N 分钟的记录 → 标记 `stale`。
哪怕不自动重跑，至少让卡死状态**可见**，用户能看懂、能重试。

**② reindex 前先 `delete(document_id)`？—— 先别急**
这会把「崩溃后残留孤儿」变成「崩溃后数据全丢」，裸奔在更糟的窗口里。
要做必须配合版本化 id：写 `doc{id}-v{content_hash[:8]}-{index}`，新版本全部写完后**再**删旧版本，实现原子切换。

**③ 清理死代码**
`vector_store.delete()` 要么接上 reindex 删除路径，要么删掉，别留着让人误以为已有清理逻辑。


---

## 2026-09-15 · chunk_id 生成机制

### Q5. `chunk_id` 如何生成的？

**结论：两段式——先有位置序号 `chunk.index`，再在入库时拼成字符串 `doc{document_id}-{chunk.index}`。**

#### 第一段：`chunk.index`（`rag/chunking.py:78`）

```python
chunks: list[TextChunk] = []
index = 0                                    # ← 计数器，从 0 开始
for chapter_index, chapter in enumerate(book.chapters):
    for kind, segment in _split_table_blocks(chapter.content):
        if kind == 'table':
            chunks.append(TextChunk(text=segment, index=index,
                                    metadata={... 'block_type': 'table'}))
            index += 1
            continue
        for piece in chunk_text(segment, chunk_size, chunk_overlap, separators):
            chunks.append(TextChunk(text=piece, index=index, metadata={...}))
            index += 1
```

两个要点：

1. **计数器是「全书全局」的**，不是每章重置。第三章第一块接着第二章尾号继续涨。
2. **表格块也占号**。`_split_table_blocks` 把表格单独切出（`TABLE_BEGIN`/`TABLE_END` 之间），它占一个 chunk 也占一个号，纯文本块与表格块共享同一条数轴。

> `chapter_index`（章节号）是另一码事，进 `metadata` 而非 id。

#### 第二段：拼字符串（`graph/ingest.py:95`）

```python
StoredChunk(
    id=f'doc{document_id}-{chunk.index}',
    ...
)
```

`document_id` 是 `documents` 表主键（`models.py:22`，`autoincrement=True` 的 int）。最终 id：

```
doc5-0
doc5-1
...
doc5-149
```

Chroma 要求 id 为字符串；这种可读格式比 UUID 好调试——一眼看出是第 5 本书的第 0 块。

#### 核心性质：位置型，不是内容型

| 字段 | 类型 | 内容变化时 |
|---|---|---|
| `file_hash` | sha256(64位hex) | 变 |
| `content_hash` | sha256(64位hex) | 变 |
| `chunk_id` | 位置坐标 | **不变** |

`chunk_id` 只回答"第几本书的第几块"，**不含任何内容指纹**。三条推论：

**① 位置唯一、内容随意 → 孤儿向量的根源（见 Q4）**
`doc5-10` 永远指"第 5 本书第 11 块"，但装什么文本取决于本次的 `chunk_size=800 / chunk_overlap=100`（`config.py:66-67`）。改参数重跑后同一 id 装的是**另一段文本**，upsert 静默覆盖，无人察觉。

**② 恢复能力来自 document_id 的稳定性**
reindex 走 `force=True` 时，`add_book` 因 `file_hash` 命中而返回**同一条** document 记录（`service.py:64`）→ document_id 不变 → id 命名空间不变 → 原地覆盖。这正是 Q4 中"手动重跑可恢复"的前提。
反例：若 reindex 新建 document 记录，id 全变，旧向量永久残留为孤儿。

**③ 隐藏脆弱点：id 里存在两个 index 概念**

```python
embedding=vectors[index]                 # ← enumerate(chunks) 的位置
id=f'doc{document_id}-{chunk.index}'     # ← chunk 自带序号
```

目前同一份 list、同一顺序，两者恒等。但这**依赖约定，并非代码强制**——只要中途插一个过滤（如 `chunks = [c for c in chunks if c.text]`），`vectors[index]` 就会取到别人的向量，而 id 仍用 `chunk.index`。**向量串块这类 bug 极难排查**，建议统一为单一来源。

#### 真实碰撞风险：数据库重建 + 残留向量库

`documents.id` 是自增主键，正常不复用 → 跨文档不撞。
**但 Chroma 持久化目录独立于 PG 存活**：若 PG 被重建（删 sqlite 重跑），新上传的第一本书拿到 id=1，而 `chroma_persist_path` 里 `doc1-*` 仍是旧书向量 → **upsert 使两本不同的书混入同一 id 命名空间，互相覆盖**。

> 💡 修法与 Q4 的「版本化 id」同源：用 `doc{content_hash[:8]}-{index}` 替代 `doc{document_id}-{index}`，
> 可一次性解决 **孤儿残留 / 数据库重建 / 跨库碰撞** 三个问题。

