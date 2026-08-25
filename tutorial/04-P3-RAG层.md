# 第 04 章 P3 RAG 层

> 对应阶段：P3 RAG 层 —— 文本分块（chunking）与向量检索（retriever）。

## 本章目标

把整本书切成「可检索的小块」（默认 800 字、重叠 100 字），并把「问题 → 向量 → top-k 原文片段」这条检索链路打通，检索结果携带章节元数据，为引用做准备。

## 步骤拆解

### 步骤 1：理解 langchain-text-splitters 1.x 的 API 变化

**关键认知**：1.x 的 `RecursiveCharacterTextSplitter.__init__` 签名是：

```python
def __init__(self, separators=None, keep_separator=True, is_separator_regex=False, **kwargs)
```

`chunk_size` / `chunk_overlap` 不在显式参数里，而是通过 `**kwargs` 透传给父类 `TextSplitter`。所以这样用是**正确**的：

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=['\n\n', '\n', '.', '!', '?', '。', '！', '？', ' ', ''],
    keep_separator=False,
)
```

**两个必须知道的细节**：

1. **separators 必须以空串 `''` 结尾**：分块器先找「第一个能匹配的分隔符」，没有分隔符可匹配时回退到 `''` 做**硬切分**。不写 `''`，长文本会原样返回不切分（实测踩过）。
2. **`keep_separator=False`**：切分结果里不保留分隔符（1.x 默认 `True` 会保留，影响文本干净度）。

### 步骤 2：修一个配置 bug（`config/chroma.yml`）

YAML **单引号字符串**里的 `\n` 是字面量反斜杠+n，不是换行符：

```yaml
# 错误：'\n\n' 在 YAML 单引号里是两个字符 \ n
separators: [ '\n\n','\n','.','!','?','。','！','？',' ','' ]
# 正确：双引号才会解析转义
separators: ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
```

这个 bug 会让分块器永远按不到段落边界。**技巧**：配置里含转义字符时用双引号，并加一个测试断言「配置里真的是换行符」防回归。

### 步骤 3：实现 `rag/chunking.py`

**做什么**：`chunk_text`（单段文本切块）与 `chunk_book`（按章节切块并保留元数据）。

```python
@dataclass
class TextChunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)

def chunk_text(text, chunk_size=None, chunk_overlap=None, separators=None) -> list[str]:
    if not text or not text.strip():
        return []
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=separators if separators is not None else settings.separators,
        keep_separator=False,
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]
```

**注意**：`chunk_overlap if chunk_overlap is not None` 这种写法是因为 `0` 也是合法值，`or` 会把 0 吞掉。

`chunk_book` 遍历 `ParsedBook.chapters`，每块元数据带 `chapter` / `chapter_index`，index 全局连续。**为什么元数据放在分块阶段**：后面检索结果要展示「出自哪一章」，章节信息必须跟着块走。

### 步骤 4：实现 `rag/retriever.py`

**做什么**：`Retriever` 把「embedding + 向量库查询」封装成一个方法。

```python
class Retriever:
    def __init__(self, vector_store: VectorStore, embedding_model: Embeddings | None = None):
        self._vector_store = vector_store
        self._embedding_model = embedding_model or get_embedding_model()

    def retrieve(self, query, top_k=None, document_id=None) -> list[RetrievedChunk]:
        k = top_k or get_settings().top_k
        embedding = self._embedding_model.embed_query(query)
        where = {'document_id': document_id} if document_id is not None else None
        hits = self._vector_store.query(embedding, top_k=k, where=where)
        return [RetrievedChunk(...) for hit in hits]
```

**`RetrievedChunk.citation`**：把文档 id 和章节拼成可展示的引用文本：

```python
@property
def citation(self) -> str:
    parts = [f'文档 {self.document_id}'] if self.document_id is not None else []
    if self.chapter:
        parts.append(self.chapter)
    return '｜'.join(parts)
```

## 测试技巧：FakeEmbeddings

真实 embedding 要网络和密钥，测试里注入固定向量的假实现：

```python
class FakeEmbeddings(Embeddings):
    def embed_query(self, text): return list(self._vector)
    def embed_documents(self, texts): return [list(self._vector) for _ in texts]
```

配合 `InMemoryVectorStore`（P2 写的），整个检索链路不碰网络即可测。

## 常见坑

1. **小块会被重新合并**：分块器内部会把小片段用分隔符拼回一个大块（直到超过 chunk_size）。所以「按句号切分」的测试若 chunk_size 太大，结果还是一整块——测试用小的 `chunk_size`。
2. **句尾分隔符被剥掉**：`keep_separator=False` 时 `'一句话。'` 会变成 `'一句话'`，断言时别期待原样。
3. **f-string 嵌套同引号**：Python 3.12+（PEP 701）允许 `f'{chunk.get('citation', '')}'` 这种写法，但为了可读性，复杂场景还是先取变量。
4. **重叠测试**：`chunks[1][:overlap] == chunks[0][-overlap:]` 是验证 overlap 行为的直接断言。

## 测试与验证

```bash
.venv/bin/pytest tests/test_rag.py -q
```

覆盖：空文本、短文本单块、长文本重叠、中文分隔符、段落边界、章节元数据、检索排序/过滤/top_k/引用格式。
