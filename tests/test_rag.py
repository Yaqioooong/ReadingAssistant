"""P3 RAG 层测试：文本分块与向量检索。"""

from langchain_core.embeddings import Embeddings

from reading_assistant.config import get_settings
from reading_assistant.parsers import Chapter, ParsedBook
from reading_assistant.rag import Retriever, chunk_book, chunk_text
from reading_assistant.storage.vector_store import InMemoryVectorStore, StoredChunk


class FakeEmbeddings(Embeddings):
    """固定向量的假 Embedding，避免测试依赖真实 API。"""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector or [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return list(self._vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class TestChunkText:
    def test_empty_text_returns_empty(self) -> None:
        assert chunk_text('') == []
        assert chunk_text('   \n ') == []

    def test_short_text_is_single_chunk(self) -> None:
        assert chunk_text('这是一个短句', chunk_size=800, chunk_overlap=100) == ['这是一个短句']

    def test_long_text_splits_with_overlap(self) -> None:
        chunks = chunk_text('字' * 2500, chunk_size=800, chunk_overlap=100)
        assert len(chunks) >= 3
        assert all(len(chunk) <= 800 for chunk in chunks)
        assert chunks[1][:100] == chunks[0][-100:]

    def test_respects_chinese_separator(self) -> None:
        chunks = chunk_text(
            '第一句。第二句。第三句。', chunk_size=4, chunk_overlap=0, separators=['。', '']
        )
        assert chunks == ['第一句', '第二句', '第三句']

    def test_splits_at_paragraph_boundaries(self) -> None:
        text = '\n\n'.join(f'段落{index}内容' * 40 for index in range(5))

        chunks = chunk_text(text, chunk_size=300, chunk_overlap=30)

        assert len(chunks) == 5
        assert all(chunk.startswith('段落') for chunk in chunks)

    def test_config_separators_contain_real_newlines(self) -> None:
        assert '\n\n' in get_settings().separators
        assert '' in get_settings().separators


class TestChunkBook:
    def test_preserves_chapter_metadata(self) -> None:
        book = ParsedBook(
            title='测试',
            chapters=[
                Chapter(title='第一章', content='内容' * 900),
                Chapter(title='第二章', content='其他内容' * 900),
            ],
        )

        chunks = chunk_book(book, chunk_size=800, chunk_overlap=100)

        assert len(chunks) > 2
        assert chunks[0].metadata['chapter'] == '第一章'
        assert chunks[0].index == 0
        assert chunks[-1].metadata['chapter'] == '第二章'
        assert chunks[-1].index == len(chunks) - 1
        assert all(chunk.text for chunk in chunks)


class TestRetriever:
    @staticmethod
    def _store() -> InMemoryVectorStore:
        store = InMemoryVectorStore()
        store.add(
            [
                StoredChunk(
                    id='c1',
                    text='张三出场',
                    metadata={'document_id': 1, 'chapter': '第一章', 'chapter_index': 0},
                    embedding=[1.0, 0.0],
                ),
                StoredChunk(
                    id='c2',
                    text='事件发展',
                    metadata={'document_id': 1, 'chapter': '第二章', 'chapter_index': 1},
                    embedding=[0.0, 1.0],
                ),
                StoredChunk(
                    id='c3',
                    text='另一本书',
                    metadata={'document_id': 2, 'chapter': '第一章', 'chapter_index': 0},
                    embedding=[0.5, 0.5],
                ),
            ]
        )
        return store

    def test_retrieve_returns_sorted_hits_with_citation(self) -> None:
        retriever = Retriever(self._store(), FakeEmbeddings([1.0, 0.0]))

        hits = retriever.retrieve('张三')

        assert [hit.chunk_id for hit in hits] == ['c1', 'c3', 'c2']
        assert hits[0].document_id == 1
        assert hits[0].chapter == '第一章'
        assert hits[0].citation == '文档 1｜第一章'

    def test_retrieve_filters_by_document_id(self) -> None:
        retriever = Retriever(self._store(), FakeEmbeddings([1.0, 0.0]))

        hits = retriever.retrieve('查询', document_id=2)

        assert [hit.chunk_id for hit in hits] == ['c3']

    def test_retrieve_respects_top_k(self) -> None:
        retriever = Retriever(self._store(), FakeEmbeddings([1.0, 0.0]))

        hits = retriever.retrieve('查询', top_k=1)

        assert len(hits) == 1
        assert hits[0].chunk_id == 'c1'
