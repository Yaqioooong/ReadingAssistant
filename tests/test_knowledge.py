from reading_assistant.graph.knowledge import KnowledgeGraph
from reading_assistant.storage.vector_store import StoredChunk


def _chunk(chunk_id: str, text: str, document_id: int = 7) -> StoredChunk:
    return StoredChunk(
        id=chunk_id,
        text=text,
        metadata={
            'document_id': document_id,
            'chapter': '第十三回 双叉岭遇妖',
            'chapter_index': 13,
        },
    )


def test_knowledge_graph_keeps_entity_relation_evidence() -> None:
    graph = KnowledgeGraph.from_chunks(
        [_chunk('c-1', '寅将军遇见唐三藏，太白金星在旁边出现。')]
    )

    relations = graph.lookup('寅将军')

    assert relations
    assert {relation.target for relation in relations} >= {'唐三藏', '太白金星'}
    assert relations[0].predicate == 'co_occurs'
    assert relations[0].chunk_id == 'c-1'
    assert relations[0].chapter_index == 13
    assert '寅将军' in relations[0].evidence


def test_knowledge_graph_resolves_punctuation_alias_and_document_scope() -> None:
    graph = KnowledgeGraph.from_chunks(
        [
            _chunk('c-1', '孙悟空与唐僧同行。'),
            _chunk('c-2', '孙悟空与猪八戒同行。', document_id=8),
        ],
        document_ids=[7],
    )

    assert graph.resolve(' 孙悟空 ') == '孙悟空'
    assert graph.resolve('孙 悟空') == '孙悟空'
    assert {relation.target for relation in graph.lookup('孙悟空')} == {'唐僧'}
