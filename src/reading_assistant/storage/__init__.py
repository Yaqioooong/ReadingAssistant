"""存储层：SQLAlchemy 模型、数据库会话、文档 repository/服务与向量库适配器。"""

from reading_assistant.storage.database import (
    create_db_engine,
    create_session_factory,
    init_db,
    session_scope,
)
from reading_assistant.storage.models import Base, ChatMessage, ChatSession, Document, HitlTask
from reading_assistant.storage.repositories import (
    create_hitl_task,
    get_document,
    get_document_by_content_hash,
    get_document_by_file_hash,
    get_hitl_task,
    insert_document,
    list_documents,
    reject_hitl_task,
    submit_hitl_clarification,
)
from reading_assistant.storage.service import (
    AddBookResult,
    DocumentService,
    normalize_text,
    sha256_hex,
)
from reading_assistant.storage.vector_store import (
    ChromaVectorStore,
    InMemoryVectorStore,
    SearchHit,
    StoredChunk,
    VectorStore,
    create_vector_store,
)

__all__ = [
    'AddBookResult',
    'Base',
    'ChatMessage',
    'ChatSession',
    'ChromaVectorStore',
    'Document',
    'DocumentService',
    'HitlTask',
    'InMemoryVectorStore',
    'SearchHit',
    'StoredChunk',
    'VectorStore',
    'create_hitl_task',
    'create_db_engine',
    'create_session_factory',
    'create_vector_store',
    'get_document',
    'get_document_by_content_hash',
    'get_document_by_file_hash',
    'get_hitl_task',
    'init_db',
    'insert_document',
    'list_documents',
    'normalize_text',
    'reject_hitl_task',
    'session_scope',
    'sha256_hex',
    'submit_hitl_clarification',
]
