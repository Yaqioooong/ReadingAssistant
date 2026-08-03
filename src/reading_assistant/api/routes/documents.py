"""文档路由：上传与列表。"""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from reading_assistant.api import schemas
from reading_assistant.api.deps import (
    get_db_session,
    get_embedding_model,
    get_session_factory,
    get_upload_dir,
    get_vector_store,
)
from reading_assistant.graph import build_ingest_graph
from reading_assistant.parsers import ParseError, UnsupportedFormatError, get_parser
from reading_assistant.storage import get_document
from reading_assistant.storage import list_documents as list_document_rows

router = APIRouter(prefix='/api/documents', tags=['documents'])


@router.get('', response_model=list[schemas.DocumentOut])
def list_documents(session: Session = Depends(get_db_session)):
    """列出已入库的文档。"""
    return list_document_rows(session)


@router.post('/upload', response_model=schemas.UploadResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    session_factory=Depends(get_session_factory),
    vector_store=Depends(get_vector_store),
    embedding_model=Depends(get_embedding_model),
    upload_dir: Path = Depends(get_upload_dir),
    session: Session = Depends(get_db_session),
):
    """上传并解析电子书，重复上传返回已有 document_id。"""
    filename = Path(file.filename or 'book').name  # 去掉路径，防止目录穿越
    try:
        get_parser(filename)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f'{uuid4().hex}_{filename}'
    target.write_bytes(file.file.read())
    try:
        graph = build_ingest_graph(session_factory, vector_store, embedding_model=embedding_model)
        result = graph.invoke(
            {'book_path': str(target), 'filename': filename},
            config={'configurable': {'thread_id': f'ingest-{uuid4().hex}'}},
        )
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    document = get_document(session, result['document_id'])
    return schemas.UploadResponse(
        id=document.id,
        filename=document.filename,
        title=document.title,
        author=document.author,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        duplicate=result['duplicate'],
    )
