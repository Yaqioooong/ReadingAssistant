"""设置路由测试：脱敏展示、PUT 校验与持久化调用（不触碰真实 .env）。"""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

import reading_assistant.api.routes.settings as settings_route
from reading_assistant.api import create_app
from reading_assistant.api.routes.settings import _mask_secret
from reading_assistant.storage import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from reading_assistant.storage.vector_store import InMemoryVectorStore


class FakeEmbeddings(Embeddings):
    def embed_query(self, text):  # noqa: ARG002
        return [1.0, 0.0]

    def embed_documents(self, texts):  # noqa: ARG002
        return [[1.0, 0.0] for _ in texts]


class FakeLLM:
    def invoke(self, prompt):  # noqa: ARG002
        return SimpleNamespace(content='测试回答。')


def _client(tmp_path: Path) -> TestClient:
    engine = create_db_engine('sqlite:///:memory:')
    init_db(engine)
    app = create_app(
        session_factory=create_session_factory(engine),
        vector_store=InMemoryVectorStore(),
        llm=FakeLLM(),
        embedding_model=FakeEmbeddings(),
        upload_dir=tmp_path / 'uploads',
    )
    return TestClient(app)


def test_get_returns_groups_and_masks_secrets(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.get('/api/settings')
        assert resp.status_code == 200
        data = resp.json()
        ids = [g['id'] for g in data['groups']]
        assert 'model' in ids and 'retrieval' in ids and 'advanced' in ids
        secrets = [
            item for g in data['groups'] for item in g['items'] if item.get('type') == 'secret'
        ]
        assert secrets
        # ⚠️ 不能断言「每个 secret 都被掩码」—— 未配置的 secret 掩码为 ''，
        # 那样测试就依赖本机 .env 恰好填了值（2026-09-21 实际假失败过一次）。
        # 这里只钉**契约**：已配置 → 掩码且短；未配置 → 空掩码。
        for item in secrets:
            state = item['state']
            assert 'configured' in state
            if state['configured']:
                masked = state['masked']
                assert '••••' in masked
                assert len(masked) < 32  # 只暴露前缀/尾号，绝非完整密钥
            else:
                assert state['masked'] == ''
        # 掩码形状本身由下面这条确定性单测负责（不依赖环境）
        assert _mask_secret('sk-abcdefghijklmnop') == 'sk-••••••mnop'
        assert _mask_secret('short') == '••••••'
        assert _mask_secret('') == ''


def test_put_unknown_key_returns_400(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.put('/api/settings', json={'updates': {'NOPE': '1'}})
        assert resp.status_code == 400


def test_put_invalid_value_returns_400(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.put('/api/settings', json={'updates': {'TOP_K': 'abc'}})
        assert resp.status_code == 400


def test_put_valid_persists_without_touching_real_env(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, str | None] = {}
    monkeypatch.setattr(settings_route, 'write_env', captured.update)
    monkeypatch.setattr(settings_route, 'apply_to_os_env', lambda updates: None)
    monkeypatch.setattr(settings_route, '_reload_runtime', lambda: None)

    with _client(tmp_path) as client:
        resp = client.put(
            '/api/settings',
            json={
                'updates': {'TOP_K': '8', 'CACHE_ENABLED': 'false', 'DEEPSEEK_API_KEY': '__CLEAR__'}
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body['ok'] is True
        assert 'TOP_K' in body['applied'] and 'CACHE_ENABLED' in body['applied']
        assert 'DEEPSEEK_API_KEY' in body['cleared']
    assert captured.get('TOP_K') == '8'
    assert captured.get('CACHE_ENABLED') == 'false'
    assert captured.get('DEEPSEEK_API_KEY') is None
