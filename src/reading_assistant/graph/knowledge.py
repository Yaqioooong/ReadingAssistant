"""轻量、可回溯的实体关系索引。

第一版刻意不把模型猜测写成关系：关系只来自同一个原文 chunk 中实际出现的
实体共现，并始终保留 chunk/document/chapter 证据。后续可以在不改变查询契约
的前提下增加带模型置信度的显式关系。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from reading_assistant.graph.rewrite import extract_entities


@dataclass(frozen=True)
class Entity:
    name: str
    aliases: tuple[str, ...] = ()
    mentions: int = 0


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    predicate: str
    chunk_id: str
    document_id: int | None = None
    chapter: str | None = None
    chapter_index: int | None = None
    evidence: str = ''


@dataclass
class KnowledgeGraph:
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    _aliases: dict[str, str] = field(default_factory=dict, repr=False)

    @staticmethod
    def _alias_key(term: str) -> str:
        return re.sub(r'[\s·・,，。、“”「」『』()（）]', '', str(term or '').strip().lower())

    def add_chunk(self, chunk: Any) -> None:
        text = str(getattr(chunk, 'text', '') or '')
        metadata = getattr(chunk, 'metadata', None) or {}
        chunk_id = str(getattr(chunk, 'id', None) or getattr(chunk, 'chunk_id', '') or '')
        if not text or not chunk_id:
            return
        names = []
        for name in extract_entities(text, limit=8):
            canonical = self._aliases.get(name, name)
            if canonical not in names:
                names.append(canonical)
            self._aliases.setdefault(name, canonical)
            self._aliases.setdefault(self._alias_key(name), canonical)
            self._aliases.setdefault(canonical, canonical)
        for name in names:
            current = self.entities.get(name)
            self.entities[name] = Entity(
                name=name,
                aliases=tuple(sorted({*(current.aliases if current else ()), name})),
                mentions=(current.mentions if current else 0) + 1,
            )
        for source, target in combinations(names, 2):
            self.relations.append(
                Relation(
                    source=source,
                    target=target,
                    predicate='co_occurs',
                    chunk_id=chunk_id,
                    document_id=metadata.get('document_id'),
                    chapter=metadata.get('chapter'),
                    chapter_index=metadata.get('chapter_index'),
                    evidence=text[:400],
                )
            )

    @classmethod
    def from_chunks(
        cls, chunks: list[Any], document_ids: list[int] | None = None
    ) -> KnowledgeGraph:
        allowed = set(document_ids or [])
        graph = cls()
        for chunk in chunks:
            metadata = getattr(chunk, 'metadata', None) or {}
            if allowed and metadata.get('document_id') not in allowed:
                continue
            graph.add_chunk(chunk)
        return graph

    def resolve(self, term: str) -> str | None:
        normalized = str(term or '').strip()
        return self._aliases.get(normalized) or self._aliases.get(self._alias_key(normalized))

    def lookup(self, term: str, limit: int = 20) -> list[Relation]:
        canonical = self.resolve(term) or str(term or '').strip()
        if not canonical:
            return []
        matches = [
            relation
            for relation in self.relations
            if relation.source == canonical or relation.target == canonical
        ]
        # 同一证据可能包含很多实体组合，稳定去重后再返回。
        unique: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in matches:
            key = (relation.source, relation.target, relation.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(relation)
            if len(unique) >= limit:
                break
        return unique
