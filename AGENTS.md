# Repository Guidelines

ReadingAssistant parses ebooks (EPUB/PDF/DOCX/TXT) and answers questions about characters and events using RAG, LangGraph, PostgreSQL, and vector search, with HITL clarification and persistent chat history. The repo is in scaffold stage: frontend and utilities are implemented; backend core modules are TODO.

## Project Structure & Module Organization

```
src/reading_assistant/
├── config.py    # done: unified settings (pydantic-settings, .env + YAML)
├── parsers/     # done: txt/epub/pdf/docx parsers + factory
├── rag/         # done: chunking + retriever
├── storage/     # done: models, db session, dedup service, vector store adapters
├── graph/       # TODO: LangGraph pipelines (ingest, QA with HITL)
├── api/         # TODO: FastAPI app, routes, request/response schemas
├── model/       # done: LLM and embedding factories
├── utils/       # done: config, logging, path helpers
└── config/      # YAML configuration files
tests/           # pytest tests (smoke + parsers)
frontend/        # Vite + Vue 3 single-page app
docker-compose.yml  # PostgreSQL 16
```

Parsers and storage are designed to be swappable: add a format as a `BookParser` subclass and register it in `parsers/factory.py`.

## Development Plan

- Development must follow [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md): complete each phase in order, including its tests and acceptance criteria, before starting the next.
- Work not listed in the plan must be added to the plan before it is implemented.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"` — install the package and dev dependencies (`pyproject.toml` is the source of truth; `requirements.txt` is a thin `-e .[dev]` alias).
- `docker compose up -d` — start PostgreSQL for local development.
- `uvicorn reading_assistant.api.main:app --reload` — run the API locally (planned; `api/main.py` is not implemented).
- `cd frontend && npm install && npm run dev` — run the frontend dev server on http://localhost:5173 (proxies `/api` to port 8000).
- `npm run build` — build the frontend into `frontend/dist`.
- `pytest` — run the test suite.
- `ruff check src tests` and `ruff format --check src tests` — lint and check formatting.

## Coding Style & Naming Conventions

- Python 3.13+, formatted with Ruff (line length 100, single quotes).
- `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Type hints required on public signatures; Pydantic models at API boundaries.
- Read settings via `get_settings()` from `config.py`; never hardcode URLs or keys.
- Frontend uses Vue 3 `<script setup>` single-file components.

## Testing Guidelines

- Use pytest; name files `test_<module>.py` and functions `test_<behavior>`.
- Each new feature needs at least one happy-path and one edge-case test; keep coverage from decreasing.
- Tests needing PostgreSQL or an LLM key use pytest marks and skip when unavailable.

## Commit & Pull Request Guidelines

- Use Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`); no commits exist yet, so establish it from the first.
- Keep commits focused; each commit makes one logical change.
- PRs describe the change and how it was tested, link related issues, and include screenshots for UI or API output changes.

## Security & Configuration Tips

- Never commit secrets; copy `.env.example` to `.env` (git-ignored).
- Treat uploaded books as user data and never log their contents.

## Agent-Specific Instructions

- Read this file first and follow existing conventions.
- Follow the phase order in DEVELOPMENT_PLAN.md; never skip ahead of the current phase.
- Do not add dependencies without justification; prefer the current stack.
- Keep scaffolds honest: mark unfinished work with TODO; never claim unimplemented features.
