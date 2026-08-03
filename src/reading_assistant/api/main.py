"""FastAPI 入口：``uvicorn reading_assistant.api.main:app --reload``。"""

from reading_assistant.api.app import create_app

app = create_app()
