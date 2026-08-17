import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.core.config import settings
from app.api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=settings.app_name,
    description="RAG-powered document Q&A service — upload documents, ask grounded questions.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return FileResponse("app/static/index.html")
