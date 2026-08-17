import os
import shutil
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.models.schemas import IngestResponse, AskRequest, AskResponse, SourceChunk, HealthResponse
from app.services.ingestion import process_document
from app.services.vectorstore import vector_store
from app.services.rag_engine import answer_question

logger = logging.getLogger(__name__)
router = APIRouter()

os.makedirs(settings.upload_dir, exist_ok=True)


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", documents_indexed=vector_store.total_indexed)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    allowed_ext = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    save_path = os.path.join(settings.upload_dir, file.filename)
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = process_document(save_path, source_name=file.filename)
        indexed_count = vector_store.add_chunks(chunks)

        return IngestResponse(filename=file.filename, chunks_indexed=indexed_count)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Failed to process document.") from e


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    try:
        answer, sources = answer_question(payload.question)
        return AskResponse(
            answer=answer,
            sources=[SourceChunk(content=s["content"][:300], source=s["source"]) for s in sources],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Question answering failed")
        raise HTTPException(status_code=500, detail="Failed to answer question.") from e
