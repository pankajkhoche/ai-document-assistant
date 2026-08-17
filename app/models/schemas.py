from typing import List, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str = "Document ingested successfully."


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question about ingested documents")


class SourceChunk(BaseModel):
    content: str
    source: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: Optional[int] = None
