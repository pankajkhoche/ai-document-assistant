"""
Handles loading raw documents (PDF, DOCX, TXT) and splitting them into
overlapping chunks suitable for embedding + retrieval.
"""
import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document as DocxDocument

from app.core.config import settings


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


LOADERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".txt": _read_txt,
}


def load_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    loader = LOADERS.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS.keys())}")
    text = loader(file_path)
    if not text.strip():
        raise ValueError("No extractable text found in document.")
    return text


def chunk_text(text: str, source: str) -> List[dict]:
    """Splits text into overlapping chunks, tagging each with its source filename."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return [{"content": chunk, "source": source} for chunk in chunks]


def process_document(file_path: str, source_name: str) -> List[dict]:
    text = load_text(file_path)
    return chunk_text(text, source_name)
