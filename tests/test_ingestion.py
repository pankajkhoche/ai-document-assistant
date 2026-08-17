import os
import tempfile

import pytest

from app.services.ingestion import load_text, chunk_text, process_document


def test_load_text_txt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world, this is a test document.")
        path = f.name

    try:
        text = load_text(path)
        assert "Hello world" in text
    finally:
        os.remove(path)


def test_load_text_unsupported_extension():
    with pytest.raises(ValueError):
        load_text("document.xyz")


def test_chunk_text_produces_chunks_with_source():
    text = "This is a sentence. " * 200  # long enough to force multiple chunks
    chunks = chunk_text(text, source="sample.txt")

    assert len(chunks) > 1
    assert all(c["source"] == "sample.txt" for c in chunks)
    assert all(isinstance(c["content"], str) and c["content"] for c in chunks)


def test_process_document_end_to_end():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Some short content for testing the pipeline end to end.")
        path = f.name

    try:
        chunks = process_document(path, source_name="pipeline_test.txt")
        assert len(chunks) >= 1
        assert chunks[0]["source"] == "pipeline_test.txt"
    finally:
        os.remove(path)
