from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "documents_indexed" in body


def test_ask_without_ingested_documents_or_key():
    # With an empty index, /ask should respond gracefully rather than error out
    response = client.post("/ask", json={"question": "What is this document about?"})
    assert response.status_code in (200, 503)


def test_ask_with_short_question_is_rejected():
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_ingest_rejects_unsupported_file_type():
    files = {"file": ("test.xyz", b"dummy content", "application/octet-stream")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 400
