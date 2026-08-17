# AI Document Assistant — RAG-Powered Q&A API

A Retrieval-Augmented Generation (RAG) service that lets users upload documents (PDF/TXT/DOCX) and ask natural-language questions against them, with cited answers.

Documents are ingested, chunked, embedded, and indexed in a vector store; questions are answered by retrieving the most relevant chunks and grounding the LLM's response in them.

## Features

- **RAG pipeline** end-to-end — ingestion, chunking, embeddings, vector search, grounded answers
- **FastAPI** service with clean separation of concerns (API / services / core / models)
- **Vector search** (FAISS) with swappable embedding backends
- **Dockerized**, config-driven, environment-based deployment
- **Pytest** coverage for API and service layers
- Basic error handling, logging, and input validation

## Architecture

```
                 ┌─────────────┐
   Upload doc →  │  FastAPI    │
                 │  /ingest    │──▶ Chunking (RecursiveCharacterTextSplitter)
                 └─────────────┘         │
                                         ▼
                                 Embeddings (sentence-transformers)
                                         │
                                         ▼
                                 FAISS Vector Store (persisted to disk)

                 ┌─────────────┐
   Ask question → │  FastAPI    │
                 │  /ask       │──▶ Retrieve top-k relevant chunks
                 └─────────────┘         │
                                         ▼
                                 LLM (pluggable via config)
                                         │
                                         ▼
                                 Answer + source citations
```

## Tech Stack

- **Backend:** FastAPI, Pydantic v2
- **RAG:** LangChain, FAISS (vector store), sentence-transformers (embeddings)
- **LLM:** Pluggable provider, configured via environment variables
- **Testing:** Pytest, httpx
- **Infra:** Docker, docker-compose
- **Language:** Python 3.11

## Project Structure

```
ai-document-assistant/
├── app/
│   ├── main.py                 # FastAPI app entrypoint
│   ├── api/
│   │   └── routes.py           # /ingest, /ask, /health endpoints
│   ├── core/
│   │   └── config.py           # env-based settings (pydantic-settings)
│   ├── services/
│   │   ├── ingestion.py        # document loading + chunking
│   │   ├── vectorstore.py      # FAISS index management
│   │   └── rag_engine.py       # retrieval + LLM answer generation
│   └── models/
│       └── schemas.py          # request/response pydantic models
├── tests/
│   ├── test_ingestion.py
│   └── test_api.py
├── data/                        # uploaded docs + persisted vector index
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

## Setup (Local)

```bash
git clone <your-repo-url>
cd ai-document-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add your LLM_API_KEY and LLM_MODEL
uvicorn app.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

## Setup (Docker)

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint  | Description                              |
|--------|-----------|-------------------------------------------|
| POST   | `/ingest` | Upload a document, chunk it, embed it     |
| POST   | `/ask`    | Ask a question, get a grounded answer     |
| GET    | `/health` | Health check                              |

### Example

```bash
curl -X POST http://localhost:8000/ingest -F "file=@handbook.pdf"

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?"}'
```

## Running Tests

```bash
pytest -v
```

## Possible Extensions

- Swap FAISS for pgvector/Postgres for multi-user persistence
- Add conversation memory for follow-up questions
- Add streaming responses (SSE) for token-by-token answers
- Add auth (JWT) for multi-tenant document isolation

## Author

Pankaj Khoche — Python Application Developer
[LinkedIn] · [GitHub]
