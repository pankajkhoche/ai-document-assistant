"""Retrieve relevant chunks from the vector store, build a grounded
prompt, and call the LLM for a cited answer."""
from typing import List, Tuple

from anthropic import Anthropic

from app.core.config import settings
from app.services.vectorstore import vector_store

_client = Anthropic(api_key=settings.llm_api_key) if settings.llm_api_key else None

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the answer isn't in the context, say you don't "
    "have enough information — never make up facts. Keep answers concise."
)


def _build_prompt(question: str, contexts: List[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in contexts
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, and mention which source(s) you used."
    )


def _call_llm(prompt: str) -> str:
    if _client is None or not settings.llm_model:
        raise RuntimeError(
            "LLM_API_KEY / LLM_MODEL not configured. Set them in your .env file to enable answers."
        )
    response = _client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def answer_question(question: str) -> Tuple[str, List[dict]]:
    results = vector_store.search(question)
    if not results:
        return (
            "No documents have been ingested yet. Upload a document via /ingest first.",
            [],
        )

    contexts = [chunk for chunk, _ in results]
    prompt = _build_prompt(question, contexts)
    answer = _call_llm(prompt)
    return answer, contexts
