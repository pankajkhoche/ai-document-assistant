"""
Thin wrapper around a FAISS vector index for storing and retrieving
document chunk embeddings. Persists to disk so the index survives restarts.
"""
import os
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

os.makedirs(settings.vectorstore_dir, exist_ok=True)

INDEX_PATH = os.path.join(settings.vectorstore_dir, "index.faiss")
METADATA_PATH = os.path.join(settings.vectorstore_dir, "metadata.npy")


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata: List[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            self.metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()

    def _persist(self):
        faiss.write_index(self.index, INDEX_PATH)
        np.save(METADATA_PATH, np.array(self.metadata, dtype=object))

    def add_chunks(self, chunks: List[dict]) -> int:
        texts = [c["content"] for c in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(chunks)
        self._persist()
        return len(chunks)

    def search(self, query: str, top_k: int = None) -> List[Tuple[dict, float]]:
        if self.index.ntotal == 0:
            return []
        top_k = top_k or settings.top_k
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(dist)))
        return results

    @property
    def total_indexed(self) -> int:
        return self.index.ntotal


# Singleton instance shared across the app's lifetime
vector_store = VectorStore()
