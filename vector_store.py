from __future__ import annotations

import hashlib
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.config import Settings


class PineconeStore:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        self.client = client or Pinecone(api_key=settings.pinecone_api_key)
        self.index = None

    def _index_names(self) -> set[str]:
        return set(self.client.list_indexes().names())

    def connect(self, create_if_missing: bool = False, dimension: int | None = None) -> Any:
        name = self.settings.pinecone_index_name
        if name not in self._index_names():
            if not create_if_missing:
                raise LookupError(f"Pinecone index '{name}' has not been created.")
            if dimension is None or dimension < 1:
                raise ValueError("A valid embedding dimension is required to create the Pinecone index.")
            self.client.create_index(
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if self.client.describe_index(name).status.get("ready", False):
                    break
                time.sleep(2)
            else:
                raise TimeoutError("Pinecone index did not become ready within 90 seconds.")
        description = self.client.describe_index(name)
        if dimension is not None and description.dimension != dimension:
            raise ValueError(
                f"Pinecone index dimension is {description.dimension}, but Ollama produced "
                f"{dimension}. Use a new index name for this embedding model."
            )
        self.index = self.client.Index(name)
        return self.index

    def replace_documents(self, vectors: list[dict[str, Any]]) -> int:
        if not vectors:
            raise ValueError("Cannot create or replace a Pinecone index with no vectors.")
        dimension = len(vectors[0]["embedding"])
        if not dimension or any(len(item["embedding"]) != dimension for item in vectors):
            raise ValueError("All document embeddings must have the same non-zero dimension.")
        index = self.connect(create_if_missing=True, dimension=dimension)
        try:
            index.delete(delete_all=True, namespace=self.settings.pinecone_namespace)
        except Exception as exc:
            if "namespace not found" not in str(exc).lower():
                raise
        records = [
            {
                "id": item["id"],
                "values": item["embedding"],
                "metadata": item["metadata"],
            }
            for item in vectors
        ]
        for start in range(0, len(records), 100):
            index.upsert(vectors=records[start : start + 100], namespace=self.settings.pinecone_namespace)
        return len(vectors)

    def search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        index = self.index or self.connect()
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.settings.pinecone_namespace,
        )
        return [
            {
                "text": str((match.metadata or {}).get("text", "")),
                "page_number": int((match.metadata or {}).get("page_number", 0)),
                "source": str((match.metadata or {}).get("source", "")),
                "score": float(match.score or 0.0),
            }
            for match in result.matches
        ]

    def stats(self) -> dict[str, int]:
        if self.settings.pinecone_index_name not in self._index_names():
            return {"vector_count": 0}
        index = self.index or self.connect()
        result = index.describe_index_stats(namespace=self.settings.pinecone_namespace)
        namespace = result.namespaces or {}
        current = namespace.get(self.settings.pinecone_namespace)
        return {"vector_count": int(current.vector_count) if current else 0}


def make_vector_id(source: str, page_number: int, start_index: int, text: str) -> str:
    identity = f"{source}:{page_number}:{start_index}:{text}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()
