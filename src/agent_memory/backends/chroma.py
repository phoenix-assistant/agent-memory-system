"""ChromaDB vector backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from agent_memory.backends.base import VectorBackend


class ChromaBackend(VectorBackend):
    """ChromaDB-based vector storage and search."""

    def __init__(
        self,
        path: Path | str | None = None,
        collection_name: str = "agent_memories",
    ) -> None:
        self.path = Path(path) if path else None
        self.collection_name = collection_name
        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        else:
            # In-memory for testing
            self._client = chromadb.Client(
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def close(self) -> None:
        """ChromaDB doesn't need explicit close."""
        pass

    @property
    def collection(self) -> chromadb.Collection:
        if not self._collection:
            raise RuntimeError("ChromaDB not initialized. Call initialize() first.")
        return self._collection

    async def add(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the collection."""
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None,
        )

    async def update(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update a vector in the collection."""
        self.collection.update(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[metadata] if metadata else None,
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete a vector by ID."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    async def search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filter_metadata,
            include=["distances"],
        )

        ids = results["ids"][0] if results["ids"] else []
        # ChromaDB returns distances, convert to similarity scores
        distances = results["distances"][0] if results["distances"] else []

        # Cosine distance to similarity: 1 - distance
        scores = [1 - d for d in distances]

        return list(zip(ids, scores))

    async def get_embedding(self, memory_id: str) -> list[float] | None:
        """Get the embedding for a memory ID."""
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["embeddings"],
            )
            if result["embeddings"] and result["embeddings"][0]:
                return list(result["embeddings"][0])
            return None
        except Exception:
            return None

    async def count(self) -> int:
        """Get total number of vectors."""
        return self.collection.count()

    async def reset(self) -> None:
        """Clear all vectors (useful for testing)."""
        if self._client:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
