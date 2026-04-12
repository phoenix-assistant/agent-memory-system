"""Qdrant vector backend."""

from __future__ import annotations

from typing import Any

from agent_memory.backends.base import VectorBackend


class QdrantBackend(VectorBackend):
    """Qdrant-based vector storage and search."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "agent_memories",
        embedding_dimensions: int = 384,
    ) -> None:
        self.url = url or "http://localhost:6333"
        self.api_key = api_key
        self.collection_name = collection_name
        self.embedding_dimensions = embedding_dimensions
        self._client: Any = None

    async def initialize(self) -> None:
        """Initialize Qdrant client and collection."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )

        # Check if collection exists
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )

    async def close(self) -> None:
        """Close Qdrant client."""
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> Any:
        if not self._client:
            raise RuntimeError("Qdrant not initialized. Call initialize() first.")
        return self._client

    async def add(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the collection."""
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload=metadata or {},
                )
            ],
        )

    async def update(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update a vector in the collection."""
        # Qdrant upsert handles both insert and update
        await self.add(memory_id, embedding, metadata)

    async def delete(self, memory_id: str) -> bool:
        """Delete a vector by ID."""
        from qdrant_client.models import PointIdsList

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[memory_id]),
            )
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
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )

        return [(str(r.id), r.score) for r in results]

    async def get_embedding(self, memory_id: str) -> list[float] | None:
        """Get the embedding for a memory ID."""
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_vectors=True,
            )
            if results and results[0].vector:
                return list(results[0].vector)
            return None
        except Exception:
            return None

    async def count(self) -> int:
        """Get total number of vectors."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    async def reset(self) -> None:
        """Clear all vectors (useful for testing)."""
        self.client.delete_collection(self.collection_name)
        await self.initialize()
