"""Abstract base classes for storage and vector backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_memory.core.models import MemoryEntry, MemoryStats


class StorageBackend(ABC):
    """Abstract base for metadata/document storage."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (create tables, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup."""
        ...

    @abstractmethod
    async def store(self, memory: MemoryEntry) -> str:
        """Store a memory entry. Returns the memory ID."""
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        ...

    @abstractmethod
    async def update(self, memory: MemoryEntry) -> None:
        """Update an existing memory."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        ...

    @abstractmethod
    async def list(
        self,
        *,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List memories with optional filtering."""
        ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[MemoryEntry]:
        """Get multiple memories by their IDs."""
        ...

    @abstractmethod
    async def search_text(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, float]]:
        """Full-text search. Returns (memory, score) tuples."""
        ...

    @abstractmethod
    async def get_stats(self) -> MemoryStats:
        """Get storage statistics."""
        ...

    @abstractmethod
    async def get_stale_memories(
        self,
        days_threshold: int,
        min_count: int,
    ) -> list[list[MemoryEntry]]:
        """Get clusters of stale memories for compression."""
        ...


class VectorBackend(ABC):
    """Abstract base for vector similarity search."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector backend."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup."""
        ...

    @abstractmethod
    async def add(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        """Add a vector to the index."""
        ...

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        """Update a vector in the index."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a vector by ID."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        limit: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors. Returns (id, score) tuples."""
        ...

    @abstractmethod
    async def get_embedding(self, memory_id: str) -> list[float] | None:
        """Get the embedding for a memory ID."""
        ...
