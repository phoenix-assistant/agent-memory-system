"""Abstract base class for embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingBackend(ABC):
    """Abstract base for embedding generation."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    async def close(self) -> None:  # noqa: B027
        """Cleanup resources."""
        pass
