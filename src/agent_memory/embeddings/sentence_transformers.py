"""Sentence Transformers embedding backend."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING

from agent_memory.embeddings.base import EmbeddingBackend

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> "SentenceTransformer":
    """Cache loaded models."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class SentenceTransformersBackend(EmbeddingBackend):
    """Local embeddings using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self._model: "SentenceTransformer | None" = None
        self._dimensions: int | None = None
    
    def _ensure_model(self) -> "SentenceTransformer":
        """Lazily load the model."""
        if self._model is None:
            self._model = _get_model(self.model_name)
            # Get dimensions from a test embedding
            test_embed = self._model.encode("test")
            self._dimensions = len(test_embed)
        return self._model
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        self._ensure_model()
        return self._dimensions or 384  # Default for MiniLM
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        loop = asyncio.get_event_loop()
        model = self._ensure_model()
        
        # Run in thread pool since sentence-transformers is sync
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        loop = asyncio.get_event_loop()
        model = self._ensure_model()
        
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True)
        )
        return [e.tolist() for e in embeddings]
