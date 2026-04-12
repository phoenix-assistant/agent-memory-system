"""OpenAI embedding backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_memory.embeddings.base import EmbeddingBackend

if TYPE_CHECKING:
    from openai import AsyncOpenAI


# Dimensions for known OpenAI models
OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIBackend(EmbeddingBackend):
    """OpenAI embeddings API backend."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        """Lazily create the client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return OPENAI_DIMENSIONS.get(self.model, 1536)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        client = self._ensure_client()
        response = await client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        client = self._ensure_client()

        # OpenAI API has a limit of ~8k tokens per request
        # For safety, batch in groups of 100
        all_embeddings: list[list[float]] = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await client.embeddings.create(
                model=self.model,
                input=batch,
            )
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return all_embeddings

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None
