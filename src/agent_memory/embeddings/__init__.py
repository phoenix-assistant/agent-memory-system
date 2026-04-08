"""Embedding backends."""

from agent_memory.embeddings.base import EmbeddingBackend
from agent_memory.embeddings.sentence_transformers import SentenceTransformersBackend

__all__ = [
    "EmbeddingBackend",
    "SentenceTransformersBackend",
]
