"""Configuration for the memory system."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryConfig(BaseSettings):
    """Configuration for Agent Memory System."""
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Storage backend
    storage_backend: Literal["sqlite", "postgres"] = Field(
        default="sqlite",
        description="Storage backend to use"
    )
    
    # SQLite settings
    sqlite_path: Path = Field(
        default=Path.home() / ".agent-memory" / "memory.db",
        description="Path to SQLite database"
    )
    
    # PostgreSQL settings
    postgres_url: str | None = Field(
        default=None,
        description="PostgreSQL connection URL"
    )
    
    # Vector backend
    vector_backend: Literal["chroma", "qdrant", "memory"] = Field(
        default="chroma",
        description="Vector database backend"
    )
    
    # ChromaDB settings
    chroma_path: Path = Field(
        default=Path.home() / ".agent-memory" / "chroma",
        description="Path to ChromaDB storage"
    )
    chroma_collection: str = Field(
        default="agent_memories",
        description="ChromaDB collection name"
    )
    
    # Qdrant settings
    qdrant_url: str | None = Field(
        default=None,
        description="Qdrant server URL"
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key"
    )
    qdrant_collection: str = Field(
        default="agent_memories",
        description="Qdrant collection name"
    )
    
    # Embedding settings
    embedding_backend: Literal["sentence_transformers", "openai"] = Field(
        default="sentence_transformers",
        description="Embedding model backend"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    embedding_dimensions: int = Field(
        default=384,
        description="Embedding vector dimensions"
    )
    
    # OpenAI settings (for embeddings and compression)
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    
    # Compression settings
    compression_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for compression/summarization"
    )
    compression_threshold_days: int = Field(
        default=30,
        description="Days after which to consider compression"
    )
    compression_min_memories: int = Field(
        default=5,
        description="Minimum memories in cluster to compress"
    )
    
    # Retrieval settings
    default_search_limit: int = Field(
        default=10,
        description="Default number of results to return"
    )
    min_relevance_score: float = Field(
        default=0.3,
        description="Minimum relevance score for results"
    )
    recency_weight: float = Field(
        default=0.3,
        description="Weight given to recency in scoring"
    )
    
    # Correction detection
    correction_weight_reduction: float = Field(
        default=0.3,
        description="How much to reduce weight on correction"
    )
    
    # Surfacing settings
    surfacing_enabled: bool = Field(
        default=True,
        description="Enable proactive memory surfacing"
    )
    surfacing_max_tokens: int = Field(
        default=500,
        description="Maximum tokens to include in surfacing"
    )
    surfacing_min_score: float = Field(
        default=0.5,
        description="Minimum score for proactive surfacing"
    )
    
    # MCP settings
    mcp_host: str = Field(
        default="127.0.0.1",
        description="MCP server host"
    )
    mcp_port: int = Field(
        default=3333,
        description="MCP server port"
    )
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def local_default(cls) -> MemoryConfig:
        """Get default config for local usage."""
        return cls(
            storage_backend="sqlite",
            vector_backend="chroma",
            embedding_backend="sentence_transformers",
        )
    
    @classmethod
    def server_default(cls, postgres_url: str, qdrant_url: str) -> MemoryConfig:
        """Get default config for server deployment."""
        return cls(
            storage_backend="postgres",
            postgres_url=postgres_url,
            vector_backend="qdrant",
            qdrant_url=qdrant_url,
            embedding_backend="sentence_transformers",
        )
