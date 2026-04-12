"""Core data models for the memory system."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Types of memories that can be stored."""

    FACT = "fact"                    # Declarative knowledge: "Project uses pnpm"
    PREFERENCE = "preference"        # User preferences: "Prefers dark mode"
    PROCEDURE = "procedure"          # How to do something: "Deploy by running X"
    ENTITY = "entity"                # Named entity: person, project, tool
    CORRECTION = "correction"        # Explicit correction of prior belief
    EPISODE = "episode"              # Event/interaction summary
    SUMMARY = "summary"              # Compressed collection of memories


class MemoryEntry(BaseModel):
    """A single memory entry in the system."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The memory content")
    memory_type: MemoryType = Field(default=MemoryType.FACT)

    # Metadata
    source: str | None = Field(default=None, description="Where this memory came from")
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)

    # Scoring
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in accuracy")
    access_count: int = Field(default=0, description="Number of times retrieved")

    # Correction tracking
    correction_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight adjustment from corrections (0 = fully suppressed)"
    )
    corrected_by: list[str] = Field(
        default_factory=list,
        description="IDs of memories that corrected this one"
    )
    corrects: list[str] = Field(
        default_factory=list,
        description="IDs of memories this one corrects"
    )

    # Compression
    is_compressed: bool = Field(default=False)
    source_memories: list[str] = Field(
        default_factory=list,
        description="IDs of memories this was compressed from"
    )

    # Embedding (stored separately, this is for convenience)
    embedding: list[float] | None = Field(default=None, exclude=True)

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

    def apply_correction(self, correction_id: str, weight_reduction: float = 0.3) -> None:
        """Apply a correction to this memory."""
        self.corrected_by.append(correction_id)
        self.correction_weight = max(0.0, self.correction_weight - weight_reduction)
        self.updated_at = datetime.utcnow()

    def effective_score(self, recency_weight: float = 0.3) -> float:
        """Calculate effective retrieval score combining all factors."""
        # Base score from importance and confidence
        base = self.importance * self.confidence * self.correction_weight

        # Recency boost (decay over 30 days)
        age_days = (datetime.utcnow() - self.accessed_at).days
        recency = max(0.0, 1.0 - (age_days / 30))

        # Access frequency boost (log scale)
        import math
        frequency = math.log1p(self.access_count) / 10

        return base * (1 - recency_weight) + recency * recency_weight + frequency * 0.1


class CorrectionSignal(BaseModel):
    """A detected correction in a conversation."""

    original_memory_id: str | None = Field(
        default=None,
        description="ID of the memory being corrected, if identified"
    )
    original_content: str | None = Field(
        default=None,
        description="The content that was wrong"
    )
    corrected_content: str = Field(..., description="The correct information")
    correction_type: str = Field(
        default="explicit",
        description="How the correction was detected: explicit, implicit, contradiction"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: str = Field(..., description="The original text containing the correction")

    # What to do with affected memories
    suppress_original: bool = Field(default=True, description="Reduce weight of original")
    create_memory: bool = Field(default=True, description="Create a new memory from correction")


class SearchResult(BaseModel):
    """Result from a memory search."""

    memory: MemoryEntry
    score: float = Field(..., ge=0.0, description="Relevance score")
    match_type: str = Field(default="semantic", description="How it matched: semantic, keyword, exact")

    def __lt__(self, other: SearchResult) -> bool:
        return self.score < other.score


class CompressionResult(BaseModel):
    """Result from memory compression."""

    compressed_memory: MemoryEntry
    source_ids: list[str]
    compression_ratio: float = Field(..., description="Ratio of source to compressed size")
    preserved_facts: list[str] = Field(
        default_factory=list,
        description="Key facts preserved in compression"
    )


class SurfacingContext(BaseModel):
    """Context for proactive memory surfacing."""

    query: str = Field(..., description="Current user input or context")
    recent_topics: list[str] = Field(default_factory=list)
    active_entities: list[str] = Field(default_factory=list)
    excluded_ids: list[str] = Field(
        default_factory=list,
        description="Memory IDs to exclude (already surfaced)"
    )
    max_tokens: int = Field(default=500, description="Maximum tokens to surface")


class MemoryStats(BaseModel):
    """Statistics about the memory store."""

    total_memories: int = 0
    memories_by_type: dict[str, int] = Field(default_factory=dict)
    total_corrections: int = 0
    compressed_memories: int = 0
    average_importance: float = 0.0
    oldest_memory: datetime | None = None
    newest_memory: datetime | None = None
    storage_bytes: int = 0
