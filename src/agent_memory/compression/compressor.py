"""Memory compression using LLM summarization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent_memory.core.models import (
    CompressionResult,
    MemoryEntry,
    MemoryType,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI


COMPRESSION_PROMPT = """You are a memory compression system. Given a list of related memories, create a single compressed summary that preserves:
1. All key facts and information
2. User preferences and corrections
3. Important context and relationships

Rules:
- Be concise but preserve all important details
- Maintain any specific values, names, or technical details
- If memories contradict, prefer the most recent/confident one
- Output a single coherent memory entry

Input memories:
{memories}

Output a JSON object with:
- "content": The compressed memory content (2-3 sentences max)
- "preserved_facts": Array of key facts preserved
- "importance": 0.0-1.0 importance score for the summary

Respond with only the JSON object."""


class MemoryCompressor:
    """Compresses multiple memories into summaries using LLM."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client: "AsyncOpenAI | None" = None
    
    def _ensure_client(self) -> "AsyncOpenAI":
        """Lazily create the client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client
    
    async def compress(
        self,
        memories: list[MemoryEntry],
    ) -> CompressionResult:
        """Compress multiple memories into a single summary.
        
        Args:
            memories: List of memories to compress
            
        Returns:
            CompressionResult with the compressed memory and metadata
        """
        if not memories:
            raise ValueError("Cannot compress empty memory list")
        
        if len(memories) == 1:
            # Nothing to compress
            return CompressionResult(
                compressed_memory=memories[0],
                source_ids=[memories[0].id],
                compression_ratio=1.0,
                preserved_facts=[memories[0].content],
            )
        
        client = self._ensure_client()
        
        # Format memories for the prompt
        memory_texts = []
        for i, mem in enumerate(memories, 1):
            memory_texts.append(
                f"{i}. [{mem.memory_type.value}] (importance: {mem.importance:.2f}, "
                f"confidence: {mem.confidence:.2f}): {mem.content}"
            )
        
        formatted_memories = "\n".join(memory_texts)
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise memory compression system.",
                },
                {
                    "role": "user",
                    "content": COMPRESSION_PROMPT.format(memories=formatted_memories),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content or "{}")
        
        # Calculate metrics
        original_chars = sum(len(m.content) for m in memories)
        compressed_chars = len(result.get("content", ""))
        compression_ratio = compressed_chars / original_chars if original_chars > 0 else 1.0
        
        # Collect all tags from source memories
        all_tags = set()
        for mem in memories:
            all_tags.update(mem.tags)
        
        # Create compressed memory
        compressed = MemoryEntry(
            content=result.get("content", ""),
            memory_type=MemoryType.SUMMARY,
            source="compression",
            tags=list(all_tags),
            importance=result.get("importance", 0.5),
            confidence=0.9,  # Slightly lower confidence for summaries
            is_compressed=True,
            source_memories=[m.id for m in memories],
        )
        
        return CompressionResult(
            compressed_memory=compressed,
            source_ids=[m.id for m in memories],
            compression_ratio=compression_ratio,
            preserved_facts=result.get("preserved_facts", []),
        )
    
    async def should_compress(
        self,
        memories: list[MemoryEntry],
        min_memories: int = 5,
        max_importance: float = 0.7,
    ) -> bool:
        """Determine if a set of memories should be compressed.
        
        Args:
            memories: Memories to evaluate
            min_memories: Minimum count to consider compression
            max_importance: Don't compress if any memory exceeds this importance
            
        Returns:
            True if compression is recommended
        """
        if len(memories) < min_memories:
            return False
        
        # Don't compress if any memory is very important
        if any(m.importance > max_importance for m in memories):
            return False
        
        # Don't compress corrections
        if any(m.memory_type == MemoryType.CORRECTION for m in memories):
            return False
        
        return True
    
    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None
