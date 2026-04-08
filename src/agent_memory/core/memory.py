"""Main Memory class - the primary interface for the memory system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_memory.backends.base import StorageBackend, VectorBackend
from agent_memory.core.config import MemoryConfig
from agent_memory.core.models import (
    CorrectionSignal,
    MemoryEntry,
    MemoryStats,
    MemoryType,
    SearchResult,
    SurfacingContext,
)
from agent_memory.embeddings.base import EmbeddingBackend

if TYPE_CHECKING:
    from agent_memory.compression.compressor import MemoryCompressor


class Memory:
    """Main interface for the Agent Memory System.
    
    Example:
        ```python
        from agent_memory import Memory, MemoryConfig
        
        async def main():
            config = MemoryConfig.local_default()
            memory = Memory(config)
            await memory.initialize()
            
            # Store a memory
            await memory.add("Project uses pnpm, not npm", tags=["project", "tooling"])
            
            # Search memories
            results = await memory.search("package manager")
            
            # Apply a correction
            await memory.correct(
                original="Uses npm for packages",
                correction="Uses pnpm, not npm",
            )
            
            await memory.close()
        ```
    """
    
    def __init__(
        self,
        config: MemoryConfig | None = None,
    ) -> None:
        self.config = config or MemoryConfig.local_default()
        self._storage: StorageBackend | None = None
        self._vector: VectorBackend | None = None
        self._embeddings: EmbeddingBackend | None = None
        self._compressor: "MemoryCompressor | None" = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all backends."""
        if self._initialized:
            return
        
        self.config.ensure_directories()
        
        # Initialize storage backend
        if self.config.storage_backend == "sqlite":
            from agent_memory.backends.sqlite import SQLiteBackend
            self._storage = SQLiteBackend(self.config.sqlite_path)
        else:
            from agent_memory.backends.postgres import PostgresBackend
            if not self.config.postgres_url:
                raise ValueError("postgres_url required for postgres backend")
            self._storage = PostgresBackend(self.config.postgres_url)
        
        await self._storage.initialize()
        
        # Initialize vector backend
        if self.config.vector_backend == "chroma":
            from agent_memory.backends.chroma import ChromaBackend
            self._vector = ChromaBackend(
                path=self.config.chroma_path,
                collection_name=self.config.chroma_collection,
            )
        elif self.config.vector_backend == "qdrant":
            from agent_memory.backends.qdrant import QdrantBackend
            self._vector = QdrantBackend(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                collection_name=self.config.qdrant_collection,
                embedding_dimensions=self.config.embedding_dimensions,
            )
        else:
            # In-memory ChromaDB for testing
            from agent_memory.backends.chroma import ChromaBackend
            self._vector = ChromaBackend(collection_name=self.config.chroma_collection)
        
        await self._vector.initialize()
        
        # Initialize embedding backend
        if self.config.embedding_backend == "openai":
            from agent_memory.embeddings.openai import OpenAIBackend
            self._embeddings = OpenAIBackend(
                api_key=self.config.openai_api_key,
                model=self.config.openai_embedding_model,
            )
        else:
            from agent_memory.embeddings.sentence_transformers import (
                SentenceTransformersBackend,
            )
            self._embeddings = SentenceTransformersBackend(
                model_name=self.config.embedding_model,
            )
        
        self._initialized = True
    
    async def close(self) -> None:
        """Close all backends and cleanup."""
        if self._storage:
            await self._storage.close()
        if self._vector:
            await self._vector.close()
        if self._embeddings:
            await self._embeddings.close()
        if self._compressor:
            await self._compressor.close()
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Memory not initialized. Call initialize() first.")
    
    @property
    def storage(self) -> StorageBackend:
        self._ensure_initialized()
        assert self._storage is not None
        return self._storage
    
    @property
    def vector(self) -> VectorBackend:
        self._ensure_initialized()
        assert self._vector is not None
        return self._vector
    
    @property
    def embeddings(self) -> EmbeddingBackend:
        self._ensure_initialized()
        assert self._embeddings is not None
        return self._embeddings
    
    async def add(
        self,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.FACT,
        source: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """Add a new memory.
        
        Args:
            content: The memory content
            memory_type: Type of memory (fact, preference, procedure, etc.)
            source: Where this memory came from
            tags: Tags for organization and filtering
            importance: Importance score 0.0-1.0
            metadata: Additional metadata
            
        Returns:
            The created MemoryEntry
        """
        self._ensure_initialized()
        
        memory = MemoryEntry(
            content=content,
            memory_type=memory_type,
            source=source,
            tags=tags or [],
            importance=importance,
            metadata=metadata or {},
        )
        
        # Store in storage backend
        await self.storage.store(memory)
        
        # Generate and store embedding
        embedding = await self.embeddings.embed(content)
        await self.vector.add(
            memory.id,
            embedding,
            metadata={"type": memory_type.value, "tags": tags or []},
        )
        
        return memory
    
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        self._ensure_initialized()
        memory = await self.storage.get(memory_id)
        if memory:
            memory.touch()
            await self.storage.update(memory)
        return memory
    
    async def update(self, memory: MemoryEntry) -> None:
        """Update an existing memory."""
        self._ensure_initialized()
        await self.storage.update(memory)
        
        # Re-embed if content changed
        embedding = await self.embeddings.embed(memory.content)
        await self.vector.update(
            memory.id,
            embedding,
            metadata={"type": memory.memory_type.value, "tags": memory.tags},
        )
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        self._ensure_initialized()
        await self.vector.delete(memory_id)
        return await self.storage.delete(memory_id)
    
    async def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        min_score: float | None = None,
        include_suppressed: bool = False,
    ) -> list[SearchResult]:
        """Search memories using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum results to return
            memory_type: Filter by memory type
            tags: Filter by tags
            min_score: Minimum relevance score
            include_suppressed: Include memories with low correction_weight
            
        Returns:
            List of SearchResult sorted by relevance
        """
        self._ensure_initialized()
        
        limit = limit or self.config.default_search_limit
        min_score = min_score or self.config.min_relevance_score
        
        # Generate query embedding
        query_embedding = await self.embeddings.embed(query)
        
        # Vector search
        filter_metadata = {}
        if memory_type:
            filter_metadata["type"] = memory_type.value
        
        vector_results = await self.vector.search(
            query_embedding,
            limit=limit * 2,  # Get extra for filtering
            filter_metadata=filter_metadata if filter_metadata else None,
        )
        
        if not vector_results:
            return []
        
        # Get full memory entries
        memory_ids = [r[0] for r in vector_results]
        score_map = {r[0]: r[1] for r in vector_results}
        
        memories = await self.storage.get_by_ids(memory_ids)
        
        results: list[SearchResult] = []
        for memory in memories:
            # Filter by tags
            if tags and not all(t in memory.tags for t in tags):
                continue
            
            # Filter by correction weight
            if not include_suppressed and memory.correction_weight < 0.3:
                continue
            
            base_score = score_map.get(memory.id, 0.0)
            # Adjust score by correction weight and effective score
            adjusted_score = base_score * memory.effective_score(self.config.recency_weight)
            
            if adjusted_score >= min_score:
                results.append(SearchResult(
                    memory=memory,
                    score=adjusted_score,
                    match_type="semantic",
                ))
        
        # Sort by score and limit
        results.sort(reverse=True)
        results = results[:limit]
        
        # Touch accessed memories
        for result in results:
            result.memory.touch()
            await self.storage.update(result.memory)
        
        return results
    
    async def search_text(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Full-text search (keyword matching)."""
        self._ensure_initialized()
        
        text_results = await self.storage.search_text(query, limit=limit)
        
        results = []
        for memory, score in text_results:
            if memory.correction_weight >= 0.3:
                results.append(SearchResult(
                    memory=memory,
                    score=score * memory.correction_weight,
                    match_type="keyword",
                ))
        
        return results
    
    async def correct(
        self,
        original: str | None = None,
        correction: str = "",
        *,
        original_id: str | None = None,
        source: str | None = None,
        confidence: float = 1.0,
    ) -> MemoryEntry:
        """Apply a correction to the memory system.
        
        This is the key innovation - corrections adjust future retrievals:
        1. Finds memories similar to the original (wrong) content
        2. Reduces their correction_weight
        3. Creates a new correction memory
        4. Links them together
        
        Args:
            original: The incorrect content to correct (searches for similar)
            correction: The correct information
            original_id: Specific memory ID to correct (if known)
            source: Source of the correction
            confidence: Confidence in the correction
            
        Returns:
            The created correction memory
        """
        self._ensure_initialized()
        
        affected_ids: list[str] = []
        
        # Find memories to correct
        if original_id:
            affected_ids.append(original_id)
        elif original:
            # Search for similar memories
            similar = await self.search(original, limit=5, min_score=0.5)
            affected_ids = [r.memory.id for r in similar]
        
        # Apply correction weight reduction to affected memories
        for mem_id in affected_ids:
            memory = await self.storage.get(mem_id)
            if memory:
                memory.apply_correction(
                    "pending",  # Will update with real ID
                    weight_reduction=self.config.correction_weight_reduction,
                )
                await self.storage.update(memory)
        
        # Create correction memory
        correction_memory = MemoryEntry(
            content=correction,
            memory_type=MemoryType.CORRECTION,
            source=source or "user_correction",
            importance=0.8,  # Corrections are important
            confidence=confidence,
            corrects=affected_ids,
            metadata={
                "original_content": original,
            },
        )
        
        await self.storage.store(correction_memory)
        
        # Store embedding
        embedding = await self.embeddings.embed(correction)
        await self.vector.add(
            correction_memory.id,
            embedding,
            metadata={"type": "correction"},
        )
        
        # Update affected memories with actual correction ID
        for mem_id in affected_ids:
            memory = await self.storage.get(mem_id)
            if memory:
                # Replace "pending" with actual ID
                memory.corrected_by = [
                    correction_memory.id if c == "pending" else c
                    for c in memory.corrected_by
                ]
                await self.storage.update(memory)
        
        return correction_memory
    
    async def surface(
        self,
        context: SurfacingContext,
    ) -> list[MemoryEntry]:
        """Proactively surface relevant memories for the given context.
        
        Args:
            context: Current context (query, recent topics, etc.)
            
        Returns:
            List of relevant memories to inject into context
        """
        self._ensure_initialized()
        
        if not self.config.surfacing_enabled:
            return []
        
        # Search based on current query
        results = await self.search(
            context.query,
            limit=10,
            min_score=self.config.surfacing_min_score,
        )
        
        # Filter out excluded memories
        results = [r for r in results if r.memory.id not in context.excluded_ids]
        
        # Budget tokens
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        surfaced: list[MemoryEntry] = []
        total_tokens = 0
        
        for result in results:
            tokens = len(enc.encode(result.memory.content))
            if total_tokens + tokens <= context.max_tokens:
                surfaced.append(result.memory)
                total_tokens += tokens
            else:
                break
        
        return surfaced
    
    async def compress_stale(self) -> int:
        """Compress stale memories into summaries.
        
        Returns:
            Number of memories compressed
        """
        self._ensure_initialized()
        
        if not self._compressor:
            from agent_memory.compression.compressor import MemoryCompressor
            self._compressor = MemoryCompressor(
                api_key=self.config.openai_api_key,
                model=self.config.compression_model,
            )
        
        # Get stale memory clusters
        clusters = await self.storage.get_stale_memories(
            days_threshold=self.config.compression_threshold_days,
            min_count=self.config.compression_min_memories,
        )
        
        compressed_count = 0
        
        for cluster in clusters:
            if not await self._compressor.should_compress(cluster):
                continue
            
            result = await self._compressor.compress(cluster)
            
            # Store compressed memory
            await self.storage.store(result.compressed_memory)
            
            # Store embedding
            embedding = await self.embeddings.embed(result.compressed_memory.content)
            await self.vector.add(
                result.compressed_memory.id,
                embedding,
                metadata={"type": "summary"},
            )
            
            # Mark source memories as compressed (don't delete - keep for audit)
            for mem_id in result.source_ids:
                memory = await self.storage.get(mem_id)
                if memory:
                    memory.is_compressed = True
                    memory.metadata["compressed_into"] = result.compressed_memory.id
                    await self.storage.update(memory)
            
            compressed_count += len(cluster)
        
        return compressed_count
    
    async def stats(self) -> MemoryStats:
        """Get memory system statistics."""
        self._ensure_initialized()
        return await self.storage.get_stats()
    
    async def list(
        self,
        *,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List memories with optional filtering."""
        self._ensure_initialized()
        return await self.storage.list(
            memory_type=memory_type.value if memory_type else None,
            tags=tags,
            limit=limit,
            offset=offset,
        )
    
    async def __aenter__(self) -> "Memory":
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        await self.close()
