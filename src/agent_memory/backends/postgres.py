"""PostgreSQL storage backend."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from agent_memory.backends.base import StorageBackend
from agent_memory.core.models import MemoryEntry, MemoryStats, MemoryType

if TYPE_CHECKING:
    import asyncpg


class PostgresBackend(StorageBackend):
    """PostgreSQL-based storage for memory entries."""
    
    def __init__(self, connection_url: str) -> None:
        self.connection_url = connection_url
        self._pool: asyncpg.Pool | None = None
    
    async def initialize(self) -> None:
        """Create database pool and tables."""
        import asyncpg
        
        self._pool = await asyncpg.create_pool(self.connection_url)
        
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'fact',
                    source TEXT,
                    tags JSONB DEFAULT '[]',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    accessed_at TIMESTAMPTZ NOT NULL,
                    importance REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    correction_weight REAL DEFAULT 1.0,
                    corrected_by JSONB DEFAULT '[]',
                    corrects JSONB DEFAULT '[]',
                    is_compressed BOOLEAN DEFAULT FALSE,
                    source_memories JSONB DEFAULT '[]'
                );
                
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at);
                CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
                CREATE INDEX IF NOT EXISTS idx_memories_compressed ON memories(is_compressed);
                CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags);
                
                -- Full-text search
                CREATE INDEX IF NOT EXISTS idx_memories_content_fts 
                    ON memories USING GIN(to_tsvector('english', content));
            """)
    
    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    @property
    def pool(self) -> "asyncpg.Pool":
        if not self._pool:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._pool
    
    def _row_to_memory(self, row: dict) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            source=row["source"],
            tags=row["tags"],
            metadata=row["metadata"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            accessed_at=row["accessed_at"],
            importance=row["importance"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            correction_weight=row["correction_weight"],
            corrected_by=row["corrected_by"],
            corrects=row["corrects"],
            is_compressed=row["is_compressed"],
            source_memories=row["source_memories"],
        )
    
    async def store(self, memory: MemoryEntry) -> str:
        """Store a memory entry."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (
                    id, content, memory_type, source, tags, metadata,
                    created_at, updated_at, accessed_at,
                    importance, confidence, access_count,
                    correction_weight, corrected_by, corrects,
                    is_compressed, source_memories
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                memory.id,
                memory.content,
                memory.memory_type.value,
                memory.source,
                json.dumps(memory.tags),
                json.dumps(memory.metadata),
                memory.created_at,
                memory.updated_at,
                memory.accessed_at,
                memory.importance,
                memory.confidence,
                memory.access_count,
                memory.correction_weight,
                json.dumps(memory.corrected_by),
                json.dumps(memory.corrects),
                memory.is_compressed,
                json.dumps(memory.source_memories),
            )
        return memory.id
    
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memories WHERE id = $1",
                memory_id,
            )
            if row is None:
                return None
            return self._row_to_memory(dict(row))
    
    async def update(self, memory: MemoryEntry) -> None:
        """Update an existing memory."""
        memory.updated_at = datetime.utcnow()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memories SET
                    content = $1, memory_type = $2, source = $3, tags = $4, metadata = $5,
                    updated_at = $6, accessed_at = $7,
                    importance = $8, confidence = $9, access_count = $10,
                    correction_weight = $11, corrected_by = $12, corrects = $13,
                    is_compressed = $14, source_memories = $15
                WHERE id = $16
                """,
                memory.content,
                memory.memory_type.value,
                memory.source,
                json.dumps(memory.tags),
                json.dumps(memory.metadata),
                memory.updated_at,
                memory.accessed_at,
                memory.importance,
                memory.confidence,
                memory.access_count,
                memory.correction_weight,
                json.dumps(memory.corrected_by),
                json.dumps(memory.corrects),
                memory.is_compressed,
                json.dumps(memory.source_memories),
                memory.id,
            )
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memories WHERE id = $1",
                memory_id,
            )
            return result == "DELETE 1"
    
    async def list(
        self,
        *,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List memories with optional filtering."""
        query = "SELECT * FROM memories WHERE TRUE"
        params: list = []
        param_count = 0
        
        if memory_type:
            param_count += 1
            query += f" AND memory_type = ${param_count}"
            params.append(memory_type)
        
        if tags:
            for tag in tags:
                param_count += 1
                query += f" AND tags ? ${param_count}"
                params.append(tag)
        
        param_count += 1
        query += f" ORDER BY updated_at DESC LIMIT ${param_count}"
        params.append(limit)
        
        param_count += 1
        query += f" OFFSET ${param_count}"
        params.append(offset)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_memory(dict(row)) for row in rows]
    
    async def get_by_ids(self, ids: list[str]) -> list[MemoryEntry]:
        """Get multiple memories by their IDs."""
        if not ids:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM memories WHERE id = ANY($1)",
                ids,
            )
            return [self._row_to_memory(dict(row)) for row in rows]
    
    async def search_text(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, float]]:
        """Full-text search."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *, ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as score
                FROM memories
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                query,
                limit,
            )
            return [(self._row_to_memory(dict(row)), row["score"]) for row in rows]
    
    async def get_stats(self) -> MemoryStats:
        """Get storage statistics."""
        async with self.pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM memories")
            
            by_type_rows = await conn.fetch(
                "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
            )
            by_type = {row["memory_type"]: row["count"] for row in by_type_rows}
            
            corrections = await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE memory_type = 'correction'"
            )
            
            compressed = await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE is_compressed = TRUE"
            )
            
            avg_importance = await conn.fetchval(
                "SELECT AVG(importance) FROM memories"
            ) or 0.0
            
            date_row = await conn.fetchrow(
                "SELECT MIN(created_at), MAX(created_at) FROM memories"
            )
            oldest = date_row[0] if date_row else None
            newest = date_row[1] if date_row else None
            
            storage_bytes = await conn.fetchval(
                "SELECT pg_total_relation_size('memories')"
            ) or 0
            
            return MemoryStats(
                total_memories=total or 0,
                memories_by_type=by_type,
                total_corrections=corrections or 0,
                compressed_memories=compressed or 0,
                average_importance=float(avg_importance),
                oldest_memory=oldest,
                newest_memory=newest,
                storage_bytes=storage_bytes,
            )
    
    async def get_stale_memories(
        self,
        days_threshold: int,
        min_count: int,
    ) -> list[list[MemoryEntry]]:
        """Get clusters of stale memories for compression."""
        threshold = datetime.utcnow() - timedelta(days=days_threshold)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM memories 
                WHERE accessed_at < $1 
                AND is_compressed = FALSE
                AND memory_type != 'correction'
                ORDER BY tags, created_at
                """,
                threshold,
            )
            memories = [self._row_to_memory(dict(row)) for row in rows]
        
        # Simple clustering by tags
        clusters: dict[str, list[MemoryEntry]] = {}
        for memory in memories:
            key = ",".join(sorted(memory.tags)) if memory.tags else "_untagged"
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(memory)
        
        return [mems for mems in clusters.values() if len(mems) >= min_count]
