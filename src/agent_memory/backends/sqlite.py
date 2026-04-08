"""SQLite storage backend."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from agent_memory.backends.base import StorageBackend
from agent_memory.core.models import MemoryEntry, MemoryStats, MemoryType

if TYPE_CHECKING:
    pass


class SQLiteBackend(StorageBackend):
    """SQLite-based storage for memory entries."""
    
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
    
    async def initialize(self) -> None:
        """Create database and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        
        await self._conn.executescript("""
            -- Main memories table
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'fact',
                source TEXT,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                correction_weight REAL DEFAULT 1.0,
                corrected_by TEXT DEFAULT '[]',
                corrects TEXT DEFAULT '[]',
                is_compressed INTEGER DEFAULT 0,
                source_memories TEXT DEFAULT '[]'
            );
            
            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_compressed ON memories(is_compressed);
            
            -- Full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id,
                content,
                tags,
                content='memories',
                content_rowid='rowid'
            );
            
            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(id, content, tags) 
                VALUES (new.id, new.content, new.tags);
            END;
            
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, id, content, tags) 
                VALUES ('delete', old.id, old.content, old.tags);
            END;
            
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, id, content, tags) 
                VALUES ('delete', old.id, old.content, old.tags);
                INSERT INTO memories_fts(id, content, tags) 
                VALUES (new.id, new.content, new.tags);
            END;
        """)
        await self._conn.commit()
    
    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
    
    @property
    def conn(self) -> aiosqlite.Connection:
        if not self._conn:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn
    
    def _row_to_memory(self, row: aiosqlite.Row) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            source=row["source"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
            importance=row["importance"],
            confidence=row["confidence"],
            access_count=row["access_count"],
            correction_weight=row["correction_weight"],
            corrected_by=json.loads(row["corrected_by"]),
            corrects=json.loads(row["corrects"]),
            is_compressed=bool(row["is_compressed"]),
            source_memories=json.loads(row["source_memories"]),
        )
    
    async def store(self, memory: MemoryEntry) -> str:
        """Store a memory entry."""
        await self.conn.execute(
            """
            INSERT INTO memories (
                id, content, memory_type, source, tags, metadata,
                created_at, updated_at, accessed_at,
                importance, confidence, access_count,
                correction_weight, corrected_by, corrects,
                is_compressed, source_memories
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.memory_type.value,
                memory.source,
                json.dumps(memory.tags),
                json.dumps(memory.metadata),
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                memory.accessed_at.isoformat(),
                memory.importance,
                memory.confidence,
                memory.access_count,
                memory.correction_weight,
                json.dumps(memory.corrected_by),
                json.dumps(memory.corrects),
                int(memory.is_compressed),
                json.dumps(memory.source_memories),
            ),
        )
        await self.conn.commit()
        return memory.id
    
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a memory by ID."""
        cursor = await self.conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)
    
    async def update(self, memory: MemoryEntry) -> None:
        """Update an existing memory."""
        memory.updated_at = datetime.utcnow()
        await self.conn.execute(
            """
            UPDATE memories SET
                content = ?, memory_type = ?, source = ?, tags = ?, metadata = ?,
                updated_at = ?, accessed_at = ?,
                importance = ?, confidence = ?, access_count = ?,
                correction_weight = ?, corrected_by = ?, corrects = ?,
                is_compressed = ?, source_memories = ?
            WHERE id = ?
            """,
            (
                memory.content,
                memory.memory_type.value,
                memory.source,
                json.dumps(memory.tags),
                json.dumps(memory.metadata),
                memory.updated_at.isoformat(),
                memory.accessed_at.isoformat(),
                memory.importance,
                memory.confidence,
                memory.access_count,
                memory.correction_weight,
                json.dumps(memory.corrected_by),
                json.dumps(memory.corrects),
                int(memory.is_compressed),
                json.dumps(memory.source_memories),
                memory.id,
            ),
        )
        await self.conn.commit()
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        cursor = await self.conn.execute(
            "DELETE FROM memories WHERE id = ?",
            (memory_id,),
        )
        await self.conn.commit()
        return cursor.rowcount > 0
    
    async def list(
        self,
        *,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List memories with optional filtering."""
        query = "SELECT * FROM memories WHERE 1=1"
        params: list = []
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        
        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')
        
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]
    
    async def get_by_ids(self, ids: list[str]) -> list[MemoryEntry]:
        """Get multiple memories by their IDs."""
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cursor = await self.conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            ids,
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]
    
    async def search_text(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, float]]:
        """Full-text search."""
        cursor = await self.conn.execute(
            """
            SELECT m.*, bm25(memories_fts) as score
            FROM memories_fts f
            JOIN memories m ON f.id = m.id
            WHERE memories_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            # BM25 returns negative scores, convert to positive
            score = -row["score"] if row["score"] else 0.0
            results.append((memory, score))
        return results
    
    async def get_stats(self) -> MemoryStats:
        """Get storage statistics."""
        # Total count
        cursor = await self.conn.execute("SELECT COUNT(*) FROM memories")
        total = (await cursor.fetchone())[0]
        
        # By type
        cursor = await self.conn.execute(
            "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
        )
        by_type = {row[0]: row[1] for row in await cursor.fetchall()}
        
        # Corrections
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE memory_type = 'correction'"
        )
        corrections = (await cursor.fetchone())[0]
        
        # Compressed
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE is_compressed = 1"
        )
        compressed = (await cursor.fetchone())[0]
        
        # Average importance
        cursor = await self.conn.execute("SELECT AVG(importance) FROM memories")
        avg_importance = (await cursor.fetchone())[0] or 0.0
        
        # Date range
        cursor = await self.conn.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM memories"
        )
        row = await cursor.fetchone()
        oldest = datetime.fromisoformat(row[0]) if row[0] else None
        newest = datetime.fromisoformat(row[1]) if row[1] else None
        
        # Storage size
        import os
        storage_bytes = os.path.getsize(self.db_path) if self.db_path.exists() else 0
        
        return MemoryStats(
            total_memories=total,
            memories_by_type=by_type,
            total_corrections=corrections,
            compressed_memories=compressed,
            average_importance=avg_importance,
            oldest_memory=oldest,
            newest_memory=newest,
            storage_bytes=storage_bytes,
        )
    
    async def get_stale_memories(
        self,
        days_threshold: int,
        min_count: int,
    ) -> list[list[MemoryEntry]]:
        """Get clusters of stale memories for compression.
        
        Returns memories grouped by similar tags/topics that are older than threshold.
        """
        threshold = datetime.utcnow()
        from datetime import timedelta
        threshold = threshold - timedelta(days=days_threshold)
        
        cursor = await self.conn.execute(
            """
            SELECT * FROM memories 
            WHERE accessed_at < ? 
            AND is_compressed = 0
            AND memory_type != 'correction'
            ORDER BY tags, created_at
            """,
            (threshold.isoformat(),),
        )
        rows = await cursor.fetchall()
        memories = [self._row_to_memory(row) for row in rows]
        
        # Simple clustering by tags
        clusters: dict[str, list[MemoryEntry]] = {}
        for memory in memories:
            key = ",".join(sorted(memory.tags)) if memory.tags else "_untagged"
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(memory)
        
        # Only return clusters with enough memories
        return [mems for mems in clusters.values() if len(mems) >= min_count]
