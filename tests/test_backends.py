"""Tests for storage and vector backends."""

import tempfile
from pathlib import Path

import pytest

from agent_memory.backends.chroma import ChromaBackend
from agent_memory.backends.sqlite import SQLiteBackend
from agent_memory.core.models import MemoryEntry, MemoryType


class TestSQLiteBackend:
    """Test SQLite storage backend."""

    @pytest.fixture
    async def backend(self):
        """Create a temporary SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path)
            await backend.initialize()
            yield backend
            await backend.close()

    async def test_store_and_get(self, backend: SQLiteBackend):
        """Test storing and retrieving a memory."""
        entry = MemoryEntry(
            content="Test content",
            memory_type=MemoryType.FACT,
            tags=["test"],
        )

        memory_id = await backend.store(entry)

        retrieved = await backend.get(memory_id)

        assert retrieved is not None
        assert retrieved.id == memory_id
        assert retrieved.content == "Test content"
        assert retrieved.tags == ["test"]

    async def test_update(self, backend: SQLiteBackend):
        """Test updating a memory."""
        entry = MemoryEntry(content="Original")
        await backend.store(entry)

        entry.content = "Updated"
        await backend.update(entry)

        retrieved = await backend.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Updated"

    async def test_delete(self, backend: SQLiteBackend):
        """Test deleting a memory."""
        entry = MemoryEntry(content="To delete")
        await backend.store(entry)

        result = await backend.delete(entry.id)

        assert result is True
        assert await backend.get(entry.id) is None

    async def test_list_with_type_filter(self, backend: SQLiteBackend):
        """Test listing with type filter."""
        await backend.store(MemoryEntry(content="Fact", memory_type=MemoryType.FACT))
        await backend.store(MemoryEntry(content="Pref", memory_type=MemoryType.PREFERENCE))

        facts = await backend.list(memory_type="fact")

        assert len(facts) == 1
        assert facts[0].content == "Fact"

    async def test_list_with_tag_filter(self, backend: SQLiteBackend):
        """Test listing with tag filter."""
        await backend.store(MemoryEntry(content="Tagged", tags=["important"]))
        await backend.store(MemoryEntry(content="Untagged", tags=[]))

        results = await backend.list(tags=["important"])

        assert len(results) == 1
        assert results[0].content == "Tagged"

    async def test_full_text_search(self, backend: SQLiteBackend):
        """Test full-text search."""
        await backend.store(MemoryEntry(content="Python is a programming language"))
        await backend.store(MemoryEntry(content="JavaScript runs in browsers"))

        results = await backend.search_text("programming")

        assert len(results) == 1
        assert "Python" in results[0][0].content

    async def test_get_by_ids(self, backend: SQLiteBackend):
        """Test getting multiple memories by IDs."""
        e1 = MemoryEntry(content="One")
        e2 = MemoryEntry(content="Two")
        e3 = MemoryEntry(content="Three")

        await backend.store(e1)
        await backend.store(e2)
        await backend.store(e3)

        results = await backend.get_by_ids([e1.id, e3.id])

        assert len(results) == 2
        contents = {r.content for r in results}
        assert "One" in contents
        assert "Three" in contents

    async def test_stats(self, backend: SQLiteBackend):
        """Test getting statistics."""
        await backend.store(MemoryEntry(content="One", memory_type=MemoryType.FACT))
        await backend.store(MemoryEntry(content="Two", memory_type=MemoryType.FACT))
        await backend.store(MemoryEntry(content="Three", memory_type=MemoryType.PREFERENCE))

        stats = await backend.get_stats()

        assert stats.total_memories == 3
        assert stats.memories_by_type["fact"] == 2
        assert stats.memories_by_type["preference"] == 1


class TestChromaBackend:
    """Test ChromaDB vector backend."""

    @pytest.fixture
    async def backend(self):
        """Create an in-memory ChromaDB backend."""
        backend = ChromaBackend(collection_name="test_memories")
        await backend.initialize()
        yield backend
        await backend.close()

    async def test_add_and_search(self, backend: ChromaBackend):
        """Test adding and searching vectors."""
        # Simple test embedding
        embedding = [0.1] * 384

        await backend.add("mem-1", embedding, {"type": "fact"})

        results = await backend.search(embedding, limit=5)

        assert len(results) == 1
        assert results[0][0] == "mem-1"
        assert results[0][1] > 0.99  # Should be near-perfect match

    async def test_update(self, backend: ChromaBackend):
        """Test updating a vector."""
        embedding1 = [0.1] * 384
        embedding2 = [0.9] * 384

        await backend.add("mem-1", embedding1)
        await backend.update("mem-1", embedding2)

        # Search with new embedding should find it
        results = await backend.search(embedding2, limit=5)

        assert len(results) == 1
        assert results[0][0] == "mem-1"

    async def test_delete(self, backend: ChromaBackend):
        """Test deleting a vector."""
        embedding = [0.5] * 384

        await backend.add("mem-1", embedding)
        await backend.delete("mem-1")

        results = await backend.search(embedding, limit=5)

        assert len(results) == 0

    async def test_get_embedding(self, backend: ChromaBackend):
        """Test retrieving an embedding."""
        embedding = [0.3] * 384

        await backend.add("mem-1", embedding)

        retrieved = await backend.get_embedding("mem-1")

        assert retrieved is not None
        assert len(retrieved) == 384
        assert abs(retrieved[0] - 0.3) < 0.01

    async def test_count(self, backend: ChromaBackend):
        """Test counting vectors."""
        embedding = [0.5] * 384

        await backend.add("mem-1", embedding)
        await backend.add("mem-2", embedding)
        await backend.add("mem-3", embedding)

        count = await backend.count()

        assert count == 3
