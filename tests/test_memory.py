"""Tests for core Memory functionality."""

import pytest

from agent_memory import Memory, MemoryConfig, MemoryEntry, MemoryType


@pytest.fixture
async def memory():
    """Create an in-memory test instance."""
    config = MemoryConfig(
        storage_backend="sqlite",
        sqlite_path=":memory:",
        vector_backend="memory",
        embedding_backend="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
    )
    mem = Memory(config)
    await mem.initialize()
    yield mem
    await mem.close()


class TestMemoryBasics:
    """Test basic memory operations."""

    async def test_add_memory(self, memory: Memory):
        """Test adding a memory."""
        entry = await memory.add(
            "Project uses pnpm for package management",
            memory_type=MemoryType.FACT,
            tags=["project", "tooling"],
            importance=0.7,
        )

        assert entry.id is not None
        assert entry.content == "Project uses pnpm for package management"
        assert entry.memory_type == MemoryType.FACT
        assert "project" in entry.tags
        assert entry.importance == 0.7

    async def test_get_memory(self, memory: Memory):
        """Test retrieving a memory by ID."""
        entry = await memory.add("Test content")

        retrieved = await memory.get(entry.id)

        assert retrieved is not None
        assert retrieved.id == entry.id
        assert retrieved.content == entry.content
        assert retrieved.access_count == 1  # Should be touched

    async def test_delete_memory(self, memory: Memory):
        """Test deleting a memory."""
        entry = await memory.add("To be deleted")

        result = await memory.delete(entry.id)

        assert result is True
        assert await memory.get(entry.id) is None

    async def test_update_memory(self, memory: Memory):
        """Test updating a memory."""
        entry = await memory.add("Original content")
        entry.content = "Updated content"
        entry.importance = 0.9

        await memory.update(entry)

        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Updated content"
        assert retrieved.importance == 0.9


class TestMemorySearch:
    """Test memory search functionality."""

    async def test_semantic_search(self, memory: Memory):
        """Test semantic search finds related content."""
        await memory.add("We use TypeScript for all frontend code", tags=["frontend"])
        await memory.add("Python is used for backend services", tags=["backend"])
        await memory.add("The deployment process uses Docker", tags=["devops"])

        results = await memory.search("What language for the UI?")

        assert len(results) > 0
        # TypeScript should be most relevant to UI/frontend
        assert "TypeScript" in results[0].memory.content

    async def test_search_with_type_filter(self, memory: Memory):
        """Test filtering search by memory type."""
        await memory.add("Prefer dark mode", memory_type=MemoryType.PREFERENCE)
        await memory.add("Dark mode is better for eyes", memory_type=MemoryType.FACT)

        results = await memory.search(
            "dark mode",
            memory_type=MemoryType.PREFERENCE,
        )

        assert len(results) == 1
        assert results[0].memory.memory_type == MemoryType.PREFERENCE

    async def test_search_with_tag_filter(self, memory: Memory):
        """Test filtering search by tags."""
        await memory.add("Use pnpm", tags=["tooling"])
        await memory.add("Use npm", tags=["legacy"])

        results = await memory.search("package manager", tags=["tooling"])

        assert len(results) == 1
        assert "pnpm" in results[0].memory.content

    async def test_search_respects_min_score(self, memory: Memory):
        """Test minimum score filtering."""
        await memory.add("Completely unrelated: cats are cute")

        results = await memory.search("deployment process", min_score=0.8)

        # Should filter out low-relevance results
        assert len(results) == 0


class TestCorrections:
    """Test correction learning functionality."""

    async def test_apply_correction(self, memory: Memory):
        """Test applying a correction."""
        wrong = await memory.add("Use npm for packages")

        correction = await memory.correct(
            original="npm",
            correction="Use pnpm, not npm",
        )

        assert correction.memory_type == MemoryType.CORRECTION
        assert correction.corrects  # Should have affected memories

        # Original should have reduced weight
        updated_wrong = await memory.get(wrong.id)
        assert updated_wrong is not None
        assert updated_wrong.correction_weight < 1.0

    async def test_correction_affects_search(self, memory: Memory):
        """Test that corrections affect search results."""
        await memory.add("Use npm for dependencies")

        await memory.correct(
            original="npm",
            correction="Use pnpm, not npm",
        )

        results = await memory.search("package manager")

        # pnpm correction should rank higher than suppressed npm
        pnpm_found = any("pnpm" in r.memory.content for r in results)
        assert pnpm_found

    async def test_suppressed_excluded_by_default(self, memory: Memory):
        """Test that heavily suppressed memories are excluded."""
        entry = await memory.add("Wrong information")

        # Apply multiple corrections to fully suppress
        for _ in range(5):
            await memory.correct(
                original_id=entry.id,
                correction="This was wrong",
            )

        results = await memory.search(
            "wrong information",
            include_suppressed=False,
        )

        # Should not find the suppressed memory
        assert not any(r.memory.id == entry.id for r in results)


class TestStats:
    """Test memory statistics."""

    async def test_stats_counts(self, memory: Memory):
        """Test that stats reflect memory counts."""
        await memory.add("Fact 1", memory_type=MemoryType.FACT)
        await memory.add("Fact 2", memory_type=MemoryType.FACT)
        await memory.add("Preference 1", memory_type=MemoryType.PREFERENCE)

        stats = await memory.stats()

        assert stats.total_memories == 3
        assert stats.memories_by_type.get("fact") == 2
        assert stats.memories_by_type.get("preference") == 1


class TestMemoryEntry:
    """Test MemoryEntry model."""

    def test_effective_score(self):
        """Test effective score calculation."""
        entry = MemoryEntry(
            content="Test",
            importance=0.8,
            confidence=1.0,
            correction_weight=1.0,
        )

        score = entry.effective_score()

        # Should be close to importance for fresh memory
        assert 0.7 < score < 1.0

    def test_correction_reduces_score(self):
        """Test that corrections reduce effective score."""
        entry = MemoryEntry(content="Test", importance=0.8)

        initial_score = entry.effective_score()

        entry.apply_correction("corr-1", weight_reduction=0.3)

        new_score = entry.effective_score()

        assert new_score < initial_score

    def test_touch_increments_access(self):
        """Test that touch updates access tracking."""
        entry = MemoryEntry(content="Test")

        assert entry.access_count == 0

        entry.touch()

        assert entry.access_count == 1
