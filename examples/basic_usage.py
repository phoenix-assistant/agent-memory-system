"""Basic usage example for Agent Memory System."""

import asyncio
from agent_memory import Memory, MemoryConfig, MemoryType


async def main():
    # Create memory with default local config (SQLite + ChromaDB)
    config = MemoryConfig.local_default()
    
    async with Memory(config) as memory:
        # Add some memories
        print("Adding memories...")
        
        await memory.add(
            "Project uses pnpm for package management",
            memory_type=MemoryType.PREFERENCE,
            tags=["project", "tooling"],
            importance=0.8,
        )
        
        await memory.add(
            "Backend is written in Python with FastAPI",
            memory_type=MemoryType.FACT,
            tags=["project", "backend"],
        )
        
        await memory.add(
            "Deploy by running: ./scripts/deploy.sh production",
            memory_type=MemoryType.PROCEDURE,
            tags=["devops"],
            importance=0.9,
        )
        
        # Search memories
        print("\nSearching for 'package manager'...")
        results = await memory.search("package manager")
        
        for result in results:
            print(f"  [{result.score:.2f}] {result.memory.content}")
        
        # Apply a correction
        print("\nApplying correction: npm -> pnpm...")
        correction = await memory.correct(
            original="npm install",
            correction="Use pnpm install, not npm install",
        )
        print(f"  Correction ID: {correction.id}")
        print(f"  Affected memories: {len(correction.corrects)}")
        
        # Search again - correction should be preferred
        print("\nSearching again after correction...")
        results = await memory.search("how to install packages")
        
        for result in results:
            print(f"  [{result.score:.2f}] {result.memory.content}")
        
        # Get stats
        print("\nMemory Statistics:")
        stats = await memory.stats()
        print(f"  Total memories: {stats.total_memories}")
        print(f"  Corrections: {stats.total_corrections}")
        print(f"  By type: {stats.memories_by_type}")


if __name__ == "__main__":
    asyncio.run(main())
