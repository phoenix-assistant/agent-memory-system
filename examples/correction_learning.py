"""Example demonstrating correction learning in action."""

import asyncio
from agent_memory import Memory, MemoryConfig, MemoryType


async def simulate_conversation():
    """Simulate a conversation where corrections are learned."""
    
    async with Memory(MemoryConfig.local_default()) as memory:
        print("=== Simulating Agent Memory with Correction Learning ===\n")
        
        # Initial setup: Agent has some knowledge
        print("1. Agent learns initial facts...")
        
        await memory.add(
            "The user prefers tabs for indentation",
            memory_type=MemoryType.PREFERENCE,
            tags=["coding", "style"],
        )
        
        await memory.add(
            "API responses should use camelCase",
            memory_type=MemoryType.PREFERENCE,
            tags=["api", "style"],
        )
        
        await memory.add(
            "Use npm to install packages",
            memory_type=MemoryType.PROCEDURE,
            tags=["tooling"],
        )
        
        # Day 1: Agent suggests npm, user corrects
        print("\n2. Simulating correction: User says 'No, we use pnpm!'...")
        
        correction = await memory.correct(
            original="npm",
            correction="Always use pnpm, never npm. The project has a pnpm-lock.yaml.",
        )
        
        print(f"   Correction applied. Affected {len(correction.corrects)} memories.")
        
        # Now search for package installation
        print("\n3. Agent searches for how to install packages...")
        
        results = await memory.search("how to install packages")
        
        print("   Top results:")
        for r in results[:3]:
            weight_indicator = "✓" if r.memory.correction_weight >= 0.7 else "↓"
            print(f"   {weight_indicator} [{r.score:.2f}] {r.memory.content[:60]}...")
        
        # Day 2: Another correction
        print("\n4. Another correction: 'Actually, we prefer spaces now'...")
        
        await memory.correct(
            original="tabs for indentation",
            correction="Switched to 2-space indentation as of March 2024",
        )
        
        # Search for coding style
        print("\n5. Agent searches for indentation style...")
        
        results = await memory.search("indentation code style")
        
        print("   Top results:")
        for r in results[:3]:
            weight_indicator = "✓" if r.memory.correction_weight >= 0.7 else "↓"
            print(f"   {weight_indicator} [{r.score:.2f}] {r.memory.content[:60]}...")
        
        # Final stats
        print("\n=== Final Statistics ===")
        stats = await memory.stats()
        print(f"Total memories: {stats.total_memories}")
        print(f"Corrections applied: {stats.total_corrections}")
        
        # Show all memories with their weights
        print("\n=== All Memories with Weights ===")
        all_memories = await memory.list(limit=20)
        
        for mem in all_memories:
            status = "🔴" if mem.correction_weight < 0.5 else "🟢" if mem.correction_weight >= 0.9 else "🟡"
            print(f"{status} [{mem.memory_type.value:10}] (w={mem.correction_weight:.2f}) {mem.content[:50]}...")


if __name__ == "__main__":
    asyncio.run(simulate_conversation())
