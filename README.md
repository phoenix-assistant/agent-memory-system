# Agent Memory System

> Memory that learns from your corrections and knows what you need before you ask.

[![CI](https://github.com/phoenix-assistant/agent-memory-system/actions/workflows/ci.yml/badge.svg)](https://github.com/phoenix-assistant/agent-memory-system/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agent-memory-system.svg)](https://pypi.org/project/agent-memory-system/)
[![Python](https://img.shields.io/pypi/pyversions/agent-memory-system.svg)](https://pypi.org/project/agent-memory-system/)
[![License](https://img.shields.io/github/license/phoenix-assistant/agent-memory-system)](LICENSE)

## Why?

Every AI agent has the same problem: **zero memory across sessions**. You explain your project structure, correct mistakes, state preferences—and next session, it's all gone.

Agent Memory System fixes this with:

- **Correction Learning** — When you say "No, we use pnpm not npm", the system learns. Future retrievals prefer the correction.
- **Intelligent Compression** — Old memories get summarized, not deleted. Important context stays accessible.
- **Proactive Surfacing** — Relevant memories inject into context before you ask.
- **Multi-Backend** — SQLite + ChromaDB locally, Postgres + Qdrant for servers.
- **MCP Integration** — Works with Claude Code and any MCP-compatible agent.

## Quick Start

### Install

```bash
pip install agent-memory-system
```

### CLI Usage

```bash
# Add a memory
mem add "Project uses pnpm for package management" --type preference --tags project,tooling

# Search memories
mem search "how to install packages"

# Apply a correction (the key innovation)
mem correct "Use pnpm, not npm" --original "npm install"

# View stats
mem stats

# List memories
mem list --type preference

# Preview proactive surfacing
mem surface "I'm about to write a deployment script"
```

### Python SDK

```python
import asyncio
from agent_memory import Memory, MemoryConfig, MemoryType

async def main():
    async with Memory(MemoryConfig.local_default()) as memory:
        # Store memories
        await memory.add(
            "Project uses pnpm for package management",
            memory_type=MemoryType.PREFERENCE,
            tags=["project", "tooling"],
            importance=0.8,
        )
        
        # Search semantically
        results = await memory.search("how do we install packages")
        for r in results:
            print(f"[{r.score:.2f}] {r.memory.content}")
        
        # Apply corrections - this is the key innovation
        await memory.correct(
            original="npm install",
            correction="Use pnpm install, never npm",
        )
        
        # Future searches now prefer the correction
        results = await memory.search("install packages")
        # -> Returns "Use pnpm install, never npm" with higher score

asyncio.run(main())
```

## Key Innovation: Correction Learning

When you correct an agent, most systems just ignore it. Agent Memory System treats corrections as first-class signals:

```python
# User says: "No, we use pnpm, not npm!"

await memory.correct(
    original="npm",
    correction="Always use pnpm, the project has pnpm-lock.yaml",
)
```

What happens:
1. **Find similar memories** — Searches for memories containing "npm"
2. **Reduce weights** — Lowers `correction_weight` on matching memories
3. **Create correction** — Stores the correction with high importance
4. **Link them** — Creates bidirectional links between old and new
5. **Future retrieval** — Suppressed memories rank lower; corrections rank higher

This means the agent **actually learns** from your feedback.

## MCP Integration

Agent Memory System works as an MCP server for Claude Code and other MCP-compatible tools.

### Setup

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "python",
      "args": ["-m", "agent_memory.mcp.server"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `memory_add` | Store a new memory |
| `memory_search` | Search memories semantically |
| `memory_correct` | Apply a correction |
| `memory_surface` | Get proactive suggestions |
| `memory_stats` | View statistics |

See [MCP Integration Guide](examples/mcp_integration.md) for details.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Memory System                       │
├─────────────────────────────────────────────────────────────┤
│  Interface Layer                                             │
│  ├── MCP Server (for Claude Code, agents)                   │
│  ├── CLI (mem add/search/correct/stats)                     │
│  └── Python SDK (async)                                      │
├─────────────────────────────────────────────────────────────┤
│  Memory Operations                                           │
│  ├── Store (with importance scoring)                        │
│  ├── Retrieve (semantic + keyword + recency)                │
│  ├── Correct (feedback loop, weight adjustment)             │
│  ├── Compress (summarize old, preserve important)           │
│  └── Surface (proactive context injection)                  │
├─────────────────────────────────────────────────────────────┤
│  Storage Backends                                            │
│  ├── SQLite + ChromaDB (local)                              │
│  └── Postgres + Qdrant (server)                             │
└─────────────────────────────────────────────────────────────┘
```

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `fact` | Declarative knowledge | "Backend uses Python 3.11" |
| `preference` | User preferences | "Prefers 2-space indentation" |
| `procedure` | How to do something | "Deploy with ./scripts/deploy.sh" |
| `entity` | Named thing | "Sarah is the PM" |
| `correction` | Explicit correction | "Use pnpm, not npm" |
| `episode` | Event summary | "Refactored auth on March 5" |
| `summary` | Compressed memories | Auto-generated |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MEMORY_STORAGE_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `AGENT_MEMORY_SQLITE_PATH` | `~/.agent-memory/memory.db` | SQLite path |
| `AGENT_MEMORY_POSTGRES_URL` | - | PostgreSQL URL |
| `AGENT_MEMORY_VECTOR_BACKEND` | `chroma` | `chroma`, `qdrant`, or `memory` |
| `AGENT_MEMORY_CHROMA_PATH` | `~/.agent-memory/chroma` | ChromaDB path |
| `AGENT_MEMORY_QDRANT_URL` | - | Qdrant server URL |
| `AGENT_MEMORY_EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` or `openai` |
| `AGENT_MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `AGENT_MEMORY_OPENAI_API_KEY` | - | For OpenAI embeddings |
| `AGENT_MEMORY_SURFACING_ENABLED` | `true` | Enable proactive surfacing |
| `AGENT_MEMORY_SURFACING_MAX_TOKENS` | `500` | Max tokens to surface |

## Docker

```bash
# Local mode (SQLite + ChromaDB)
docker-compose --profile local up

# Server mode (Postgres + Qdrant)
docker-compose --profile server up
```

## Development

```bash
# Clone
git clone https://github.com/phoenix-assistant/agent-memory-system
cd agent-memory-system

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src tests

# Type check
mypy src
```

## Roadmap

- [x] Core memory operations
- [x] SQLite + ChromaDB backend
- [x] Correction learning
- [x] CLI interface
- [x] MCP server
- [ ] Postgres + Qdrant backend testing
- [ ] LLM-based compression
- [ ] Proactive surfacing optimization
- [ ] Web UI for memory browsing
- [ ] Team/shared memory support

## License

MIT - see [LICENSE](LICENSE)

## Related Projects

- [Mem0](https://github.com/mem0ai/mem0) - Memory for AI agents (no correction learning)
- [Zep](https://github.com/getzep/zep) - Memory for LLM applications
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/) - Framework-specific memory

Agent Memory System differentiates through **correction learning** — the system actually improves from your feedback.
