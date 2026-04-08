# MCP Integration Guide

This guide explains how to use Agent Memory System with Claude Code and other MCP-compatible tools.

## Setup

### 1. Install the package

```bash
pip install agent-memory-system
```

### 2. Configure Claude Code

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

Or with explicit path:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "agent_memory.mcp.server"],
      "env": {
        "AGENT_MEMORY_SQLITE_PATH": "/path/to/memory.db"
      }
    }
  }
}
```

### 3. Verify Connection

Restart Claude Code. You should see the memory tools available:

- `memory_add` - Store new memories
- `memory_search` - Search memories semantically
- `memory_correct` - Apply corrections
- `memory_surface` - Get proactive suggestions
- `memory_stats` - View statistics

## Available Tools

### memory_add

Store a new memory:

```json
{
  "content": "Project uses pnpm for package management",
  "type": "preference",
  "tags": ["project", "tooling"],
  "importance": 0.8
}
```

**Types:** `fact`, `preference`, `procedure`, `entity`, `episode`

### memory_search

Search memories semantically:

```json
{
  "query": "how do we install packages",
  "limit": 5,
  "type": "procedure",
  "tags": ["tooling"]
}
```

### memory_correct

Apply a correction (this is the key innovation):

```json
{
  "correction": "Use pnpm install, not npm install",
  "original": "npm install"
}
```

This will:
1. Find memories containing "npm install"
2. Reduce their retrieval weight
3. Create a new correction memory
4. Future searches will prefer the correction

### memory_surface

Get proactive memory suggestions:

```json
{
  "context": "I'm about to write a deployment script",
  "max_tokens": 500
}
```

Returns memories that might be relevant to inject as context.

### memory_stats

Get system statistics:

```json
{}
```

Returns counts, types, and storage info.

## Example Workflow

```
User: "How do I add a dependency?"

[Agent internally calls memory_search("add dependency")]
[Gets result: "Use npm install <package>"]

Agent: "You can use npm install..."

User: "No, we use pnpm!"

[Agent calls memory_correct with original="npm", correction="pnpm"]

User: "How do I add a dependency?" (later)

[Agent internally calls memory_search("add dependency")]
[Gets result: "Use pnpm, never npm" with higher score]

Agent: "You can use pnpm add <package>..."
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MEMORY_SQLITE_PATH` | `~/.agent-memory/memory.db` | SQLite database path |
| `AGENT_MEMORY_CHROMA_PATH` | `~/.agent-memory/chroma` | ChromaDB storage path |
| `AGENT_MEMORY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `AGENT_MEMORY_SURFACING_ENABLED` | `true` | Enable proactive surfacing |

## Tips

1. **Tag consistently** - Use tags to organize memories by project, topic, etc.
2. **Set importance** - Higher importance memories surface more often
3. **Correct explicitly** - When correcting, be specific about what was wrong
4. **Review stats** - Check `memory_stats` to see correction patterns
