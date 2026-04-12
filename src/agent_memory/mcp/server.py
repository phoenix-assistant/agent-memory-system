"""MCP (Model Context Protocol) server for Agent Memory System.

This allows Claude Code and other MCP-compatible agents to use the memory system.
"""

from __future__ import annotations

import json
from typing import Any

from agent_memory.core.config import MemoryConfig
from agent_memory.core.memory import Memory
from agent_memory.core.models import MemoryType, SurfacingContext

# MCP tool definitions
TOOLS = [
    {
        "name": "memory_add",
        "description": "Add a new memory to the system. Use this to remember facts, preferences, procedures, or corrections.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store"
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "preference", "procedure", "entity", "episode"],
                    "description": "Type of memory",
                    "default": "fact"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for organization"
                },
                "importance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Importance score (0-1)",
                    "default": 0.5
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "memory_search",
        "description": "Search memories semantically. Returns relevant memories based on meaning, not just keywords.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 5
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "preference", "procedure", "entity", "correction", "episode"],
                    "description": "Filter by memory type"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_correct",
        "description": "Apply a correction. Use when the user corrects a mistake - this adjusts memory weights so the correction is preferred in future.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "correction": {
                    "type": "string",
                    "description": "The correct information"
                },
                "original": {
                    "type": "string",
                    "description": "The incorrect content being corrected"
                },
                "original_id": {
                    "type": "string",
                    "description": "ID of specific memory to correct (if known)"
                }
            },
            "required": ["correction"]
        }
    },
    {
        "name": "memory_surface",
        "description": "Get proactively surfaced memories for the current context. Returns memories that might be relevant.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Current context or user query"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to surface",
                    "default": 500
                }
            },
            "required": ["context"]
        }
    },
    {
        "name": "memory_stats",
        "description": "Get statistics about the memory system.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


class MCPMemoryServer:
    """MCP server wrapping the Memory system."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig.local_default()
        self._memory: Memory | None = None

    async def initialize(self) -> None:
        """Initialize the memory system."""
        self._memory = Memory(self.config)
        await self._memory.initialize()

    async def close(self) -> None:
        """Close the memory system."""
        if self._memory:
            await self._memory.close()

    @property
    def memory(self) -> Memory:
        if not self._memory:
            raise RuntimeError("Server not initialized")
        return self._memory

    def list_tools(self) -> list[dict[str, Any]]:
        """Return available tools."""
        return TOOLS

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool and return results."""

        if name == "memory_add":
            entry = await self.memory.add(
                content=arguments["content"],
                memory_type=MemoryType(arguments.get("type", "fact")),
                tags=arguments.get("tags", []),
                importance=arguments.get("importance", 0.5),
            )
            return {
                "success": True,
                "memory_id": entry.id,
                "content": entry.content,
            }

        elif name == "memory_search":
            results = await self.memory.search(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                memory_type=MemoryType(arguments["type"]) if arguments.get("type") else None,
                tags=arguments.get("tags"),
            )
            return {
                "results": [
                    {
                        "id": r.memory.id,
                        "content": r.memory.content,
                        "type": r.memory.memory_type.value,
                        "score": r.score,
                        "tags": r.memory.tags,
                        "importance": r.memory.importance,
                    }
                    for r in results
                ]
            }

        elif name == "memory_correct":
            entry = await self.memory.correct(
                correction=arguments["correction"],
                original=arguments.get("original"),
                original_id=arguments.get("original_id"),
            )
            return {
                "success": True,
                "correction_id": entry.id,
                "affected_memories": entry.corrects,
            }

        elif name == "memory_surface":
            ctx = SurfacingContext(
                query=arguments["context"],
                max_tokens=arguments.get("max_tokens", 500),
            )
            memories = await self.memory.surface(ctx)
            return {
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "type": m.memory_type.value,
                    }
                    for m in memories
                ]
            }

        elif name == "memory_stats":
            stats = await self.memory.stats()
            return {
                "total_memories": stats.total_memories,
                "corrections": stats.total_corrections,
                "compressed": stats.compressed_memories,
                "by_type": stats.memories_by_type,
            }

        else:
            return {"error": f"Unknown tool: {name}"}


def create_server(config: MemoryConfig | None = None) -> MCPMemoryServer:
    """Create an MCP server instance."""
    return MCPMemoryServer(config)


# Stdio transport for MCP
async def run_stdio_server(config: MemoryConfig | None = None) -> None:
    """Run the MCP server over stdio.

    This is the entry point when running as: agent-memory-mcp
    """
    import sys

    server = create_server(config)
    await server.initialize()

    try:
        # Read JSON-RPC messages from stdin
        while True:
            line = sys.stdin.readline()
            if not line:
                break

            try:
                request = json.loads(line)
                method = request.get("method", "")
                params = request.get("params", {})
                request_id = request.get("id")

                if method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {}
                            },
                            "serverInfo": {
                                "name": "agent-memory-system",
                                "version": "0.1.0"
                            }
                        }
                    }

                elif method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": server.list_tools()
                        }
                    }

                elif method == "tools/call":
                    tool_name = params.get("name", "")
                    tool_args = params.get("arguments", {})
                    result = await server.call_tool(tool_name, tool_args)
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2)
                                }
                            ]
                        }
                    }

                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }

                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError:
                pass

    finally:
        await server.close()


def main() -> None:
    """CLI entry point for MCP server."""
    import asyncio
    asyncio.run(run_stdio_server())


if __name__ == "__main__":
    main()
