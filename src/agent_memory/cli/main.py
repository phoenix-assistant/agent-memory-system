"""CLI for Agent Memory System."""

from __future__ import annotations

import asyncio
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from agent_memory.core.config import MemoryConfig
from agent_memory.core.memory import Memory
from agent_memory.core.models import MemoryType

console = Console()


def run_async(coro: Any) -> Any:
    """Run an async function."""
    return asyncio.get_event_loop().run_until_complete(coro)


def get_memory() -> Memory:
    """Get a Memory instance with default config."""
    return Memory(MemoryConfig.local_default())


@click.group()
@click.version_option()
def cli() -> None:
    """Agent Memory System - Memory that learns from corrections."""
    pass


@cli.command()
@click.argument("content")
@click.option("--type", "-t", "memory_type", default="fact",
              type=click.Choice(["fact", "preference", "procedure", "entity", "episode"]),
              help="Type of memory")
@click.option("--tags", "-g", multiple=True, help="Tags for the memory")
@click.option("--importance", "-i", default=0.5, type=float, help="Importance score (0-1)")
@click.option("--source", "-s", help="Source of the memory")
def add(
    content: str,
    memory_type: str,
    tags: tuple[str, ...],
    importance: float,
    source: str | None,
) -> None:
    """Add a new memory.

    Example:
        mem add "Project uses pnpm" --type preference --tags project,tooling
    """
    async def _add() -> None:
        async with get_memory() as memory:
            entry = await memory.add(
                content,
                memory_type=MemoryType(memory_type),
                tags=list(tags),
                importance=importance,
                source=source,
            )
            console.print(f"[green]✓[/green] Added memory: {entry.id[:8]}...")
            console.print(f"  Content: {content}")
            console.print(f"  Type: {memory_type}")
            if tags:
                console.print(f"  Tags: {', '.join(tags)}")

    run_async(_add())


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Maximum results")
@click.option("--type", "-t", "memory_type", help="Filter by type")
@click.option("--tags", "-g", multiple=True, help="Filter by tags")
@click.option("--min-score", default=0.3, type=float, help="Minimum relevance score")
def search(
    query: str,
    limit: int,
    memory_type: str | None,
    tags: tuple[str, ...],
    min_score: float,
) -> None:
    """Search memories semantically.

    Example:
        mem search "package manager"
        mem search "deployment" --type procedure
    """
    async def _search() -> None:
        async with get_memory() as memory:
            results = await memory.search(
                query,
                limit=limit,
                memory_type=MemoryType(memory_type) if memory_type else None,
                tags=list(tags) if tags else None,
                min_score=min_score,
            )

            if not results:
                console.print("[yellow]No matching memories found.[/yellow]")
                return

            table = Table(title=f"Search Results for: {query}")
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Type", style="green", width=12)
            table.add_column("Content", style="white")
            table.add_column("Tags", style="dim", width=20)

            for result in results:
                mem = result.memory
                table.add_row(
                    f"{result.score:.3f}",
                    mem.memory_type.value,
                    mem.content[:80] + "..." if len(mem.content) > 80 else mem.content,
                    ", ".join(mem.tags[:3]) + ("..." if len(mem.tags) > 3 else ""),
                )

            console.print(table)
            console.print(f"\n[dim]Found {len(results)} memories[/dim]")

    run_async(_search())


@cli.command()
@click.argument("correction")
@click.option("--original", "-o", help="Original incorrect content to correct")
@click.option("--id", "original_id", help="ID of specific memory to correct")
@click.option("--source", "-s", help="Source of the correction")
def correct(
    correction: str,
    original: str | None,
    original_id: str | None,
    source: str | None,
) -> None:
    """Apply a correction to the memory system.

    This adjusts weights of similar memories and creates a correction record.

    Example:
        mem correct "We use pnpm, not npm" --original "npm install"
        mem correct "Deploy to prod requires approval" --id abc123
    """
    if not original and not original_id:
        console.print("[red]Error: Must provide either --original or --id[/red]")
        return

    async def _correct() -> None:
        async with get_memory() as memory:
            entry = await memory.correct(
                original=original,
                correction=correction,
                original_id=original_id,
                source=source,
            )

            console.print(f"[green]✓[/green] Correction applied: {entry.id[:8]}...")
            console.print(f"  Correction: {correction}")
            if entry.corrects:
                console.print(f"  Affected {len(entry.corrects)} memories")

    run_async(_correct())


@cli.command()
def stats() -> None:
    """Show memory system statistics."""
    async def _stats() -> None:
        async with get_memory() as memory:
            s = await memory.stats()

            console.print("\n[bold]Memory System Statistics[/bold]\n")

            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Total Memories", str(s.total_memories))
            table.add_row("Corrections", str(s.total_corrections))
            table.add_row("Compressed", str(s.compressed_memories))
            table.add_row("Avg Importance", f"{s.average_importance:.2f}")

            if s.oldest_memory:
                table.add_row("Oldest", s.oldest_memory.strftime("%Y-%m-%d"))
            if s.newest_memory:
                table.add_row("Newest", s.newest_memory.strftime("%Y-%m-%d"))

            # Convert bytes to human readable
            if s.storage_bytes > 1_000_000:
                size = f"{s.storage_bytes / 1_000_000:.1f} MB"
            elif s.storage_bytes > 1_000:
                size = f"{s.storage_bytes / 1_000:.1f} KB"
            else:
                size = f"{s.storage_bytes} B"
            table.add_row("Storage Size", size)

            console.print(table)

            if s.memories_by_type:
                console.print("\n[bold]By Type:[/bold]")
                for mtype, count in s.memories_by_type.items():
                    console.print(f"  {mtype}: {count}")

    run_async(_stats())


@cli.command("list")
@click.option("--type", "-t", "memory_type", help="Filter by type")
@click.option("--tags", "-g", multiple=True, help="Filter by tags")
@click.option("--limit", "-n", default=20, help="Maximum results")
@click.option("--offset", default=0, help="Offset for pagination")
def list_memories(
    memory_type: str | None,
    tags: tuple[str, ...],
    limit: int,
    offset: int,
) -> None:
    """List memories with optional filtering.

    Example:
        mem list --type preference
        mem list --tags project --limit 50
    """
    async def _list() -> None:
        async with get_memory() as memory:
            memories = await memory.list(
                memory_type=MemoryType(memory_type) if memory_type else None,
                tags=list(tags) if tags else None,
                limit=limit,
                offset=offset,
            )

            if not memories:
                console.print("[yellow]No memories found.[/yellow]")
                return

            table = Table(title="Memories")
            table.add_column("ID", style="dim", width=10)
            table.add_column("Type", style="green", width=12)
            table.add_column("Content", style="white")
            table.add_column("Importance", style="cyan", width=10)
            table.add_column("Weight", style="magenta", width=8)

            for mem in memories:
                # Show weight differently if suppressed
                weight_style = "red" if mem.correction_weight < 0.5 else "magenta"
                table.add_row(
                    mem.id[:8] + "...",
                    mem.memory_type.value,
                    mem.content[:60] + "..." if len(mem.content) > 60 else mem.content,
                    f"{mem.importance:.2f}",
                    f"[{weight_style}]{mem.correction_weight:.2f}[/{weight_style}]",
                )

            console.print(table)
            console.print(f"\n[dim]Showing {len(memories)} memories (offset: {offset})[/dim]")

    run_async(_list())


@cli.command()
@click.argument("memory_id")
def get(memory_id: str) -> None:
    """Get details of a specific memory.

    Example:
        mem get abc12345
    """
    async def _get() -> None:
        async with get_memory() as memory:
            # Try to find by prefix
            memories = await memory.list(limit=100)
            matches = [m for m in memories if m.id.startswith(memory_id)]

            if not matches:
                console.print(f"[red]Memory not found: {memory_id}[/red]")
                return

            if len(matches) > 1:
                console.print(f"[yellow]Multiple matches for '{memory_id}':[/yellow]")
                for m in matches:
                    console.print(f"  {m.id[:12]}... - {m.content[:40]}...")
                return

            mem = matches[0]

            console.print(f"\n[bold]Memory: {mem.id}[/bold]\n")
            console.print(f"[cyan]Type:[/cyan] {mem.memory_type.value}")
            console.print(f"[cyan]Content:[/cyan] {mem.content}")
            console.print(f"[cyan]Tags:[/cyan] {', '.join(mem.tags) if mem.tags else 'none'}")
            console.print(f"[cyan]Source:[/cyan] {mem.source or 'unknown'}")
            console.print(f"[cyan]Importance:[/cyan] {mem.importance:.2f}")
            console.print(f"[cyan]Confidence:[/cyan] {mem.confidence:.2f}")
            console.print(f"[cyan]Correction Weight:[/cyan] {mem.correction_weight:.2f}")
            console.print(f"[cyan]Access Count:[/cyan] {mem.access_count}")
            console.print(f"[cyan]Created:[/cyan] {mem.created_at.strftime('%Y-%m-%d %H:%M')}")
            console.print(f"[cyan]Last Accessed:[/cyan] {mem.accessed_at.strftime('%Y-%m-%d %H:%M')}")

            if mem.corrected_by:
                console.print(f"[yellow]Corrected by:[/yellow] {', '.join(c[:8] for c in mem.corrected_by)}")
            if mem.corrects:
                console.print(f"[green]Corrects:[/green] {', '.join(c[:8] for c in mem.corrects)}")

    run_async(_get())


@cli.command()
@click.argument("memory_id")
@click.confirmation_option(prompt="Are you sure you want to delete this memory?")
def delete(memory_id: str) -> None:
    """Delete a memory by ID.

    Example:
        mem delete abc12345
    """
    async def _delete() -> None:
        async with get_memory() as memory:
            # Try to find by prefix
            memories = await memory.list(limit=100)
            matches = [m for m in memories if m.id.startswith(memory_id)]

            if not matches:
                console.print(f"[red]Memory not found: {memory_id}[/red]")
                return

            if len(matches) > 1:
                console.print(f"[yellow]Multiple matches for '{memory_id}'. Be more specific.[/yellow]")
                return

            if await memory.delete(matches[0].id):
                console.print(f"[green]✓[/green] Deleted memory: {matches[0].id[:8]}...")
            else:
                console.print("[red]Failed to delete memory[/red]")

    run_async(_delete())


@cli.command()
def compress() -> None:
    """Compress stale memories into summaries.

    This reduces storage and improves retrieval by summarizing
    old, less-accessed memories.
    """
    async def _compress() -> None:
        async with get_memory() as memory:
            console.print("[dim]Analyzing memories for compression...[/dim]")
            count = await memory.compress_stale()

            if count > 0:
                console.print(f"[green]✓[/green] Compressed {count} memories")
            else:
                console.print("[yellow]No memories eligible for compression[/yellow]")

    run_async(_compress())


@cli.command()
@click.argument("query")
@click.option("--max-tokens", "-t", default=500, help="Maximum tokens to surface")
def surface(query: str, max_tokens: int) -> None:
    """Preview what memories would be proactively surfaced.

    Example:
        mem surface "How do I deploy the app?"
    """
    from agent_memory.core.models import SurfacingContext

    async def _surface() -> None:
        async with get_memory() as memory:
            ctx = SurfacingContext(
                query=query,
                max_tokens=max_tokens,
            )

            surfaced = await memory.surface(ctx)

            if not surfaced:
                console.print("[yellow]No memories would be surfaced for this query.[/yellow]")
                return

            console.print(f"\n[bold]Memories to surface for: {query}[/bold]\n")

            for i, mem in enumerate(surfaced, 1):
                console.print(f"[cyan]{i}.[/cyan] [{mem.memory_type.value}] {mem.content}")

            console.print(f"\n[dim]{len(surfaced)} memories would be injected[/dim]")

    run_async(_surface())


if __name__ == "__main__":
    cli()
