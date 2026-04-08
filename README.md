# Agent Memory System

> **One-liner:** Memory that learns from your corrections and knows what you need before you ask.

## Problem

**Persona:** Alex, senior engineer using Claude/Cursor daily for coding

**Pain:**
- Explains the same project context every new session (monorepo structure, coding standards, deployment process)
- Agent makes the same mistake twice—uses `npm` when project uses `pnpm`, suggests patterns already rejected
- Context windows fill up with repeated explanations instead of actual work
- When Alex corrects the agent, the correction disappears next session

**Quantified:**
- 15-20 minutes/day re-establishing context = 6+ hours/week wasted
- Average developer does 3-5 "actually, we don't do it that way" corrections per hour
- 30% of context window consumed by repeated project context
- Zero corrections persist across sessions (100% knowledge loss)

## Solution

**What:** Memory layer that sits between user and agent:
- **Learns from corrections** — "We use pnpm, not npm" gets stored and applied automatically
- **Compresses intelligently** — Extracts patterns from conversations, not raw transcripts
- **Surfaces proactively** — Injects relevant context before you have to ask
- **Forgets strategically** — Decays stale info, promotes frequently-accessed knowledge

**How:**
1. Intercepts agent conversations (proxy or plugin model)
2. Detects correction patterns ("actually...", "no, we...", explicit "remember this")
3. Extracts structured knowledge: facts, preferences, procedures, entities
4. Builds semantic index for retrieval
5. Injects relevant memories into system prompt automatically

**Why Us:**
- Built OpenClaw's memory system, understand the patterns
- Obsessed with context efficiency (see: token budget work)
- Can dogfood immediately in our own agent stack

## Why Now

1. **Context windows plateaued** — 200k tokens isn't infinite, compression matters
2. **Agent usage going daily** — People have persistent relationships with agents now
3. **MCP provides integration path** — Memory-as-MCP-server is natural
4. **RAG infrastructure is commoditized** — Vector DBs, embeddings are table stakes
5. **Correction-learning is novel** — No one's productized this specific loop

## Market Landscape

**TAM:** $12B (AI developer tools)
**SAM:** $2B (AI context/memory management)
**SOM:** $50M (premium memory layer for power users, Year 3)

### Competitors & Gaps

| Competitor | What They Do | Gap |
|------------|--------------|-----|
| **Mem0** | Memory layer for agents | Generic storage, no correction learning, no proactive surfacing |
| **Zep** | Memory for LLM apps | Dev infrastructure, not end-user facing |
| **LangMem** | LangChain memory | Framework-locked, requires dev integration |
| **Notion AI** | Knowledge base + AI | Pull-based, no automatic injection |
| **Obsidian + plugins** | Local knowledge graph | Manual maintenance, no agent integration |
| **Claude Projects** | Project knowledge | Static, no learning, limited to Claude |

**White space:** Correction-aware memory that learns passively and surfaces proactively.

## Competitive Advantages

1. **Correction-learning moat** — Novel capability, hard to replicate without similar obsession
2. **Cross-agent portability** — Works with Claude, GPT, local models—not locked in
3. **Compression IP** — Intelligent summarization/extraction is defensible
4. **User trust** — Memory is intimate; early trust = long retention
5. **Network effects** — Shared team memories, organizational knowledge

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│         (Chat, CLI, IDE Plugin, MCP Client)             │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                 Memory Proxy Layer                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │  Intercept  │ │   Inject    │ │   Route     │       │
│  │ Corrections │ │   Context   │ │  to Model   │       │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘       │
└─────────┼───────────────┼───────────────┼───────────────┘
          │               │               │
┌─────────▼───────────────▼───────────────────────────────┐
│                 Memory Core                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Knowledge Extractor                 │   │
│  │  • Correction detector (fine-tuned classifier)  │   │
│  │  • Entity extractor (NER + custom rules)        │   │
│  │  • Preference parser (structured output)        │   │
│  │  • Procedure learner (multi-turn patterns)      │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Memory Index                        │   │
│  │  • Semantic search (embeddings + vector DB)     │   │
│  │  • Temporal decay (access frequency + recency)  │   │
│  │  • Conflict resolution (newer > older)          │   │
│  │  • Confidence scoring (explicit > inferred)     │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────┐
│                   Storage Layer                          │
│  • SQLite (local) or PostgreSQL (cloud)                 │
│  • Qdrant/Chroma for vector search                      │
│  • Encrypted at rest, user-controlled keys              │
└─────────────────────────────────────────────────────────┘
```

**Stack:**
- Proxy: Node.js/Bun with streaming support
- Extraction: GPT-4o-mini for classification, structured output for parsing
- Embeddings: OpenAI text-embedding-3-small or local (nomic-embed)
- Vector DB: Qdrant (self-hosted) or Chroma (local)
- Storage: SQLite for local-first, PostgreSQL for cloud
- Deployment: Local daemon, VS Code extension, MCP server

## Build Plan

| Week | Milestone |
|------|-----------|
| 1-2 | Proxy scaffold, intercept Claude API calls, basic logging |
| 3-4 | Correction detector (classifier training), entity extraction |
| 5-6 | Memory storage, vector indexing, basic retrieval |
| 7-8 | Proactive injection, relevance ranking, context budgeting |
| 9-10 | CLI interface, memory browser UI, private beta |
| 11-12 | VS Code extension, MCP server, feedback iteration |
| 13-14 | Temporal decay, conflict resolution, production hardening |
| 15-16 | Team features (shared memories), launch |

**Team:** 1 full-stack dev (you), potential ML contractor for classifier fine-tuning

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Privacy concerns kill adoption | Medium | High | Local-first default, E2E encryption, open source core |
| Injection makes responses worse | Medium | Medium | Confidence thresholds, user feedback loop, easy disable |
| Correction detection is inaccurate | High | Medium | Start conservative (explicit only), expand with training data |
| API providers block proxying | Low | High | MCP server mode (no proxy needed), plugin model |
| Mem0 ships same features | Medium | Medium | Move fast, correction-learning is novel, execution matters |

## Monetization

**Model:** Freemium with local-first

| Tier | Price | Includes |
|------|-------|----------|
| Local | Free | Local storage, 10k memories, single agent |
| Pro | $15/mo | Cloud sync, unlimited memories, multi-agent, mobile access |
| Team | $12/user/mo | Shared team memories, admin controls, SSO |
| Enterprise | Custom | On-prem, audit logs, compliance features |

**Path to $1M ARR:**

- **Target:** Mix of Pro + Team
  - 3,000 Pro users @ $15/mo = $540k ARR
  - 50 teams × 10 users @ $12/mo = $460k ARR
  - Total: $1M ARR
  
- **Funnel:**
  - 50,000 free users (dev community, HN, ProductHunt)
  - 10% convert to Pro = 5,000 Pro users (exceeds target)
  - 2% are team leads who bring team = 100 teams

- **Timeline:** 12-18 months post-launch

**Expansion:** Enterprise sales to companies standardizing on AI tools

## Verdict

### 🟢 BUILD

**Reasoning:**
1. **Solves real pain we experience daily** — Dogfood value is immediate
2. **Novel angle** — Correction-learning is differentiated vs. generic memory
3. **Multiple form factors** — CLI, extension, MCP server, API—many GTM paths
4. **Local-first is defensible** — Privacy-conscious devs won't use cloud-only alternatives
5. **Compounds** — Memory gets more valuable over time, strong retention

**Caveats:**
- Correction detection accuracy is critical—invest in training data early
- Must nail the "proactive but not annoying" balance
- Mem0 has funding and head start—speed matters

**First step:** Build correction detector on our own OpenClaw conversation history. If it works well on real data, proceed to full build.
