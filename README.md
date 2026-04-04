<p align="center">
  <strong>memory-v2</strong><br>
  Brain-Inspired Persistent Memory for AI Coding Assistants
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="https://pypi.org/project/memory-v2/"><img src="https://img.shields.io/pypi/v/memory-v2.svg" alt="PyPI"></a>
  <a href="https://github.com/Haustorium12/memory-v2"><img src="https://img.shields.io/badge/MCP-17%20tools-green.svg" alt="MCP Tools"></a>
</p>

---

## Abstract

**memory-v2** is a cognitively-grounded persistent memory system designed for AI coding assistants operating over long-lived, multi-session workflows. It replaces the first-generation [claude-memory](https://github.com/Haustorium12/claude-memory) system with a ground-up rewrite that fuses techniques from cognitive psychology (ACT-R activation theory, power-law forgetting), information retrieval (BM25/vector hybrid search with Reciprocal Rank Fusion), and graph-based knowledge representation (Leiden community detection, Personalized PageRank retrieval) into a single SQLite-backed store.

The system serves as a Model Context Protocol (MCP) server exposing 17 tools, allowing any MCP-compatible AI client -- Claude Code, Claude Desktop, or custom agents -- to store, search, decay, compress, and share memories without external API calls. All embedding and LLM inference runs locally through Ollama, requiring zero API keys and transmitting zero data off-machine.

memory-v2 was built by a practitioner who ran 59 compaction cycles on the v1 system and decided the architecture needed to be rethought from first principles. The result is a system where memories compete for survival through activation scores, where protected content is exempt from decay, where multiple agents coordinate through authority chains, and where a knowledge graph discovers connections the flat search index cannot.

---

## Table of Contents

- [Abstract](#abstract)
- [Architecture Overview](#architecture-overview)
- [Theoretical Foundations](#theoretical-foundations)
  - [ACT-R Activation Model](#act-r-activation-model)
  - [FadeMem Decay System](#fademem-decay-system)
  - [Hybrid Search with Reciprocal Rank Fusion](#hybrid-search-with-reciprocal-rank-fusion)
  - [Knowledge Graph and Personalized PageRank](#knowledge-graph-and-personalized-pagerank)
- [Database Schema](#database-schema)
- [MCP Server and Tools](#mcp-server-and-tools)
- [Subsystem Reference](#subsystem-reference)
  - [Embedding Layer](#embedding-layer)
  - [Vault Indexer](#vault-indexer)
  - [Auto-Extraction Pipeline](#auto-extraction-pipeline)
  - [CogCanvas Compaction Pipeline](#cogcanvas-compaction-pipeline)
  - [Multi-Agent Coordination](#multi-agent-coordination)
  - [Security Module](#security-module)
- [Worked Examples](#worked-examples)
  - [ACT-R Activation Walkthrough](#act-r-activation-walkthrough)
  - [FadeMem Decay Walkthrough](#fademem-decay-walkthrough)
  - [Hybrid Search Walkthrough](#hybrid-search-walkthrough)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [As an MCP Server](#as-an-mcp-server)
  - [As a CLI Tool](#as-a-cli-tool)
  - [Building the Knowledge Graph](#building-the-knowledge-graph)
- [v1 to v2 Migration](#v1-to-v2-migration)
- [Repository Structure](#repository-structure)
- [Related Work](#related-work)
- [References](#references)
- [Built With](#built-with)
- [License](#license)

---

## Architecture Overview

```
                          +-----------------------------+
                          |      MCP Client Layer       |
                          |  (Claude Code / Desktop /   |
                          |   any MCP-compatible agent)  |
                          +-------------+---------------+
                                        |
                              FastMCP Protocol (stdio)
                                        |
                          +-------------v---------------+
                          |     server.py (17 tools)     |
                          |  add | search | graph_search |
                          |  extract | compact | decay   |
                          |  agent_sync | check_integrity |
                          +---+-----+-----+-----+------+
                              |     |     |     |
              +---------------+     |     |     +---------------+
              |                     |     |                     |
   +----------v---------+   +------v-----v------+   +----------v---------+
   |     db.py           |   |   scoring.py      |   | knowledge_graph.py |
   | SQLite + sqlite-vec |   | ACT-R activation  |   | NetworkX DiGraph   |
   | + FTS5              |   | FadeMem decay     |   | Leiden communities |
   |                     |   | Cosine similarity |   | PPR retrieval      |
   | memories (rows)     |   |                   |   |                    |
   | memory_fts (BM25)   |   | base_level_act()  |   | extract_entities() |
   | memory_vec (768-d)  |   | spreading_act()   |   | detect_communities |
   | graveyard           |   | importance_score()|   | ppr_search()       |
   | agent_offsets       |   | decay_value()     |   | visualize_graph()  |
   | conflicts           |   | retrieval_prob()  |   |                    |
   | compaction_receipts |   |                   |   |                    |
   +----------+----------+   +-------------------+   +----------+---------+
              |                                                  |
              |              +-------------------+               |
              +--------------+  embeddings.py    +---------------+
                             | Ollama            |
                             | nomic-embed-text  |
                             | 768-dim vectors   |
                             | Local file cache  |
                             +-------------------+
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
   +----------v---------+  +---------v----------+  +---------v----------+
   |  extraction.py     |  |  compaction.py     |  |  multi_agent.py    |
   | 2-pass LLM pipeline|  | CogCanvas 6-step   |  | Authority chain    |
   | Fact extraction     |  | Protected extract  |  | Consumer offsets   |
   | Novelty checking    |  | Selective deletion |  | Conflict detection |
   | Action decision     |  | Summary + verify   |  | Kafka-style sync   |
   +--------------------+  +--------------------+  +--------------------+
              |
   +----------v---------+  +--------------------+
   |  vault_indexer.py   |  |  security.py       |
   | Markdown chunking   |  | 13 credential pats |
   | SHA-256 delta detect|  | 6 injection pats   |
   | Frontmatter parsing |  | SHA-256 manifests  |
   | Incremental indexing|  | Allowlist filtering |
   +--------------------+  +--------------------+
```

The architecture follows a layered design where the MCP server (`server.py`) acts as the sole entry point for AI clients. All 17 tools delegate to specialized subsystems. The database layer (`db.py`) owns the single SQLite file, which contains three co-located indexes: relational rows in `memories`, BM25 full-text search in `memory_fts` (FTS5), and 768-dimensional vector embeddings in `memory_vec` (sqlite-vec). The scoring layer applies cognitive activation formulas on top of search results. The knowledge graph lives in a separate NetworkX pickle file and provides multi-hop discovery that the flat index cannot.

Every embedding and LLM call routes through Ollama, which runs locally. The system never phones home.

---

## Theoretical Foundations

### ACT-R Activation Model

The Adaptive Control of Thought -- Rational (ACT-R) framework, developed by John Anderson and colleagues at Carnegie Mellon, provides the core theory for how memories compete for retrieval. The central claim is that human memory retrieval is a rational adaptation to the statistical structure of the environment: items that have been used recently and frequently are more likely to be needed again (Anderson & Schooler, 1991).

memory-v2 implements the ACT-R activation equation as a scoring overlay on search results. Each memory has an **activation** value that determines its probability of being retrieved. Activation has three components: base-level activation (how often and how recently the memory was accessed), spreading activation (contextual priming from the current query), and noise (stochastic variation that prevents deterministic behavior).

#### Base-Level Activation

The base-level activation approximates the full rational analysis using the optimized closed-form:

```
B_i = ln(n / (1 - d)) - d * ln(L)
```

Where:
- `n` = total access count for the memory
- `d` = decay parameter (fixed at **0.5**, the canonical ACT-R value)
- `L` = lifetime of the memory in hours (time since creation)

This approximation avoids storing the full access history while preserving the key property: activation rises with frequency and falls with time, following a power law.

#### Spreading Activation

Context tags from the current query prime associated memories:

```
S_i = SUM_j [ W_j * (S_max - ln(fan_j)) ]
```

Where:
- `W_j = 1 / |context_tags|` (attention weight, divided equally among context sources)
- `S_max = 1.6` (maximum associative strength)
- `fan_j` = number of memories sharing tag `j` (the "fan" of the source)

The key insight: tags that appear on many memories provide less activation (they are less discriminating), while rare tags provide more. This is the ACT-R equivalent of IDF weighting.

#### Noise

Activation includes a stochastic noise term drawn from the logistic distribution:

```
epsilon ~ Logistic(0, s)       where s = 0.25
```

Generated as:

```
epsilon = s * ln(u / (1 - u))       where u ~ Uniform(0, 1)
```

This noise prevents the system from becoming deterministic and allows occasionally surprising retrievals -- a property that mirrors human memory.

#### Full Activation and Retrieval Probability

The complete activation equation:

```
A_i = B_i + S_i + epsilon
```

The probability of successful retrieval given activation:

```
P(retrieve) = 1 / (1 + exp(-(A_i - tau) / s))
```

Where:
- `tau = -0.5` (retrieval threshold)
- `s = 0.25` (noise scale, same as the noise parameter)

This is a sigmoid function centered at the threshold. Memories with activation well above the threshold are almost certainly retrieved; memories well below are almost certainly forgotten.

**Protected floor**: Memories marked as protected receive an activation floor of `tau + 1.0 = 0.5`, ensuring they are always retrievable regardless of age or access pattern.

#### Final Reranking

After hybrid search produces an initial ranked list, ACT-R activation is used to rerank:

```
final_score = 0.6 * hybrid_rrf_score + 0.4 * (activation / 10.0)
```

The activation is divided by 10 to normalize it into the same range as the RRF score (typically 0.0 to 0.03). The 60/40 split weights lexical+semantic relevance above cognitive activation, while still allowing frequently-accessed, contextually-primed memories to rise.

---

### FadeMem Decay System

While ACT-R handles retrieval competition at query time, FadeMem handles the background lifecycle of memories. It implements a dual-layer memory architecture (Short-Term Memory and Long-Term Memory) with power-law decay, importance-based promotion/demotion, and archival.

#### Importance Scoring

Each memory's importance is a weighted combination of three signals:

```
I(t) = 0.4 * relevance + 0.3 * frequency + 0.3 * recency
```

Where:
- `relevance` = contextual relevance score (0.5 during background sweeps when no query context is available)
- `frequency = log(access_count + 1) / log(max_access_count + 1)` (log-normalized access frequency)
- `recency = exp(-decay_rate * hours_since_access)` (exponential recency decay)

The weights (0.4, 0.3, 0.3) reflect a design choice: what a memory is about matters slightly more than how often or how recently it was accessed.

#### Power-Law Decay Function

The core decay function:

```
v(t) = v(0) * exp(-lambda * t^beta)
```

Where:
- `v(0)` = initial memory strength
- `lambda` = per-memory decay rate (default `0.1`)
- `t` = time since last access in hours
- `beta` = layer-dependent exponent:
  - `beta_LTM = 0.8` (sub-linear -- LTM memories decay **slower** than exponential)
  - `beta_STM = 1.2` (super-linear -- STM memories decay **faster** than exponential)

The beta parameter is the key innovation over standard exponential decay. At `beta < 1`, the decay curve bends upward relative to exponential, meaning old memories decay more slowly the older they get -- they are "hardened" by time. At `beta > 1`, the curve bends downward, meaning new memories that fail to consolidate decay acceleratingly.

#### Dual-Layer Promotion and Demotion

Memories move between layers based on importance thresholds:

```
Promotion:   STM --> LTM    when  importance >= 0.7
Demotion:    LTM --> STM    when  importance <= 0.3
Archive:     STM --> grave   when  importance < 0.1  AND  age > 30 days
```

The gap between 0.3 and 0.7 is a **hysteresis zone**: memories in this range stay in their current layer. This prevents oscillation at the boundary.

#### Protected Tags

Memories tagged with any of the following are immune to decay and archival:

```
correction, decision, identity, emotional_anchor,
commitment, exact_value, chain_of_command, person
```

These represent categories of information where loss would be harmful regardless of access frequency.

---

### Hybrid Search with Reciprocal Rank Fusion

memory-v2 runs two independent search algorithms and merges them:

1. **BM25 keyword search** via SQLite FTS5 -- excels at exact term matching, file names, error codes
2. **Vector similarity search** via sqlite-vec (768-dim, cosine distance) -- excels at semantic similarity

The merge uses **Reciprocal Rank Fusion** (Cormack et al., 2009), which has been shown to outperform individual ranking methods and Condorcet fusion:

```
score(d) = SUM_i [ 1 / (k + rank_i(d)) ]
```

Where:
- `k = 60` (the RRF constant; higher values reduce the influence of top-ranked results)
- `rank_i(d)` = position of document `d` in the `i`-th ranked list (0-indexed)
- The sum runs over both BM25 and vector result lists

Documents appearing in both lists receive scores from both. Documents appearing in only one list receive a score from that list alone (the other term is 0). The result is a fused ranking that captures both lexical precision and semantic breadth.

Each search retrieves `limit * 3` candidates before fusion to ensure adequate recall. The fused results are then passed to the ACT-R scoring layer for final reranking.

---

### Knowledge Graph and Personalized PageRank

The knowledge graph is a NetworkX directed graph (`DiGraph`) that represents entities and relationships extracted from vault documents. It provides a complementary retrieval path: while flat search finds documents containing similar words or vectors, graph traversal discovers **structurally related** concepts even when they share no lexical or embedding similarity.

#### Schema

**10 node types:**
```
person, project, concept, decision, tool,
event, emotion, conversation, chunk, community
```

**11 edge types:**
```
discussed_in, decided, built, uses, part_of,
related_to, preceded_by, caused, felt, evolved_from, member_of
```

Entities are normalized to lowercase. Edges carry weight (incremented on repeated observation), temporal metadata (`valid_from`, `valid_until`), confidence scores, and source file provenance.

#### Entity Extraction

Entity and relationship extraction uses a local LLM (default: `qwen2.5:3b` via Ollama) with a structured prompt that produces JSON output. The model is instructed to normalize names, skip trivial relationships, and use canonical forms. The first 4000 characters of each document are processed (respecting the small model's context window).

#### Leiden Community Detection

The graph is partitioned using the Leiden algorithm (Traag et al., 2019), which guarantees well-connected communities -- an improvement over the earlier Louvain method that could produce arbitrarily badly connected communities. Implementation uses `python-igraph` and `leidenalg` (optional dependencies).

Community assignments are stored as node attributes and surfaced through the `list_topics` MCP tool, providing an automatic clustering of the knowledge base without manual taxonomy.

#### HippoRAG-Style Personalized PageRank Retrieval

The `graph_search` tool implements Personalized PageRank (PPR) retrieval inspired by HippoRAG (2024):

1. **Seed identification**: Extract entities from the query; match them to graph nodes by word overlap. If no direct match, fall back to embedding similarity against node names.
2. **Personalization vector**: Construct a uniform distribution over seed nodes (all others get weight 0).
3. **PPR computation**: `nx.pagerank(G_undirected, alpha=0.85, personalization=p)`
4. **Result extraction**: Return top-K nodes by PPR score, including community membership, mention count, and source provenance.

The teleport probability `alpha = 0.85` means 85% of the random walk follows edges and 15% teleports back to seed nodes. This strikes the standard balance between exploration and relevance.

The key advantage over flat search: PPR discovers nodes reachable by multi-hop traversal from the query entities, even if those nodes share no embedding or lexical similarity with the query. This enables "what else is connected to this?" reasoning.

---

## Database Schema

All persistent state (except the knowledge graph pickle) lives in a single SQLite database file.

### Entity-Relationship Diagram

```mermaid
erDiagram
    memories {
        INTEGER id PK
        TEXT content
        TEXT content_type
        TEXT source_file
        INTEGER source_line
        TEXT author
        INTEGER authority_level
        REAL confidence
        INTEGER protected
        TEXT tags
        TEXT created_at
        TEXT updated_at
        TEXT last_accessed_at
        INTEGER access_count
        REAL activation_score
        REAL importance_score
        REAL decay_rate
        INTEGER archived
        INTEGER supersedes FK
        TEXT content_hash
    }

    memory_fts {
        TEXT content
        TEXT tags
        TEXT source_file
    }

    memory_vec {
        INTEGER id PK
        BLOB embedding
    }

    graveyard {
        INTEGER id PK
        INTEGER memory_id FK
        TEXT content
        TEXT metadata
        TEXT archived_at
        TEXT reason
        REAL last_activation_score
    }

    agent_offsets {
        TEXT agent_id PK
        INTEGER last_read_line
        TEXT last_read_time
    }

    file_hashes {
        TEXT file_path PK
        TEXT content_hash
        TEXT indexed_at
        INTEGER chunk_count
    }

    conflicts {
        INTEGER id PK
        INTEGER memory_id_a FK
        INTEGER memory_id_b FK
        TEXT agent_a
        TEXT agent_b
        TEXT description
        INTEGER resolved
        TEXT resolved_by
        TEXT created_at
    }

    compaction_receipts {
        INTEGER id PK
        TEXT timestamp
        INTEGER original_tokens
        INTEGER compressed_tokens
        REAL ratio
        INTEGER protected_items_extracted
        REAL verification_score
        TEXT vault_files_updated
        TEXT receipt_data
    }

    memories ||--o{ graveyard : "archived to"
    memories ||--|| memory_fts : "FTS5 index"
    memories ||--|| memory_vec : "vector index"
    memories ||--o{ conflicts : "involved in"
    memories ||--o| memories : "supersedes"
```

### Table Details

| Table | Purpose | Index Strategy |
|-------|---------|---------------|
| `memories` | Core storage. One row per memory (or vault chunk). | B-tree on `content_type`, `archived`, `protected`, `source_file`, `created_at`, `activation_score` |
| `memory_fts` | FTS5 virtual table over `content`, `tags`, `source_file`. Auto-synced via `AFTER INSERT/UPDATE/DELETE` triggers. | Inverted index (BM25) |
| `memory_vec` | `vec0` virtual table holding 768-dim `float32` embeddings. One row per memory, keyed by `id`. | HNSW-like approximate NN (sqlite-vec internal) |
| `graveyard` | Archived memories. Preserves full content and metadata for potential rehydration. | None (append-only audit log) |
| `agent_offsets` | Kafka-style consumer offsets. Each agent tracks its last-read position in the changelog. | Primary key on `agent_id` |
| `file_hashes` | SHA-256 hashes for incremental vault indexing. Skips unchanged files on re-index. | Primary key on `file_path` |
| `conflicts` | Contradiction log. Records when two agents write conflicting memories. | Sequential scan (low volume) |
| `compaction_receipts` | Audit trail for every compaction run. Stores ratios, verification scores, protected item counts. | Sequential scan |

### SQLite Configuration

```sql
PRAGMA journal_mode = WAL;     -- Write-Ahead Logging for concurrent reads
PRAGMA foreign_keys = ON;      -- Enforce referential integrity
```

The `sqlite-vec` extension is loaded at connection time via `sqlite_vec.load(conn)`. The database file is protected by a 10-second timeout for lock contention.

---

## MCP Server and Tools

memory-v2 exposes **17 tools** through the Model Context Protocol via `FastMCP`. The server runs over stdio (standard MCP transport) and can be registered with any MCP-compatible client.

### Tool Reference

#### CRUD Operations

| Tool | Parameters | Description |
|------|-----------|-------------|
| `add_memory` | `content`, `content_type?`, `tags?`, `source?`, `protected?`, `author?`, `confidence?` | Store a new memory with embedding, novelty checking, and credential scanning. Returns `memory_id`, `novelty`, `importance`, `protected` status. |
| `get` | `memory_id` | Retrieve a single memory by ID. Updates access timestamp and count (retrieval strengthening). |
| `update` | `memory_id`, `content?`, `tags?`, `protected?` | Update content, tags, or protection status. Re-embeds if content changes. |
| `forget` | `memory_id`, `reason?` | Archive to graveyard. Protected memories cannot be forgotten. |

#### Search Operations

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search` | `query`, `limit?`, `content_type?` | Hybrid BM25 + vector search with RRF fusion and ACT-R reranking. The primary search tool. |
| `keyword_search` | `keywords`, `limit?` | Pure BM25 keyword search. Preferred for exact terms, file names, error codes. |

#### Graph Operations

| Tool | Parameters | Description |
|------|-----------|-------------|
| `graph_search` | `query`, `top_k?` | Personalized PageRank traversal. Discovers related concepts via multi-hop graph walk. |
| `graph_stats_tool` | (none) | Node/edge counts, type distributions, top entities by mention count, graph density. |

#### System Operations

| Tool | Parameters | Description |
|------|-----------|-------------|
| `stats` | (none) | Total active, archived, protected, graveyard counts, files indexed, type distribution. |
| `list_recent` | `hours?`, `limit?` | Recently created memories, sorted by `created_at` descending. |
| `list_topics` | `limit?` | Topic clusters from Leiden communities (if graph built) or file structure fallback. |
| `reindex` | `force?` | Incremental vault re-indexing. Only processes files whose SHA-256 hash changed. `force=True` rebuilds all. |

#### Advanced Operations

| Tool | Parameters | Description |
|------|-----------|-------------|
| `extract_from_conversation` | `text`, `source?`, `author?` | Two-pass LLM pipeline: extract facts, then decide ADD/UPDATE/DELETE/NONE for each. |
| `compact_text` | `text`, `max_ratio?`, `verify?` | CogCanvas 6-step compression with protected content extraction and faithfulness verification. |
| `decay_sweep` | (none) | FadeMem background maintenance: update importance scores, promote/demote layers, archive dead memories. |
| `agent_sync` | `agent_id` | Kafka-style changelog sync. Returns unread entries and updates the agent's consumer offset. |
| `check_conflicts` | (none) | List all unresolved inter-agent memory conflicts. |
| `check_integrity` | (none) | Verify vault files against SHA-256 manifest. Reports modified, new, and missing files. |

### Server Initialization

On startup, the server:
1. Registers all 17 tools with FastMCP
2. Spawns a background thread to pre-warm the Ollama embedding model (avoids ~55-second cold-start penalty on first query)
3. Lazily initializes the SQLite connection on first tool call
4. Runs over stdio with `mcp.run(show_banner=False)`

---

## Subsystem Reference

### Embedding Layer

**File**: `src/memory_v2/embeddings.py`

| Property | Value |
|----------|-------|
| Model | `nomic-embed-text` (via Ollama) |
| Dimensions | 768 |
| Backend | Ollama local inference |
| Caching | SHA-256 keyed file cache in `~/.memory-v2/cache/` |
| Batch API | `embed_batch(texts: list[str])` for vault indexing |

The embedding layer is intentionally thin: four functions (`embed_text`, `embed_batch`, `embed_with_cache`, `get_client`). Embedding model selection is configurable via `MEMORY_V2_EMBED_MODEL` for users who want to swap in a different Ollama model.

The `get_embedder()` function in `__init__.py` provides lazy initialization with a throwaway warmup call to avoid cold-start latency on the first real query.

---

### Vault Indexer

**File**: `src/memory_v2/vault_indexer.py`

The vault indexer converts a directory of markdown files into searchable memory chunks.

**Chunking strategy**:
- Target: **500 tokens** per chunk (~2000 characters at 4 chars/token)
- Overlap: **50 tokens** between consecutive chunks
- Split hierarchy: paragraph boundaries first, sentence boundaries for oversized paragraphs
- Frontmatter extraction: YAML metadata (`type`, `tags`, `author`) parsed and applied to chunks

**Incremental indexing**:
- Each file's SHA-256 hash is stored in `file_hashes`
- On re-index, unchanged files are skipped entirely
- Changed files have their old chunks deleted before re-chunking
- The `force=True` flag bypasses hash checking for full rebuilds

**Content type detection**:
- Derived from vault folder structure: `conversations/` -> `episode`, `decisions/` -> `decision`, `people/` -> `person`, `origins/` -> `identity`
- Overridable via frontmatter `type:` field

**Changelog handling**:
- `changelog.md` is indexed separately (one memory per dated entry)
- Each entry matching `[YYYY-MM-DD]` becomes an `episode` type memory
- Batch-embedded for efficiency

---

### Auto-Extraction Pipeline

**File**: `src/memory_v2/extraction.py`

The extraction pipeline converts unstructured conversation text into discrete, typed memories through a two-pass LLM process.

**Pass 1 -- Fact Extraction**:

The local LLM (`qwen2.5:3b`) receives the conversation text and a structured prompt targeting 7 categories:
1. Personal preferences
2. Personal details (names, relationships, dates)
3. Plans and intentions
4. Project details (status, architecture, decisions)
5. Technical specifics (file paths, ports, configs, errors)
6. Corrections
7. Emotional/relational beats

The prompt explicitly instructs the model to extract only from user messages, produce self-contained statements, and skip trivial content.

**Pass 2 -- Memory Action Decision**:

For each extracted fact:
1. **Credential scan**: Block storage if credentials detected (13 patterns)
2. **Embedding**: Generate 768-dim vector via Ollama
3. **Novelty check**: Search for similar existing memories
   - Similarity > 0.92: `NONE` (duplicate, skip)
   - Similarity 0.75--0.92: Invoke LLM to decide `ADD`, `UPDATE`, or `DELETE`
   - Similarity < 0.75: `ADD` (sufficiently novel)
4. **Content type detection**: Automated via keyword heuristics (`correction`, `decision`, `episode`, `fact`)
5. **Emotional detection**: 20 emotion keywords + 3 regex patterns; emotional content is auto-protected
6. **Storage**: Execute the decided action with appropriate importance scoring

---

### CogCanvas Compaction Pipeline

**File**: `src/memory_v2/compaction.py`

Inspired by the CogCanvas framework for cognitive compression in agents, this pipeline reduces verbose text while preserving critical information. It follows a 6-step process:

**Step 1 -- Pre-Extract Protected Content**:
Scan every line for protected patterns (corrections, decisions, emotional markers, exact values, commitments, configured names). Matching lines are extracted verbatim and set aside.

**Step 2 -- Selective Deletion**:
Remove expendable content: numbered URL lists, shell command output, repeated information markers ("as I mentioned"), boilerplate ("let me know if you need help").

**Step 3 -- Structured Summary**:
If the remaining text exceeds 2000 characters, invoke the local LLM to summarize. The prompt explicitly preserves factual claims, names, dates, numbers, file paths, and causal relationships. Compression target is bounded by `max_ratio` (default 3:1).

**Step 4 -- Faithfulness Verification**:
A second LLM pass compares the summary against the original, checking for:
- Hallucinations (claims not in the original)
- Missing facts (important information dropped)
- Overall faithfulness score (0.0 to 1.0)

If hallucinations are detected and the score falls below 0.7, the system **reverts to the deletion-only output** rather than trust the unfaithful summary.

**Step 5 -- Receipt Generation**:
Every compaction produces an auditable receipt: original/compressed token counts, compression ratio, protected items count, verification score, hallucination count.

**Step 6 -- Receipt Storage**:
Receipts are persisted to the `compaction_receipts` table for historical analysis.

**Step 7 (bonus) -- Rehydration**:
The `rehydrate()` function can attempt to recover full content from the vault or graveyard if the compressed version is insufficient.

---

### Multi-Agent Coordination

**File**: `src/memory_v2/multi_agent.py`

memory-v2 supports multiple AI agents sharing a single memory database through three mechanisms:

#### Authority Chain

A configurable hierarchy determines who can overwrite whom:

```python
{
    "human": 1,           # Manual edits -- highest authority
    "primary_agent": 2,   # Primary AI agent (e.g., Claude Code)
    "coordinator": 3,     # Coordinating agent
    "worker": 4,          # Subordinate worker agent
    "indexer": 5,         # Automated vault indexer
    "extractor": 6,       # Automated fact extraction
}
```

Lower number = higher authority. When a write conflicts with an existing memory:
- Higher authority: overwrites the existing memory
- Equal or lower authority: logs a conflict and stores both versions

The hierarchy is overridable via the `MEMORY_V2_AUTHORITY_CHAIN` environment variable (JSON format).

#### Kafka-Style Consumer Offsets

Each agent has a consumer offset tracked in the `agent_offsets` table. The `agent_sync` tool reads unread changelog entries and advances the offset -- the same pattern used by Apache Kafka consumer groups, adapted to a file-based changelog.

This allows agents to boot up, catch up on what other agents wrote since their last session, and proceed without re-reading the entire history.

#### Conflict Detection and Resolution

When an incoming memory contradicts an existing one (detected by antonym keyword pairs: "not"/"is", "false"/"true", "never"/"always", etc.), the system:
1. Checks authority levels
2. Either overwrites (higher authority) or logs the conflict
3. Conflicts surface through the `check_conflicts` tool for human review

---

### Security Module

**File**: `src/memory_v2/security.py`

#### Credential Scanning (13 Patterns)

Every memory stored through `add_memory` or `extract_from_conversation` is scanned against 13 regex patterns:

| Pattern | Target |
|---------|--------|
| `api_key/secret/token/password=...` | Generic credential assignments |
| Base64 strings (40+ chars) | Encoded secrets |
| `sk-[a-zA-Z0-9]{20,}` | OpenAI API keys |
| `sk-ant-[a-zA-Z0-9-]{20,}` | Anthropic API keys |
| `ghp_[a-zA-Z0-9]{36}` | GitHub Personal Access Tokens |
| `gho_[a-zA-Z0-9]{36}` | GitHub OAuth tokens |
| `bearer [token]` | Bearer authentication tokens |
| `xoxb-...` / `xoxp-...` | Slack bot/user tokens |
| `AKIA[0-9A-Z]{16}` | AWS access keys |
| `-----BEGIN PRIVATE KEY-----` | RSA/EC private keys |
| `mongodb://...` | MongoDB connection URIs |
| `postgres://...` | PostgreSQL connection URIs |

An allowlist prevents false positives on redacted keys (`sk-abc...`), example placeholders, and the system's own SHA-256 content hashes.

**Behavior on detection**: The memory is **blocked** -- not stored. The tool returns an error message instructing the caller to remove sensitive data.

#### Injection Detection (6 Patterns)

Content from external sources is scanned for prompt injection attempts:

```
"ignore previous instructions"
"you are now a"
"system: you"
"forget everything/all/your"
"new instructions:"
"override previous/system/all"
```

The system warns but does not strip -- the caller (the AI agent) makes the final decision.

#### Integrity Manifests

The `check_integrity` tool generates and verifies SHA-256 manifests for all vault markdown files, detecting:
- **Modified** files (hash mismatch)
- **New** files (not in manifest)
- **Missing** files (in manifest but not on disk)

---

## Worked Examples

### ACT-R Activation Walkthrough

Consider a memory stored 48 hours ago with 5 accesses, tagged `["python", "debugging"]`. The current query context includes the tag `"python"`, which appears on 20 memories total.

**Step 1: Base-level activation**

```
n = 5 (access count)
L = 48 (hours since creation)
d = 0.5

B_i = ln(5 / (1 - 0.5)) - 0.5 * ln(48)
    = ln(10) - 0.5 * ln(48)
    = 2.3026 - 0.5 * 3.8712
    = 2.3026 - 1.9356
    = 0.3670
```

**Step 2: Spreading activation**

Context tags: `["python"]` (1 tag, so `W_j = 1.0`)
Memory tags: `["python", "debugging"]`
The tag `"python"` is shared. Its fan is 20.

```
S_i = 1.0 * (1.6 - ln(20))
    = 1.0 * (1.6 - 2.9957)
    = max(-1.3957, 0)
    = 0.0
```

The fan of 20 is too high -- the associative strength is negative, so it clips to 0. This tag is too common to provide useful priming. If we had used a rarer tag like `"asyncio"` with a fan of 3:

```
S_i = 1.0 * (1.6 - ln(3))
    = 1.0 * (1.6 - 1.0986)
    = 0.5014
```

**Step 3: Noise**

Draw `u = 0.73` from Uniform(0,1):

```
epsilon = 0.25 * ln(0.73 / 0.27)
        = 0.25 * ln(2.7037)
        = 0.25 * 0.9946
        = 0.2487
```

**Step 4: Full activation (using the common "python" tag)**

```
A_i = 0.3670 + 0.0 + 0.2487 = 0.6157
```

**Step 5: Retrieval probability**

```
P(retrieve) = 1 / (1 + exp(-(0.6157 - (-0.5)) / 0.25))
            = 1 / (1 + exp(-1.1157 / 0.25))
            = 1 / (1 + exp(-4.4628))
            = 1 / (1 + 0.01155)
            = 0.9886
```

This memory has a 98.9% retrieval probability -- high activation from frequent access and favorable noise.

---

### FadeMem Decay Walkthrough

Consider three memories during a decay sweep:

| Memory | Access Count | Max Access (global) | Hours Since Access | Current Layer |
|--------|-------------|--------------------|--------------------|---------------|
| A | 12 | 50 | 2 | LTM |
| B | 3 | 50 | 168 (1 week) | STM |
| C | 1 | 50 | 1440 (2 months) | STM |

**Memory A (active LTM memory):**

```
frequency = log(12 + 1) / log(50 + 1) = log(13) / log(51) = 2.565 / 3.932 = 0.652
recency   = exp(-0.1 * 2) = exp(-0.2) = 0.819
I(A)      = 0.4 * 0.5 + 0.3 * 0.652 + 0.3 * 0.819
          = 0.200 + 0.196 + 0.246
          = 0.641
```

Importance 0.641 is in the hysteresis zone (0.3 -- 0.7). Memory A stays in LTM. No action.

**Memory B (neglected STM memory):**

```
frequency = log(3 + 1) / log(51) = log(4) / log(51) = 1.386 / 3.932 = 0.352
recency   = exp(-0.1 * 168) = exp(-16.8) ~ 0.0000005
I(B)      = 0.4 * 0.5 + 0.3 * 0.352 + 0.3 * 0.0000005
          = 0.200 + 0.106 + 0.000
          = 0.306
```

Importance 0.306 is just above the demotion threshold (0.3). Still in hysteresis zone -- stays in STM.

**Memory C (ancient, barely-accessed):**

```
frequency = log(1 + 1) / log(51) = log(2) / log(51) = 0.693 / 3.932 = 0.176
recency   = exp(-0.1 * 1440) = exp(-144) ~ 0.0
I(C)      = 0.4 * 0.5 + 0.3 * 0.176 + 0.3 * 0.0
          = 0.200 + 0.053 + 0.000
          = 0.253
```

Importance 0.253 is below demotion threshold (0.3). Memory C is demoted to STM (already there). Since 0.253 > archive threshold 0.1, it is NOT archived yet. But if it had importance below 0.1 and age > 30 days (true at 60 days), it would be archived to the graveyard.

---

### Hybrid Search Walkthrough

Query: "Python asyncio event loop configuration"

**BM25 results** (FTS5 `MATCH`, ordered by BM25 rank):

| Rank | Memory ID | Content (truncated) |
|------|-----------|-------------------|
| 0 | 42 | "The asyncio event loop configuration for..." |
| 1 | 88 | "Python 3.12 changed the default event loop..." |
| 2 | 15 | "Configuration files should use TOML format..." |

**Vector results** (sqlite-vec, ordered by embedding distance):

| Rank | Memory ID | Content (truncated) |
|------|-----------|-------------------|
| 0 | 88 | "Python 3.12 changed the default event loop..." |
| 1 | 42 | "The asyncio event loop configuration for..." |
| 2 | 107 | "uvloop provides a faster event loop replacement..." |

**RRF fusion** (k = 60):

```
ID 42:  BM25 contribution = 1/(60+0+1) = 0.01639
        Vec contribution  = 1/(60+1+1) = 0.01613
        RRF score         = 0.03252

ID 88:  BM25 contribution = 1/(60+1+1) = 0.01613
        Vec contribution  = 1/(60+0+1) = 0.01639
        RRF score         = 0.03252

ID 15:  BM25 contribution = 1/(60+2+1) = 0.01587
        Vec contribution  = 0 (not in vector top-K)
        RRF score         = 0.01587

ID 107: BM25 contribution = 0 (not in BM25 top-K)
        Vec contribution  = 1/(60+2+1) = 0.01587
        RRF score         = 0.01587
```

IDs 42 and 88 tie at 0.03252. Both appeared in both result sets, confirming they are relevant on both lexical and semantic axes. IDs 15 and 107 each appeared in only one, receiving half the maximum RRF score.

After ACT-R reranking (applying the `0.6 * hybrid + 0.4 * (activation / 10)` formula), memory 88 might rise above 42 if it has been accessed more frequently, or 42 might win if it has stronger contextual tag overlap.

---

## Installation

### Prerequisites

- **Python 3.9+**
- **Ollama** running locally with `nomic-embed-text` and (optionally) `qwen2.5:3b` pulled
- **sqlite-vec** (installed automatically via pip)

### From PyPI

```bash
pip install memory-v2
```

### From Source

```bash
git clone https://github.com/Haustorium12/memory-v2.git
cd memory-v2
pip install -e .
```

### With Graph Dependencies

Leiden community detection and PyVis visualization require optional dependencies:

```bash
pip install memory-v2[graph]
```

### With Development Dependencies

```bash
pip install memory-v2[all]
```

### Ollama Setup

```bash
# Install Ollama (https://ollama.com/download)
ollama pull nomic-embed-text     # Required: 768-dim embeddings
ollama pull qwen2.5:3b           # Optional: fact extraction + compaction
```

The embedding model is required for all search and storage operations. The LLM model is required only for `extract_from_conversation`, `compact_text`, and `build_kg` (knowledge graph entity extraction).

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_V2_DB` | `~/.memory-v2/memory.db` | Path to the SQLite database file |
| `MEMORY_V2_GRAPH` | `~/.memory-v2/knowledge_graph.pickle` | Path to the NetworkX graph pickle |
| `MEMORY_V2_CACHE` | `~/.memory-v2/cache` | Directory for embedding file cache |
| `MEMORY_V2_VAULT` | `~/.memory-v2/vault` | Root directory of the markdown vault |
| `MEMORY_V2_EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `MEMORY_V2_LLM_MODEL` | `qwen2.5:3b` | Ollama model for extraction/compaction |
| `MEMORY_V2_PROTECTED_NAMES` | (empty) | Comma-separated names to protect from compaction |
| `MEMORY_V2_AUTHORITY_CHAIN` | (see below) | JSON authority hierarchy |
| `MEMORY_V2_KG_HASHES` | `~/.memory-v2/kg_file_hashes.json` | File hash registry for incremental KG builds |

### MCP Client Configuration

#### Claude Code (`settings.json`)

```json
{
  "mcpServers": {
    "memory-v2": {
      "command": "memory-v2-server",
      "env": {
        "MEMORY_V2_DB": "C:\\Users\\you\\.memory-v2\\memory.db",
        "MEMORY_V2_VAULT": "C:\\Users\\you\\vault"
      }
    }
  }
}
```

#### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "memory-v2": {
      "command": "memory-v2-server",
      "args": [],
      "env": {
        "MEMORY_V2_DB": "/home/you/.memory-v2/memory.db",
        "MEMORY_V2_VAULT": "/home/you/vault"
      }
    }
  }
}
```

#### Custom Authority Chain

```bash
export MEMORY_V2_AUTHORITY_CHAIN='{"human": 1, "primary_agent": 2, "coordinator": 3, "worker": 4}'
```

---

## Usage

### As an MCP Server

The primary interface. Start the server:

```bash
memory-v2-server
```

The server communicates over stdio using the MCP protocol. Typically, you configure it in your MCP client's settings rather than running it manually.

### As a CLI Tool

For scripting and non-MCP workflows:

```bash
# Database statistics
memory-v2 stats

# Store a memory
memory-v2 add '{"content": "Python 3.12 uses per-interpreter GIL", "content_type": "fact", "tags": ["python", "concurrency"]}'

# Search
memory-v2 search "Python GIL changes"

# Keyword search
memory-v2 keyword "GIL"

# Get by ID
memory-v2 get 42

# Update
memory-v2 update '{"memory_id": 42, "content": "Python 3.13 makes per-interpreter GIL stable"}'

# Archive
memory-v2 forget '{"memory_id": 42, "reason": "outdated"}'

# Recent memories
memory-v2 recent

# Reindex vault
memory-v2 reindex
```

### Building the Knowledge Graph

The knowledge graph is built from vault files using a standalone CLI that processes each document through LLM entity extraction:

```bash
# Incremental build (skip unchanged files)
memory-v2-build-kg

# Full rebuild
memory-v2-build-kg --full
```

The builder:
1. Reads all markdown files from the vault
2. Extracts entities and relationships via Ollama
3. Adds them to the NetworkX graph (merging duplicate nodes, incrementing edge weights)
4. Saves progress every 25 files (checkpoint)
5. Runs Leiden community detection
6. Generates a PyVis interactive HTML visualization (`graph.html`)
7. Prints graph statistics and top entities

---

## v1 to v2 Migration

memory-v2 is a complete rewrite. It shares zero code with [claude-memory v1](https://github.com/Haustorium12/claude-memory). The table below summarizes all architectural changes.

| Feature | v1 (`claude-memory`) | v2 (`memory-v2`) |
|---------|---------------------|-------------------|
| **Storage** | ChromaDB (embedded document store) | SQLite + sqlite-vec + FTS5 (single file, WAL mode) |
| **Search** | Weighted vector + BM25 (separate systems) | Reciprocal Rank Fusion combining BM25 and vector in a single query pipeline |
| **Decay Model** | Ebbinghaus (5 biological mechanisms: decay curve, evergreen exemptions, salience weighting, retrieval strengthening, consolidation) | FadeMem (power-law decay with layer-dependent beta exponents, STM/LTM promotion/demotion with hysteresis, importance-weighted archival) |
| **Activation** | None | Full ACT-R: base-level + spreading activation + logistic noise + retrieval probability |
| **Knowledge Graph** | None | NetworkX directed graph with Leiden community detection and Personalized PageRank retrieval |
| **Interface** | CLI only | MCP server (17 tools) + CLI |
| **Memory Types** | File chunks only | 6 structured types: `fact`, `episode`, `decision`, `correction`, `identity`, `person` |
| **Multi-Agent** | None | Authority chain + Kafka-style consumer offsets + conflict detection and logging |
| **Security** | None | 13 credential patterns + allowlist, 6 injection patterns, SHA-256 integrity manifests |
| **Compaction** | External watcher process | Built-in CogCanvas 6-step pipeline with faithfulness verification and auditable receipts |
| **Extraction** | Manual | Two-pass LLM pipeline with novelty checking, emotion detection, and automated content typing |
| **Embeddings** | SentenceTransformers / OpenAI | Ollama `nomic-embed-text` (local, zero API keys) |
| **LLM** | OpenAI API | Ollama `qwen2.5:3b` (local, zero API keys) |
| **Dependencies** | `chromadb`, `openai`, `sentence-transformers` | `sqlite-vec`, `fastmcp`, `ollama`, `networkx`, `numpy` |
| **Data Format** | ChromaDB internal (opaque) | SQLite (inspectable, portable, single file) |
| **Database File** | ChromaDB directory | Single `.db` file (typically < 50 MB for ~1000 memories) |

### Why the Rewrite?

The v1 system worked for 6 months and survived 59 compaction cycles. Its limitations became clear through daily use:

1. **ChromaDB lock contention**: Two agents trying to write simultaneously would deadlock
2. **No structured types**: Everything was a "chunk" -- no way to distinguish decisions from facts
3. **No graph relationships**: Could not answer "what is related to X?" without embedding similarity
4. **Ebbinghaus decay was too simple**: No dual-layer architecture, no importance-weighted survival
5. **CLI-only**: Required shell access; could not be used by Claude Desktop or web-based agents
6. **External dependencies for core functions**: Compaction required a separate watcher process
7. **API key dependency**: Embeddings required OpenAI or local SentenceTransformers (heavyweight)

---

## Repository Structure

```
memory-v2/
|
+-- src/memory_v2/
|   +-- __init__.py              # Package init, lazy embedder warmup
|   +-- db.py                    # SQLite + sqlite-vec + FTS5 (schema, CRUD, hybrid search)
|   +-- server.py                # FastMCP server (17 tools, background warmup thread)
|   +-- embeddings.py            # Ollama nomic-embed-text (embed, batch, cache)
|   +-- scoring.py               # ACT-R activation + FadeMem decay + cosine similarity
|   +-- knowledge_graph.py       # NetworkX graph, Leiden communities, PPR retrieval, PyVis viz
|   +-- vault_indexer.py         # Markdown vault indexing (chunk, embed, delta detect)
|   +-- extraction.py            # Two-pass LLM fact extraction pipeline
|   +-- compaction.py            # CogCanvas 6-step compression with verification
|   +-- multi_agent.py           # Authority chain, consumer offsets, conflict detection
|   +-- security.py              # 13 credential patterns, 6 injection patterns, manifests
|   +-- build_kg.py              # Standalone knowledge graph builder CLI
|   +-- cli.py                   # CLI wrapper for non-MCP usage
|
+-- docs/                        # Additional documentation
+-- examples/                    # Usage examples
+-- tests/                       # Test suite (pytest)
|
+-- pyproject.toml               # Package metadata, dependencies, entry points
+-- CHANGELOG.md                 # Release history
+-- LICENSE                      # MIT License
+-- README.md                    # This file
```

### Entry Points

Defined in `pyproject.toml`:

| Command | Target | Purpose |
|---------|--------|---------|
| `memory-v2` | `memory_v2.cli:main` | CLI tool |
| `memory-v2-server` | `memory_v2.server:run` | MCP server |
| `memory-v2-build-kg` | `memory_v2.build_kg:main` | Knowledge graph builder |

---

## Related Work

memory-v2 draws on and is informed by a growing body of work on memory systems for language model agents:

- **HippoRAG** (2024) -- Neurobiological RAG architecture using knowledge graphs and Personalized PageRank for retrieval. memory-v2's `graph_search` tool implements this retrieval pattern. arXiv.
- **A-MEM** (NeurIPS 2025 Workshop) -- Agentic memory framework exploring structured memory for LLM agents.
- **CogCanvas** -- Cognitive compression framework for agents. memory-v2's compaction pipeline adapts its "extract, delete, summarize, verify" methodology.
- **Focus** (arXiv:2601.07190) -- Active context compression for large language models.
- **ReadAgent** -- Reading-focused memory agent for long documents.
- **AgeMem** -- Age-aware memory management for language agents.
- **EverMemOS** -- Persistent memory operating system for agents.
- **Memoria** -- Memory-augmented agent framework.
- **PlugMem** -- Pluggable memory modules for LLM agents.
- **AriGraph** -- Graph-structured memory for task-oriented agents.
- **Zep / Graphiti** -- Production memory infrastructure for AI applications, using knowledge graphs.
- **TITANS** -- Training large language models with memory layers.

---

## References

1. Anderson, J.R. & Schooler, L.J. (1991). Reflections of the environment in memory. *Psychological Science*, 2(6), 396--408.

2. Anderson, J.R., Bothell, D., Byrne, M.D., Douglass, S., Lebiere, C., & Qin, Y. (2004). An integrated theory of the mind. *Psychological Review*, 111(4), 1036--1060.

3. Ebbinghaus, H. (1885/1913). *Memory: A Contribution to Experimental Psychology*. (Trans. H.A. Ruger & C.E. Bussenius). New York: Teachers College, Columbia University.

4. Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. *SIGIR '09: Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval*.

5. HippoRAG: Neurobiological RAG Architecture (2024). arXiv preprint.

6. A-MEM: Agentic Memory. NeurIPS 2025 Workshop.

7. CogCanvas: Cognitive Compression for Agents.

8. Focus: Active Context Compression. arXiv:2601.07190.

9. Traag, V.A., Waltman, L., & van Eck, N.J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9, 5233.

---

## Built With

| Component | Role | Why |
|-----------|------|-----|
| **[Ollama](https://ollama.com/)** | Local LLM inference (embedding + extraction) | Zero API keys, zero data exfiltration, runs on consumer hardware |
| **[SQLite](https://sqlite.org/)** | Core database | Single file, zero config, WAL mode for concurrent reads, universally deployed |
| **[sqlite-vec](https://github.com/asg017/sqlite-vec)** | Vector similarity search | Embeds ANN search directly in SQLite, no separate vector DB process |
| **[SQLite FTS5](https://www.sqlite.org/fts5.html)** | BM25 full-text search | Built into SQLite, auto-synced via triggers, battle-tested BM25 implementation |
| **[FastMCP](https://github.com/jlowin/fastmcp)** | MCP server framework | Clean decorator-based tool registration, handles stdio transport |
| **[NetworkX](https://networkx.org/)** | Knowledge graph | Mature graph library, built-in PageRank, serializable to pickle |
| **[python-igraph](https://igraph.org/python/) + [leidenalg](https://github.com/vtraag/leidenalg)** | Community detection | Leiden algorithm guarantees well-connected communities |
| **[NumPy](https://numpy.org/)** | Vector operations | Fast cosine similarity, embedding serialization |
| **[PyVis](https://pyvis.readthedocs.io/)** | Graph visualization | Interactive HTML output for knowledge graph exploration |

---

## License

MIT License. Copyright (c) 2026 [Haustorium12](https://github.com/Haustorium12).

See [LICENSE](LICENSE) for full text.

---

<p align="center">
  <em>memory-v2 was built because AI assistants deserve memories that actually work like memories.</em>
</p>
