"""
memory-v2 -- MCP Server
FastMCP server exposing the memory system to any AI coding assistant.
17 tools: add_memory, search, keyword_search, get, update,
         forget, list_recent, list_topics, stats, reindex,
         extract_from_conversation, graph_search, graph_stats_tool,
         compact_text, agent_sync, check_conflicts, check_integrity,
         decay_sweep
"""

import json
import os
import threading
from pathlib import Path

from fastmcp import FastMCP

from . import db, embeddings
from .vault_indexer import index_vault, index_changelog, VAULT_PATH

# ---------- Server setup ----------

mcp = FastMCP(
    "memory-v2",
    instructions=(
        "memory-v2: Brain-inspired memory system with hybrid BM25+vector search, "
        "ACT-R activation scoring, FadeMem decay, and knowledge graph traversal. "
        "The vault (markdown files) is the source of truth. This server is the search index."
    ),
)

# Pre-warm the embedding model in a background thread so it's ready
# before the first search. Avoids the 55-second cold-start penalty
# from importing sentence_transformers + PyTorch on first tool call.


def _prewarm_embedder():
    try:
        from . import get_embedder
        get_embedder()
    except Exception:
        pass


_warmup_thread = threading.Thread(target=_prewarm_embedder, daemon=True)
_warmup_thread.start()

# Lazy-init connection
_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        try:
            _conn = db.init_db()
        except Exception as e:
            raise RuntimeError(f"Memory DB unavailable: {e}. Another process may hold the lock.")
    return _conn


# ---------- Tools ----------

@mcp.tool()
def add_memory(
    content: str,
    content_type: str = "fact",
    tags: list[str] | None = None,
    source: str | None = None,
    protected: bool = False,
    author: str = "claude-code",
    confidence: float = 0.95,
) -> str:
    """Store a new memory. Types: fact, episode, decision, correction, identity, person.
    Protected memories never decay. Returns the memory ID."""
    conn = _get_conn()

    # Credential scanning
    from .security import scan_for_credentials
    if scan_for_credentials(content):
        return json.dumps({
            "error": "BLOCKED: Content contains potential credentials/secrets. Not stored.",
            "action": "Review and remove sensitive data before storing."
        })

    # Generate embedding
    try:
        emb = embeddings.embed_text(content)
    except Exception as e:
        return json.dumps({"error": f"Embedding failed: {e}"})

    # Novelty check
    existing = db.hybrid_search(conn, content[:100], emb, limit=3)
    novelty = 1.0
    if existing:
        from .scoring import cosine_similarity
        import numpy as np
        for mem in existing:
            # Get embedding of existing memory
            row = conn.execute(
                "SELECT embedding FROM memory_vec WHERE id = ?", (mem["id"],)
            ).fetchone()
            if row:
                existing_emb = np.frombuffer(row[0], dtype=np.float32).tolist()
                sim = cosine_similarity(emb, existing_emb)
                novelty = min(novelty, 1.0 - sim)

    # Set importance based on type and novelty
    importance = 0.5
    if protected or content_type in ("decision", "correction", "identity"):
        importance = 0.9
        protected = True
    elif novelty > 0.7:
        importance = 0.7
    elif novelty < 0.3:
        # Very similar to existing -- consider skipping
        importance = 0.3

    memory_id = db.add_memory(
        conn,
        content=content,
        embedding=emb,
        content_type=content_type,
        source_file=source,
        author=author,
        confidence=confidence,
        protected=protected,
        tags=tags,
        importance_score=importance,
    )

    return json.dumps({
        "memory_id": memory_id,
        "novelty": round(novelty, 3),
        "importance": round(importance, 3),
        "protected": protected,
    })


@mcp.tool()
def search(
    query: str,
    limit: int = 10,
    content_type: str | None = None,
) -> str:
    """Hybrid search: BM25 keywords + vector similarity + ACT-R scoring.
    Returns ranked results with scores. Use this for any 'what do I know about X' query."""
    conn = _get_conn()

    try:
        query_emb = embeddings.embed_text(query)
    except Exception as e:
        return json.dumps({"error": f"Embedding failed: {e}"})

    results = db.hybrid_search(
        conn, query, query_emb,
        limit=limit, content_type=content_type,
    )

    # Apply ACT-R scoring
    try:
        from .scoring import apply_actr_scoring
        results = apply_actr_scoring(conn, results, query)
    except ImportError:
        pass

    # Format for output
    output = []
    for mem in results:
        output.append({
            "id": mem["id"],
            "content": mem["content"][:500],
            "content_type": mem["content_type"],
            "source_file": mem["source_file"],
            "tags": mem.get("tags", "[]"),
            "protected": bool(mem["protected"]),
            "hybrid_score": round(mem.get("hybrid_score", 0), 4),
            "created_at": mem["created_at"],
            "access_count": mem["access_count"],
        })

    return json.dumps({
        "query": query,
        "results": output,
        "count": len(output),
    })


@mcp.tool()
def keyword_search(keywords: str, limit: int = 10) -> str:
    """Pure BM25 keyword search. Use for exact terms, file names, error codes."""
    conn = _get_conn()
    results = db.keyword_search(conn, keywords, limit)

    output = []
    for mem in results:
        output.append({
            "id": mem["id"],
            "content": mem["content"][:500],
            "source_file": mem["source_file"],
            "bm25_rank": mem.get("bm25_rank", 0),
        })

    return json.dumps({"keywords": keywords, "results": output, "count": len(output)})


@mcp.tool()
def get(memory_id: int) -> str:
    """Get a single memory by ID with full content and metadata."""
    conn = _get_conn()
    mem = db.get_memory(conn, memory_id)
    if mem is None:
        return json.dumps({"error": f"Memory {memory_id} not found"})
    return json.dumps(mem)


@mcp.tool()
def update(
    memory_id: int,
    content: str | None = None,
    tags: list[str] | None = None,
    protected: bool | None = None,
) -> str:
    """Update an existing memory's content, tags, or protected status."""
    conn = _get_conn()
    success = db.update_memory(conn, memory_id, content, tags, protected)
    if not success:
        return json.dumps({"error": f"Memory {memory_id} not found or no changes"})

    # Update embedding if content changed
    if content:
        try:
            emb = embeddings.embed_text(content)
            conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                (memory_id, db._serialize_f32(emb))
            )
            conn.commit()
        except Exception:
            pass

    return json.dumps({"success": True, "memory_id": memory_id})


@mcp.tool()
def forget(memory_id: int, reason: str = "manual") -> str:
    """Archive a memory to the graveyard. Protected memories cannot be forgotten."""
    conn = _get_conn()
    success = db.archive_memory(conn, memory_id, reason)
    if not success:
        return json.dumps({
            "error": f"Memory {memory_id} not found, already archived, or protected"
        })
    return json.dumps({"success": True, "memory_id": memory_id, "archived": True})


@mcp.tool()
def list_recent(hours: int = 24, limit: int = 20) -> str:
    """List recently created memories."""
    conn = _get_conn()
    results = db.list_recent(conn, hours, limit)
    output = []
    for mem in results:
        output.append({
            "id": mem["id"],
            "content": mem["content"][:200],
            "content_type": mem["content_type"],
            "created_at": mem["created_at"],
            "source_file": mem["source_file"],
        })
    return json.dumps({"results": output, "count": len(output)})


@mcp.tool()
def list_topics(limit: int = 20) -> str:
    """List topic clusters from the knowledge graph (if built) or content types."""
    conn = _get_conn()

    # Try knowledge graph communities first
    try:
        from .knowledge_graph import get_communities
        communities = get_communities(limit)
        if communities:
            return json.dumps({"source": "knowledge_graph", "topics": communities})
    except (ImportError, Exception):
        pass

    # Fallback: group by source file directories
    rows = conn.execute("""
        SELECT
            CASE
                WHEN source_file LIKE 'conversations/%' THEN 'conversations'
                WHEN source_file LIKE 'decisions/%' THEN 'decisions'
                WHEN source_file LIKE 'projects/%' THEN 'projects'
                WHEN source_file LIKE 'intelligence/%' THEN 'intelligence'
                WHEN source_file LIKE 'people/%' THEN 'people'
                WHEN source_file LIKE 'briefings/%' THEN 'briefings'
                ELSE 'other'
            END as topic,
            COUNT(*) as count
        FROM memories
        WHERE archived = 0 AND source_file IS NOT NULL
        GROUP BY topic
        ORDER BY count DESC
        LIMIT ?
    """, (limit,)).fetchall()

    topics = [{"topic": r["topic"], "count": r["count"]} for r in rows]
    return json.dumps({"source": "file_structure", "topics": topics})


@mcp.tool()
def stats() -> str:
    """Get memory system statistics."""
    conn = _get_conn()
    s = db.get_stats(conn)
    return json.dumps(s)


@mcp.tool()
def reindex(force: bool = False) -> str:
    """Re-index the vault. Incremental by default (only changed files).
    Set force=True to re-index everything."""
    conn = _get_conn()

    def progress(current, total, name):
        pass  # Silent during MCP calls

    vault_stats = index_vault(conn, force=force, progress_callback=progress)
    changelog_stats = index_changelog(conn)

    return json.dumps({
        "vault": vault_stats,
        "changelog": changelog_stats,
    })


# ---------- Advanced Tools ----------

@mcp.tool()
def extract_from_conversation(
    text: str,
    source: str | None = None,
    author: str = "claude-code",
) -> str:
    """Auto-extract facts from a conversation and store them.
    Runs the 2-pass pipeline: extract facts, then decide ADD/UPDATE/DELETE/NONE."""
    conn = _get_conn()
    from .extraction import process_conversation
    result = process_conversation(conn, text, source, author)
    return json.dumps(result)


@mcp.tool()
def graph_search(query: str, top_k: int = 10) -> str:
    """Knowledge graph search using Personalized PageRank.
    Discovers related concepts via graph traversal (multi-hop reasoning)."""
    from .knowledge_graph import ppr_search
    results = ppr_search(query, top_k=top_k)
    return json.dumps({"query": query, "results": results, "count": len(results)})


@mcp.tool()
def graph_stats_tool() -> str:
    """Get knowledge graph statistics: node/edge counts, types, top entities."""
    from .knowledge_graph import graph_stats
    return json.dumps(graph_stats())


@mcp.tool()
def compact_text(
    text: str,
    max_ratio: float = 3.0,
    verify: bool = True,
) -> str:
    """Compact text using CogCanvas pipeline: extract protected content,
    delete expendable, summarize remainder, verify faithfulness."""
    conn = _get_conn()
    from .compaction import compact
    result = compact(text, max_ratio, verify, conn)
    # Don't return full compressed text in JSON (too large)
    return json.dumps({
        "receipt": result["receipt"],
        "protected_items_count": len(result["protected_items"]),
        "verification": result.get("verification", {}),
        "compressed_length": len(result["compressed_text"]),
    })


@mcp.tool()
def agent_sync(agent_id: str) -> str:
    """Sync an agent: get unread changelog entries and update offset.
    Use on boot to catch up on what other agents did."""
    conn = _get_conn()
    from .multi_agent import sync_agent
    result = sync_agent(agent_id, conn)
    return json.dumps(result)


@mcp.tool()
def check_conflicts() -> str:
    """Get unresolved memory conflicts between agents."""
    conn = _get_conn()
    from .multi_agent import get_unresolved_conflicts
    conflicts = get_unresolved_conflicts(conn)
    return json.dumps({"conflicts": conflicts, "count": len(conflicts)})


@mcp.tool()
def check_integrity() -> str:
    """Check vault file integrity against saved manifest.
    Returns lists of ok, modified, new, and missing files."""
    from .security import check_integrity as _check, save_manifest
    result = _check(str(VAULT_PATH))
    if "error" in result:
        # Generate manifest first
        save_manifest(str(VAULT_PATH))
        result = _check(str(VAULT_PATH))
    return json.dumps({
        "ok_count": len(result.get("ok", [])),
        "modified": result.get("modified", []),
        "new": result.get("new", []),
        "missing": result.get("missing", []),
    })


@mcp.tool()
def decay_sweep() -> str:
    """Run FadeMem decay sweep: update importance scores,
    promote/demote between STM/LTM, archive low-importance memories."""
    conn = _get_conn()
    from .scoring import importance_score, should_archive, get_layer
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    rows = conn.execute(
        "SELECT * FROM memories WHERE archived = 0 AND protected = 0"
    ).fetchall()

    max_access = max((r["access_count"] for r in rows), default=1)

    sweep_stats = {"swept": 0, "archived": 0, "promoted": 0, "demoted": 0}

    for row in rows:
        mem = dict(row)
        try:
            last_access = datetime.fromisoformat(mem["last_accessed_at"])
            hours_since = (now - last_access).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_since = 720  # 30 days default

        try:
            created = datetime.fromisoformat(mem["created_at"])
            age_days = (now - created).total_seconds() / 86400
        except (ValueError, TypeError):
            age_days = 30

        # Compute new importance
        new_importance = importance_score(
            relevance=0.5,  # No active query context during sweep
            access_count=mem["access_count"],
            max_access_count=max_access,
            hours_since_access=hours_since,
        )

        # Update importance
        conn.execute(
            "UPDATE memories SET importance_score = ? WHERE id = ?",
            (round(new_importance, 4), mem["id"])
        )

        # Check for archival
        try:
            tags = json.loads(mem.get("tags", "[]"))
        except (json.JSONDecodeError, TypeError):
            tags = []

        if should_archive(new_importance, age_days, tags):
            db.archive_memory(conn, mem["id"], reason="decay_sweep")
            sweep_stats["archived"] += 1

        # Track layer changes
        layer = get_layer(new_importance)
        if layer == "LTM":
            sweep_stats["promoted"] += 1
        elif layer == "STM":
            sweep_stats["demoted"] += 1

        sweep_stats["swept"] += 1

    conn.commit()
    return json.dumps(sweep_stats)


# ---------- Entry point ----------

def run():
    """Entry point for the MCP server."""
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run()
