"""
memory-v2 -- CLI helper
Lets you call memory tools without MCP integration.
Usage: memory-v2 <command> [json_args]
"""
import json
import sys
import os
import inspect
from pathlib import Path

from . import db, embeddings


_conn = None


def get_conn():
    global _conn
    if _conn is None:
        _conn = db.init_db()
    return _conn


def cmd_stats():
    conn = get_conn()
    return db.get_stats(conn)


def cmd_add(content, content_type="fact", tags=None, source=None,
            protected=False, author="claude-code", confidence=0.95):
    conn = get_conn()
    from .security import scan_for_credentials
    if scan_for_credentials(content):
        return {"error": "BLOCKED: credentials detected"}

    emb = embeddings.embed_text(content)

    existing = db.hybrid_search(conn, content[:100], emb, limit=3)
    novelty = 1.0
    if existing:
        from .scoring import cosine_similarity
        import numpy as np
        for mem in existing:
            row = conn.execute(
                "SELECT embedding FROM memory_vec WHERE id = ?", (mem["id"],)
            ).fetchone()
            if row:
                existing_emb = np.frombuffer(row[0], dtype=np.float32).tolist()
                sim = cosine_similarity(emb, existing_emb)
                novelty = min(novelty, 1.0 - sim)

    importance = 0.5
    if protected or content_type in ("decision", "correction", "identity"):
        importance = 0.9
        protected = True
    elif novelty > 0.7:
        importance = 0.7
    elif novelty < 0.3:
        importance = 0.3

    if tags and isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    memory_id = db.add_memory(
        conn, content=content, embedding=emb, content_type=content_type,
        source_file=source, author=author, confidence=confidence,
        protected=protected, tags=tags, importance_score=importance,
    )
    return {"memory_id": memory_id, "novelty": round(novelty, 3),
            "importance": round(importance, 3), "protected": protected}


def cmd_search(query, limit=10, content_type=None):
    conn = get_conn()
    query_emb = embeddings.embed_text(query)
    results = db.hybrid_search(conn, query, query_emb, limit=limit,
                               content_type=content_type)
    try:
        from .scoring import apply_actr_scoring
        results = apply_actr_scoring(conn, results, query)
    except ImportError:
        pass

    output = []
    for mem in results:
        output.append({
            "id": mem["id"], "content": mem["content"][:500],
            "content_type": mem["content_type"],
            "protected": bool(mem["protected"]),
            "hybrid_score": round(mem.get("hybrid_score", 0), 4),
            "created_at": mem["created_at"],
        })
    return {"query": query, "results": output, "count": len(output)}


def cmd_keyword(keywords, limit=10):
    conn = get_conn()
    results = db.keyword_search(conn, keywords, limit)
    output = []
    for mem in results:
        output.append({
            "id": mem["id"], "content": mem["content"][:500],
            "source_file": mem["source_file"],
        })
    return {"keywords": keywords, "results": output, "count": len(output)}


def cmd_get(memory_id):
    conn = get_conn()
    mem = db.get_memory(conn, int(memory_id))
    if mem is None:
        return {"error": f"Memory {memory_id} not found"}
    return mem


def cmd_update(memory_id, content=None, tags=None, protected=None):
    conn = get_conn()
    if protected is not None:
        protected = protected in (True, "true", "True", "1")
    success = db.update_memory(conn, int(memory_id), content, tags, protected)
    if not success:
        return {"error": f"Memory {memory_id} not found or no changes"}
    if content:
        emb = embeddings.embed_text(content)
        conn.execute("DELETE FROM memory_vec WHERE id = ?", (int(memory_id),))
        conn.execute(
            "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
            (int(memory_id), db._serialize_f32(emb))
        )
        conn.commit()
    return {"success": True, "memory_id": int(memory_id)}


def cmd_forget(memory_id, reason="manual"):
    conn = get_conn()
    success = db.archive_memory(conn, int(memory_id), reason)
    if not success:
        return {"error": f"Memory {memory_id} not found, already archived, or protected"}
    return {"success": True, "memory_id": int(memory_id), "archived": True}


def cmd_recent(hours=24, limit=20):
    conn = get_conn()
    results = db.list_recent(conn, int(hours), int(limit))
    output = []
    for mem in results:
        output.append({
            "id": mem["id"], "content": mem["content"][:200],
            "content_type": mem["content_type"],
            "created_at": mem["created_at"],
        })
    return {"results": output, "count": len(output)}


def cmd_reindex(force=False):
    conn = get_conn()
    from .vault_indexer import index_vault, index_changelog
    vault_stats = index_vault(conn, force=force)
    changelog_stats = index_changelog(conn)
    return {"vault": vault_stats, "changelog": changelog_stats}


COMMANDS = {
    "stats": cmd_stats,
    "add": cmd_add,
    "search": cmd_search,
    "keyword": cmd_keyword,
    "get": cmd_get,
    "update": cmd_update,
    "forget": cmd_forget,
    "recent": cmd_recent,
    "reindex": cmd_reindex,
}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: memory-v2 <command> [json_args]",
                          "commands": list(COMMANDS.keys())}))
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(json.dumps({"error": f"Unknown command: {cmd}",
                          "commands": list(COMMANDS.keys())}))
        sys.exit(1)

    args = {}
    if len(sys.argv) > 2:
        raw = " ".join(sys.argv[2:])
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            # Treat as positional first arg
            sig = inspect.signature(COMMANDS[cmd])
            first_param = list(sig.parameters.keys())[0]
            args = {first_param: raw}

    result = COMMANDS[cmd](**args)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
