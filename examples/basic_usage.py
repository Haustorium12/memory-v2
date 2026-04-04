"""
basic_usage.py -- Quick tour of the memory-v2 Python API.

This script initializes a temporary database, adds a memory with an
embedding, runs a hybrid search, and performs a FadeMem decay sweep.

Note: Requires Ollama running locally with nomic-embed-text pulled.
For tests, mock embeddings are used instead (see tests/).
"""

from memory_v2.db import init_db, add_memory, hybrid_search, get_memory
from memory_v2.embeddings import embed_text

# 1. Initialize a fresh database (in-memory for this demo)
conn = init_db(":memory:")

# 2. Embed and store a memory
text = "Sean prefers native Windows services over Docker containers."
embedding = embed_text(text)

memory_id = add_memory(
    conn,
    content=text,
    embedding=embedding,
    content_type="fact",
    author="demo-script",
    tags=["preferences", "infrastructure"],
    protected=True,
)
print(f"Stored memory #{memory_id}")

# 3. Search for related memories
query = "Docker vs native services"
results = hybrid_search(
    conn,
    query_text=query,
    query_embedding=embed_text(query),
    limit=5,
)

print(f"\nSearch results for '{query}':")
for r in results:
    print(f"  [{r['id']}] (score={r['final_score']:.3f}) {r['content'][:80]}")

# 4. Run a decay sweep (same logic the MCP server exposes)
from memory_v2.scoring import importance_score, should_archive
from datetime import datetime, timezone

now = datetime.now(timezone.utc)
rows = conn.execute(
    "SELECT * FROM memories WHERE archived = 0 AND protected = 0"
).fetchall()

archived = 0
for row in rows:
    mem = dict(row)
    last_access = datetime.fromisoformat(mem["last_accessed_at"])
    hours_since = (now - last_access).total_seconds() / 3600
    created = datetime.fromisoformat(mem["created_at"])
    age_days = (now - created).total_seconds() / 86400
    max_access = max(r["access_count"] for r in rows) or 1

    score = importance_score(
        access_count=mem["access_count"],
        max_access_count=max_access,
        hours_since_access=hours_since,
        age_days=age_days,
        is_protected=bool(mem["protected"]),
        confidence=mem["confidence"],
        decay_rate=mem["decay_rate"],
    )

    if should_archive(score, hours_since):
        archived += 1

print(f"\nDecay sweep: {len(rows)} non-protected memories checked, {archived} would be archived")

conn.close()
print("Done.")
