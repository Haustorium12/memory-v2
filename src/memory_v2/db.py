"""
memory-v2 -- Database Layer (sqlite-vec + FTS5)
Hybrid BM25 + vector search in a single SQLite database.
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec
import numpy as np

# ---------- constants ----------

DB_PATH = os.environ.get(
    "MEMORY_V2_DB",
    str(Path.home() / ".memory-v2" / "memory.db")
)

EMBEDDING_DIM = 768  # nomic-embed-text (Ollama local)


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to raw bytes for sqlite-vec."""
    return np.array(vec, dtype=np.float32).tobytes()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- connection factory ----------

def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with sqlite-vec loaded."""
    db_path = db_path or DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------- schema ----------

SCHEMA_SQL = """
-- Core memories table
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_type TEXT DEFAULT 'fact',
    source_file TEXT,
    source_line INTEGER,
    author TEXT DEFAULT 'claude-code',
    authority_level INTEGER DEFAULT 2,
    confidence REAL DEFAULT 0.95,
    protected INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 1,
    activation_score REAL DEFAULT 0.0,
    importance_score REAL DEFAULT 0.5,
    decay_rate REAL DEFAULT 0.1,
    archived INTEGER DEFAULT 0,
    supersedes INTEGER,
    content_hash TEXT
);

-- FTS5 full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    tags,
    source_file,
    content='memories',
    content_rowid='id'
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content, tags, source_file)
    VALUES (new.id, new.content, new.tags, new.source_file);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags, source_file)
    VALUES ('delete', old.id, old.content, old.tags, old.source_file);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags, source_file)
    VALUES ('delete', old.id, old.content, old.tags, old.source_file);
    INSERT INTO memory_fts(rowid, content, tags, source_file)
    VALUES (new.id, new.content, new.tags, new.source_file);
END;

-- Vector search table (created separately via sqlite-vec API)
-- See init_vec_table()

-- Graveyard for archived memories
CREATE TABLE IF NOT EXISTS graveyard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    archived_at TEXT NOT NULL,
    reason TEXT,
    last_activation_score REAL
);

-- Consumer offsets for multi-agent sync
CREATE TABLE IF NOT EXISTS agent_offsets (
    agent_id TEXT PRIMARY KEY,
    last_read_line INTEGER DEFAULT 0,
    last_read_time TEXT
);

-- File hash registry (for incremental indexing)
CREATE TABLE IF NOT EXISTS file_hashes (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0
);

-- Conflict log
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id_a INTEGER,
    memory_id_b INTEGER,
    agent_a TEXT,
    agent_b TEXT,
    description TEXT,
    resolved INTEGER DEFAULT 0,
    resolved_by TEXT,
    created_at TEXT NOT NULL
);

-- Compaction receipts
CREATE TABLE IF NOT EXISTS compaction_receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    original_tokens INTEGER,
    compressed_tokens INTEGER,
    ratio REAL,
    protected_items_extracted INTEGER,
    verification_score REAL,
    vault_files_updated TEXT,
    receipt_data TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(content_type);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_protected ON memories(protected);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_file);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_activation ON memories(activation_score);
"""


def init_vec_table(conn: sqlite3.Connection):
    """Create the vec0 virtual table for vector search."""
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)
    conn.commit()


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialize the database with full schema."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    init_vec_table(conn)
    conn.commit()
    return conn


# ---------- CRUD operations ----------

def add_memory(
    conn: sqlite3.Connection,
    content: str,
    embedding: list[float],
    content_type: str = "fact",
    source_file: str | None = None,
    source_line: int | None = None,
    author: str = "claude-code",
    authority_level: int = 2,
    confidence: float = 0.95,
    protected: bool = False,
    tags: list[str] | None = None,
    importance_score: float = 0.5,
    decay_rate: float = 0.1,
    supersedes: int | None = None,
) -> int:
    """Add a new memory with embedding. Returns the memory ID."""
    now = _now_iso()
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    tags_json = json.dumps(tags or [])

    cursor = conn.execute(
        """INSERT INTO memories
        (content, content_type, source_file, source_line, author,
         authority_level, confidence, protected, tags,
         created_at, updated_at, last_accessed_at,
         importance_score, decay_rate, supersedes, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, content_type, source_file, source_line, author,
         authority_level, confidence, 1 if protected else 0, tags_json,
         now, now, now,
         importance_score, decay_rate, supersedes, content_hash)
    )
    memory_id = cursor.lastrowid

    # Insert embedding into vec0
    conn.execute(
        "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
        (memory_id, _serialize_f32(embedding))
    )
    conn.commit()
    return memory_id


def get_memory(conn: sqlite3.Connection, memory_id: int) -> dict | None:
    """Retrieve a single memory by ID. Updates access timestamp."""
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND archived = 0",
        (memory_id,)
    ).fetchone()
    if row is None:
        return None

    # Update access stats
    now = _now_iso()
    conn.execute(
        """UPDATE memories
        SET last_accessed_at = ?, access_count = access_count + 1
        WHERE id = ?""",
        (now, memory_id)
    )
    conn.commit()
    return dict(row)


def update_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    content: str | None = None,
    tags: list[str] | None = None,
    protected: bool | None = None,
    importance_score: float | None = None,
) -> bool:
    """Update an existing memory."""
    updates = []
    params = []

    if content is not None:
        updates.append("content = ?")
        params.append(content)
        updates.append("content_hash = ?")
        params.append(hashlib.sha256(content.encode()).hexdigest()[:16])
    if tags is not None:
        updates.append("tags = ?")
        params.append(json.dumps(tags))
    if protected is not None:
        updates.append("protected = ?")
        params.append(1 if protected else 0)
    if importance_score is not None:
        updates.append("importance_score = ?")
        params.append(importance_score)

    if not updates:
        return False

    updates.append("updated_at = ?")
    params.append(_now_iso())
    params.append(memory_id)

    conn.execute(
        f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
        params
    )
    conn.commit()
    return True


def archive_memory(
    conn: sqlite3.Connection,
    memory_id: int,
    reason: str = "decay"
) -> bool:
    """Archive a memory to the graveyard."""
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND archived = 0",
        (memory_id,)
    ).fetchone()
    if row is None:
        return False

    mem = dict(row)
    if mem["protected"]:
        return False  # Never archive protected memories

    # Write to graveyard
    conn.execute(
        """INSERT INTO graveyard
        (memory_id, content, metadata, archived_at, reason, last_activation_score)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (memory_id, mem["content"], json.dumps(mem), _now_iso(),
         reason, mem["activation_score"])
    )

    # Mark as archived
    conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,))

    # Remove from vec index
    conn.execute("DELETE FROM memory_vec WHERE id = ?", (memory_id,))
    conn.commit()
    return True


# ---------- Search ----------

def hybrid_search(
    conn: sqlite3.Connection,
    query_text: str,
    query_embedding: list[float],
    limit: int = 10,
    content_type: str | None = None,
    include_archived: bool = False,
) -> list[dict]:
    """
    Hybrid search: BM25 (FTS5) + vector similarity (vec0).
    Uses Reciprocal Rank Fusion to combine results.
    """
    k = 60  # RRF constant

    # --- BM25 keyword search ---
    fts_results = {}
    try:
        rows = conn.execute(
            """SELECT rowid, rank FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY rank LIMIT ?""",
            (query_text, limit * 3)
        ).fetchall()
        for rank_pos, row in enumerate(rows):
            fts_results[row["rowid"]] = 1.0 / (k + rank_pos + 1)
    except Exception:
        pass  # FTS match syntax errors are non-fatal

    # --- Vector similarity search ---
    vec_results = {}
    rows = conn.execute(
        """SELECT id, distance FROM memory_vec
        WHERE embedding MATCH ?
        ORDER BY distance LIMIT ?""",
        (_serialize_f32(query_embedding), limit * 3)
    ).fetchall()
    for rank_pos, row in enumerate(rows):
        vec_results[row["id"]] = 1.0 / (k + rank_pos + 1)

    # --- Reciprocal Rank Fusion ---
    all_ids = set(fts_results.keys()) | set(vec_results.keys())
    fused = []
    for mid in all_ids:
        score = fts_results.get(mid, 0) + vec_results.get(mid, 0)
        fused.append((mid, score))

    fused.sort(key=lambda x: x[1], reverse=True)

    # Fetch full records
    results = []
    for mid, score in fused[:limit * 2]:
        archived_filter = "" if include_archived else "AND archived = 0"
        type_filter = f"AND content_type = '{content_type}'" if content_type else ""
        row = conn.execute(
            f"SELECT * FROM memories WHERE id = ? {archived_filter} {type_filter}",
            (mid,)
        ).fetchone()
        if row:
            mem = dict(row)
            mem["hybrid_score"] = score
            mem["bm25_score"] = fts_results.get(mid, 0)
            mem["vec_score"] = vec_results.get(mid, 0)
            results.append(mem)

        if len(results) >= limit:
            break

    # Update access timestamps for returned results
    now = _now_iso()
    for mem in results:
        conn.execute(
            """UPDATE memories
            SET last_accessed_at = ?, access_count = access_count + 1
            WHERE id = ?""",
            (now, mem["id"])
        )
    conn.commit()

    return results


def keyword_search(
    conn: sqlite3.Connection,
    keywords: str,
    limit: int = 10,
) -> list[dict]:
    """Pure BM25 keyword search via FTS5."""
    try:
        rows = conn.execute(
            """SELECT rowid, rank FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY rank LIMIT ?""",
            (keywords, limit)
        ).fetchall()
    except Exception:
        return []

    results = []
    now = _now_iso()
    for row in rows:
        mem_row = conn.execute(
            "SELECT * FROM memories WHERE id = ? AND archived = 0",
            (row["rowid"],)
        ).fetchone()
        if mem_row:
            mem = dict(mem_row)
            mem["bm25_rank"] = row["rank"]
            results.append(mem)
            conn.execute(
                """UPDATE memories
                SET last_accessed_at = ?, access_count = access_count + 1
                WHERE id = ?""",
                (now, mem["id"])
            )
    conn.commit()
    return results


def list_recent(
    conn: sqlite3.Connection,
    hours: int = 24,
    limit: int = 20,
) -> list[dict]:
    """Get recently created or accessed memories."""
    rows = conn.execute(
        """SELECT * FROM memories
        WHERE archived = 0
        ORDER BY created_at DESC
        LIMIT ?""",
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get database statistics."""
    total = conn.execute("SELECT COUNT(*) FROM memories WHERE archived = 0").fetchone()[0]
    archived = conn.execute("SELECT COUNT(*) FROM memories WHERE archived = 1").fetchone()[0]
    protected = conn.execute("SELECT COUNT(*) FROM memories WHERE protected = 1 AND archived = 0").fetchone()[0]
    graveyard = conn.execute("SELECT COUNT(*) FROM graveyard").fetchone()[0]
    files_indexed = conn.execute("SELECT COUNT(*) FROM file_hashes").fetchone()[0]

    type_counts = {}
    for row in conn.execute(
        "SELECT content_type, COUNT(*) as cnt FROM memories WHERE archived = 0 GROUP BY content_type"
    ).fetchall():
        type_counts[row["content_type"]] = row["cnt"]

    return {
        "total_active": total,
        "archived": archived,
        "protected": protected,
        "graveyard": graveyard,
        "files_indexed": files_indexed,
        "type_counts": type_counts,
    }
