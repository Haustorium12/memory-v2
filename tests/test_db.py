"""
Tests for memory_v2.db -- CRUD, hybrid search, keyword search, stats.
All embedding calls are mocked via conftest._patch_embeddings.
"""

import json

import pytest

from memory_v2.db import (
    add_memory,
    get_memory,
    update_memory,
    archive_memory,
    hybrid_search,
    keyword_search,
    get_stats,
    init_vec_table,
    SCHEMA_SQL,
)


# ---------- Schema ----------

def test_init_db(db_conn):
    """Schema creation should succeed and core tables should exist."""
    tables = [
        row[0]
        for row in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "memories" in tables
    assert "graveyard" in tables
    assert "memory_fts" in tables
    assert "memory_vec" in tables


# ---------- CRUD ----------

def test_add_and_get_memory(db_conn, mock_embedding):
    """Round-trip: add a memory then retrieve it by ID."""
    emb = mock_embedding("test memory content")
    mid = add_memory(
        db_conn,
        content="test memory content",
        embedding=emb,
        content_type="fact",
        tags=["test", "unit"],
    )
    assert mid is not None and mid > 0

    mem = get_memory(db_conn, mid)
    assert mem is not None
    assert mem["content"] == "test memory content"
    assert mem["content_type"] == "fact"
    assert json.loads(mem["tags"]) == ["test", "unit"]


def test_update_memory(db_conn, mock_embedding):
    """Updating content and tags should persist."""
    emb = mock_embedding("original content")
    mid = add_memory(db_conn, content="original content", embedding=emb)

    update_memory(db_conn, mid, content="updated content", tags=["changed"])

    mem = get_memory(db_conn, mid)
    assert mem["content"] == "updated content"
    assert json.loads(mem["tags"]) == ["changed"]


def test_archive_memory(db_conn, mock_embedding):
    """Archiving a non-protected memory should move it to the graveyard."""
    emb = mock_embedding("ephemeral fact")
    mid = add_memory(db_conn, content="ephemeral fact", embedding=emb, protected=False)

    result = archive_memory(db_conn, mid, reason="test")
    assert result is True

    # Should no longer be retrievable
    mem = get_memory(db_conn, mid)
    assert mem is None

    # Should exist in graveyard
    grave = db_conn.execute(
        "SELECT * FROM graveyard WHERE memory_id = ?", (mid,)
    ).fetchone()
    assert grave is not None
    assert grave["reason"] == "test"


def test_archive_protected_memory(db_conn, mock_embedding):
    """Archiving a protected memory should fail (return False)."""
    emb = mock_embedding("protected fact")
    mid = add_memory(db_conn, content="protected fact", embedding=emb, protected=True)

    result = archive_memory(db_conn, mid, reason="test")
    assert result is False

    # Memory should still be accessible
    mem = get_memory(db_conn, mid)
    assert mem is not None


# ---------- Search ----------

def test_hybrid_search(db_conn, mock_embedding):
    """Hybrid search should return ranked results from 3 distinct memories."""
    contents = [
        "Python is a great programming language",
        "JavaScript runs in the browser",
        "Python testing with pytest is powerful",
    ]
    for c in contents:
        add_memory(db_conn, content=c, embedding=mock_embedding(c), tags=["code"])

    results = hybrid_search(
        db_conn,
        query_text="Python programming",
        query_embedding=mock_embedding("Python programming"),
        limit=10,
    )
    assert len(results) >= 1
    # The top result should mention Python
    assert "Python" in results[0]["content"]


def test_keyword_search(db_conn, mock_embedding):
    """Keyword search should find memories matching FTS5 query."""
    add_memory(db_conn, content="SQLite is an embedded database", embedding=mock_embedding("SQLite is an embedded database"), tags=["db"])
    add_memory(db_conn, content="Redis is an in-memory store", embedding=mock_embedding("Redis is an in-memory store"), tags=["db"])

    results = keyword_search(db_conn, keywords="SQLite embedded", limit=5)
    assert len(results) >= 1
    assert "SQLite" in results[0]["content"]


# ---------- Stats ----------

def test_get_stats(db_conn, mock_embedding):
    """Stats should reflect the current database state."""
    add_memory(db_conn, content="fact one", embedding=mock_embedding("fact one"), content_type="fact", protected=True)
    add_memory(db_conn, content="episode two", embedding=mock_embedding("episode two"), content_type="episode")

    stats = get_stats(db_conn)
    assert stats["total_active"] == 2
    assert stats["protected"] == 1
    assert stats["archived"] == 0
    assert stats["type_counts"]["fact"] == 1
    assert stats["type_counts"]["episode"] == 1
