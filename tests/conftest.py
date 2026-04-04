"""
Shared fixtures for memory-v2 tests.
All tests run WITHOUT Ollama -- embeddings are mocked with deterministic SHA-256 vectors.
"""

import hashlib
import sqlite3

import numpy as np
import pytest
import sqlite_vec

from memory_v2 import embeddings as embeddings_mod
from memory_v2.db import SCHEMA_SQL, init_vec_table


# ---------- Mock embedding ----------

def mock_embed(text: str) -> list[float]:
    """Deterministic 768-dim unit vector derived from SHA-256 of input text."""
    h = hashlib.sha256(text.encode()).digest()
    arr = np.frombuffer(h * 96, dtype=np.uint8)[:768].astype(np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def mock_embed_batch(texts: list[str]) -> list[list[float]]:
    return [mock_embed(t) for t in texts]


# ---------- Fixtures ----------

@pytest.fixture()
def mock_embedding():
    """Return the mock_embed callable for direct use in tests."""
    return mock_embed


@pytest.fixture()
def db_conn():
    """
    In-memory SQLite database with full memory-v2 schema (including sqlite-vec).
    Yields a connection; closes it after the test.
    """
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    init_vec_table(conn)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _patch_embeddings(monkeypatch):
    """Monkeypatch embed_text and embed_batch so Ollama is never called."""
    monkeypatch.setattr(embeddings_mod, "embed_text", mock_embed)
    monkeypatch.setattr(embeddings_mod, "embed_batch", mock_embed_batch)
