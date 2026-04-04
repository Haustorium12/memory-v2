"""
memory-v2 -- Embedding Layer
Uses Ollama nomic-embed-text for local vector generation.
No API key needed. Runs entirely on local hardware.
"""

import os
import hashlib
import json
from pathlib import Path

import ollama

MODEL = os.environ.get("MEMORY_V2_EMBED_MODEL", "nomic-embed-text")
DIM = 768  # nomic-embed-text dimension
CACHE_DIR = Path(os.environ.get(
    "MEMORY_V2_CACHE",
    str(Path.home() / ".memory-v2" / "cache")
))


def embed_text(text: str) -> list[float]:
    """Embed a single text string using Ollama. Returns 768-dim float vector."""
    resp = ollama.embed(model=MODEL, input=text)
    return resp.embeddings[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Ollama."""
    resp = ollama.embed(model=MODEL, input=texts)
    return resp.embeddings


def embed_with_cache(text: str) -> list[float]:
    """Embed with local file cache to avoid re-embedding identical content."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{text_hash}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    embedding = embed_text(text)
    cache_file.write_text(json.dumps(embedding))
    return embedding


def get_client():
    """Compatibility shim -- not needed for Ollama but keeps API consistent."""
    return None
