"""
memory-v2: Brain-inspired persistent memory for AI coding assistants.

Hybrid BM25 + vector search, ACT-R activation scoring, FadeMem decay,
knowledge graph traversal, and multi-agent coordination.
"""

__version__ = "1.0.0"
__author__ = "Sean Pembroke"

_embedder = None


def get_embedder():
    """Lazy-init and cache the embedding function. Avoids cold-start penalty."""
    global _embedder
    if _embedder is None:
        from .embeddings import embed_text
        # Warm up the model with a throwaway embedding
        embed_text("warmup")
        _embedder = embed_text
    return _embedder
