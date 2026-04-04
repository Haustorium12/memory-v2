"""
memory-v2 -- Scoring Module
ACT-R activation scoring + FadeMem decay + cosine similarity.
"""

import math
import json
import random
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ---------- Cosine Similarity ----------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ---------- ACT-R Activation ----------

# Constants from Anderson & Schooler 1991 / ACT-R 7.0
DECAY_D = 0.5          # Power-law decay exponent
S_MAX = 1.6            # Maximum associative strength
NOISE_S = 0.25         # Logistic noise scale
RETRIEVAL_TAU = -0.5   # Retrieval threshold


def base_level_activation(
    access_count: int,
    created_at: str,
    last_accessed_at: str,
) -> float:
    """
    Compute ACT-R base-level activation using optimized approximation.
    B_i = ln(n / (1 - d)) - d * ln(L)

    Where n = access count, L = lifetime in hours, d = decay parameter.
    """
    now = datetime.now(timezone.utc)

    try:
        created = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        created = now

    lifetime_hours = max((now - created).total_seconds() / 3600, 0.01)

    if access_count <= 0:
        access_count = 1

    # Optimized approximation (avoids storing full access history)
    b = math.log(access_count / (1.0 - DECAY_D)) - DECAY_D * math.log(lifetime_hours)
    return b


def spreading_activation(
    memory_tags: list[str],
    context_tags: list[str],
    tag_fan: dict[str, int],
) -> float:
    """
    Compute spreading activation from current context.
    S_i = sum(W_j * (S_max - ln(fan_j)))

    tag_fan: dict mapping each tag to how many memories it appears in.
    """
    if not context_tags or not memory_tags:
        return 0.0

    total_attention = 1.0
    w_per_source = total_attention / len(context_tags)

    s_total = 0.0
    for tag in context_tags:
        if tag in memory_tags:
            fan = tag_fan.get(tag, 1)
            strength = S_MAX - math.log(max(fan, 1))
            s_total += w_per_source * max(strength, 0)

    return s_total


def activation_noise() -> float:
    """Generate ACT-R logistic noise."""
    # Logistic distribution with location=0, scale=NOISE_S
    u = random.random()
    u = max(u, 1e-10)
    u = min(u, 1.0 - 1e-10)
    return NOISE_S * math.log(u / (1.0 - u))


def full_activation(
    access_count: int,
    created_at: str,
    last_accessed_at: str,
    memory_tags: list[str],
    context_tags: list[str],
    tag_fan: dict[str, int],
    protected: bool = False,
) -> float:
    """
    Compute full ACT-R activation: B_i + S_i + noise
    Protected memories get an activation floor.
    """
    b = base_level_activation(access_count, created_at, last_accessed_at)
    s = spreading_activation(memory_tags, context_tags, tag_fan)
    noise = activation_noise()

    activation = b + s + noise

    # Protected memories never fall below threshold
    if protected:
        activation = max(activation, RETRIEVAL_TAU + 1.0)

    return activation


def retrieval_probability(activation: float) -> float:
    """
    Probability of successful retrieval given activation.
    P(retrieve) = 1 / (1 + exp(-(A - tau) / s))
    """
    try:
        return 1.0 / (1.0 + math.exp(-(activation - RETRIEVAL_TAU) / NOISE_S))
    except OverflowError:
        return 0.0 if activation < RETRIEVAL_TAU else 1.0


# ---------- FadeMem Decay ----------

PROTECTED_TAGS = {
    "correction", "decision", "identity", "emotional_anchor",
    "commitment", "exact_value", "chain_of_command", "person",
}

# Dual-layer thresholds
PROMOTE_THRESHOLD = 0.7    # STM -> LTM
DEMOTE_THRESHOLD = 0.3     # LTM -> STM
ARCHIVE_THRESHOLD = 0.1    # STM -> graveyard (if age > 30 days)

LTM_BETA = 0.8   # Slower decay
STM_BETA = 1.2   # Faster decay


def importance_score(
    relevance: float,
    access_count: int,
    max_access_count: int,
    hours_since_access: float,
    decay_rate: float = 0.1,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3,
) -> float:
    """
    FadeMem importance scoring.
    I(t) = alpha * relevance + beta * frequency + gamma * recency
    """
    frequency = math.log(access_count + 1) / math.log(max(max_access_count, 2) + 1)
    recency = math.exp(-decay_rate * hours_since_access)
    return alpha * relevance + beta * frequency + gamma * recency


def decay_value(
    initial_strength: float,
    hours_since_access: float,
    decay_rate: float = 0.1,
    beta: float = 0.8,
) -> float:
    """
    FadeMem decay function.
    v(t) = v(0) * exp(-lambda * t^beta)
    """
    try:
        return initial_strength * math.exp(-decay_rate * (hours_since_access ** beta))
    except OverflowError:
        return 0.0


def should_archive(
    importance: float,
    age_days: float,
    tags: list[str],
) -> bool:
    """Check if a memory should be archived."""
    # Protected tags are immune
    if any(t in PROTECTED_TAGS for t in tags):
        return False
    return importance < ARCHIVE_THRESHOLD and age_days > 30


def get_layer(importance: float) -> str:
    """Determine which memory layer a memory belongs to."""
    if importance >= PROMOTE_THRESHOLD:
        return "LTM"
    elif importance <= DEMOTE_THRESHOLD:
        return "STM"
    return "current"  # In hysteresis zone, don't move


# ---------- Apply scoring to search results ----------

def apply_actr_scoring(
    conn,
    results: list[dict],
    query: str,
    context_tags: list[str] | None = None,
) -> list[dict]:
    """
    Apply ACT-R activation scoring to search results and re-rank.
    """
    if not results:
        return results

    # Build tag fan counts
    tag_fan = {}
    rows = conn.execute(
        "SELECT tags FROM memories WHERE archived = 0"
    ).fetchall()
    for row in rows:
        try:
            tags = json.loads(row["tags"]) if row["tags"] else []
        except (json.JSONDecodeError, TypeError):
            tags = []
        for t in tags:
            tag_fan[t] = tag_fan.get(t, 0) + 1

    # Extract context tags from query
    if context_tags is None:
        context_tags = [w.lower() for w in query.split() if len(w) > 3]

    for mem in results:
        try:
            mem_tags = json.loads(mem.get("tags", "[]"))
        except (json.JSONDecodeError, TypeError):
            mem_tags = []

        activation = full_activation(
            access_count=mem.get("access_count", 1),
            created_at=mem.get("created_at", ""),
            last_accessed_at=mem.get("last_accessed_at", ""),
            memory_tags=[t.lower() for t in mem_tags],
            context_tags=context_tags,
            tag_fan=tag_fan,
            protected=bool(mem.get("protected", False)),
        )

        mem["activation_score"] = round(activation, 4)
        mem["retrieval_prob"] = round(retrieval_probability(activation), 4)

        # Combined score: hybrid search score + activation
        hybrid = mem.get("hybrid_score", 0)
        mem["final_score"] = round(0.6 * hybrid + 0.4 * (activation / 10.0), 4)

    # Re-sort by final score
    results.sort(key=lambda m: m.get("final_score", 0), reverse=True)
    return results
