"""
Tests for memory_v2.scoring -- ACT-R activation, FadeMem decay, cosine similarity.
"""

import math
from datetime import datetime, timezone, timedelta

import pytest

from memory_v2.scoring import (
    cosine_similarity,
    base_level_activation,
    spreading_activation,
    retrieval_probability,
    importance_score,
    should_archive,
    get_layer,
    RETRIEVAL_TAU,
)


# ---------- Cosine similarity ----------

def test_cosine_similarity_identical(mock_embedding):
    """Identical vectors should have cosine similarity ~1.0."""
    vec = mock_embedding("hello world")
    sim = cosine_similarity(vec, vec)
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have cosine similarity ~0.0."""
    # Build two vectors that are exactly orthogonal in 768 dims.
    a = [0.0] * 768
    b = [0.0] * 768
    a[0] = 1.0
    b[1] = 1.0
    sim = cosine_similarity(a, b)
    assert sim == pytest.approx(0.0, abs=1e-5)


# ---------- ACT-R base-level activation ----------

def test_base_level_activation():
    """Base-level activation should be a finite, reasonable number for known inputs."""
    now = datetime.now(timezone.utc)
    created = (now - timedelta(hours=10)).isoformat()
    last_acc = (now - timedelta(hours=1)).isoformat()

    b = base_level_activation(access_count=5, created_at=created, last_accessed_at=last_acc)
    # With n=5, lifetime=10h, d=0.5:  ln(5/0.5) - 0.5*ln(10) ≈ 2.303 - 1.151 ≈ 1.15
    assert math.isfinite(b)
    assert b > 0, "Activation should be positive for a recently accessed memory"


# ---------- Spreading activation ----------

def test_spreading_activation():
    """Overlapping tags should produce positive spreading activation."""
    memory_tags = ["python", "testing", "memory"]
    context_tags = ["python", "testing"]
    tag_fan = {"python": 5, "testing": 3, "memory": 2}

    s = spreading_activation(memory_tags, context_tags, tag_fan)
    assert s > 0.0, "Overlapping tags must give positive spreading activation"


def test_spreading_activation_no_overlap():
    """Non-overlapping tags should produce zero spreading activation."""
    memory_tags = ["python", "testing"]
    context_tags = ["javascript", "react"]
    tag_fan = {"python": 5, "testing": 3, "javascript": 4, "react": 2}

    s = spreading_activation(memory_tags, context_tags, tag_fan)
    assert s == 0.0


# ---------- Retrieval probability ----------

def test_retrieval_probability_high_activation():
    """Very high activation should give retrieval probability close to 1.0."""
    p = retrieval_probability(10.0)
    assert p > 0.99


def test_retrieval_probability_low_activation():
    """Very low activation should give retrieval probability close to 0.0."""
    p = retrieval_probability(-10.0)
    assert p < 0.01


# ---------- FadeMem importance ----------

def test_importance_score():
    """Importance should equal the weighted sum of relevance, frequency, recency."""
    # With access_count=10, max_access_count=10 => frequency = log(11)/log(11) = 1.0
    # With hours_since_access=0 => recency = exp(0) = 1.0
    # relevance = 1.0
    # I = 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
    score = importance_score(
        relevance=1.0,
        access_count=10,
        max_access_count=10,
        hours_since_access=0.0,
    )
    assert score == pytest.approx(1.0, abs=1e-5)


# ---------- should_archive ----------

def test_should_archive_old_low_importance():
    """Old memory with low importance and no protected tags should be archived."""
    result = should_archive(importance=0.05, age_days=60, tags=["fact"])
    assert result is True


def test_should_archive_protected_tag():
    """Memory with a protected tag should never be archived."""
    result = should_archive(importance=0.01, age_days=365, tags=["identity"])
    assert result is False


# ---------- get_layer ----------

def test_get_layer_ltm():
    """Importance >= 0.7 should map to LTM."""
    assert get_layer(0.7) == "LTM"
    assert get_layer(0.9) == "LTM"


def test_get_layer_stm():
    """Importance <= 0.3 should map to STM."""
    assert get_layer(0.3) == "STM"
    assert get_layer(0.1) == "STM"


def test_get_layer_hysteresis():
    """Importance strictly between 0.3 and 0.7 should stay 'current' (hysteresis zone)."""
    assert get_layer(0.5) == "current"
    assert get_layer(0.31) == "current"
    assert get_layer(0.69) == "current"
