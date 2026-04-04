"""
memory-v2 -- Auto-Extraction Pipeline
Two-pass LLM extraction: extract facts, then decide memory actions.
Uses Ollama for local LLM inference (no API key needed).
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import ollama

from . import db, embeddings
from .security import scan_for_credentials
from .scoring import cosine_similarity

# LLM model for extraction (small, fast, local)
EXTRACTION_MODEL = os.environ.get("MEMORY_V2_LLM_MODEL", "qwen2.5:3b")

# ---------- Extraction Prompts ----------

EXTRACT_PROMPT = """You are a Personal Information Organizer. Extract discrete facts from this conversation text.

Focus on these 7 categories:
1. Personal preferences (likes, dislikes, working style)
2. Personal details (names, relationships, dates)
3. Plans and intentions (next steps, commitments)
4. Project details (status, architecture, decisions)
5. Technical specifics (file paths, ports, configs, errors)
6. Corrections (anything the user corrected)
7. Emotional/relational beats (breakthroughs, frustrations, energy)

Rules:
- Only extract from USER messages, not assistant messages
- Each fact should be a single, self-contained statement
- Include dates when mentioned
- Do NOT extract trivial greetings or filler
- Today's date: {date}

Return ONLY a JSON object: {{"facts": ["fact1", "fact2", ...]}}

Conversation text:
{text}"""

ACTION_PROMPT = """You are a Memory Manager. For each new fact, decide what action to take given existing memories.

New fact: {new_fact}

Existing similar memories:
{existing}

Decide ONE action:
- ADD: This is new information not captured in any existing memory
- UPDATE: This refines or updates an existing memory (specify which ID)
- DELETE: This contradicts an existing memory (specify which ID)
- NONE: This is already captured, no action needed

Return ONLY a JSON object: {{"action": "ADD|UPDATE|DELETE|NONE", "target_id": null_or_integer, "reason": "brief explanation"}}"""


# ---------- Emotional Detection ----------

EMOTION_WORDS = {
    "breakthrough", "frustrated", "excited", "quiet", "energy",
    "surprised", "amazing", "terrible", "love", "hate",
    "brilliant", "struggle", "finally", "eureka", "painful",
    "beautiful", "proud", "disappointed", "thrilled", "annoyed",
}

EMOTION_PATTERNS = [
    re.compile(r'!{2,}'),                     # Multiple exclamation marks
    re.compile(r'I Remember', re.IGNORECASE),  # Memory anchors
    re.compile(r'\b(wow|whoa|damn|holy)\b', re.IGNORECASE),
]


def detect_emotion(text: str) -> bool:
    """Detect if text contains emotional content."""
    text_lower = text.lower()
    if any(word in text_lower for word in EMOTION_WORDS):
        return True
    if any(p.search(text) for p in EMOTION_PATTERNS):
        return True
    return False


# ---------- Content Type Detection ----------

def detect_content_type(fact: str) -> str:
    """Detect the content type of an extracted fact."""
    fact_lower = fact.lower()

    if any(w in fact_lower for w in ["corrected", "correction", "actually", "not that", "wrong"]):
        return "correction"
    if any(w in fact_lower for w in ["decided", "decision", "chose", "architecture", "design"]):
        return "decision"
    if any(w in fact_lower for w in ["path", "port", "config", "file", "directory", "url"]):
        return "fact"
    if detect_emotion(fact):
        return "episode"

    return "fact"


# ---------- Pass 1: Fact Extraction ----------

def extract_facts(text: str, date: str | None = None) -> list[str]:
    """
    Extract discrete facts from conversation text using local LLM.
    Returns list of fact strings.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    prompt = EXTRACT_PROMPT.format(date=date, text=text[:8000])  # Limit context

    try:
        resp = ollama.chat(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 2000},
        )
        content = resp.message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in content:
            content = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            content = content.group(1) if content else "{}"

        data = json.loads(content)
        return data.get("facts", [])
    except (json.JSONDecodeError, Exception) as e:
        # Try to extract facts even from malformed response
        try:
            facts = re.findall(r'"([^"]{10,})"', content)
            return facts if facts else []
        except Exception:
            return []


# ---------- Pass 2: Memory Action Decision ----------

def decide_action(
    conn,
    new_fact: str,
    new_embedding: list[float],
) -> dict:
    """
    Decide what to do with a new fact: ADD, UPDATE, DELETE, or NONE.
    Checks against existing similar memories.
    """
    import numpy as np

    # Find similar existing memories
    similar = db.hybrid_search(conn, new_fact[:100], new_embedding, limit=5)

    if not similar:
        return {"action": "ADD", "target_id": None, "reason": "No similar memories found"}

    # Check novelty
    max_sim = 0.0
    most_similar_id = None
    for mem in similar:
        row = conn.execute(
            "SELECT embedding FROM memory_vec WHERE id = ?", (mem["id"],)
        ).fetchone()
        if row:
            existing_emb = np.frombuffer(row[0], dtype=np.float32).tolist()
            sim = cosine_similarity(new_embedding, existing_emb)
            if sim > max_sim:
                max_sim = sim
                most_similar_id = mem["id"]

    # High similarity = probably duplicate
    if max_sim > 0.92:
        return {
            "action": "NONE",
            "target_id": most_similar_id,
            "reason": f"Very similar to memory {most_similar_id} (sim={max_sim:.3f})",
            "novelty": round(1.0 - max_sim, 3),
        }

    # Moderate similarity = might be an update
    if max_sim > 0.75:
        # Use LLM to decide
        existing_text = "\n".join(
            f"ID {m['id']}: {m['content'][:200]}" for m in similar[:3]
        )
        prompt = ACTION_PROMPT.format(new_fact=new_fact, existing=existing_text)

        try:
            resp = ollama.chat(
                model=EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 500},
            )
            content = resp.message.content.strip()
            if "```" in content:
                content = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                content = content.group(1) if content else "{}"
            result = json.loads(content)
            result["novelty"] = round(1.0 - max_sim, 3)
            return result
        except Exception:
            pass

    # Low similarity = new information
    return {
        "action": "ADD",
        "target_id": None,
        "reason": "Sufficiently novel information",
        "novelty": round(1.0 - max_sim, 3),
    }


# ---------- Full Pipeline ----------

def process_conversation(
    conn,
    text: str,
    source: str | None = None,
    author: str = "claude-code",
    date: str | None = None,
) -> dict:
    """
    Full auto-extraction pipeline:
    1. Extract facts from conversation
    2. For each fact: credential scan, embed, decide action, store
    Returns processing stats.
    """
    stats = {
        "facts_extracted": 0,
        "added": 0,
        "updated": 0,
        "deleted": 0,
        "skipped": 0,
        "blocked": 0,
        "errors": [],
    }

    # Pass 1: Extract facts
    facts = extract_facts(text, date)
    stats["facts_extracted"] = len(facts)

    for fact in facts:
        # Credential scan
        creds = scan_for_credentials(fact)
        if creds:
            stats["blocked"] += 1
            continue

        # Detect content type and emotional content
        content_type = detect_content_type(fact)
        is_emotional = detect_emotion(fact)
        protected = content_type in ("correction", "decision") or is_emotional

        # Embed
        try:
            emb = embeddings.embed_text(fact)
        except Exception as e:
            stats["errors"].append(f"Embedding failed: {e}")
            continue

        # Pass 2: Decide action
        action = decide_action(conn, fact, emb)

        if action["action"] == "ADD":
            importance = 0.8 if protected else (0.7 if action.get("novelty", 1.0) > 0.5 else 0.4)
            db.add_memory(
                conn,
                content=fact,
                embedding=emb,
                content_type=content_type,
                source_file=source,
                author=author,
                protected=protected,
                importance_score=importance,
                tags=["auto-extracted", content_type],
            )
            stats["added"] += 1

        elif action["action"] == "UPDATE" and action.get("target_id"):
            target_id = action["target_id"]
            db.update_memory(conn, target_id, content=fact)
            # Update embedding
            conn.execute("DELETE FROM memory_vec WHERE id = ?", (target_id,))
            conn.execute(
                "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                (target_id, db._serialize_f32(emb))
            )
            conn.commit()
            stats["updated"] += 1

        elif action["action"] == "DELETE" and action.get("target_id"):
            db.archive_memory(conn, action["target_id"], reason=f"Contradicted by: {fact[:100]}")
            # Add the new fact
            db.add_memory(
                conn,
                content=fact,
                embedding=emb,
                content_type=content_type,
                source_file=source,
                author=author,
                protected=protected,
                tags=["auto-extracted", "replacement"],
            )
            stats["deleted"] += 1
            stats["added"] += 1

        else:
            stats["skipped"] += 1

    return stats
