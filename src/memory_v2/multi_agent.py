"""
memory-v2 -- Multi-Agent Memory Sharing
Consumer offsets, authority chain, conflict detection/surfacing.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .vault_indexer import VAULT_PATH

# ---------- Authority Chain ----------
# Configurable hierarchy. Lower number = higher authority.
# Override with MEMORY_V2_AUTHORITY_CHAIN env var as JSON.

import os

_default_authority = {
    "human": 1,           # Manual edits by the user -- highest authority
    "primary_agent": 2,   # Primary AI agent (e.g., Claude Code)
    "coordinator": 3,     # Coordinating agent
    "worker": 4,          # Subordinate worker agent
    "indexer": 5,         # Automated vault indexer
    "extractor": 6,       # Automated fact extraction
}

_custom_authority = os.environ.get("MEMORY_V2_AUTHORITY_CHAIN", "")
if _custom_authority:
    try:
        AUTHORITY_LEVELS = json.loads(_custom_authority)
    except json.JSONDecodeError:
        AUTHORITY_LEVELS = _default_authority
else:
    AUTHORITY_LEVELS = _default_authority


def get_authority_level(author: str) -> int:
    """Get authority level for an author. Lower = higher authority."""
    return AUTHORITY_LEVELS.get(author.lower(), 99)


def can_overwrite(writer_author: str, existing_author: str) -> bool:
    """Check if writer has sufficient authority to overwrite existing memory."""
    writer_level = get_authority_level(writer_author)
    existing_level = get_authority_level(existing_author)
    return writer_level <= existing_level


# ---------- Consumer Offsets (Kafka-style) ----------

def get_offset(conn, agent_id: str) -> dict:
    """Get the last-read position for an agent."""
    row = conn.execute(
        "SELECT * FROM agent_offsets WHERE agent_id = ?",
        (agent_id,)
    ).fetchone()

    if row is None:
        return {"agent_id": agent_id, "last_read_line": 0, "last_read_time": None}
    return dict(row)


def set_offset(conn, agent_id: str, last_read_line: int):
    """Update the last-read position for an agent."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO agent_offsets
        (agent_id, last_read_line, last_read_time)
        VALUES (?, ?, ?)""",
        (agent_id, last_read_line, now)
    )
    conn.commit()


def get_unread_changelog(
    agent_id: str,
    conn=None,
    changelog_path: Path | None = None,
) -> dict:
    """
    Get changelog entries since this agent's last read.
    Returns new entries and the new offset.
    """
    if conn is None:
        conn = db.init_db()

    changelog_path = changelog_path or (VAULT_PATH / "changelog.md")
    if not changelog_path.exists():
        return {"entries": [], "new_offset": 0}

    lines = changelog_path.read_text(encoding="utf-8", errors="replace").splitlines()
    offset_info = get_offset(conn, agent_id)
    last_line = offset_info["last_read_line"]

    # Get new entries (lines after offset that start with [date])
    new_entries = []
    for i, line in enumerate(lines):
        if i <= last_line:
            continue
        if re.match(r'\[\d{4}-\d{2}-\d{2}\]', line.strip()):
            new_entries.append({"line": i, "content": line.strip()})

    return {
        "entries": new_entries,
        "new_offset": len(lines) - 1,
        "total_lines": len(lines),
        "unread_count": len(new_entries),
    }


def sync_agent(agent_id: str, conn=None) -> dict:
    """
    Sync an agent: read unread changelog, update offset.
    Returns unread entries.
    """
    if conn is None:
        conn = db.init_db()

    result = get_unread_changelog(agent_id, conn)
    if result["entries"]:
        set_offset(conn, agent_id, result["new_offset"])

    return result


# ---------- Conflict Detection ----------

def detect_conflict(
    conn,
    new_content: str,
    new_author: str,
    similar_memories: list[dict],
) -> dict | None:
    """
    Check if a new memory conflicts with existing ones.
    Returns conflict info if found, None otherwise.
    """
    for mem in similar_memories:
        # Same author updating their own memory = not a conflict
        if mem.get("author") == new_author:
            continue

        # Check for contradictory keywords
        contradiction_signals = [
            ("not", "is"), ("false", "true"), ("wrong", "correct"),
            ("never", "always"), ("removed", "added"), ("deleted", "created"),
        ]

        new_lower = new_content.lower()
        old_lower = mem["content"].lower()

        for neg, pos in contradiction_signals:
            if (neg in new_lower and pos in old_lower) or \
               (pos in new_lower and neg in old_lower):
                return {
                    "memory_id_a": mem["id"],
                    "agent_a": mem.get("author", "unknown"),
                    "agent_b": new_author,
                    "description": (
                        f"Potential contradiction detected.\n"
                        f"Existing (by {mem.get('author')}): {mem['content'][:200]}\n"
                        f"New (by {new_author}): {new_content[:200]}"
                    ),
                }

    return None


def log_conflict(conn, conflict: dict):
    """Log a detected conflict to the database."""
    conn.execute(
        """INSERT INTO conflicts
        (memory_id_a, agent_a, agent_b, description, created_at)
        VALUES (?, ?, ?, ?, ?)""",
        (
            conflict.get("memory_id_a"),
            conflict.get("agent_a"),
            conflict.get("agent_b"),
            conflict.get("description"),
            datetime.now(timezone.utc).isoformat(),
        )
    )
    conn.commit()


def get_unresolved_conflicts(conn) -> list[dict]:
    """Get all unresolved conflicts for briefing."""
    rows = conn.execute(
        "SELECT * FROM conflicts WHERE resolved = 0 ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_conflict(conn, conflict_id: int, resolved_by: str = "human"):
    """Mark a conflict as resolved."""
    conn.execute(
        "UPDATE conflicts SET resolved = 1, resolved_by = ? WHERE id = ?",
        (resolved_by, conflict_id)
    )
    conn.commit()


# ---------- Authority-Enforced Write ----------

def authority_write(
    conn,
    content: str,
    embedding: list[float],
    author: str,
    content_type: str = "fact",
    **kwargs,
) -> dict:
    """
    Write a memory with authority chain enforcement.
    Higher authority can overwrite lower. Lower can only flag.
    """
    # Check for similar existing memories
    similar = db.hybrid_search(conn, content[:100], embedding, limit=3)

    # Detect conflicts
    conflict = detect_conflict(conn, content, author, similar)
    if conflict:
        # Check authority
        existing_author = conflict.get("agent_a", "")
        if can_overwrite(author, existing_author):
            # Higher authority: overwrite
            target_id = conflict.get("memory_id_a")
            if target_id:
                db.update_memory(conn, target_id, content=content)
                # Update embedding
                conn.execute("DELETE FROM memory_vec WHERE id = ?", (target_id,))
                conn.execute(
                    "INSERT INTO memory_vec (id, embedding) VALUES (?, ?)",
                    (target_id, db._serialize_f32(embedding))
                )
                conn.commit()
                return {
                    "action": "overwritten",
                    "memory_id": target_id,
                    "conflict_logged": True,
                }

        # Lower authority: flag the conflict
        log_conflict(conn, conflict)
        # Still store the new memory but mark the conflict
        memory_id = db.add_memory(
            conn, content=content, embedding=embedding,
            content_type=content_type, author=author,
            authority_level=get_authority_level(author),
            **kwargs,
        )
        return {
            "action": "added_with_conflict",
            "memory_id": memory_id,
            "conflict_logged": True,
            "conflict": conflict["description"][:200],
        }

    # No conflict: normal add
    memory_id = db.add_memory(
        conn, content=content, embedding=embedding,
        content_type=content_type, author=author,
        authority_level=get_authority_level(author),
        **kwargs,
    )
    return {"action": "added", "memory_id": memory_id}
