"""
memory-v2 -- Vault Indexer
Reads all markdown files from a vault, chunks them, embeds them,
and stores them in sqlite-vec + FTS5.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

from . import db, embeddings

VAULT_PATH = Path(os.environ.get(
    "MEMORY_V2_VAULT",
    str(Path.home() / ".memory-v2" / "vault")
))

# Chunk settings
CHUNK_SIZE = 500       # target tokens per chunk (approx 4 chars/token)
CHUNK_OVERLAP = 50     # overlap tokens between chunks
CHARS_PER_TOKEN = 4    # rough approximation


def _safe_parse_tags(tags):
    """Parse tags from frontmatter, handling malformed values."""
    if tags is None:
        return []
    if isinstance(tags, list):
        return tags
    if isinstance(tags, str):
        try:
            parsed = json.loads(tags)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (json.JSONDecodeError, ValueError):
            return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def _file_hash(path: Path) -> str:
    """SHA-256 hash of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def _extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter if present, return (metadata, body)."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            front = text[3:end].strip()
            body = text[end + 3:].strip()
            meta = {}
            for line in front.split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()
            return meta, body
    return {}, text


def _detect_content_type(path: Path) -> str:
    """Detect memory type from vault folder structure."""
    parts = path.parts
    for part in parts:
        pl = part.lower()
        if pl == "conversations":
            return "episode"
        elif pl == "decisions":
            return "decision"
        elif pl in ("projects", "intelligence", "intel"):
            return "fact"
        elif pl == "people":
            return "person"
        elif pl == "briefings":
            return "fact"
        elif pl == "origins":
            return "identity"
    return "fact"


def _chunk_text(text: str, source_file: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    # Split on paragraph boundaries first
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_len = 0
    chunk_start_line = 1
    line_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para) // CHARS_PER_TOKEN

        # If a single paragraph exceeds chunk size, split by sentences
        if para_len > CHUNK_SIZE:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_len = len(sent) // CHARS_PER_TOKEN
                if current_len + sent_len > CHUNK_SIZE and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "source_file": source_file,
                        "source_line": chunk_start_line,
                        "char_count": len(chunk_text),
                    })
                    # Keep overlap
                    overlap_text = chunk_text[-(CHUNK_OVERLAP * CHARS_PER_TOKEN):]
                    current_chunk = [overlap_text] if overlap_text else []
                    current_len = len(overlap_text) // CHARS_PER_TOKEN
                    chunk_start_line = line_count
                current_chunk.append(sent)
                current_len += sent_len
        else:
            if current_len + para_len > CHUNK_SIZE and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "source_file": source_file,
                    "source_line": chunk_start_line,
                    "char_count": len(chunk_text),
                })
                overlap_text = chunk_text[-(CHUNK_OVERLAP * CHARS_PER_TOKEN):]
                current_chunk = [overlap_text] if overlap_text else []
                current_len = len(overlap_text) // CHARS_PER_TOKEN
                chunk_start_line = line_count
            current_chunk.append(para)
            current_len += para_len

        line_count += para.count("\n") + 2

    # Final chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "source_file": source_file,
            "source_line": chunk_start_line,
            "char_count": len(chunk_text),
        })

    return chunks


def _get_vault_files(vault_path: Path) -> list[Path]:
    """Get all markdown files from vault, excluding changelog (too large)."""
    files = []
    for f in vault_path.rglob("*.md"):
        # Skip very large files that should be queried differently
        name = f.name.lower()
        if name == "changelog.md":
            continue  # 1570+ entries, index separately
        if f.stat().st_size > 0:
            files.append(f)
    return sorted(files)


def index_vault(
    conn=None,
    vault_path: Path | None = None,
    force: bool = False,
    progress_callback=None,
) -> dict:
    """
    Index all vault markdown files into sqlite-vec + FTS5.
    Skips files whose hash hasn't changed (incremental).

    Returns stats dict.
    """
    vault_path = vault_path or VAULT_PATH
    if conn is None:
        conn = db.init_db()

    files = _get_vault_files(vault_path)

    stats = {
        "total_files": len(files),
        "files_indexed": 0,
        "files_skipped": 0,
        "chunks_created": 0,
        "errors": [],
    }

    for i, fpath in enumerate(files):
        rel_path = str(fpath.relative_to(vault_path)).replace("\\", "/")

        if progress_callback:
            progress_callback(i + 1, len(files), rel_path)

        # Check if file has changed
        current_hash = _file_hash(fpath)
        existing = conn.execute(
            "SELECT content_hash FROM file_hashes WHERE file_path = ?",
            (rel_path,)
        ).fetchone()

        if existing and existing["content_hash"] == current_hash and not force:
            stats["files_skipped"] += 1
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            stats["errors"].append(f"{rel_path}: {e}")
            continue

        meta, body = _extract_frontmatter(text)
        content_type = meta.get("type", _detect_content_type(fpath))

        # Remove old chunks for this file
        old_ids = conn.execute(
            "SELECT id FROM memories WHERE source_file = ?",
            (rel_path,)
        ).fetchall()
        for row in old_ids:
            conn.execute("DELETE FROM memory_vec WHERE id = ?", (row["id"],))
        conn.execute("DELETE FROM memories WHERE source_file = ?", (rel_path,))

        # Chunk the document
        chunks = _chunk_text(body, rel_path)
        if not chunks:
            stats["files_skipped"] += 1
            continue

        # Embed all chunks in batch
        chunk_texts = [c["text"] for c in chunks]
        try:
            chunk_embeddings = embeddings.embed_batch(chunk_texts)
        except Exception as e:
            stats["errors"].append(f"{rel_path}: embedding error: {e}")
            continue

        # Store chunks
        tags = meta.get("tags", "[]")
        if isinstance(tags, str) and not tags.startswith("["):
            tags = json.dumps([t.strip() for t in tags.split(",")])

        protected = content_type in ("decision", "identity", "correction")
        author = meta.get("author", "vault-indexer")

        for chunk, emb in zip(chunks, chunk_embeddings):
            db.add_memory(
                conn,
                content=chunk["text"],
                embedding=emb,
                content_type=content_type,
                source_file=rel_path,
                source_line=chunk["source_line"],
                author=author,
                protected=protected,
                tags=_safe_parse_tags(tags),
                importance_score=0.7 if protected else 0.5,
            )
            stats["chunks_created"] += 1

        # Update file hash
        conn.execute(
            """INSERT OR REPLACE INTO file_hashes
            (file_path, content_hash, indexed_at, chunk_count)
            VALUES (?, ?, ?, ?)""",
            (rel_path, current_hash, db._now_iso(), len(chunks))
        )
        conn.commit()
        stats["files_indexed"] += 1

    return stats


def index_changelog(
    conn=None,
    vault_path: Path | None = None,
) -> dict:
    """Index changelog.md separately -- each entry becomes a memory."""
    vault_path = vault_path or VAULT_PATH
    changelog = vault_path / "changelog.md"
    if not changelog.exists():
        return {"error": "changelog.md not found"}

    if conn is None:
        conn = db.init_db()

    text = changelog.read_text(encoding="utf-8", errors="replace")

    # Parse entries: each starts with [date]
    entries = re.findall(
        r'(\[\d{4}-\d{2}-\d{2}\]\s+.+?)(?=\n\[\d{4}-\d{2}-\d{2}\]|\Z)',
        text, re.DOTALL
    )

    # Check if already indexed
    existing = conn.execute(
        "SELECT content_hash FROM file_hashes WHERE file_path = 'changelog.md'"
    ).fetchone()
    current_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
    if existing and existing["content_hash"] == current_hash:
        return {"skipped": True, "entries": len(entries)}

    # Remove old changelog entries
    old_ids = conn.execute(
        "SELECT id FROM memories WHERE source_file = 'changelog.md'"
    ).fetchall()
    for row in old_ids:
        conn.execute("DELETE FROM memory_vec WHERE id = ?", (row["id"],))
    conn.execute("DELETE FROM memories WHERE source_file = 'changelog.md'")

    # Embed and store each entry
    if entries:
        entry_embeddings = embeddings.embed_batch(entries)
        for entry, emb in zip(entries, entry_embeddings):
            db.add_memory(
                conn,
                content=entry.strip(),
                embedding=emb,
                content_type="episode",
                source_file="changelog.md",
                author="vault-indexer",
            )

    conn.execute(
        """INSERT OR REPLACE INTO file_hashes
        (file_path, content_hash, indexed_at, chunk_count)
        VALUES ('changelog.md', ?, ?, ?)""",
        (current_hash, db._now_iso(), len(entries))
    )
    conn.commit()
    return {"entries_indexed": len(entries)}


if __name__ == "__main__":
    print("memory-v2 -- Vault Indexer")
    print(f"Vault: {VAULT_PATH}")
    conn = db.init_db()

    def progress(current, total, name):
        print(f"  [{current}/{total}] {name}")

    print("\nIndexing vault files...")
    stats = index_vault(conn, progress_callback=progress)
    print(f"\nVault indexing complete:")
    print(f"  Files: {stats['files_indexed']} indexed, {stats['files_skipped']} skipped")
    print(f"  Chunks: {stats['chunks_created']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for e in stats["errors"][:5]:
            print(f"    - {e}")

    print("\nIndexing changelog...")
    cl_stats = index_changelog(conn)
    print(f"  {cl_stats}")

    print("\nDatabase stats:")
    s = db.get_stats(conn)
    for k, v in s.items():
        print(f"  {k}: {v}")
