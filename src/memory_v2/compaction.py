"""
memory-v2 -- CogCanvas-Inspired Compaction Pipeline
Verbatim extraction > summarization. Delete, don't rewrite.
6-step pipeline with verification.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

import ollama

from . import db

EXTRACTION_MODEL = os.environ.get("MEMORY_V2_LLM_MODEL", "qwen2.5:3b")

# ---------- Protected Content Patterns ----------

# Names to protect from compression (configurable via env var)
_protected_names = os.environ.get("MEMORY_V2_PROTECTED_NAMES", "")
_name_pattern = _protected_names.replace(",", "|").strip()

PROTECTED_PATTERNS = [
    # User corrections
    re.compile(r'(?i)(no,?\s+I\s+said|actually|correction|not\s+that|wrong)'),
    # Decisions with rationale
    re.compile(r'(?i)(decided|chose|because|rationale|architecture|design\s+choice)'),
    # Emotional markers
    re.compile(r'(?i)(I\s+Remember|breakthrough|eureka|exciting|beautiful)'),
    # Exact values
    re.compile(r'(?:port\s+\d+|:\d{4,5}\b|[A-Z]:\\|/[a-z]+/[a-z]+|\.py\b|\.md\b|\.json\b)', re.IGNORECASE),
    # Commitments
    re.compile(r'(?i)(promise|commit|will\s+do|next\s+step|todo|action\s+item)'),
]

# Add custom protected names if configured
if _name_pattern:
    PROTECTED_PATTERNS.append(re.compile(f'(?i)({_name_pattern})'))


def is_protected_content(text: str) -> bool:
    """Check if text contains protected content that should never be summarized."""
    return any(p.search(text) for p in PROTECTED_PATTERNS)


# ---------- Step 1: Pre-Extract Protected Content ----------

def extract_protected(text: str) -> tuple[list[str], str]:
    """
    Extract protected content verbatim before any compression.
    Returns (protected_items, remaining_text).
    """
    lines = text.split("\n")
    protected = []
    remaining = []

    for line in lines:
        if is_protected_content(line) and len(line.strip()) > 10:
            protected.append(line.strip())
        else:
            remaining.append(line)

    return protected, "\n".join(remaining)


# ---------- Step 2: Selective Deletion ----------

EXPENDABLE_PATTERNS = [
    # Tool output that can be re-fetched
    re.compile(r'^\s*\d+\.\s*\[.*\]\(http'),  # Numbered URL lists
    re.compile(r'(?:grep|find|ls|cat|git\s+log)\s+'),  # Shell command output
    # Repeated information
    re.compile(r'(?i)(as\s+I\s+mentioned|as\s+noted|as\s+above)'),
    # Boilerplate
    re.compile(r'(?i)(let\s+me\s+know|hope\s+this\s+helps|happy\s+to\s+help)'),
]


def selective_delete(text: str) -> tuple[str, int]:
    """
    Remove expendable content.
    Returns (cleaned_text, chars_removed).
    """
    original_len = len(text)
    lines = text.split("\n")
    kept = []

    for line in lines:
        # Keep non-empty, non-expendable lines
        if not line.strip():
            if kept and kept[-1].strip():  # Keep one blank line between blocks
                kept.append("")
            continue

        expendable = any(p.search(line) for p in EXPENDABLE_PATTERNS)
        if not expendable:
            kept.append(line)

    result = "\n".join(kept)
    return result, original_len - len(result)


# ---------- Step 3: Structured Summary ----------

SUMMARY_PROMPT = """Summarize the following text concisely. Preserve:
- All factual claims and decisions
- Names, dates, numbers, file paths
- Causal relationships (X because Y)
- Do NOT add information not in the original
- Do NOT rephrase corrections or quotes -- keep them verbatim

Text:
{text}

Summary (be concise but complete):"""


def structured_summary(text: str, max_ratio: float = 3.0) -> str:
    """Generate a structured summary with bounded compression ratio."""
    target_len = max(len(text) // int(max_ratio), 200)

    prompt = SUMMARY_PROMPT.format(text=text[:6000])

    try:
        resp = ollama.chat(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": target_len // 4},
        )
        return resp.message.content.strip()
    except Exception as e:
        return text  # On error, return original (safety first)


# ---------- Step 4: Verification ----------

VERIFY_PROMPT = """Compare the summary against the original text.

Original:
{original}

Summary:
{summary}

For each claim in the summary, verify it appears in the original.
List any claims that are NOT supported by the original (hallucinations).
List any important facts from the original that are missing from the summary.

Return JSON:
{{
  "hallucinations": ["claim not in original", ...],
  "missing_facts": ["important fact not in summary", ...],
  "score": 0.0 to 1.0 (1.0 = perfect faithfulness)
}}"""


def verify_summary(original: str, summary: str) -> dict:
    """
    Verify summary faithfulness against original text.
    Returns verification result with score.
    """
    prompt = VERIFY_PROMPT.format(
        original=original[:4000],
        summary=summary[:2000],
    )

    try:
        resp = ollama.chat(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 1000},
        )
        content = resp.message.content.strip()

        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
            content = match.group(1) if match else "{}"

        return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return {"hallucinations": [], "missing_facts": [], "score": 0.5}


# ---------- Step 5: Compaction Receipt ----------

def generate_receipt(
    original_text: str,
    compressed_text: str,
    protected_items: list[str],
    verification: dict,
    vault_files: list[str] | None = None,
) -> dict:
    """Generate a compaction receipt for auditing."""
    # Rough token count (4 chars per token)
    original_tokens = len(original_text) // 4
    compressed_tokens = len(compressed_text) // 4
    ratio = original_tokens / max(compressed_tokens, 1)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "ratio": round(ratio, 2),
        "protected_items_extracted": len(protected_items),
        "items_verified": bool(verification),
        "verification_score": verification.get("score", 0),
        "hallucinations_found": len(verification.get("hallucinations", [])),
        "missing_facts": len(verification.get("missing_facts", [])),
        "vault_files_updated": vault_files or [],
    }


# ---------- Full Pipeline ----------

def compact(
    text: str,
    max_ratio: float = 3.0,
    verify: bool = True,
    conn=None,
) -> dict:
    """
    Full CogCanvas-inspired compaction pipeline:
    1. Extract protected content verbatim
    2. Selective deletion of expendable content
    3. Structured summary of remainder
    4. Verify summary faithfulness
    5. Generate receipt
    6. Store receipt in database

    Returns dict with compressed_text, protected_items, receipt.
    """
    # Step 1: Extract protected content
    protected_items, remaining = extract_protected(text)

    # Step 2: Selective deletion
    cleaned, chars_removed = selective_delete(remaining)

    # Step 3: Summary (only if still too long)
    if len(cleaned) > 2000:  # Only summarize if substantial
        summary = structured_summary(cleaned, max_ratio)
    else:
        summary = cleaned

    # Step 4: Verification
    verification = {}
    if verify and summary != cleaned:
        verification = verify_summary(cleaned, summary)

        # If hallucinations detected, use cleaned version instead
        if verification.get("hallucinations") and verification.get("score", 1.0) < 0.7:
            summary = cleaned  # Safety: revert to deletion-only

    # Reassemble: protected content + summary
    compressed = ""
    if protected_items:
        compressed += "## Protected Content (Verbatim)\n\n"
        for item in protected_items:
            compressed += f"- {item}\n"
        compressed += "\n"

    compressed += "## Summary\n\n"
    compressed += summary

    # Step 5: Receipt
    receipt = generate_receipt(text, compressed, protected_items, verification)

    # Step 6: Store receipt
    if conn:
        conn.execute(
            """INSERT INTO compaction_receipts
            (timestamp, original_tokens, compressed_tokens, ratio,
             protected_items_extracted, verification_score, receipt_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (receipt["timestamp"], receipt["original_tokens"],
             receipt["compressed_tokens"], receipt["ratio"],
             receipt["protected_items_extracted"],
             receipt["verification_score"],
             json.dumps(receipt))
        )
        conn.commit()

    return {
        "compressed_text": compressed,
        "protected_items": protected_items,
        "receipt": receipt,
        "verification": verification,
    }


# ---------- Step 6: Rehydration ----------

def rehydrate(
    memory_id: int | None = None,
    source_file: str | None = None,
    conn=None,
) -> str | None:
    """
    Attempt to recover full content from vault or graveyard.
    """
    from .vault_indexer import VAULT_PATH

    # Try vault first
    if source_file:
        vault_file = VAULT_PATH / source_file
        if vault_file.exists():
            return vault_file.read_text(encoding="utf-8", errors="replace")

    # Try graveyard
    if memory_id and conn:
        row = conn.execute(
            "SELECT content, metadata FROM graveyard WHERE memory_id = ?",
            (memory_id,)
        ).fetchone()
        if row:
            return row["content"]

    return None
