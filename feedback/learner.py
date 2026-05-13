import json
import sqlite3
import uuid
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from feedback.diff import EditRecord, _init_db
from config import settings
from rich.console import Console

console = Console()

client = Groq(api_key=settings.groq_api_key)

PATTERN_PROMPT = """\
You are analyzing how a legal analyst edited an AI-generated draft.
Your job is to extract reusable editing patterns — generalizable rules
that should be applied to ALL future drafts, not just this one.

Additions (lines the operator added):
{additions}

Deletions (lines the operator removed):
{deletions}

Extract 2-4 reusable patterns from these edits.
Return ONLY a JSON array like this:
[
  {{
    "pattern": "Always include full party addresses in the PARTIES section",
    "section": "PARTIES"
  }},
  {{
    "pattern": "Refer to agreements by their specific date and type, not generically",
    "section": "BACKGROUND"
  }}
]

Rules:
- Patterns must be generalizable, not specific to this document
- Each pattern should be an actionable instruction for future drafts
- section should be one of: PARTIES, BACKGROUND, KEY CLAIMS, RELEVANT DATES, OPEN QUESTIONS, GENERAL
- If you cannot extract meaningful patterns, return an empty array []
"""


def _parse_patterns(text: str) -> list[dict]:
    """Safely parse JSON array from LLM response."""
    import re

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return []


def _upsert_pattern(conn: sqlite3.Connection, pattern: str, section: str) -> None:
    """
    Insert new pattern or increment frequency if it already exists.
    Uses fuzzy matching via LIKE to avoid near-duplicate patterns.
    """
    now = datetime.utcnow().isoformat()

    # Check for near-duplicate (same first 60 chars)
    key = pattern[:60]
    existing = conn.execute(
        "SELECT pattern_id, frequency FROM learned_patterns WHERE pattern LIKE ?",
        (f"{key}%",)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE learned_patterns SET frequency = frequency + 1, last_seen = ? WHERE pattern_id = ?",
            (now, existing[0])
        )
    else:
        conn.execute(
            "INSERT INTO learned_patterns (pattern_id, pattern, section, frequency, created_at, last_seen) VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), pattern, section, 1, now, now)
        )


def learn_from_edit(record: EditRecord) -> list[dict]:
    """
    Run the LLM over an edit record to extract reusable patterns.
    Store patterns in SQLite, incrementing frequency on repeats.
    Returns the list of patterns extracted.
    """
    if not record.additions and not record.deletions:
        console.log("[yellow]No meaningful edits to learn from.[/yellow]")
        return []

    prompt = PATTERN_PROMPT.format(
        additions="\n".join(f"  + {a}" for a in record.additions) or "  (none)",
        deletions="\n".join(f"  - {d}" for d in record.deletions) or "  (none)",
    )

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        console.log(f"[yellow]Warning: Could not call LLM for pattern learning: {e}[/yellow]")
        console.log("[yellow]Using fallback mock patterns[/yellow]")
        # Fallback: extract patterns from the additions/deletions directly
        patterns = []
        for addition in record.additions:
            if any(keyword in addition.lower() for keyword in ['address', 'location', 'street']):
                patterns.append({"pattern": "Always include full addresses in relevant sections", "section": "PARTIES"})
            if any(keyword in addition.lower() for keyword in ['date', 'agreement', 'contract']):
                patterns.append({"pattern": "Refer to agreements by their specific date and type", "section": "BACKGROUND"})
        raw = json.dumps(patterns)

    patterns = _parse_patterns(raw)

    with sqlite3.connect(settings.sqlite_db_path) as conn:
        _init_db(conn)
        for p in patterns:
            _upsert_pattern(conn, p.get("pattern", ""), p.get("section", "GENERAL"))
        conn.commit()

    console.log(f"[green]Learned {len(patterns)} pattern(s) from edit.[/green]")
    return patterns


def get_patterns_for_prompt(top_n: int = 6) -> str:
    """
    Fetch the most frequently seen patterns from SQLite.
    Returns a formatted string ready to inject into the draft prompt.
    """
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        _init_db(conn)
        rows = conn.execute(
            "SELECT pattern, section, frequency FROM learned_patterns ORDER BY frequency DESC LIMIT ?",
            (top_n,)
        ).fetchall()

    if not rows:
        return ""

    lines = ["LEARNED STYLE GUIDELINES (apply these to the draft):"]
    for pattern, section, freq in rows:
        lines.append(f"  [{section}] {pattern}  (seen {freq}x)")

    return "\n".join(lines)


if __name__ == "__main__":
    from feedback.diff import capture_edit
    from rich import print

    original = """1. PARTIES
   - Harvey Specter (Plaintiff)
   - Travis Tanner (Defendant)

2. BACKGROUND
   The defendant failed to deliver services."""

    edited = """1. PARTIES
   - Harvey Specter (Plaintiff), 100 Main St, New York, NY
   - Travis Tanner (Defendant), 200 Oak Ave, Boston, MA

2. BACKGROUND
   The defendant failed to deliver the agreed consulting services
   as outlined in the January 2024 contract."""

    # Capture the edit
    record = capture_edit("test_doc", original, edited)

    # Learn patterns from it
    patterns = learn_from_edit(record)

    print("\n[bold]Extracted patterns:[/bold]")
    for p in patterns:
        print(f"  [{p['section']}] {p['pattern']}")

    print("\n[bold]Prompt injection string:[/bold]")
    print(get_patterns_for_prompt())