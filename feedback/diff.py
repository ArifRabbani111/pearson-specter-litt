import difflib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from rich.console import Console

console = Console()


@dataclass
class EditRecord:
    edit_id: str
    doc_id: str
    original_draft: str
    edited_draft: str
    diff_lines: list[str]
    timestamp: str
    additions: list[str] = field(default_factory=list)   # lines operator added
    deletions: list[str] = field(default_factory=list)   # lines operator removed


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edit_records (
            edit_id     TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            original    TEXT NOT NULL,
            edited      TEXT NOT NULL,
            diff        TEXT NOT NULL,
            additions   TEXT NOT NULL,
            deletions   TEXT NOT NULL,
            timestamp   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learned_patterns (
            pattern_id  TEXT PRIMARY KEY,
            pattern     TEXT NOT NULL,
            section     TEXT,
            frequency   INTEGER DEFAULT 1,
            created_at  TEXT NOT NULL,
            last_seen   TEXT NOT NULL
        )
    """)
    conn.commit()


def capture_edit(
    doc_id: str,
    original_draft: str,
    edited_draft: str,
) -> EditRecord:
    """
    Diff the original and edited draft.
    Store the edit record in SQLite.
    Return a structured EditRecord for downstream pattern extraction.
    """
    diff = list(difflib.unified_diff(
        original_draft.splitlines(),
        edited_draft.splitlines(),
        lineterm="",
        n=2,
    ))

    additions = [
        line[1:].strip()
        for line in diff
        if line.startswith("+") and not line.startswith("+++")
    ]
    deletions = [
        line[1:].strip()
        for line in diff
        if line.startswith("-") and not line.startswith("---")
    ]

    record = EditRecord(
        edit_id=str(uuid.uuid4()),
        doc_id=doc_id,
        original_draft=original_draft,
        edited_draft=edited_draft,
        diff_lines=diff,
        timestamp=datetime.utcnow().isoformat(),
        additions=additions,
        deletions=deletions,
    )

    # Persist to SQLite
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        _init_db(conn)
        conn.execute("""
            INSERT INTO edit_records
            (edit_id, doc_id, original, edited, diff, additions, deletions, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.edit_id,
            record.doc_id,
            record.original_draft,
            record.edited_draft,
            json.dumps(record.diff_lines),
            json.dumps(record.additions),
            json.dumps(record.deletions),
            record.timestamp,
        ))
        conn.commit()

    console.log(f"[green]Edit captured: {len(additions)} additions, {len(deletions)} deletions.[/green]")
    return record


def get_edit_history(doc_id: str = None) -> list[dict]:
    """Fetch all edit records, optionally filtered by doc_id."""
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        _init_db(conn)
        if doc_id:
            rows = conn.execute(
                "SELECT * FROM edit_records WHERE doc_id = ? ORDER BY timestamp DESC",
                (doc_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM edit_records ORDER BY timestamp DESC"
            ).fetchall()

    return [
        {
            "edit_id": r[0],
            "doc_id": r[1],
            "additions": json.loads(r[5]),
            "deletions": json.loads(r[6]),
            "timestamp": r[7],
        }
        for r in rows
    ]


if __name__ == "__main__":
    # Simulate an operator edit
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

    record = capture_edit("test_doc", original, edited)
    print(f"\nEdit ID  : {record.edit_id}")
    print(f"Added    : {record.additions}")
    print(f"Deleted  : {record.deletions}")