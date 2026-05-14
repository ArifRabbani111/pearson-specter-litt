"""
End-to-end demo script for Pearson Specter Litt Legal AI.

Runs the full pipeline:
  1. Document extraction
  2. Structured field extraction
  3. Chunking + embedding
  4. Grounded draft generation
  5. Operator edit simulation
  6. Pattern learning
  7. Improved draft generation

Usage:
    python tests/demo.py
    python tests/demo.py --file data/samples/your_file.pdf
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# Make sure root is on path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.extractor import extract
from ingest.structurer import structure
from retrieval.chunker import chunk
from retrieval.embedder import embed_and_store
from drafting.generator import generate_draft
from feedback.diff import capture_edit
from feedback.learner import learn_from_edit, get_patterns_for_prompt

console = Console()

SAMPLE_FILE = "data/samples/sample_case.txt"

# Simulated operator edit — mimics a real analyst improving the draft
SIMULATED_EDIT = {
    "additions": [
        "- Harvey Specter (Plaintiff), 100 Main St, New York, NY 10001",
        "- Travis Tanner (Defendant), 200 Oak Ave, Boston, MA 02101",
        "The defendant failed to deliver the agreed consulting services as outlined in the January 15, 2024 written contract.",
        "Plaintiff seeks compensatory damages of $500,000 plus attorney fees.",
    ],
    "deletions": [
        "- Harvey Specter (Plaintiff)",
        "- Travis Tanner (Defendant)",
        "The defendant failed to deliver services.",
        "Plaintiff seeks damages of $500,000.",
    ]
}


def section(title: str):
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))


def success(msg: str):
    console.print(f"  [green]✓[/green] {msg}")


def info(msg: str):
    console.print(f"  [dim]{msg}[/dim]")


def save_output(filename: str, content: str):
    out_path = Path("data/outputs") / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    success(f"Saved → {out_path}")


def run_demo(file_path: str):
    console.print(Panel.fit(
        "[bold white]Pearson Specter Litt — Legal AI Demo[/bold white]\n"
        f"[dim]Running full pipeline on: {file_path}[/dim]",
        border_style="cyan"
    ))

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------ #
    # STEP 1: Extract
    # ------------------------------------------------------------------ #
    section("Step 1 — Document Extraction")

    extracted = extract(file_path)
    success(f"Extracted {extracted.page_count} page(s) via '{extracted.extraction_method}'")

    if extracted.warnings:
        for w in extracted.warnings:
            console.print(f"  [yellow]⚠ {w}[/yellow]")

    info(f"Raw text length: {len(extracted.raw_text):,} characters")

    # ------------------------------------------------------------------ #
    # STEP 2: Structure
    # ------------------------------------------------------------------ #
    section("Step 2 — Structured Field Extraction")

    structured = structure(extracted)
    success(f"Doc type    : {structured.doc_type}")
    success(f"Case number : {structured.case_number or 'not found'}")
    success(f"Parties     : {', '.join(structured.parties) or 'not found'}")
    success(f"Jurisdiction: {structured.jurisdiction or 'not found'}")
    success(f"Dates       : {', '.join(structured.dates) or 'not found'}")
    console.print(f"\n  [bold]Summary:[/bold] {structured.summary_one_line}")

    if structured.key_facts:
        console.print("\n  [bold]Key facts:[/bold]")
        for i, fact in enumerate(structured.key_facts, 1):
            console.print(f"    {i}. {fact}")

    # ------------------------------------------------------------------ #
    # STEP 3: Chunk + Embed
    # ------------------------------------------------------------------ #
    section("Step 3 — Chunking + Embedding")

    chunks = chunk(structured)
    success(f"Created {len(chunks)} chunks (size={400}, overlap={80})")

    embed_and_store(chunks)
    success("Chunks embedded and stored in ChromaDB")

    doc_id = hashlib.md5(structured.source_path.encode()).hexdigest()[:8]
    info(f"doc_id: {doc_id}")

    # ------------------------------------------------------------------ #
    # STEP 4: First draft (no learned patterns yet)
    # ------------------------------------------------------------------ #
    section("Step 4 — Draft Generation (baseline)")

    output_v1 = generate_draft(structured)
    success(f"Draft generated using {len(output_v1.retrieved_chunks)} retrieved chunks")
    success(f"Model: {output_v1.model_used}")

    console.print("\n[bold yellow]--- DRAFT v1 (baseline) ---[/bold yellow]")
    console.print(output_v1.draft)

    console.print("\n[bold]Citations:[/bold]")
    for c in output_v1.citations:
        console.print(f"  [{c['chunk_id']}] page {c['page_number']} | score {c['score']}")
        console.print(f"  [dim]{c['excerpt']}[/dim]\n")

    save_output(f"draft_v1_{timestamp}.txt", output_v1.draft)

    # ------------------------------------------------------------------ #
    # STEP 5: Simulate operator edit
    # ------------------------------------------------------------------ #
    section("Step 5 — Operator Edit Simulation")

    # Build a realistic before/after from our simulated edit
    original_snippet = "\n".join(SIMULATED_EDIT["deletions"])
    edited_snippet = "\n".join(SIMULATED_EDIT["additions"])

    record = capture_edit(
        doc_id=doc_id,
        original_draft=original_snippet,
        edited_draft=edited_snippet,
    )

    success(f"Edit captured — ID: {record.edit_id}")
    info(f"  {len(record.additions)} additions, {len(record.deletions)} deletions")

    console.print("\n  [bold]Additions:[/bold]")
    for a in record.additions:
        console.print(f"    [green]+ {a}[/green]")

    console.print("\n  [bold]Deletions:[/bold]")
    for d in record.deletions:
        console.print(f"    [red]- {d}[/red]")

    # ------------------------------------------------------------------ #
    # STEP 6: Learn patterns
    # ------------------------------------------------------------------ #
    section("Step 6 — Pattern Learning")

    patterns = learn_from_edit(record)
    success(f"Extracted {len(patterns)} reusable pattern(s)")

    for p in patterns:
        console.print(f"  [bold cyan][{p.get('section', 'GENERAL')}][/bold cyan] {p.get('pattern')}")

    console.print("\n  [bold]Prompt injection string:[/bold]")
    console.print(f"  [dim]{get_patterns_for_prompt()}[/dim]")

    # ------------------------------------------------------------------ #
    # STEP 7: Improved draft
    # ------------------------------------------------------------------ #
    section("Step 7 — Draft Generation (with learned patterns)")

    output_v2 = generate_draft(structured)
    success(f"Improved draft generated")

    console.print("\n[bold yellow]--- DRAFT v2 (with learned patterns) ---[/bold yellow]")
    console.print(output_v2.draft)

    save_output(f"draft_v2_{timestamp}.txt", output_v2.draft)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    section("Demo Complete")

    summary = {
        "timestamp": timestamp,
        "file": file_path,
        "doc_id": doc_id,
        "doc_type": structured.doc_type,
        "pages": structured.page_count,
        "chunks": len(chunks),
        "patterns_learned": len(patterns),
        "draft_v1_length": len(output_v1.draft),
        "draft_v2_length": len(output_v2.draft),
    }

    save_output(f"demo_summary_{timestamp}.json", json.dumps(summary, indent=2))

    console.print(Panel.fit(
        "[bold green]All steps completed successfully.[/bold green]\n"
        f"[dim]Outputs saved to data/outputs/[/dim]",
        border_style="green"
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pearson Specter Litt — Legal AI Demo")
    parser.add_argument(
        "--file",
        default=SAMPLE_FILE,
        help="Path to document to process (default: data/samples/sample_case.txt)"
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        console.print(f"[red]File not found: {args.file}[/red]")
        sys.exit(1)

    run_demo(args.file)