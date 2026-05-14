#!/usr/bin/env python
"""Debug version of demo with explicit print statements."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("[1] Importing extract...")
from ingest.extractor import extract
print("[2] Import extract OK")

print("[3] Importing structure...")
from ingest.structurer import structure
print("[4] Import structure OK")

print("[5] Importing chunk...")
from retrieval.chunker import chunk
print("[6] Import chunk OK")

print("[7] Importing embed_and_store...")
from retrieval.embedder import embed_and_store
print("[8] Import embed_and_store OK")

print("[9] Importing generate_draft...")
from drafting.generator import generate_draft
print("[10] Import generate_draft OK")

print("[11] Importing capture_edit...")
from feedback.diff import capture_edit
print("[12] Import capture_edit OK")

print("[13] Importing learn_from_edit...")
from feedback.learner import learn_from_edit, get_patterns_for_prompt
print("[14] Import learn_from_edit OK")

print("\n=== All imports successful ===\n")

print("[15] Extracting document...")
extracted = extract("data/samples/sample_case.txt")
print(f"[16] Extraction OK: {extracted.page_count} pages")

print("[17] Structuring...")
structured = structure(extracted)
print(f"[18] Structuring OK: {structured.doc_type}")

print("[19] Chunking...")
chunks = chunk(structured)
print(f"[20] Chunking OK: {len(chunks)} chunks")

print("[21] Embedding...")
embed_and_store(chunks, "doc123")
print("[22] Embedding OK")

print("[23] Generating draft v1...")
output_v1 = generate_draft(structured, "doc123")
print(f"[24] Draft v1 OK: {len(output_v1.draft)} chars")

print("[25] Simulating edit...")
edited_v1 = output_v1.draft.replace("Key Facts", "Background Information")
record = capture_edit("doc123", output_v1.draft, edited_v1)
print(f"[26] Edit capture OK: {len(record.additions)} additions")

print("[27] Learning patterns...")
patterns = learn_from_edit(record)
print(f"[28] Learn patterns OK: {len(patterns)} patterns")

print("[29] Getting patterns for prompt...")
patterns_str = get_patterns_for_prompt()
print(f"[30] Patterns for prompt OK: {len(patterns_str)} chars")

print("[31] Generating draft v2...")
output_v2 = generate_draft(structured, "doc123")
print(f"[32] Draft v2 OK: {len(output_v2.draft)} chars")

print("\n=== Demo completed successfully ===")
