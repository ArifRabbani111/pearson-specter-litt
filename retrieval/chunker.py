from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.structurer import StructuredDocument
from config import settings


@dataclass
class Chunk:
    chunk_id: str        # e.g. "doc_abc123_chunk_0"
    doc_id: str          # derived from source path
    text: str
    page_number: int
    chunk_index: int
    metadata: dict       # carries structured fields for filtering


def _make_doc_id(source_path: str) -> str:
    """Stable short ID from file path."""
    import hashlib
    return hashlib.md5(source_path.encode()).hexdigest()[:8]


def chunk(doc: StructuredDocument) -> list[Chunk]:
    """
    Split document pages into overlapping fixed-size chunks.
    Each chunk carries metadata from the structured fields.
    """
    doc_id = _make_doc_id(doc.source_path)
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks = []
    chunk_index = 0

    for page_num, page_text in enumerate(doc.pages):
        words = page_text.split()

        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                text=chunk_text,
                page_number=page_num + 1,
                chunk_index=chunk_index,
                metadata={
                    "doc_id": doc_id,
                    "source_path": doc.source_path,
                    "doc_type": doc.doc_type,
                    "case_number": doc.case_number or "",
                    "parties": ", ".join(doc.parties),
                    "jurisdiction": doc.jurisdiction or "",
                    "page_number": page_num + 1,
                }
            ))

            chunk_index += 1
            if end == len(words):
                break
            start += size - overlap

    return chunks


if __name__ == "__main__":
    from ingest.extractor import extract
    from ingest.structurer import structure

    doc = structure(extract("data/samples/sample_case.txt"))
    chunks = chunk(doc)

    print(f"Total chunks: {len(chunks)}")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}] page {c.page_number}")
        print(c.text[:200])