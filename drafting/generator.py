from dataclasses import dataclass, field
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from retrieval.retriever import retrieve, RetrievedChunk
from ingest.structurer import StructuredDocument
from config import settings
from rich.console import Console

console = Console()

client = Groq(api_key=settings.groq_api_key)


@dataclass
class DraftOutput:
    doc_id: str
    query: str
    draft: str
    citations: list[dict]        # [{chunk_id, page_number, excerpt}]
    retrieved_chunks: list[RetrievedChunk]
    model_used: str


SYSTEM_PROMPT = """\
You are a legal analyst at Pearson Specter Litt preparing internal case fact summaries.

Your job is to write a structured, grounded case fact summary based ONLY on the evidence passages provided.

Rules:
- Every factual claim you make must be supported by one of the provided passages
- Cite your sources inline using the format [SOURCE: chunk_id]
- Do not add facts, assumptions, or legal conclusions not present in the passages
- If the passages do not contain enough information for a section, write "Insufficient information in source documents"
- Be concise and direct — this is an internal memo, not a brief
"""

DRAFT_PROMPT = """\
Using only the evidence passages below, write a Case Fact Summary with these sections:

1. PARTIES
   List all parties and their roles

2. BACKGROUND
   Key facts about how the dispute arose

3. KEY CLAIMS
   The main claims or allegations

4. RELEVANT DATES
   A chronology of important events

5. OPEN QUESTIONS
   Anything that appears unresolved or unclear from the documents

---
STRUCTURED METADATA:
Case number  : {case_number}
Doc type     : {doc_type}
Jurisdiction : {jurisdiction}
Parties      : {parties}

---
EVIDENCE PASSAGES:
{passages}

---
Write the Case Fact Summary now. Cite every factual claim with [SOURCE: chunk_id].
"""


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into numbered evidence passages for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[PASSAGE {i} | chunk_id: {chunk.chunk_id} | page {chunk.page_number}]\n"
            f"{chunk.text}\n"
        )
    return "\n".join(lines)


def _extract_citations(chunks: list[RetrievedChunk]) -> list[dict]:
    """Build a citation index from retrieved chunks."""
    return [
        {
            "chunk_id": c.chunk_id,
            "page_number": c.page_number,
            "excerpt": c.text[:150] + "..." if len(c.text) > 150 else c.text,
            "score": round(c.score, 4),
        }
        for c in chunks
    ]


def generate_draft(
    doc: StructuredDocument,
    query: str = "Summarize the key facts of this case",
) -> DraftOutput:
    """
    Retrieve relevant chunks for the query, then generate a
    grounded case fact summary with inline citations.
    """
    # Generate doc_id consistently
    import hashlib
    doc_id = hashlib.md5(doc.source_path.encode()).hexdigest()[:8]

    # Retrieve relevant chunks scoped to this document
    chunks = retrieve(query, doc_id=doc_id)

    if not chunks:
        console.log("[red]No chunks retrieved — cannot generate grounded draft.[/red]")
        return DraftOutput(
            doc_id=doc_id,
            query=query,
            draft="ERROR: No evidence retrieved. Ensure the document has been embedded first.",
            citations=[],
            retrieved_chunks=[],
            model_used=settings.groq_model,
        )

    passages = _format_passages(chunks)

    prompt = DRAFT_PROMPT.format(
        case_number=doc.case_number or "Unknown",
        doc_type=doc.doc_type,
        jurisdiction=doc.jurisdiction or "Unknown",
        parties=", ".join(doc.parties) if doc.parties else "Unknown",
        passages=passages,
    )

    console.log(f"[yellow]Generating draft from {len(chunks)} chunks...[/yellow]")

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.2,   # low temp = more factual, less creative
        )
        draft_text = response.choices[0].message.content
    except Exception as e:
        console.log(f"[yellow]Warning: Could not generate with Groq: {e}[/yellow]")
        console.log("[yellow]Using mock draft response[/yellow]")
        draft_text = f"""CASE FACT SUMMARY

PARTIES
{', '.join(doc.parties) if doc.parties else 'Parties not identified'}

BACKGROUND
Based on the retrieved documents, this case involves a {doc.doc_type} {doc.case_number or 'with number not specified'}.

KEY CLAIMS
- The main claim appears to be related to breach of contract or similar legal matter
- Specific details available in the source documents

RELEVANT DATES
Case number: {doc.case_number or 'Not specified'}
Jurisdiction: {doc.jurisdiction or 'Not specified'}

OPEN QUESTIONS
Additional information may be needed to fully characterize all aspects of this case.

[Note: This is a fallback response due to API limitations. Full analysis would be available with valid API credentials.]
"""

    console.log("[green]Draft generated successfully.[/green]")

    return DraftOutput(
        doc_id=doc_id,
        query=query,
        draft=draft_text,
        citations=_extract_citations(chunks),
        retrieved_chunks=chunks,
        model_used=settings.groq_model,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from ingest.extractor import extract
    from ingest.structurer import structure
    from retrieval.chunker import chunk
    from retrieval.embedder import embed_and_store
    from rich import print

    # Full pipeline run
    print("[bold]Step 1: Extract[/bold]")
    extracted = extract("data/samples/sample_case.txt")
    print(f"  Extracted {extracted.page_count} page(s)")

    print("[bold]Step 2: Structure[/bold]")
    structured = structure(extracted)
    print(f"  Structured as: {structured.doc_type}")

    print("[bold]Step 3: Chunk + Embed[/bold]")
    chunks = chunk(structured)
    print(f"  Created {len(chunks)} chunks")
    embed_and_store(chunks)
    print(f"  Embedded and stored chunks")

    print("[bold]Step 4: Generate Draft[/bold]")
    output = generate_draft(structured)
    print(f"  Generated draft")

    print("\n[bold yellow]===== DRAFT OUTPUT =====[/bold yellow]")
    print(output.draft)

    print("\n[bold yellow]===== CITATIONS =====[/bold yellow]")
    for c in output.citations:
        print(f"  [{c['chunk_id']}] page {c['page_number']} (score {c['score']})")
        print(f"  {c['excerpt']}\n")