import hashlib
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import tempfile

from ingest.extractor import extract
from ingest.structurer import structure
from retrieval.chunker import chunk
from retrieval.embedder import embed_and_store
from drafting.generator import generate_draft
from feedback.diff import capture_edit
from feedback.learner import learn_from_edit, get_patterns_for_prompt
from rich.console import Console

console = Console()

app = FastAPI(
    title="Pearson Specter Litt — Legal AI",
    description="Document ingestion, grounded drafting, and improvement from operator edits.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / Response models
# --------------------------------------------------------------------------- #

class DraftRequest(BaseModel):
    doc_id: str
    query: str = "Summarize the key facts of this case"


class EditRequest(BaseModel):
    doc_id: str
    original_draft: str
    edited_draft: str


class PatternResponse(BaseModel):
    patterns: str


# --------------------------------------------------------------------------- #
# In-memory doc store (maps doc_id → StructuredDocument)
# In production you'd persist this — fine for take-home scope
# --------------------------------------------------------------------------- #

_doc_store: dict = {}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Pearson Specter Litt Legal AI",
        "endpoints": [
            "POST /ingest",
            "POST /draft",
            "POST /edit",
            "GET  /patterns",
            "GET  /health",
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a legal document (PDF, image, or .txt).
    Extracts text, structures fields, chunks, and embeds into ChromaDB.
    Returns doc_id and structured metadata.
    """
    allowed = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".tiff"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}"
        )

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        console.log(f"[yellow]Ingesting: {file.filename}[/yellow]")

        extracted = extract(tmp_path)
        structured = structure(extracted)
        chunks = chunk(structured)
        embed_and_store(chunks)

        doc_id = hashlib.md5(structured.source_path.encode()).hexdigest()[:8]
        _doc_store[doc_id] = structured

        console.log(f"[green]Ingested doc_id: {doc_id}[/green]")

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "page_count": structured.page_count,
            "extraction_method": structured.extraction_method,
            "doc_type": structured.doc_type,
            "case_number": structured.case_number,
            "parties": structured.parties,
            "dates": structured.dates,
            "jurisdiction": structured.jurisdiction,
            "summary": structured.summary_one_line,
            "warnings": structured.warnings,
            "chunks_stored": len(chunks),
        }

    except Exception as e:
        console.log(f"[red]Ingest failed: {e}[/red]")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/draft")
def generate(req: DraftRequest):
    """
    Generate a grounded case fact summary for an ingested document.
    Requires the doc_id returned from /ingest.
    """
    doc = _doc_store.get(req.doc_id)

    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"doc_id '{req.doc_id}' not found. Ingest the document first."
        )

    try:
        output = generate_draft(doc, query=req.query)

        return {
            "doc_id": req.doc_id,
            "query": req.query,
            "draft": output.draft,
            "citations": output.citations,
            "model_used": output.model_used,
            "chunks_used": len(output.retrieved_chunks),
        }

    except Exception as e:
        console.log(f"[red]Draft generation failed: {e}[/red]")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit")
def submit_edit(req: EditRequest):
    """
    Submit an operator-edited draft.
    Captures the diff, extracts reusable patterns, stores in SQLite.
    Returns the patterns learned from this edit.
    """
    if req.original_draft == req.edited_draft:
        return {
            "message": "No changes detected.",
            "patterns_learned": []
        }

    try:
        record = capture_edit(
            doc_id=req.doc_id,
            original_draft=req.original_draft,
            edited_draft=req.edited_draft,
        )

        patterns = learn_from_edit(record)

        return {
            "edit_id": record.edit_id,
            "additions_count": len(record.additions),
            "deletions_count": len(record.deletions),
            "patterns_learned": patterns,
        }

    except Exception as e:
        console.log(f"[red]Edit capture failed: {e}[/red]")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patterns")
def get_patterns():
    """
    Return the current learned style guidelines
    that are being injected into draft prompts.
    """
    patterns_str = get_patterns_for_prompt()

    return {
        "patterns": patterns_str if patterns_str else "No patterns learned yet.",
    }