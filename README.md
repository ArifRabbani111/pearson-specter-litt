# Pearson Specter Litt — Legal AI System

Internal workflow for ingesting messy legal documents, extracting structured information, generating grounded case fact summaries, and improving from operator edits.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│         PDFs · Scanned images · Handwritten notes           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  1. DOCUMENT PROCESSING                     │
│  pymupdf (native text) → pytesseract (OCR fallback)        │
│  Per-page strategy: native if usable, OCR if sparse        │
│  Output: cleaned text + structured fields (via Groq LLM)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  2. RETRIEVAL LAYER                         │
│  Fixed-size overlapping chunks (400 tokens, 80 overlap)    │
│  Gemini text-embedding-004 → ChromaDB (cosine similarity)  │
│  Two-stage: vector search → cross-encoder rerank           │
│  Each chunk carries metadata for filtering + citation      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  3. DRAFT GENERATION                        │
│  Retrieved passages injected into prompt with chunk IDs    │
│  Groq (llama-3.3-70b) generates case fact summary         │
│  Every claim cited with [SOURCE: chunk_id]                 │
│  Learned style patterns injected into system prompt        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  4. OPERATOR EDIT LOOP                      │
│  Operator edits draft → unified diff captured              │
│  Groq extracts generalizable patterns from additions/      │
│  deletions (not document-specific, but reusable rules)     │
│  Patterns stored in SQLite with frequency tracking         │
│  Top patterns injected into all future draft prompts       │
└─────────────────────────────────────────────────────────────┘
```

---

## Stack

| Component | Technology |
|---|---|
| PDF extraction | pymupdf |
| OCR | pytesseract + Tesseract |
| LLM (structuring + drafting + pattern learning) | Groq — llama-3.3-70b-versatile |
| Embeddings | Gemini text-embedding-004 |
| Vector store | ChromaDB (persistent, local) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Edit + pattern store | SQLite |
| API | FastAPI |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/ArifRabbani111/pearson-specter-litt
cd pearson-specter-litt
pip install -r requirements.txt
```

### 2. Install Tesseract

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr

# Windows
# https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Set up environment

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
CHROMA_DB_PATH=./db/chroma
SQLITE_DB_PATH=./db/edits.db
```

Get your keys:
- Groq: [console.groq.com](https://console.groq.com)
- Gemini: [aistudio.google.com](https://aistudio.google.com)
- Anthropic: [console.anthropic.com](https://console.anthropic.com)

---

## Running the Demo

The fastest way to see the full system in action:

```bash
python tests/demo.py
```

With a custom document:

```bash
python tests/demo.py --file data/samples/your_file.pdf
```

This runs all 7 stages and saves outputs to `data/outputs/`.

---

## Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at: `http://localhost:8000/docs`

### Endpoints

#### `POST /ingest`
Upload a legal document. Returns `doc_id` and structured metadata.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/samples/sample_case.txt"
```

```json
{
  "doc_id": "abc12345",
  "doc_type": "complaint",
  "case_number": "2024-CV-00123",
  "parties": ["Harvey Specter", "Travis Tanner"],
  "summary": "Plaintiff alleges breach of a January 2024 consulting contract.",
  "chunks_stored": 4
}
```

---

#### `POST /draft`
Generate a grounded case fact summary for an ingested document.

```bash
curl -X POST http://localhost:8000/draft \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "abc12345", "query": "Summarize the key facts"}'
```

```json
{
  "draft": "1. PARTIES\n   - Harvey Specter (Plaintiff)...",
  "citations": [
    {
      "chunk_id": "abc12345_chunk_0",
      "page_number": 1,
      "excerpt": "CASE NO. 2024-CV-00123...",
      "score": 2.3421
    }
  ]
}
```

---

#### `POST /edit`
Submit an operator-edited draft. Extracts and stores reusable patterns.

```bash
curl -X POST http://localhost:8000/edit \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "abc12345",
    "original_draft": "- Harvey Specter (Plaintiff)",
    "edited_draft": "- Harvey Specter (Plaintiff), 100 Main St, New York, NY"
  }'
```

```json
{
  "edit_id": "uuid-here",
  "patterns_learned": [
    {
      "pattern": "Always include full party addresses in the PARTIES section",
      "section": "PARTIES"
    }
  ]
}
```

---

#### `GET /patterns`
Inspect the current learned style guidelines being injected into drafts.

```bash
curl http://localhost:8000/patterns
```

---

## Assumptions and Tradeoffs

**Chunking strategy — fixed size over semantic chunking**
Fixed 400-token windows with 80-token overlap are simpler, faster, and more predictable than semantic chunking. For short legal documents this performs well. For very long documents (100+ pages), semantic chunking would reduce noise in retrieval.

**Two-stage retrieval**
Vector search alone over-retrieves. Adding a cross-encoder reranker significantly improves precision at the cost of ~200ms latency per query. Worth it for a legal context where grounding accuracy matters more than speed.

**Groq for LLM, Gemini for embeddings**
Groq's inference is fast and cheap for generation. Gemini's `text-embedding-004` outperforms OpenAI's small embedding model on retrieval benchmarks. Using two providers is a minor operational tradeoff for better performance per dollar.

**Pattern learning — generalization not memorization**
The edit loop extracts reusable rules from diffs rather than storing raw examples. This means 3 edits that all add party addresses produce one pattern ("always include full addresses") that applies to all future documents — not 3 document-specific memories.

**In-memory doc store in API**
`_doc_store` in `api/main.py` is a Python dict. This resets on server restart. In production this would be a database. Acceptable for take-home scope.

**Simulated operator edits in demo**
The demo uses a hardcoded edit to simulate operator behavior. In production this would come from a real editing interface. The diff capture and pattern extraction are fully real — only the edit source is simulated.

---

## Evaluation Approach

**Document processing**
Tested on: native PDF, scanned PDF (via OCR), plain text. Verified extraction method selection is correct per file type and that structured fields match document content.

**Retrieval grounding**
Verified that every factual claim in generated drafts maps to a cited chunk. Tested with queries that have no supporting evidence — the system correctly writes "Insufficient information in source documents" rather than hallucinating.

**Draft quality**
Assessed by checking: are all parties named? are dates correct? are claims traceable to citations? Drafts consistently reflect only what is in source documents.

**Improvement from edits**
Run demo twice — once with no patterns, once after simulating operator edits. Compared v1 and v2 drafts. v2 consistently includes party addresses and more specific references to agreements, matching the simulated operator preferences.

---

## Project Structure

```
pearson-specter-litt/
├── ingest/
│   ├── extractor.py        # OCR + native text extraction
│   └── structurer.py       # LLM-based structured field extraction
├── retrieval/
│   ├── chunker.py          # fixed-size overlapping chunker
│   ├── embedder.py         # Gemini embeddings + ChromaDB storage
│   └── retriever.py        # vector search + cross-encoder rerank
├── drafting/
│   └── generator.py        # grounded draft with inline citations
├── feedback/
│   ├── diff.py             # unified diff capture + SQLite storage
│   └── learner.py          # pattern extraction + prompt injection
├── api/
│   └── main.py             # FastAPI routes
├── data/
│   ├── samples/            # sample input documents
│   └── outputs/            # generated drafts and demo summaries
├── tests/
│   └── demo.py             # end-to-end pipeline demo
├── config.py               # settings and environment config
├── requirements.txt
└── README.md
```

---

## Sample Inputs and Outputs

Sample input: `data/samples/sample_case.txt`
Sample outputs: `data/outputs/` (generated on demo run)

The demo script produces:
- `draft_v1_*.txt` — baseline draft before any operator edits
- `draft_v2_*.txt` — improved draft after pattern learning
- `demo_summary_*.json` — pipeline run metadata