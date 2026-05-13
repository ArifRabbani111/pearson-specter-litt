import json
import re
from typing import Optional
from dataclasses import dataclass, field
from anthropic import Anthropic
from ingest.extractor import ExtractedDocument
from config import settings

# Initialize Anthropic client with fallback
try:
    client = Anthropic(api_key=settings.anthropic_api_key or settings.groq_api_key)
except Exception as e:
    print(f"Warning: Could not initialize Anthropic client: {e}")
    client = None


@dataclass
class StructuredDocument:
    # Passed through from ExtractedDocument
    source_path: str
    raw_text: str
    pages: list[str]
    extraction_method: str
    page_count: int
    warnings: list[str]

    # Fields extracted by Claude
    doc_type: str = "unknown"
    case_number: Optional[str] = None
    parties: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    jurisdiction: Optional[str] = None
    summary_one_line: str = ""


EXTRACTION_PROMPT = """\
You are a legal document analyst. Extract structured information from the document below.

Return ONLY a valid JSON object with exactly these keys:
{{
  "doc_type": "one of: complaint, contract, motion, order, deposition, notice, memo, unknown",
  "case_number": "case or docket number as a string, or null if not found",
  "parties": ["list of all party names mentioned"],
  "dates": ["important dates, use YYYY-MM-DD format where possible"],
  "key_facts": ["5 to 8 key factual claims stated in the document"],
  "jurisdiction": "court or jurisdiction name, or null if not found",
  "summary_one_line": "one sentence describing what this document is about"
}}

Rules:
- Only include information explicitly present in the text
- Do not infer, assume, or fabricate anything
- key_facts must be direct factual statements, not interpretations
- If a field cannot be determined, use null or an empty list

Document text:
{text}
"""


def _parse_json_safe(text: str) -> dict:
    """
    Try to extract a JSON object from Claude's response.
    Handles direct JSON, markdown code blocks, and loose JSON in text.
    """
    # 1. Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Inside ```json ... ``` block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Any JSON object in the response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def structure(extracted: ExtractedDocument) -> StructuredDocument:
    """
    Run Claude over the extracted text to pull out structured fields.
    Only sends the first 3000 characters — enough for metadata, stays cheap.
    """
    preview = extracted.raw_text[:3000]

    # If client is not available, use mock response
    if client is None:
        print("[yellow]Using mock response due to Anthropic client initialization error[/yellow]")
        parsed = {
            "doc_type": "complaint",
            "case_number": "2024-CV-00123",
            "parties": ["Harvey Specter", "Travis Tanner"],
            "dates": ["2024-01-15"],
            "key_facts": [
                "Defendant entered into a written agreement with plaintiff",
                "Defendant failed to deliver services as agreed",
                "Plaintiff seeks damages of $500,000"
            ],
            "jurisdiction": "United States District Court",
            "summary_one_line": "Breach of contract complaint seeking $500,000 in damages"
        }
    else:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(text=preview)
            }]
        )

        raw = response.content[0].text
        parsed = _parse_json_safe(raw)

    return StructuredDocument(
        source_path=extracted.source_path,
        raw_text=extracted.raw_text,
        pages=extracted.pages,
        extraction_method=extracted.extraction_method,
        page_count=extracted.page_count,
        warnings=extracted.warnings,
        doc_type=parsed.get("doc_type", "unknown"),
        case_number=parsed.get("case_number"),
        parties=parsed.get("parties", []),
        dates=parsed.get("dates", []),
        key_facts=parsed.get("key_facts", []),
        jurisdiction=parsed.get("jurisdiction"),
        summary_one_line=parsed.get("summary_one_line", ""),
    )

if __name__ == "__main__":
    import sys
    from ingest.extractor import extract
    from rich import print

    path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/sample_case.txt"
    extracted = extract(path)
    structured = structure(extracted)

    print("\n[bold]Structured fields:[/bold]")
    print(f"  doc_type        : {structured.doc_type}")
    print(f"  case_number     : {structured.case_number}")
    print(f"  parties         : {structured.parties}")
    print(f"  dates           : {structured.dates}")
    print(f"  jurisdiction    : {structured.jurisdiction}")
    print(f"  summary         : {structured.summary_one_line}")
    print(f"\n[bold]Key facts:[/bold]")
    for i, fact in enumerate(structured.key_facts, 1):
        print(f"  {i}. {fact}")