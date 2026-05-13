import fitz  # pymupdf
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import re
from pathlib import Path
from dataclasses import dataclass, field
from rich.console import Console

console = Console()


@dataclass
class ExtractedDocument:
    source_path: str
    raw_text: str
    pages: list[str]
    extraction_method: str  # "native" | "ocr" | "mixed"
    page_count: int
    warnings: list[str] = field(default_factory=list)


def _is_text_usable(text: str, min_chars: int = 100) -> bool:
    """Native PDF text is usable if there's enough non-whitespace content."""
    return len(re.sub(r'\s+', ' ', text).strip()) >= min_chars


def _extract_native(pdf_path: str) -> tuple[list[str], bool]:
    """Pull text directly from PDF text layer."""
    doc = fitz.open(pdf_path)
    pages = []
    all_usable = True

    for page in doc:
        text = page.get_text("text")
        pages.append(text)
        if not _is_text_usable(text):
            all_usable = False

    doc.close()
    return pages, all_usable


def _extract_ocr(pdf_path: str) -> list[str]:
    """Convert PDF pages to images and OCR each one."""
    console.log(f"[yellow]Running OCR on {pdf_path}[/yellow]")
    images = convert_from_path(pdf_path, dpi=300)
    pages = []

    for i, img in enumerate(images):
        gray = img.convert("L")  # grayscale improves Tesseract accuracy
        text = pytesseract.image_to_string(gray, config="--psm 6")
        pages.append(text)
        console.log(f"  OCR page {i+1}/{len(images)}")

    return pages


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip OCR noise."""
    text = re.sub(r'\f', '\n\n', text)           # form feeds → paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)           # collapse spaces and tabs
    text = re.sub(r'\n{3,}', '\n\n', text)        # max 2 consecutive newlines
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # strip non-ASCII garbage
    return text.strip()


def extract(file_path: str) -> ExtractedDocument:
    """
    Extract text from a PDF, image, or .txt file.
    Strategy:
    - PDFs: try native text layer first, fall back to OCR per-page if sparse
    - Images: go straight to OCR
    - .txt: read directly
    """
    path = Path(file_path)
    warnings = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        img = Image.open(file_path).convert("L")
        text = pytesseract.image_to_string(img, config="--psm 6")
        pages = [text]
        method = "ocr"

    elif suffix == ".pdf":
        native_pages, all_usable = _extract_native(file_path)

        if all_usable:
            pages = native_pages
            method = "native"
        else:
            ocr_pages = _extract_ocr(file_path)
            pages = []
            method = "mixed"

            for i, (native, ocr) in enumerate(zip(native_pages, ocr_pages)):
                if _is_text_usable(native):
                    pages.append(native)
                else:
                    pages.append(ocr)
                    warnings.append(f"Page {i+1} used OCR fallback")

    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        pages = [text]
        method = "native"

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    cleaned_pages = [_clean_text(p) for p in pages]
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(cleaned_pages)

    if warnings:
        console.log(f"[yellow]Warnings: {warnings}[/yellow]")

    console.log(f"[green]Extracted {len(cleaned_pages)} page(s) via '{method}'[/green]")

    return ExtractedDocument(
        source_path=str(path.resolve()),
        raw_text=full_text,
        pages=cleaned_pages,
        extraction_method=method,
        page_count=len(cleaned_pages),
        warnings=warnings,
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.extractor <path_to_file>")
        sys.exit(1)

    result = extract(sys.argv[1])
    print(f"\nPages     : {result.page_count}")
    print(f"Method    : {result.extraction_method}")
    print(f"Warnings  : {result.warnings}")
    print(f"\n--- First 500 chars ---\n{result.raw_text[:500]}")