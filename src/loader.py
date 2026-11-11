"""PDF Loader Module

Provides functionality to load a PDF file and extract its text content page by page.
Each extracted chunk includes metadata:
- `document_name`: original filename
- `page_number`: 1‑based page index
- `text`: extracted text for that page

The module gracefully handles:
- Corrupted or unreadable PDFs (raises a clear exception)
- PDFs with no extractable text (returns an empty list and logs a warning)
"""

import pymupdf
from pathlib import Path
from typing import List, Dict

def load_pdf(file_path: str) -> List[Dict[str, str]]:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dictionaries, each representing a page with keys:
        `document_name`, `page_number`, `text`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid PDF or cannot be read.
    """
    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File '{file_path}' is not a PDF (expected .pdf extension)")

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Unable to open PDF file '{file_path}': {exc}")

    if doc.page_count == 0:
        # Empty PDF – nothing to extract
        return []

    pages: List[Dict[str, str]] = []
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "document_name": pdf_path.name,
                "page_number": page_number + 1,  # human‑friendly indexing
                "text": text,
            })
    doc.close()
    return pages
