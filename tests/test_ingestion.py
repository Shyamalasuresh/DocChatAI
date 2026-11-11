"""Tests for the document ingestion pipeline and Phase 8 upload validation."""

import pymupdf

from src.ingestion import validate_pdf_upload, ingest_pdf_files
from src.vector_store import get_collection


def _make_pdf_bytes(text: str = "Sample PDF content for DocuChat AI.") -> bytes:
    """Create a valid 1-page PDF in memory and return its bytes."""
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_validate_pdf_upload():
    valid_bytes = _make_pdf_bytes()
    assert validate_pdf_upload("notes.pdf", valid_bytes) == (True, None)

    # Reject non-PDF extension
    ok, err = validate_pdf_upload("notes.txt", valid_bytes)
    assert not ok and "only `.pdf`" in err

    # Reject empty file
    ok, err = validate_pdf_upload("empty.pdf", b"")
    assert not ok and "empty" in err

    # Reject oversized file
    ok, err = validate_pdf_upload("huge.pdf", valid_bytes, max_size_bytes=10)
    assert not ok and "exceeds" in err

    # Reject disguised non-PDF content (missing %PDF- signature)
    ok, err = validate_pdf_upload("fake.pdf", b"This is plain text, not a PDF")
    assert not ok and "not a valid PDF" in err


def test_ingest_pdf_files_and_duplicate_skip(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    pdf_bytes = _make_pdf_bytes("Procure to Pay involves Purchase Orders and Invoices.")

    result = ingest_pdf_files(
        files=[("p2p_guide.pdf", pdf_bytes)],
        session_id="sess_test_1",
        existing_docs=set(),
        collection_name="ingestion_test",
        persist_directory=persist_dir,
        chunk_size=200,
        overlap=20,
    )

    assert "p2p_guide.pdf" in result["indexed"]
    assert result["indexed"]["p2p_guide.pdf"]["pages"] == 1
    assert result["total_chunks"] >= 1
    assert result["errors"] == []

    col = get_collection("ingestion_test", persist_directory=persist_dir)
    assert col.count() == result["total_chunks"]

    # Re-uploading same filename should be skipped
    dup_result = ingest_pdf_files(
        files=[("p2p_guide.pdf", pdf_bytes)],
        session_id="sess_test_1",
        existing_docs={"p2p_guide.pdf"},
        collection_name="ingestion_test",
        persist_directory=persist_dir,
    )
    assert dup_result["total_chunks"] == 0
    assert any("already indexed" in w for w in dup_result["warnings"])


def test_ingest_blank_pdf_warns(tmp_path):
    persist_dir = str(tmp_path / "chroma_blank")
    blank_pdf = _make_pdf_bytes(text="")

    result = ingest_pdf_files(
        files=[("scanned.pdf", blank_pdf)],
        session_id="sess_blank",
        collection_name="blank_test",
        persist_directory=persist_dir,
    )
    assert result["indexed"] == {}
    assert any("No extractable text" in w for w in result["warnings"])


def test_path_traversal_filename_sanitized(tmp_path):
    persist_dir = str(tmp_path / "chroma_sec")
    pdf_bytes = _make_pdf_bytes("Safe content")

    result = ingest_pdf_files(
        files=[("../../secret.pdf", pdf_bytes)],
        session_id="sess_sec",
        collection_name="sec_test",
        persist_directory=persist_dir,
    )
    assert "secret.pdf" in result["indexed"]
    assert "../../secret.pdf" not in result["indexed"]

