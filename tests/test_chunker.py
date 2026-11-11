import sys, os
sys.path.append(os.path.abspath('.'))
import pytest
from src.chunker import chunk_text


def test_chunk_basic():
    text = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 chars
    chunks = chunk_text(text, document_name="sample.pdf", page_number=1, chunk_size=200, overlap=50)
    # Expect ceil((len - overlap) / (chunk_size - overlap)) chunks
    # Calculation: first chunk 0-200, second starts at 150 -> 150-350, third starts at 300 -> 300-500, fourth starts at 450 -> 450-520
    assert len(chunks) == 4
    # Verify overlap content
    assert chunks[1]["text"].startswith(chunks[0]["text"][-50:])
    assert chunks[2]["text"].startswith(chunks[1]["text"][-50:])
    assert chunks[3]["text"].startswith(chunks[2]["text"][-50:])
    # Verify metadata
    for i, chunk in enumerate(chunks, start=1):
        assert chunk["document_name"] == "sample.pdf"
        assert chunk["page_number"] == 1
        assert chunk["chunk_id"] == i


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("text", "doc.pdf", 1, chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("text", "doc.pdf", 1, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("text", "doc.pdf", 1, chunk_size=100, overlap=100)


def test_no_redundant_trailing_chunk():
    """Short text fitting within a single chunk should return exactly 1 chunk."""
    text = "a" * 180
    chunks = chunk_text(text, "short.pdf", 1, chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
