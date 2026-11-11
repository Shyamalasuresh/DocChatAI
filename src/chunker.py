"""Text Chunker Module

Provides functionality to split a large text into smaller overlapping chunks.
Each chunk retains metadata about the source document and page number.
"""

from typing import List, Dict


def chunk_text(
    text: str,
    document_name: str,
    page_number: int,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """Split `text` into overlapping chunks.

    Args:
        text: Full text to be chunked.
        document_name: Name of the source PDF file.
        page_number: Page number the text originates from.
        chunk_size: Desired maximum length of each chunk (in characters).
        overlap: Number of characters each chunk should overlap with the previous one.

    Returns:
        A list of dictionaries, each containing:
            - ``document_name``
            - ``page_number``
            - ``chunk_id`` (sequential integer starting at 1)
            - ``text`` (the chunk content)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict[str, str]] = []
    start = 0
    chunk_id = 1
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end]
        chunks.append(
            {
                "document_name": document_name,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "text": piece,
            }
        )
        if end == text_len:
            break
        # Move start forward by chunk_size - overlap for next chunk
        start += chunk_size - overlap
        chunk_id += 1
    return chunks
