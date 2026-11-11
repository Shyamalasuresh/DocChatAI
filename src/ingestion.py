"""Document Ingestion Pipeline Module.

Coordinates validation, PDF text extraction, chunking, embedding generation,
and persistent vector storage in ChromaDB with session-based isolation.
"""

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from .loader import load_pdf
from .chunker import chunk_text
from .embeddings import embed_texts
from .vector_store import add_embeddings

DEFAULT_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def validate_pdf_upload(
    filename: str,
    file_bytes: bytes,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> Tuple[bool, Optional[str]]:
    """Validate an uploaded PDF's name, size, and magic header bytes.

    Returns:
        (is_valid, error_message_or_none)
    """
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        return False, f"Skipped `{safe_name}` — only `.pdf` files are supported."

    size = len(file_bytes)
    if size == 0:
        return False, f"Skipped `{safe_name}` — file is empty."

    if size > max_size_bytes:
        limit_mb = max_size_bytes // (1024 * 1024)
        return False, f"Skipped `{safe_name}` — exceeds {limit_mb} MB limit."

    # Check actual PDF magic signature rather than trusting filename extension alone
    if not file_bytes.lstrip().startswith(b"%PDF-"):
        return False, f"Skipped `{safe_name}` — file content is not a valid PDF."

    return True, None


def ingest_pdf_files(
    files: Iterable[Tuple[str, bytes]],
    session_id: str,
    existing_docs: Optional[Set[str]] = None,
    collection_name: str = "docuchat",
    persist_directory: Optional[str] = "chroma_db",
    chunk_size: int = 500,
    overlap: int = 50,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Validate, extract, chunk, embed, and store uploaded PDFs in ChromaDB.

    Args:
        files: Iterable of `(filename, raw_bytes)` tuples.
        session_id: Session identifier used to isolate documents per session.
        existing_docs: Set of filenames already indexed (to prevent duplicates).
        collection_name: Target ChromaDB collection name.
        persist_directory: Directory where ChromaDB persists data (or None for in-memory).
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between adjacent chunks.
        max_size_bytes: Maximum allowed file size in bytes.
        on_progress: Optional callback receiving status messages for UI display.

    Returns:
        Summary dictionary with keys ``indexed``, ``warnings``, ``errors``, ``total_chunks``.
    """
    existing = existing_docs or set()
    all_chunks: List[Dict[str, Any]] = []
    indexed: Dict[str, Dict[str, int]] = {}
    warnings: List[str] = []
    errors: List[str] = []

    def _notify(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for raw_name, file_bytes in files:
            safe_name = Path(raw_name).name

            if safe_name in existing or safe_name in indexed:
                warnings.append(
                    f"Skipped `{safe_name}` — already indexed in this session."
                )
                continue

            is_valid, err_msg = validate_pdf_upload(
                safe_name, file_bytes, max_size_bytes=max_size_bytes
            )
            if not is_valid:
                if err_msg and "empty" in err_msg:
                    warnings.append(err_msg)
                elif err_msg:
                    errors.append(err_msg)
                continue

            _notify(f"📖 Extracting text from `{safe_name}`...")
            temp_pdf_path = Path(tmp_dir) / safe_name
            temp_pdf_path.write_bytes(file_bytes)

            try:
                pages = load_pdf(str(temp_pdf_path))
            except Exception as exc:
                errors.append(f"Failed to read `{safe_name}`: {exc}")
                continue

            if not pages:
                warnings.append(
                    f"No extractable text found in `{safe_name}` "
                    "(it may be a scanned/image-only PDF)."
                )
                continue

            doc_chunks: List[Dict[str, Any]] = []
            for page in pages:
                chunks = chunk_text(
                    page["text"],
                    document_name=safe_name,
                    page_number=page["page_number"],
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                doc_chunks.extend(chunks)

            if not doc_chunks:
                warnings.append(f"No text chunks created for `{safe_name}`.")
                continue

            all_chunks.extend(doc_chunks)
            indexed[safe_name] = {
                "pages": len(pages),
                "chunks": len(doc_chunks),
            }

    if all_chunks:
        _notify(f"🧠 Generating embeddings for {len(all_chunks)} chunks...")
        docs = [c["text"] for c in all_chunks]
        embeddings = embed_texts(docs)

        ids = [
            f"{session_id}_{c['document_name']}_p{c['page_number']}_c{idx + 1}"
            for idx, c in enumerate(all_chunks)
        ]
        metadatas = [
            {
                "session_id": session_id,
                "document_name": c["document_name"],
                "page_number": c["page_number"],
            }
            for c in all_chunks
        ]

        _notify("💾 Saving vectors to ChromaDB...")
        add_embeddings(
            collection_name=collection_name,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=docs,
            persist_directory=persist_directory,
        )

    return {
        "indexed": indexed,
        "warnings": warnings,
        "errors": errors,
        "total_chunks": len(all_chunks),
    }

