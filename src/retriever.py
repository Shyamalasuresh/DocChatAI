"""Retriever module for DocuChat AI.

Provides a simple interface to query the ChromaDB collection created in Phase 4
and return the top‑k most similar document chunks.
"""

from typing import List, Dict, Any, Optional

from .vector_store import query_collection, get_collection

DEFAULT_COLLECTION = "docuchat"


def retrieve_chunks(
    query_embeddings: List[List[float]],
    top_k: int = 5,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the top‑k most similar chunks for the given query embeddings.

    Args:
        query_embeddings: List of embedding vectors (usually length 1).
        top_k: Number of results to return.
        collection_name: Name of the Chroma collection.
        persist_directory: Optional path to Chroma persistence directory.
        where: Optional ChromaDB metadata filter.
    Returns:
        A dictionary containing `ids`, `documents`, `metadatas`,
        and `distances` for a single query.
    """
    results = query_collection(
        collection_name=collection_name,
        query_embeddings=query_embeddings,
        n_results=top_k,
        persist_directory=persist_directory,
        where=where,
    )
    # `results` is a list with one entry per query embedding; return the first entry.
    return results[0] if results else {}
