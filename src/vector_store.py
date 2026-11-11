"""Vector Store Module using ChromaDB.

Provides functions to create a Chroma collection (with optional persistence),
add embeddings with metadata, and query for similar chunks.
"""

from typing import List, Dict, Any, Optional
import chromadb

# Cache of Chroma clients per persistence directory to avoid recreating
_clients: Dict[Optional[str], Any] = {}


def _get_client(persist_directory: Optional[str] = None) -> Any:
    """Return a Chroma client.

    If ``persist_directory`` is provided, Chroma will store data there;
    otherwise an in‑memory client is used.
    """
    key = persist_directory or None
    if key not in _clients:
        if persist_directory:
            _clients[key] = chromadb.PersistentClient(path=persist_directory)
        else:
            _clients[key] = chromadb.Client()
    return _clients[key]


def get_collection(name: str = "docuchat", persist_directory: Optional[str] = None) -> Any:
    """Get (or create) a collection for storing document chunks.

    ``persist_directory`` is forwarded to the underlying Chroma client.
    """
    client = _get_client(persist_directory)
    return client.get_or_create_collection(name=name)


def add_embeddings(
    collection_name: str,
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    documents: List[str],
    persist_directory: Optional[str] = None,
) -> None:
    """Add a batch of embeddings to the specified collection.

    The function validates input lengths and stores data in the collection
    identified by ``collection_name``.
    """
    if not (len(ids) == len(embeddings) == len(metadatas) == len(documents)):
        raise ValueError("All input lists must have the same length.")
    collection = get_collection(collection_name, persist_directory)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )


def query_collection(
    collection_name: str,
    query_embeddings: List[List[float]],
    n_results: int = 5,
    persist_directory: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve the top‑k most similar chunks for each query embedding.

    Returns a list (one per query) of dictionaries containing ``ids``, ``documents``,
    ``metadatas`` and ``distances``.
    """
    collection = get_collection(collection_name, persist_directory)
    if collection.count() == 0:
        return [
            {"ids": [], "documents": [], "metadatas": [], "distances": []}
            for _ in query_embeddings
        ]

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": query_embeddings,
        "n_results": min(n_results, collection.count()),
        "include": ["metadatas", "documents", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)
    # Format raw Chroma dict-of-lists into a list of dicts (one dict per query embedding)
    formatted = []
    num_queries = len(query_embeddings)
    for i in range(num_queries):
        formatted.append({
            "ids": results["ids"][i] if results.get("ids") else [],
            "documents": results["documents"][i] if results.get("documents") else [],
            "metadatas": results["metadatas"][i] if results.get("metadatas") else [],
            "distances": results["distances"][i] if results.get("distances") else [],
        })
    return formatted


def delete_document_embeddings(
    collection_name: str,
    document_name: str,
    session_id: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> None:
    """Delete all stored chunks belonging to ``document_name`` (and ``session_id`` if given)."""
    collection = get_collection(collection_name, persist_directory)
    if collection.count() == 0:
        return
    if session_id:
        where_filter: Dict[str, Any] = {
            "$and": [{"session_id": session_id}, {"document_name": document_name}]
        }
    else:
        where_filter = {"document_name": document_name}
    collection.delete(where=where_filter)
