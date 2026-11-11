"""Tests for the ChromaDB vector store wrapper.

These tests use a temporary directory for the Chroma persistence layer to avoid
polluting the project workspace.
"""

import os
import shutil
import tempfile
from typing import List

import numpy as np

from src.vector_store import (
    add_embeddings,
    query_collection,
    get_collection,
    delete_document_embeddings,
)


_rng = np.random.default_rng(42)

def _random_embedding(dim: int = 5) -> List[float]:
    """Generate a deterministic random embedding for testing purposes."""
    return _rng.random(dim).tolist()


def test_add_and_query_embeddings():
    # Use a temporary directory for Chroma persistence
    temp_dir = tempfile.mkdtemp()
    try:
        collection_name = "test_collection"
        # Prepare dummy data
        ids = ["doc1_chunk1", "doc2_chunk1"]
        embeddings = [_random_embedding(), _random_embedding()]
        metadatas = [{"document_name": "doc1.pdf", "page_number": 1}, {"document_name": "doc2.pdf", "page_number": 1}]
        documents = ["Hello world", "Goodbye world"]

        # Add embeddings to the collection
        add_embeddings(
            collection_name=collection_name,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            persist_directory=temp_dir,
        )

        # Query with one of the embeddings (should retrieve the same chunk as most similar)
        results = query_collection(
            collection_name=collection_name,
            query_embeddings=[embeddings[0]],
            n_results=1,
            persist_directory=temp_dir,
        )

        assert len(results) == 1
        assert results[0]["ids"][0] == ids[0]
        assert results[0]["documents"][0] == documents[0]
        assert results[0]["metadatas"][0]["document_name"] == "doc1.pdf"
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_filter_and_delete_embeddings():
    """Verify session/document metadata filtering and document deletion."""
    temp_dir = tempfile.mkdtemp()
    try:
        collection_name = "filter_delete_test"
        emb1 = _random_embedding()
        emb2 = _random_embedding()
        add_embeddings(
            collection_name=collection_name,
            ids=["s1_doc1", "s1_doc2"],
            embeddings=[emb1, emb2],
            metadatas=[
                {"session_id": "s1", "document_name": "doc1.pdf", "page_number": 1},
                {"session_id": "s1", "document_name": "doc2.pdf", "page_number": 1},
            ],
            documents=["Alpha text", "Beta text"],
            persist_directory=temp_dir,
        )

        # Query restricted to doc2.pdf even though query embedding matches emb1
        filtered = query_collection(
            collection_name=collection_name,
            query_embeddings=[emb1],
            n_results=2,
            persist_directory=temp_dir,
            where={"$and": [{"session_id": "s1"}, {"document_name": {"$in": ["doc2.pdf"]}}]},
        )
        assert filtered[0]["ids"] == ["s1_doc2"]
        assert filtered[0]["documents"] == ["Beta text"]

        # Delete doc1.pdf and confirm collection count drops from 2 to 1
        delete_document_embeddings(
            collection_name=collection_name,
            document_name="doc1.pdf",
            session_id="s1",
            persist_directory=temp_dir,
        )
        col = get_collection(collection_name, persist_directory=temp_dir)
        assert col.count() == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
