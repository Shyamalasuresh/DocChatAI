"""Tests for the retriever module.

These tests add dummy embeddings to a temporary Chroma collection and verify that
`retrieve_chunks` returns the expected most similar chunk.
"""

import os
import shutil
import tempfile
from typing import List

import numpy as np

from src.vector_store import add_embeddings
from src.retriever import retrieve_chunks


_rng = np.random.default_rng(999)

def _random_embedding(dim: int = 5) -> List[float]:
    return _rng.random(dim).tolist()


def test_retrieve_chunks():
    temp_dir = tempfile.mkdtemp()
    try:
        collection_name = "retriever_test"
        ids = ["chunk1", "chunk2"]
        embeddings = [_random_embedding(), _random_embedding()]
        metadatas = [{"document_name": "doc1.pdf", "page_number": 1}, {"document_name": "doc2.pdf", "page_number": 2}]
        documents = ["First chunk", "Second chunk"]
        # Add embeddings
        add_embeddings(
            collection_name=collection_name,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            persist_directory=temp_dir,
        )
        # Query using first embedding
        results = retrieve_chunks(
            [embeddings[0]],
            top_k=1,
            collection_name=collection_name,
            persist_directory=temp_dir,
        )
        assert results["ids"][0] == "chunk1"
        assert results["documents"][0] == "First chunk"
        assert results["metadatas"][0]["document_name"] == "doc1.pdf"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
