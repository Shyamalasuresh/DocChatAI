import sys, os
sys.path.append(os.path.abspath('.'))
import pytest
from src.embeddings import embed_texts


def test_embed_basic():
    texts = ["Hello world", "DocuChat AI"]
    vectors = embed_texts(texts)
    assert len(vectors) == 2
    # MiniLM model produces 384‑dimensional vectors
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_embed_empty():
    assert embed_texts([]) == []
