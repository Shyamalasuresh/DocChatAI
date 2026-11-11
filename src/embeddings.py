"""Embeddings Module

Provides a simple wrapper around a Sentence‑Transformers model to generate
vector embeddings for a list of text chunks.
"""

from typing import List, Optional, Any

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[Any] = None


def get_model() -> Any:
    """Lazy-load and cache the SentenceTransformer model (local cache first)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a list of embedding vectors for the given texts.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (list of floats) matching the input order.
    """
    if not texts:
        return []
    model = get_model()
    # encode returns a numpy array; convert to list of lists for JSON‑serializable output
    return model.encode(texts, show_progress_bar=False).tolist()
