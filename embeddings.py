"""Creates text chunks and normalized embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL_NAME

_model = None

def get_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def make_chunks(text, chunk_size=800, chunk_overlap=100):
    """Split text into overlapping chunks measured in characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller "
            "than chunk_size."
        )
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    print(f"Created {len(chunks)} chunks.")
    print(f"Target chunk size: {chunk_size} characters")
    print(f"Requested overlap: {chunk_overlap} characters")
    return chunks

def get_embeddings(texts):
    """Create normalized semantic embeddings for a list of texts."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )