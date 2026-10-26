"""Create text chunks and embeddings using Langchain."""

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def make_chunks(text, chunk_size=800, chunk_overlap=100):
    """
    Create chunks using Langchain's RecursiveCharacterTextSplitter.
    Chunk size and overlap are measured in characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    if not text or not text.strip():
        return []

    # Langchain splits by paragraph, then sentence, then word based on the separators list
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    print(f"Created {len(chunks)} chunks using Langchain.")
    print(f"Target chunk size: {chunk_size} characters")
    print(f"Requested overlap: {chunk_overlap} characters")
    if chunks:
        print("Actual maximum chunk length:", max(len(chunk) for chunk in chunks))

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
        show_progress_bar=True
    )