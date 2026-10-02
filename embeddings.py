"""This script creates embeddings for a text using BAAI's BGE model locally."""
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def make_chunks(text, size=800, overlap=100):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def get_embeddings(texts):
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings