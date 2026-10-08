"""This script creates embeddings for a text using BAAI's BGE model locally."""
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def make_chunks(text, size=800, overlap=100):
    """
    Split text into chunks without cutting paragraphs whenever possible.

    size: maximum approximate number of characters per chunk
    overlap: number of characters from the previous chunk to carry over
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current_chunk = ""
    for paragraph in paragraphs: 
        if len(current_chunk) + len(paragraph) <= size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(paragraph) > size: # for large paragraphs, split into smaller pieces
                start = 0
                while start < len(paragraph):
                    end = start + size
                    piece = paragraph[start:end]
                    chunks.append(piece)
                    start = end - overlap
                current_chunk = ""
            else:
                current_chunk = paragraph
    if current_chunk: # the funal chunk
        chunks.append(current_chunk)
    return chunks

def get_embeddings(texts):
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings