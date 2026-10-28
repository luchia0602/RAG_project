"""Saves and loads extracted PDF text, chunks, and embeddings."""

import hashlib
import json
from pathlib import Path

import numpy as np

from config import CACHE_DIR, EMBEDDING_MODEL_NAME
from embeddings import make_chunks, get_embeddings
from pdf_reader import read_pdf

PROCESSING_VERSION = 4
EMBEDDING_DIMENSION = 384


def get_pdf_hash(pdf_path):
    """Return a content hash so PDF identity doesn't depend on its path."""
    digest = hashlib.sha256()

    with open(pdf_path, "rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def get_rag_data(
    pdf_path,
    cache_dir=None,
    chunk_size=800,
    overlap=100,
    force_rebuild=False,
):
    """Load a compatible saved cache or rebuild it from the PDF."""

    pdf_path = Path(pdf_path).expanduser().resolve() if pdf_path else None

    if cache_dir is None:
        cache_dir = CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    text_file = cache_dir / "parsed_text.txt"
    chunks_file = cache_dir / "chunks.json"
    embeddings_file = cache_dir / "embeddings.npy"
    metadata_file = cache_dir / "metadata.json"

    cache_files = [
        text_file,
        chunks_file,
        embeddings_file,
        metadata_file,
    ]

    pdf_exists = pdf_path is not None and pdf_path.is_file()
    current_pdf_hash = get_pdf_hash(pdf_path) if pdf_exists else None

    # Try the saved cache first.
    if not force_rebuild and all(path.is_file() for path in cache_files):
        try:
            saved_metadata = json.loads(
                metadata_file.read_text(encoding="utf-8")
            )

            compatible = (
                saved_metadata.get("chunk_size") == chunk_size
                and saved_metadata.get("overlap") == overlap
                and saved_metadata.get("embedding_model")
                == EMBEDDING_MODEL_NAME
                and saved_metadata.get("processing_version")
                == PROCESSING_VERSION
            )

            # If a PDF is available, make sure the cache belongs to it.
            # If no PDF is available, use the saved cache as a snapshot.
            if pdf_exists:
                compatible = compatible and (
                    saved_metadata.get("pdf_sha256") == current_pdf_hash
                )

            if compatible:
                text = text_file.read_text(encoding="utf-8")
                chunks = json.loads(
                    chunks_file.read_text(encoding="utf-8")
                )
                embeddings = np.load(embeddings_file)

                valid_cache = (
                    isinstance(chunks, list)
                    and len(chunks) > 0
                    and embeddings.ndim == 2
                    and embeddings.shape
                    == (len(chunks), EMBEDDING_DIMENSION)
                )

                if valid_cache:
                    print("Loaded saved PDF cache.")
                    print("Characters:", len(text))
                    print("Chunks:", len(chunks))
                    print("Embeddings shape:", embeddings.shape)
                    return text, chunks, embeddings

            print("Cache is incompatible or invalid; rebuilding if possible.")

        except (OSError, ValueError, json.JSONDecodeError):
            print("Cache is invalid or incomplete; rebuilding if possible.")

    elif force_rebuild:
        print("Force rebuild requested.")
    else:
        print("Saved cache not found or incomplete.")

    # Rebuilding requires the original PDF.
    if not pdf_exists:
        raise FileNotFoundError(
            f"No usable cache was found in '{cache_dir}', "
            "and the source PDF is unavailable. "
            "Provide the PDF or restore the repository's cache files."
        )

    print("Reading PDF...")
    text = read_pdf(str(pdf_path))

    print("Creating chunks...")
    chunks = make_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    print("Creating embeddings...")
    embeddings = get_embeddings(chunks)

    metadata = {
        "pdf_sha256": current_pdf_hash,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "processing_version": PROCESSING_VERSION,
    }

    text_file.write_text(text, encoding="utf-8")

    chunks_file.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    np.save(embeddings_file, embeddings)

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Cache created successfully.")
    print("Characters:", len(text))
    print("Chunks:", len(chunks))
    print("Embeddings shape:", embeddings.shape)
    return text, chunks, embeddings