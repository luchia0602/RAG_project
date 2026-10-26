"""This script saves and loads extracted PDF text, chunks, and embeddings so we don't have to recompute it every time"""

import json
from pathlib import Path
import numpy as np
from pdf_reader import read_pdf
from embeddings import make_chunks, get_embeddings, MODEL_NAME

PROCESSING_VERSION = 3

def get_rag_data(
    pdf_path,
    cache_dir="cache",
    chunk_size=800,
    overlap=100,
    force_rebuild=False
):
    """Loads a valid cache or rebuilds it from the PDF"""

    pdf_path = Path(pdf_path).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_file = cache_dir / "parsed_text.txt"
    chunks_file = cache_dir / "chunks.json"
    embeddings_file = cache_dir / "embeddings.npy"
    metadata_file = cache_dir / "metadata.json"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    stat = pdf_path.stat()
    current_metadata = {
        "pdf_path": str(pdf_path),
        "pdf_size": stat.st_size,
        "pdf_modified": stat.st_mtime,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": MODEL_NAME,
        "processing_version": PROCESSING_VERSION
    }
    cache_files = [
        text_file,
        chunks_file,
        embeddings_file,
        metadata_file
    ]

    if not force_rebuild and all(
        path.exists() for path in cache_files
    ):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                saved_metadata = json.load(f)
            if saved_metadata == current_metadata:
                text = text_file.read_text(encoding="utf-8")
                with open(chunks_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                embeddings = np.load(embeddings_file)
                valid_cache = ( # Check if cache is valid
                    isinstance(chunks, list)
                    and len(chunks) > 0
                    and embeddings.ndim == 2
                    and embeddings.shape[0] == len(chunks)
                    and embeddings.shape[1] == 384
                )
                if valid_cache:
                    print("Loaded saved PDF.")
                    print("Characters:", len(text))
                    print("Chunks:", len(chunks))
                    print("Embeddings shape:", embeddings.shape)
                    return text, chunks, embeddings
                print("Rebuilding PDF...")
            else:
                print("Rebuilding PDF...")

        except (OSError, ValueError, json.JSONDecodeError):
            print("Cache is invalid or incomplete. Rebuilding...")

    elif force_rebuild:
        print("Force rebuild")
    else:
        print("Cache not found")

    print(f"Reading PDF...")
    text = read_pdf(str(pdf_path))

    print("Creating chunks...")
    chunks = make_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    print("Creating embeddings...")
    embeddings = get_embeddings(chunks)
    text_file.write_text(text, encoding="utf-8")

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(embeddings_file, embeddings)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(current_metadata, f, indent=2)

    print("Cache created successfully.")
    print("Characters:", len(text))
    print("Chunks:", len(chunks))
    print("Embeddings shape:", embeddings.shape)
    return text, chunks, embeddings