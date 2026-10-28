
"""Central configuration for the RAG application."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Absolute path to the repository root, independent of the terminal's directory.
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the repository's .env file.
load_dotenv(BASE_DIR / ".env")

# Paths
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"

# PDF path can be overridden in .env. The default is data/Core Rules.pdf.
PDF_PATH = Path(
    os.getenv("PDF_PATH", "data/Core Rules.pdf")
).expanduser()

if not PDF_PATH.is_absolute():
    PDF_PATH = (BASE_DIR / PDF_PATH).resolve()

# Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Models
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-small-en-v1.5",
)

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-base",
)

GENERATION_MODELS = [
    model.strip()
    for model in os.getenv(
        "GENERATION_MODELS",
        "gemini-2.5-flash,gemini-2.5-flash-lite",
    ).split(",")
    if model.strip()
]

# Retrieval and chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
INITIAL_K = int(os.getenv("INITIAL_K", "15"))
TOP_K = int(os.getenv("TOP_K", "5"))
SCORE_MARGIN = float(os.getenv("SCORE_MARGIN", "0.25"))

# Validate configuration early.
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than zero.")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE."
    )

if INITIAL_K <= 0 or TOP_K <= 0:
    raise ValueError("INITIAL_K and TOP_K must be greater than zero.")

if SCORE_MARGIN < 0:
    raise ValueError("SCORE_MARGIN must be non-negative.")