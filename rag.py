"""This script retrieves candidate chunks using embeddings, then reranks and filters them"""

import numpy as np
from sentence_transformers import CrossEncoder
from embeddings import get_embeddings

INITIAL_K = 15 # This many candidates through semantic similarity
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base" # For reranking
SCORE_MARGIN = 0.25 # Threshold for filtering reranked results
_reranker = None

def get_reranker():
    """Loads the reranker once and reuses it."""
    global _reranker
    if _reranker is None:
        print(f"Loading reranker: {RERANKER_MODEL_NAME}")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker

def search(
    question,
    chunks,
    embeddings,
    top_k=5,
    initial_k=INITIAL_K,
    score_margin=SCORE_MARGIN,
):
    """
    1. Retrieves initial_k chunks using semantic similarity;
    2. Reranks candidates using a cross-encoder;
    3. Filters weak results relative to the highest reranker score;
    4. Returns up to top_k chunks, best first.
    """
    embeddings = np.asarray(embeddings)
    question_embedding = get_embeddings([question])[0]
    similarity_scores = embeddings @ question_embedding
    candidate_count = min(initial_k, len(chunks))
    candidate_indices = np.argsort(
        similarity_scores
    )[-candidate_count:][::-1]
    candidate_chunks = [
        chunks[int(i)] for i in candidate_indices
    ]
    print(
        f"\nInitial retrieval: {len(candidate_chunks)} candidates"
    )
    # Reranking
    reranker = get_reranker()
    pairs = [
        (question, chunk)
        for chunk in candidate_chunks
    ]
    rerank_scores = np.asarray(
        reranker.predict(
            pairs,
            show_progress_bar=False
        )
    ).reshape(-1)

    # Sorting candidate positions by reranker score (highest first)
    reranked_positions = np.argsort(rerank_scores)[::-1]
    best_score = float(rerank_scores[reranked_positions[0]])
    score_threshold = best_score - score_margin
    eligible_positions = [
        int(position)
        for position in reranked_positions
        if float(rerank_scores[position]) >= score_threshold
    ]

    selected_positions = eligible_positions[:top_k]
    print(f"Best reranker score: {best_score:.4f}")
    print(f"Score threshold: {score_threshold:.4f}")
    print(
        f"Chunks passing score filter: {len(eligible_positions)}"
    )
    print(
        f"Returning {len(selected_positions)} chunks "
        "after filtering"
    )

    results = []
    for rank, position in enumerate(selected_positions, start=1):
        original_index = int(candidate_indices[position])
        chunk = chunks[original_index]
        semantic_score = float(
            similarity_scores[original_index]
        )
        rerank_score = float(rerank_scores[position])
        print(
            f"Rank: {rank}"
            f"Semantic similarity: {semantic_score:.4f}"
            f"Reranker score: {rerank_score:.4f}"
        )
        print(chunk[:500])
        results.append(chunk)

    return results
