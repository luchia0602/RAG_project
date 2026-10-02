"""This script computes the similarity between question and chunks, returns 3 best chunks"""
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import get_embeddings

def search(question, chunks, embeddings):
    question_embedding = get_embeddings(
        [question]
    )
    scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]
    top = scores.argsort()[-3:][::-1]
    return [
        chunks[i]
        for i in top
    ]