
"""Run a single question through the RAG pipeline."""

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INITIAL_K,
    PDF_PATH,
    TOP_K,
)
from cache import get_rag_data
from rag import search
from generator import generate_answer

def main():
    print("Loading knowledge base...")

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}\n"
            "Set PDF_PATH in your .env file to the location of your PDF."
        )

    text, chunks, embeddings = get_rag_data(
        pdf_path=PDF_PATH,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError(
            "No text chunks were created. "
            "Check whether the PDF contains extractable text."
        )

    print("\nKnowledge base ready.")
    print("PDF:", PDF_PATH.name)
    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    results = search(
        question,
        chunks,
        embeddings,
        top_k=TOP_K,
        initial_k=INITIAL_K,
    )

    if not results:
        print("No relevant context was retrieved.")
        return

    context = "\n\n".join(results)

    print("\nGenerating answer...")
    answer = generate_answer(question, context)

    print("\nGENERATED ANSWER:")
    print(answer)

    print("\nFINAL RETRIEVED CONTEXT:")
    print(context)


if __name__ == "__main__":
    main()