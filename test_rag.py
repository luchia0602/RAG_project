
from cache import get_rag_data
from rag import search
from generator import generate_answer
from dotenv import load_dotenv
load_dotenv()  # Reads the .env file and sets the environment variables

PDF_PATH = r"C:\Users\user\Documents\Studies\PS\RAG\Core Rules.pdf"
INITIAL_K = 15
TOP_K = 5

def main():
    print("Loading knowledge base...")

    text, chunks, embeddings = get_rag_data(
        pdf_path=PDF_PATH,
        chunk_size=800,
        overlap=100
    )

    print("\nKnowledge base ready.")
    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # Retrieve 15 candidates, rerank them, and keep the best 5.
    results = search(
        question,
        chunks,
        embeddings,
        top_k=TOP_K,
        initial_k=INITIAL_K
    )

    context = "\n\n".join(results)

    print("\nGenerating answer...")
    answer = generate_answer(question, context)

    print("\n--- GENERATED ANSWER ---")
    print(answer)

    print("\n--- FINAL RETRIEVED CONTEXT ---")
    print(context)

if __name__ == "__main__":
    main()
