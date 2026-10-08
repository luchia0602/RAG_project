from pdf_reader import read_pdf
from embeddings import make_chunks, get_embeddings
from rag import search
from generator import generate_answer

# Replace this path with your path!
PDF_PATH = r"C:\\Users\\user\\Documents\\Studies\\PS\\RAG\\Core Rules.pdf"
text = read_pdf(PDF_PATH)
print("PDF loaded, total number ofcharacters: ", len(text))

chunks = make_chunks(
    text,
    size=800,
    overlap=100
)
print("Total number of chunks:", len(chunks))

embeddings = get_embeddings(chunks)
print("Embeddings created")

# Try changing the question if you want
question = "What are the two steps of the Fight phase?"
results = search(
    question,
    chunks,
    embeddings
)

print("QUESTION:", question)
print("RETRIEVED CHUNKS:")

for i, result in enumerate(results, 1):
    print(f"\n--- RESULT {i} ---\n")
    print(result)

context = "\n\n".join(results)

answer = generate_answer(
    question,
    context
)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)