from pdf_reader import read_pdf
from embeddings import make_chunks, get_embeddings
from rag import search

# Replace this path with your path!
PDF_PATH = r"C:\...\Core Rules.pdf"
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
question = "What happens during the Fight phase?"
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