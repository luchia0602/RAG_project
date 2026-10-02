"""This script reads a PDF file and extracts its text content."""
from pypdf import PdfReader
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text