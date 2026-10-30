
# Context-Aware RAG Assistant

Liudmila SHLYAKHTINA, Tori BARAL

A retrieval-augmented generation (RAG) system that answers
questions using information retrieved from a user-provided
PDF and, where configured, structured database content.

## Features

- PDF text extraction and cleaning
- Recursive text chunking with configurable overlap
- Semantic retrieval using sentence embeddings
- Cross-encoder reranking
- Context-grounded answer generation using the Gemini API
- Configurable models and retrieval parameters
- Local caching of processed documents and embeddings

## How It Works

1. Extract text from the PDF.
2. Split the text into overlapping chunks.
3. Generate and cache normalized embeddings.
4. Retrieve candidate chunks using semantic similarity.
5. Rerank the candidates using a cross-encoder.
6. Generate an answer using the retrieved context.

## Requirements

- Python 3.10 or a compatible Python version
- A Gemini API key
- Internet access for API calls and initial model downloads

## Installation

Clone the repository and enter its directory.

Create and activate a virtual environment:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

## Configuration

Copy the example environment file:

    Copy-Item .env.example .env

Set your own GOOGLE_API_KEY and PDF_PATH in `.env`.

For example:

    GOOGLE_API_KEY=your_api_key
    PDF_PATH=data/my_document.pdf

Place your PDF at the configured path.

## Run

    python router_graph.py

Enter your question when prompted.

## Limitations

Answers depend on the information successfully extracted
and retrieved from the source documents. The system may
not answer questions when the relevant information is
missing from the retrieved context.

## Implementation details

Please find a detailed description of all the scripts in this repository below.

    api.py
    
This script handles the configuration and retrieval of the Google API key required to access Google's Gemini API. It loads environment variables from a local .env file, keeping sensitive credentials separate from the source code. It ensures that values defined in the .env file take precedence over existing environment variables. This approach avoids hardcoding sensitive credentials in the code and improves portability across environments.
