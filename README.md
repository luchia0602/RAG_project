
# Context-Aware RAG Assistant

Liudmila SHLYAKHTINA, Tori BARAL

A retrieval-augmented generation (RAG) system that answers
questions using information retrieved from a user-provided
PDF and structured database content.

## Features

- PDF text extraction and cleaning
- Recursive text chunking with configurable overlap
- Semantic retrieval using sentence embeddings
- Cross-encoder reranking
- Structured data retrieval from a local SQLite database
- Natural-language-to-SQL query generation using Gemini
- Dynamic database schema inspection and SQL validation
- Read-only SQL execution with query timeouts and result limits
- Combined retrieval from PDF rules and structured unit data
- Context-grounded answer generation using the Gemini API
- Configurable models and retrieval parameters
- Local caching of processed documents and embeddings

## How it works

1. Extract and clean text from the PDF.
2. Split the text into overlapping chunks.
3. Generate and cache normalized embeddings.
4. Retrieve candidate chunks using semantic similarity.
5. Rerank the candidates using a cross-encoder.
6. Use a Gemini-powered rerouter within a LangGraph workflow to analyze the user's question and decide whether to retrieve information from the PDF, the SQLite database, or both.
7. For database queries, inspect the schema and use Gemini to translate the natural-language question into SQL.
8. Validate and execute the generated SQL using read-only database access.
9. Combine the retrieved PDF passages and structured database results into a single context.
10. Generate a context-grounded answer using Gemini.

## Requirements

- Python 3.10 or a compatible Python version
- A Gemini API key
- Internet access for API calls and initial model downloads

## Installation

Clone the repository and enter its directory. Create and activate a virtual environment:

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

Type `exit` to stop.

You may find example questions in eval_questions.json testing both PDF and SQL understanding.

## Limitations

Answers depend on the information successfully extracted
and retrieved from the source documents. The system may
not answer questions when the relevant information is
missing from the retrieved context.

## Implementation details

Please find a detailed description of all the scripts in this repository below.

    api_check.py (debugging)
    
This script handles the configuration and retrieval of the Google API key required to access Google's Gemini API. It loads environment variables from a local .env file, keeping sensitive credentials separate from the source code. It ensures that values defined in the .env file take precedence over existing environment variables. This approach avoids hardcoding sensitive credentials in the code and improves portability across environments.

    cache.py

This script manages the storage and retrieval of preprocessed PDF data to avoid repeating expensive preprocessing operations every time the RAG pipeline runs. An important implementation choice is storing embeddings in NumPy .npy format and metadata separately in JSON, allowing efficient loading and validation while also making it possible to reuse a previously generated cache even when the original PDF is unavailable.

    config.py

This script centralizes the application's configuration, including file paths, API credentials, model names, and RAG parameters. 

    embeddings.py

This script handles text segmentation and embedding generation for the RAG pipeline. It uses LangChain's RecursiveCharacterTextSplitter to divide the extracted PDF text into chunks of 800 characters with 100-character overlaps, prioritizing paragraph and sentence boundaries to preserve contextual coherence. It uses the BAAI/bge-small-en-v1.5 model to convert each chunk into a 384-dimensional semantic vector, with L2 normalization enabled to facilitate similarity comparisons. Here we use the lazy model loading with a global singleton to ensure the embedding model is loaded only once and reused throughout execution, reducing unnecessary loading time and memory overhead.

    evaluate.py

This script implements an evaluation pipeline for the RAG application using the RAGAS framework with four dimensions: faithfulness, answer relevancy, context precision, and context recall. It uses a small sample of questions with predefined ground-truth answers, runs them through the retrieval and generation pipeline, and evaluates the resulting answers using Gemini as the evaluator LLM and Google's embedding model. Due to API token and quota limitations, we were unable to complete the full evaluation reliably; however, we implemented the evaluation pipeline and attempted to run it on a small sample of three questions.

    generator.py

This script handles answer generation using Google's Gemini API, taking the user's question and the context retrieved by the RAG pipeline as input. It uses a carefully designed prompt that restricts the model to information explicitly present in the provided context, prohibits unsupported inferences, and specifies an exact fallback response when insufficient information is available, helping reduce hallucinations. The implementation includes lazy client initialization, configurable model selection with automatic fallback if a model fails, API retries and a 60-second timeout; generation uses a low temperature (0.2) to encourage consistent, focused answers. The output is limited to 4,096 tokens, and the script also checks for empty or truncated responses to handle incomplete generation.

    models_check.py (debugging)

This script retrieves and displays the Gemini models available through the configured Google API key using the google-genai SDK. This was used to verify API access and identify available generation models, helping select compatible models for the RAG application's answer-generation component.

    pdf_reader.py

This script extracts and cleans text from the Warhammer Core Rules PDF using the pypdf library, processing the document page by page to preserve its original structure. It normalizes whitespace and line endings, removes null characters, and filters out repeated layout elements such as page numbers and running headers using frequency-based detection across pages. An important implementation choice is preserving page boundaries with form-feed characters, allowing subsequent chunking to distinguish between pages rather than accidentally merging unrelated text across page boundaries.

    rag.py (legacy)

The initial implementation with the PDF reader (not used anymore, replaced by router_graph.py). First, it calculates cosine similarity between the question embedding and precomputed document embeddings to select the 15 most relevant chunks; then, it uses the BAAI/bge-reranker-base cross-encoder to reassess and reorder these candidates based on their relevance to the question. A score-margin filter removes weak candidates (those scoring more than 0.25 below the highest reranker score), and up to five relevant chunks are returned as context for answer generation.

    router_graph.py

This script is the main entry point of the application and orchestrates the RAG pipeline using LangGraph. It first uses a Gemini-based router to determine whether a question requires information from the Core Rules PDF, the SQLite unit database, or both; the selected retrieval branches are then executed in parallel when necessary. The retrieved information is combined into a unified context, with PDF content and database results clearly distinguished, before being passed to the Gemini answer generator. An important implementation choice is using a graph-based architecture to separate routing, retrieval, and generation into independent nodes, making the pipeline modular and easier to extend; the script also includes execution-time logging, error handling, and model warm-up to improve reliability and responsiveness.

    sqlite_reader.py

This script provides a secure interface for retrieving structured information from the Warhammer SQLite database. It supports both predefined retrieval methods and dynamically generated SQL queries, using parameterized queries to prevent SQL injection. It enforces read-only database access, combined with SQLite authorization rules that block modification commands, a query execution timeout, and a maximum result limit to protect the database from unsafe or excessively expensive queries. 

    test_rag.py (legacy)

This is an older version of the RAG pipeline. It loads the PDF knowledge base, retrieves relevant chunks using semantic search and reranking, and generates an answer to a single user question. The script handles the full workflow directly, including user input, retrieval, and answer generation. It has since been replaced by router_graph.py, which is the current entry point and uses LangGraph to orchestrate the workflow, including routing questions between the PDF knowledge base and the SQL database.

    text_to_sql.py

This script implements the SQL-based retrieval component of the project, translating natural-language questions about Warhammer 40,000 units into SQLite queries using Gemini. It dynamically retrieves the database schema to guide SQL generation, applies validation checks, and executes queries through the read-only SQLiteReader, with a three-second timeout and a maximum of 100 returned rows. The results are then converted into concise natural-language answers by Gemini, using only the retrieved database information.
