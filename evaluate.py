import json
import os
import time
import re
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pdf_reader import read_pdf
from embeddings import make_chunks, get_embeddings
from rag import search
from generator import generate_answer
from dotenv import load_dotenv

load_dotenv(override=True)

PDF_PATH = r"C:\Users\user\Documents\Studies\PS\RAG\Core Rules.pdf"
QUESTIONS_PATH = "eval_questions.json"
CHECKPOINT_PATH = "ragas_checkpoint.json"
RESULTS_PATH = "ragas_results.csv"

SAMPLE_SIZE = 3
WAIT_SECONDS = 15
MAX_RETRIES = 5
QUOTA_WAIT_SECONDS = 65

def save_checkpoint(data):
    temp_path = CHECKPOINT_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, CHECKPOINT_PATH)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def generate_with_retry(question, context):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answer = generate_answer(question, context)
            return answer
        except Exception as error:
            error_text = str(error)
            is_quota_error = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            )
            if not is_quota_error:
                raise
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Question failed after {MAX_RETRIES} attempts."
                ) from error
            match = re.search(
                r"retry in ([0-9.]+)s",
                error_text,
                flags=re.IGNORECASE,
            )
            wait_time = (
                float(match.group(1)) + 3
                if match
                else QUOTA_WAIT_SECONDS
            )
            wait_time = max(wait_time, QUOTA_WAIT_SECONDS)

            print(
                f"Gemini quota limit reached. "
                f"Attempt {attempt}/{MAX_RETRIES}"
            )
            print(f"Waiting {wait_time:.0f} seconds before retrying")
            time.sleep(wait_time)
    raise RuntimeError("Generation failed")

with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
    test_questions = json.load(file)[:SAMPLE_SIZE]

checkpoint = load_checkpoint()
print(f"Questions in dataset: {len(test_questions)}")
print(f"Previously completed: {len(checkpoint)}")
print("Loading PDF...")

text = read_pdf(PDF_PATH)
chunks = make_chunks(text, chunk_size=800, chunk_overlap=100)
embeddings = get_embeddings(chunks)

for i, item in enumerate(test_questions):
    key = str(i)
    if key in checkpoint:
        print(f"Skipping question {i + 1}: answer already saved")
        continue
    question = item["question"]
    ground_truth = item["ground_truth"]
    print(f"Evaluating question {i + 1}/{len(test_questions)}")
    print("Question:", question)
    try:
        retrieved_chunks = search(question, chunks, embeddings)
        context = "\n\n".join(retrieved_chunks)
        answer = generate_with_retry(question, context)

    except Exception as error:
        print(f"ERROR on question {i + 1}: {error}")
        raise

    checkpoint[key] = {
        "question": question,
        "answer": answer,
        "contexts": retrieved_chunks,
        "ground_truth": ground_truth,
    }

    save_checkpoint(checkpoint)
    print("Answer:", answer)
    print("Progress saved")
    if i < len(test_questions) - 1:
        print(f"Waiting {WAIT_SECONDS} seconds...")
        time.sleep(WAIT_SECONDS)

records = [
    checkpoint[str(i)]
    for i in range(len(test_questions))
]

dataset = Dataset.from_dict({
    "question": [r["question"] for r in records],
    "answer": [r["answer"] for r in records],
    "contexts": [r["contexts"] for r in records],
    "ground_truth": [r["ground_truth"] for r in records],
})

print("Starting Ragas evaluation...")
run_config = RunConfig(
    max_workers=1,
    max_retries=3,
    timeout=180,
)
evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )
)

results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=run_config,
    raise_exceptions=False,
)

results_df = results.to_pandas()
results_df.to_csv(RESULTS_PATH, index=False)
print("Results:")
print(results)