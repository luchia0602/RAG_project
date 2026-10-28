"""This script builds a graph of nodes that route a question to the appropriate knowledge source(s)"""

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, START, END

from cache import get_rag_data
from rag import search, warmup
from generator import generate_answer
from sqlite_reader import SQLiteReader
from text_to_sql import generate_sql, validate_sql, MAX_ROWS
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "Core Rules.pdf"
DB_PATH = BASE_DIR / "data" / "wahadb (1).sqlite"
INITIAL_K = 15
TOP_K = 5

load_dotenv(override=True)

# For routing, we are using small cheap models. If the first one does not work, we can try the next one
ROUTER_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]
ROUTER_PROMPT = """You are a router for a Warhammer 40,000 question-answering system.
Decide which knowledge source(s) are needed to answer the user's question.

SOURCES:
- "sql": a database of unit datasheets. Contains concrete data per unit and
  faction: unit names, models, weapon profiles, stats (movement, toughness,
  save, wounds, attacks, strength, AP, damage), points costs, abilities
  listed on datasheets, loadouts, factions, stratagems.
- "pdf": the Core Rules text. Explains how the game works: phases, movement,
  charging, shooting, fighting, cover, objectives, stratagems, keywords and
  what they do, terrain, game terms and general mechanics.

Pick "sql" when the answer is a fact about specific units/factions.
Pick "pdf" when the answer is a rule or how a mechanic works.
Pick both when the question needs unit data AND a rule to answer
(e.g. applying a rule to a specific unit, or asking what a keyword on a
unit does).

EXAMPLES:
Q: How many attacks does a Terminator Sergeant have? -> {{"sources": ["sql"]}}
Q: List all units in the Orks faction costing under 100 points. -> {{"sources": ["sql"]}}
Q: How does the charge phase work? -> {{"sources": ["pdf"]}}
Q: What happens during the Command phase? -> {{"sources": ["pdf"]}}
Q: What does Lethal Hits do? -> {{"sources": ["pdf"]}}
Q: Does Intercessor get cover from a ruin when shot at? -> {{"sources": ["pdf", "sql"]}}
Q: Can a unit with Deep Strike charge on the turn it arrives? -> {{"sources": ["pdf"]}}

Answer with JSON only, no explanation, in this format:
{{"sources": ["pdf"]}} or {{"sources": ["sql"]}} or {{"sources": ["pdf", "sql"]}}

QUESTION:
{question}
"""

def call_router_llm(client, prompt):
    """Single place to change the router model (e.g. to a local one)."""
    last_error = None
    for model in ROUTER_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            if response.text:
                return response.text
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"All router models failed: {last_error}")

def route_question(client, question):
    """Return a non-empty list from {"pdf", "sql"}. Falls back to both."""
    try:
        raw = call_router_llm(client, ROUTER_PROMPT.format(question=question))
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0))
        sources = [s for s in data.get("sources", []) if s in ("pdf", "sql")]
        if sources:
            return list(dict.fromkeys(sources))  
    except Exception as exc:
        print(f"[router] failed ({exc}); using both sources.")
    return ["pdf", "sql"]

class State(TypedDict):
    question: str
    sources: list
    pdf_context: str
    sql_context: str
    answer: str

def timed(name, fn):
    """Print how long a graph node takes."""
    def wrapper(state):
        t0 = time.perf_counter()
        out = fn(state)
        print(f"[timing] {name}: {time.perf_counter() - t0:.1f}s")
        return out
    return wrapper


def build_app(client, reader, chunks, embeddings):
    def router_node(state: State):
        sources = route_question(client, state["question"])
        print(f"[router] -> {sources}")
        return {"sources": sources}

    def pdf_node(state: State):
        results = search(
            state["question"],
            chunks,
            embeddings,
            top_k=TOP_K,
            initial_k=INITIAL_K,
        )
        return {"pdf_context": "\n\n".join(results)}

    def sql_node(state: State):
        try:
            # generate_sql returns (sql, model_used)
            sql, _model = generate_sql(client, reader, state["question"])
            sql = validate_sql(sql)
            rows = reader.execute_readonly_query(
                sql, params=(), max_rows=MAX_ROWS, timeout_seconds=3.0
            )
            if not rows:
                return {"sql_context": "The database query returned no rows."}
            lines = [str(dict(r)) for r in rows]
            return {"sql_context": "\n".join(lines)}
        except Exception as exc:
            print(f"[sql] error: {exc}")
            traceback.print_exc()
            return {"sql_context": f"The database query failed: {exc}"}

    def generate_node(state: State):
        parts = []
        if state.get("pdf_context"):
            parts.append("RULES (from Core Rules PDF):\n" + state["pdf_context"])
        if state.get("sql_context"):
            parts.append("UNIT DATA (from database):\n" + state["sql_context"])
        context = "\n\n".join(parts)
        try:
            return {"answer": generate_answer(state["question"], context)}
        except Exception as exc:
            print(f"[generate] error: {exc}")
            traceback.print_exc()
            return {"answer": f"Sorry, answer generation failed: {exc}"}

    def dispatch(state: State):
        return state["sources"]  # one or both; both run in parallel

    graph = StateGraph(State)
    graph.add_node("router", timed("router", router_node))
    graph.add_node("pdf", timed("pdf", pdf_node))
    graph.add_node("sql", timed("sql", sql_node))
    graph.add_node("generate", timed("generate", generate_node))

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", dispatch, ["pdf", "sql"])
    # Separate edges (not a list) so generate runs after whichever branch(es)
    # the router selected, instead of waiting for both.
    graph.add_edge("pdf", "generate")
    graph.add_edge("sql", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

def main():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY (e.g. in .env).")
        return

    print("Loading knowledge base...")
    _, chunks, embeddings = get_rag_data(
        pdf_path=PDF_PATH, chunk_size=800, overlap=100
    )
    # Load the embedding model + reranker now, not on the first question, to make it faster
    warmup()

    client = genai.Client()
    reader = SQLiteReader(DB_PATH)
    app = build_app(client, reader, chunks, embeddings)
    print("Ready. Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue

        t0 = time.perf_counter()
        result = app.invoke({
            "question": question,
            "sources": [],
            "pdf_context": "",
            "sql_context": "",
            "answer": "",
        })
        print("\nANSWER:")
        print(result["answer"])
        print(f"\n[total: {time.perf_counter() - t0:.1f}s]")

if __name__ == "__main__":
    main()