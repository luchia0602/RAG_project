
"""Generate an answer using Gemini and the context retrieved by the RAG pipeline."""

from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, GENERATION_MODELS

_client = None

MAX_OUTPUT_TOKENS = 4096
THINKING_LEVEL = None
DEBUG_CONTEXT = False # Change to True if you want to see what context is sent to the generator


def get_client():
    """Create the Gemini client only when it is first needed."""
    global _client

    if _client is None:
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
            raise RuntimeError(
                "Gemini API key not configured. "
                "Copy .env.example to .env and set GOOGLE_API_KEY."
            )

        _client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=types.HttpOptions(
                timeout=60_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    return _client


def build_prompt(question, context):
    """Build a prompt using the retrieved information."""

    return f"""
You are a precise assistant answering questions about Warhammer 40,000.

Your task is to answer the user's question using ONLY the supplied context.

The context can contain information from two sources:

1. RULES: Text extracted from the Warhammer Core Rules PDF.
2. UNIT DATA: Structured information retrieved from the Warhammer database.

HOW TO USE THE CONTEXT:
- Use RULES to answer questions about game mechanics and procedures.
- Use UNIT DATA to answer questions about units, weapons, characteristics,
  abilities, factions, and other database information.
- You may combine information from RULES and UNIT DATA when both sources
  explicitly provide the information needed for the answer.
- You may combine multiple relevant passages from the same source.
- Do not invent missing information or rely on your general knowledge.
- Do not treat information from one source as evidence for something
  that it does not explicitly establish.

IMPORTANT:
- Distinguish carefully between "Fights First", "eligible to fight",
  and "made a Charge move".
- Do not confuse a unit's characteristics with rules governing those
  characteristics.
- If the context contains the answer, provide it directly.
- Do not reject an answer merely because the relevant information
  appears in several separate passages.
- If the context genuinely does not contain enough information,
  say exactly:
  "I couldn't find enough information in the provided rules."

RESPONSE FORMAT:
- Answer the specific question directly.
- Use complete sentences.
- Include relevant names, values, and units.
- Explain the relevant steps when asked about a procedure.
- Be concise but sufficiently detailed.
- Do not mention SQL, retrieval, or the internal implementation.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

def generate_answer(question, context):
    """Generate an answer based on retrieved context."""

    if not context or not context.strip():
        print("[generator] Empty context received.")
        return "I couldn't find enough information in the provided rules."

    print(f"[generator] Context length: {len(context)} characters")

    if DEBUG_CONTEXT:
        print("CONTEXT RECEIVED BY GENERATOR:")
        print(context)

    prompt = build_prompt(question, context)

    config_kwargs = {
        "temperature": 0.1,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }

    if THINKING_LEVEL:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=THINKING_LEVEL
        )

    client = get_client()
    last_error = None

    for model in GENERATION_MODELS:
        try:
            print(f"[generator] Trying model: {model}")

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            if not response.candidates:
                print(f"[generator] {model} returned no candidates.")
                continue

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            print(f"[generator model: {model}]")
            print(f"[Gemini finish reason: {finish_reason}]")

            if response.text is None:
                print(f"[generator] {model} returned no text.")
                continue

            answer = response.text.strip()

            if not answer:
                print(f"[generator] {model} returned an empty answer.")
                continue

            return answer

        except Exception as exc:
            last_error = exc
            print(f"[generator] {model} failed: {exc}")

    raise RuntimeError(
        f"All generator models failed. Last error: {last_error}"
    )
