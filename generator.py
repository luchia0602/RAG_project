"""This script generates an answer using the Gemini API, given a question and context retrieved from the knowledge base"""

from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, GENERATION_MODELS
_client = None

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
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return _client

MAX_OUTPUT_TOKENS = 4096
THINKING_LEVEL = None # For faster answers

def generate_answer(question, context):
    prompt = f"""You are a Warhammer rules assistant.

Answer the question using ONLY the information explicitly stated
in the provided context.

The context may contain a RULES section (text from the Core Rules)
and/or a UNIT DATA section (values from the unit database). Treat
both as part of the provided context. Unit data values may be used
together with the rules to answer, but only apply rules that are
stated in the context.

STRICT RULES:
- Do not use your general knowledge of Warhammer.
- Do not add facts that are not explicitly stated in the context.
- Do not combine separate rules, abilities, or conditions.
- Do not infer that one condition causes or grants another condition
  unless the context explicitly says so.
- Preserve the exact distinction between "Fights First",
  "eligible to fight", and "made a Charge move".
- Do not add examples that are not explicitly present in the context.
- If something is unclear or incomplete, do not guess.
- If the context does not contain enough information to answer,
  say exactly:
  "I couldn't find enough information in the provided rules."
- Answer the specific question directly.
- When asked about a phase or procedure, summarize the relevant
  rules and include the steps explicitly present in the context.
- Use complete sentences. Do not stop mid-sentence.
- Be concise, but include all information needed to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    config_kwargs = {
        "temperature": 0.2,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if THINKING_LEVEL:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=THINKING_LEVEL
        )

    response = None
    last_error = None
    client = get_client()

    for model in GENERATION_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            print(f"[generator model: {model}]")
            break
        except Exception as exc:
            last_error = exc
            print(f"[generator] {model} failed: {exc}")

    if response is None:
        raise RuntimeError(
            f"All generator models failed. Last error: {last_error}"
        )

    if not response.candidates: # For empty/incomplete responses
        return "I couldn't generate an answer. Please try again."
    candidate = response.candidates[0]
    finish_reason = candidate.finish_reason
    print(f"[Gemini finish reason: {finish_reason}]")

    if response.text is None:
        return "I couldn't generate an answer. Please try again."
    answer = response.text.strip()

    if str(finish_reason) == "MAX_TOKENS" or str(finish_reason).endswith(
        ".MAX_TOKENS"
    ):
        print("[Warning: The answer may have been truncated.]")
    return answer