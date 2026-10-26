"""This script generates an answer using the Gemini API, given a question and context retrieved from the knowledge base"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GOOGLE_API_KEY"]
)

MODEL_NAME = "gemini-3.8-flash"

def generate_answer(question, context):
    prompt = f"""You are a Warhammer rules assistant.

Answer the question using ONLY the information explicitly stated
in the provided context.

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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=500
        )
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