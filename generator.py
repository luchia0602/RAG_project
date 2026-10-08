from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


def generate_answer(question, context):
    prompt = f"""You are a Warhammer rules assistant.

Answer the question using ONLY the information explicitly stated in the provided context.

STRICT RULES:
- Do not use your general knowledge of Warhammer.
- Do not add facts that are not explicitly stated in the context.
- Do not combine separate rules, abilities, or conditions.
- Do not infer that one condition causes or grants another condition unless the context explicitly says so.
- Preserve the exact distinction between "Fights First", "eligible to fight", and "made a Charge move".
- Do not add examples that are not explicitly present in the context.
- If something is unclear or incomplete in the context, do not guess.
- If the context does not contain enough information to answer the question, say:
  "I couldn't find enough information in the provided rules."

Give a concise answer based only on the context.

Context:
{context}

Question:
{question}

Answer:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.2
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )
    return answer