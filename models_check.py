import os
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

print("Supported generateContent models for your key:")
for model in client.models.list():
    # Filter for models that support text generation
    supported = getattr(model, "supported_generation_methods", []) or getattr(model, "supported_actions", [])
    methods_str = " ".join(supported).lower()
    if "generatecontent" in methods_str or not supported:
        print(f" - {model.name}")