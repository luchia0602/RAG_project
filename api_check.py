import os
from dotenv import load_dotenv

load_dotenv(override=True)
key = os.environ.get("GOOGLE_API_KEY", "")
print("Key in use ends with:", key[-6:] if key else "NO KEY FOUND")