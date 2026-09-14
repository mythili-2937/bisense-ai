from google import genai
from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Say: Gemini connection successful."
)

print(interaction.output_text)