from google import genai
from dotenv import load_dotenv
from pathlib import Path
import os
import json

# Optional Streamlit import so this file still works from terminal.
try:
    import streamlit as st
except ImportError:
    st = None


# -------------------------------------------------
# ENVIRONMENT / API KEY
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Local development: load .env if it exists.
load_dotenv(BASE_DIR / ".env")


def get_api_key():
    """
    Load Gemini API key from:
    1. Local environment / .env
    2. Streamlit Secrets (for Streamlit Community Cloud)
    """

    # Local .env / environment variable
    key = os.getenv("GEMINI_API_KEY")

    if key:
        return key

    # Streamlit Cloud secrets
    if st is not None:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    raise RuntimeError(
        "GEMINI_API_KEY was not found. "
        "For local use, add it to .env. "
        "For Streamlit Cloud, add it under App Settings > Secrets."
    )


api_key = get_api_key()

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.6-flash"


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def _clean_json_text(text):
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # Keep only the outer JSON object if the model adds extra text.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


# -------------------------------------------------
# REQUIREMENT ANALYSIS
# -------------------------------------------------

def analyze_requirement(requirement, response_language="English"):
    """
    Understand a procurement requirement in English, Tamil, Hindi,
    or mixed-language input.

    All display fields are returned in the selected response language.
    search_query is always returned in English for semantic search.
    """

    prompt = f"""
You are an AI assistant for procurement standards discovery.

The user may write the procurement requirement in English, Tamil, Hindi,
or a mixture of these languages.

PROCUREMENT REQUIREMENT:
{requirement}

SELECTED RESPONSE LANGUAGE:
{response_language}

Tasks:
1. Understand the requirement regardless of the input language.
2. Extract the product, category, specifications, application, and likely
   compliance needs.
3. Return the extracted display fields in {response_language}.
4. Create "search_query" in ENGLISH ONLY. It will be used by an English
   semantic-search system against a standards knowledge base.
5. Do NOT invent Indian Standard numbers.
6. Keep the extracted values concise and procurement-focused.

Return ONLY valid JSON in exactly this structure:

{{
  "product": "",
  "category": "",
  "specifications": [],
  "application": "",
  "compliance_needs": [],
  "search_query": ""
}}
"""

    response = client.interactions.create(
        model=MODEL_NAME,
        input=prompt
    )

    text = _clean_json_text(response.output_text)

    try:
        data = json.loads(text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned an invalid JSON response. "
            f"Raw response: {response.output_text}"
        ) from exc

    # Normalize structure so Streamlit can safely use it.
    data.setdefault("product", "")
    data.setdefault("category", "")
    data.setdefault("specifications", [])
    data.setdefault("application", "")
    data.setdefault("compliance_needs", [])
    data.setdefault("search_query", "")

    if not isinstance(data["specifications"], list):
        data["specifications"] = [str(data["specifications"])]

    if not isinstance(data["compliance_needs"], list):
        data["compliance_needs"] = [str(data["compliance_needs"])]

    return data


# -------------------------------------------------
# RECOMMENDATION EXPLANATION
# -------------------------------------------------

def explain_recommendations(
    requirement,
    standards,
    response_language="English"
):
    """
    Explain retrieved standards using only supplied retrieval results.
    The explanation is returned in the selected response language.
    """

    prompt = f"""
You are an AI procurement standards assistant.

ORIGINAL PROCUREMENT REQUIREMENT:
{requirement}

RETRIEVED STANDARDS:
{standards}

SELECTED RESPONSE LANGUAGE:
{response_language}

Explain briefly why each retrieved standard may be relevant.

Rules:
- Use ONLY the retrieved standards supplied above.
- Do NOT invent IS numbers.
- Do NOT invent certification requirements.
- Do NOT claim that a recommendation is legally mandatory unless the supplied
  data explicitly says so.
- Do NOT use HTML tags.
- Use clean Markdown bullet points.
- Keep each explanation short and clear.
- Mention whether a standard appears to be a product, safety, testing,
  or related standard when this is evident from the supplied data.
- Write the explanation in {response_language}.

Return only the explanation in clean Markdown.
"""

    response = client.interactions.create(
        model=MODEL_NAME,
        input=prompt
    )

    return (response.output_text or "").strip()


# -------------------------------------------------
# OPTIONAL TERMINAL TEST
# -------------------------------------------------

if __name__ == "__main__":

    requirement = input(
        "Enter procurement requirement: "
    )

    language = input(
        "Response language (English/Tamil/Hindi): "
    ).strip() or "English"

    result = analyze_requirement(
        requirement,
        language
    )

    print("\nAI Requirement Analysis")
    print("------------------------")
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )
