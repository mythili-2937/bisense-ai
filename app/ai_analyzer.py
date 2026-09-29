from google import genai
from dotenv import load_dotenv
from pathlib import Path
import os
import json
import time
import random

try:
    import streamlit as st
except ImportError:
    st = None

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

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


def _clean_json_text(text):
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


def _is_rate_limit_error(exc):
    name = type(exc).__name__.lower()
    message = str(exc).lower()

    indicators = [
        "ratelimit",
        "rate limit",
        "429",
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "too many requests",
    ]

    return any(
        indicator in name or indicator in message
        for indicator in indicators
    )


def _call_gemini(prompt, retries=3):
    last_error = None

    for attempt in range(retries):
        try:
            return client.interactions.create(
                model=MODEL_NAME,
                input=prompt
            )

        except Exception as exc:
            last_error = exc

            if _is_rate_limit_error(exc):
                if attempt < retries - 1:
                    wait_seconds = (4 * (attempt + 1)) + random.uniform(0, 1.5)
                    time.sleep(wait_seconds)
                    continue

                raise RuntimeError(
                    "Gemini API rate limit or quota has been reached. "
                    "Please wait for the quota window to reset, or check the "
                    "Gemini API usage/quota for the API key used by this app."
                ) from exc

            raise RuntimeError(
                f"Gemini request failed: {type(exc).__name__}. "
                "Please check the Streamlit logs for more details."
            ) from exc

    raise RuntimeError(
        "Gemini request failed after multiple attempts."
    ) from last_error


def analyze_requirement(requirement, response_language="English"):
    if not requirement or not requirement.strip():
        raise ValueError("Procurement requirement cannot be empty.")

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

    response = _call_gemini(prompt)
    text = _clean_json_text(response.output_text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned an invalid JSON response. Please try again."
        ) from exc

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


def explain_recommendations(
    requirement,
    standards,
    response_language="English"
):
    if not standards:
        return "No relevant standards were retrieved for this requirement."

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

    try:
        response = _call_gemini(prompt)
        return (response.output_text or "").strip()

    except RuntimeError as exc:
        message = str(exc).lower()

        if "rate limit" not in message and "quota" not in message:
            raise

        lines = [
            "### Recommendation Summary",
            "",
            "Gemini explanation is temporarily unavailable due to API limits.",
            "The retrieved standards are still shown from the semantic search results:",
            ""
        ]

        for standard in standards:
            standard_id = standard.get("standard_id", "Unknown")
            title = standard.get("title", "Untitled standard")
            product = standard.get("product", "")
            category = standard.get("category", "")
            score = standard.get("score", "")

            detail_parts = []

            if product:
                detail_parts.append(f"Product: {product}")

            if category:
                detail_parts.append(f"Category: {category}")

            if score != "":
                detail_parts.append(f"Semantic match: {score}%")

            details = " | ".join(detail_parts)

            if details:
                lines.append(
                    f"- **{standard_id} — {title}**  \n  {details}"
                )
            else:
                lines.append(
                    f"- **{standard_id} — {title}**"
                )

        return "\n".join(lines)


if __name__ == "__main__":
    requirement = input("Enter procurement requirement: ")
    language = input(
        "Response language (English/Tamil/Hindi): "
    ).strip() or "English"

    try:
        result = analyze_requirement(requirement, language)

        print("\nAI Requirement Analysis")
        print("------------------------")
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )

    except Exception as exc:
        print("\nError:")
        print(str(exc))
