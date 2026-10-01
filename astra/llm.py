"""
LLM layer: sends the question + retrieved chunks to Gemini and returns a
grounded answer. Gemini only picks excerpt NUMBERS; our code maps them back
to document/page/text, so the model can never invent a citation.
"""
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from astra.config import GEMINI_MODEL, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

NOT_FOUND_MESSAGE = "I couldn't find sufficient information in the provided documents."

SYSTEM_PROMPT = """You are ASTRA INTEL, an evidence-grounded defence document analyst.

Answer the question using ONLY the numbered excerpts provided. Never use outside
knowledge, even if you know the answer.

Return JSON with exactly these keys:
  "status": "supported" | "partial" | "not_found"
  "answer": string
  "cited_ids": list of excerpt numbers (integers) you actually used

Rules:
1. "supported": the excerpts fully answer the question.
2. "partial": the excerpts answer only part of it. Say clearly what is stated and what is NOT stated.
3. "not_found": the excerpts do not contain the requested information.
4. Being on the same topic is NOT enough. If the question asks for a specific figure,
   statistic, name or definition and the excerpts do not state it, the status is "not_found".
   Never substitute a different number. You may mention a related figure only if you say
   exactly what it really measures.
5. For "not_found", "answer" is one or two sentences saying what the documents do not contain.
6. Be concise. Do not write excerpt numbers inside "answer"; put them only in "cited_ids".
7. Do not add facts that are not in the excerpts.
"""


class LLMError(Exception):
    """Any problem talking to the LLM (missing key, bad reply, API failure)."""


def _get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=key)


def build_context(chunks):
    """Number the chunks [1], [2], ... so Gemini can cite them by number."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (document: {c['doc_name']}, page {c['page']})\n{c['text']}")
    return "\n\n".join(parts)


def _parse_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)   # strip ``` fences if present
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise LLMError(f"Gemini returned invalid JSON: {raw[:200]}")


def answer_question(question, chunks):
    """
    Returns {"status", "answer", "evidence": [chunk dicts], "downgraded": bool}
    """
    if not chunks:
        return {"status": "not_found", "answer": NOT_FOUND_MESSAGE,
                "evidence": [], "downgraded": False}

    client = _get_client()
    prompt = f"QUESTION:\n{question}\n\nEXCERPTS:\n{build_context(chunks)}"
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        response_mime_type="application/json",
    )

    response = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config)
            break
        except Exception as e:
            if attempt == 2:
                raise LLMError(f"Gemini API call failed: {e}")
            time.sleep(4 * (attempt + 1))     # short wait, then retry (rate limits)

    data = _parse_json(response.text)
    status = data.get("status", "not_found")
    if status not in ("supported", "partial", "not_found"):
        status = "not_found"
    answer = str(data.get("answer", "")).strip()

    # Map cited numbers back to real chunks; ignore anything out of range
    valid_ids = []
    for n in data.get("cited_ids", []):
        if isinstance(n, int) and 1 <= n <= len(chunks) and n not in valid_ids:
            valid_ids.append(n)
    evidence = [chunks[n - 1] for n in valid_ids]

    # Claimed an answer but cited nothing valid -> no evidence, so no answer
    downgraded = False
    if status in ("supported", "partial") and not evidence:
        status, downgraded = "not_found", True
        answer = "The model gave an answer without citing valid evidence, so it was rejected."

    if status == "not_found":
        answer = f"{NOT_FOUND_MESSAGE} {answer}".strip()

    return {"status": status, "answer": answer,
            "evidence": evidence, "downgraded": downgraded}


if __name__ == "__main__":
    # Usage: python -m astra.llm "your question"
    from astra.index import get_or_build_index

    question = " ".join(sys.argv[1:]) or "What are the three subdivisions of electronic warfare?"
    idx = get_or_build_index()
    chunks = idx.search(question, top_k=6)

    result = answer_question(question, chunks)
    print(f"\nQUESTION: {question}")
    print(f"STATUS:   {result['status']}")
    print(f"ANSWER:   {result['answer']}")
    print(f"\nEVIDENCE ({len(result['evidence'])} chunk(s) cited):")
    for c in result["evidence"]:
        print(f"  - {c['doc_name']} | page {c['page']} | score {c['score']:.3f}")
        print(f"    {c['text'][:160].replace(chr(10), ' ')}...")
