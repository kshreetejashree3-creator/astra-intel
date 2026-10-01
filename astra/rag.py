"""
RAG pipeline: question -> balanced retrieval -> grounded answer -> evidence.
This is the single entry point the UI will call: ask(question, index).
"""
import sys

from astra.config import PER_DOC_K, MIN_CHUNK_SCORE, MIN_BEST_SCORE
from astra.index import get_or_build_index
from astra.llm import answer_question, LLMError, NOT_FOUND_MESSAGE


def retrieve(question, index):
    """Top chunks from EACH document, minus anything below the noise floor."""
    results = index.search_balanced(question, per_doc=PER_DOC_K)
    return [r for r in results if r["score"] >= MIN_CHUNK_SCORE]


def ask(question, index):
    """
    Returns a dict:
      question, status ("supported"/"partial"/"not_found"), answer,
      evidence   (chunks the answer is actually based on),
      retrieved  (ALL chunks sent to the LLM - for debugging),
      refused_by (None, "retrieval" or "llm"),
      downgraded (True if the LLM answered without valid citations)
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question is empty.")

    retrieved = retrieve(question, index)

    # Refusal layer 1: retrieval found nothing relevant -> don't even call the LLM
    if not retrieved or retrieved[0]["score"] < MIN_BEST_SCORE:
        return {
            "question": question, "status": "not_found", "answer": NOT_FOUND_MESSAGE,
            "evidence": [], "retrieved": retrieved,
            "refused_by": "retrieval", "downgraded": False,
        }

    # Refusal layer 2 lives inside answer_question(): the LLM may still say not_found
    result = answer_question(question, retrieved)
    result["question"] = question
    result["retrieved"] = retrieved
    result["refused_by"] = "llm" if result["status"] == "not_found" else None
    return result


def print_result(result, debug=False):
    print(f"\nQUESTION: {result['question']}")
    refused = f"  (refused by: {result['refused_by']})" if result["refused_by"] else ""
    print(f"STATUS:   {result['status']}{refused}")
    print(f"ANSWER:   {result['answer']}")

    if result["evidence"]:
        print("\nEVIDENCE")
        for n, c in enumerate(result["evidence"], start=1):
            passage = c["text"].replace("\n", " ")[:300]
            print(f"  [{n}] Source: {c['doc_name']} | Page: {c['page']} | score {c['score']:.3f}")
            print(f"      Passage: {passage}...")

    if debug:
        cited = {c["chunk_id"] for c in result["evidence"]}
        print(f"\nDEBUG: {len(result['retrieved'])} chunk(s) sent to the LLM")
        for c in result["retrieved"]:
            mark = "CITED" if c["chunk_id"] in cited else "     "
            snippet = c["text"].replace("\n", " ")[:90]
            print(f"  {mark} {c['score']:.3f} | {c['doc_name'][:32]:32} | p{c['page']:<3} | {snippet}...")


if __name__ == "__main__":
    # Usage: python -m astra.rag "your question" [--debug]
    debug = "--debug" in sys.argv
    words = [a for a in sys.argv[1:] if a != "--debug"]
    question = " ".join(words) or "Name one specific UGV program and its developer."

    idx = get_or_build_index()
    try:
        print_result(ask(question, idx), debug=debug)
    except LLMError as e:
        print(f"\n[LLM ERROR] {e}")
