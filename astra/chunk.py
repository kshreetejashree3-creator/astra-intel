"""
Text chunking: splits extracted page text into overlapping chunks,
each tagged with document name, page number, and a unique chunk ID.

Also drops "tail" pages (References, External links, Further reading, etc.)
since these are citation lists, not real content, and would pollute retrieval.
"""

CHUNK_SIZE = 700       # characters per chunk
CHUNK_OVERLAP = 100    # characters repeated between consecutive chunks

# Headings that mark the start of the "junk tail" of a Wikipedia-style article.
# Once we see one of these as a standalone-ish line, we stop chunking that
# document from that page onward.
TAIL_HEADINGS = [
    "references",
    "external links",
    "further reading",
    "see also",
    "sources",
    "notes",
]


def _page_starts_tail(page_text):
    """
    Heuristic: check the first ~50 characters of a page for a tail heading.
    Real body-text pages don't usually START with one of these words alone.
    """
    head = page_text.strip()[:50].lower()
    for heading in TAIL_HEADINGS:
        if head.startswith(heading):
            return True
    return False


def chunk_document(pages):
    """
    pages: list of dicts from extract_pdf_pages(), e.g.
        {"doc_name": "...", "page": 3, "text": "..."}

    Returns a list of chunk dicts:
        {"chunk_id": "...", "doc_name": "...", "page": 3, "text": "..."}
    """
    chunks = []
    chunk_counter = 0
    in_tail = False

    for page in pages:
        if _page_starts_tail(page["text"]):
            in_tail = True  # once we hit references, everything after is tail too

        if in_tail:
            continue  # skip this page entirely

        text = page["text"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            piece = text[start:end].strip()

            if piece:  # skip empty/whitespace-only chunks
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"{page['doc_name']}_p{page['page']}_c{chunk_counter}",
                    "doc_name": page["doc_name"],
                    "page": page["page"],
                    "text": piece
                })

            start += (CHUNK_SIZE - CHUNK_OVERLAP)  # advance, leaving overlap behind

    return chunks


if __name__ == "__main__":
    # Quick manual test: python -m astra.chunk
    from astra.extract import extract_pdf_pages
    from astra.config import OFFICIAL_DOCS_DIR

    pdf_files = list(OFFICIAL_DOCS_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:
        pages = extract_pdf_pages(pdf_file)
        total_pages = len(pages)
        chunks = chunk_document(pages)

        last_page_kept = chunks[-1]["page"] if chunks else 0

        print(f"--- {pdf_file.name} ---")
        print(f"  Total pages extracted: {total_pages}")
        print(f"  Chunks created:        {len(chunks)}")
        print(f"  Last page kept:        {last_page_kept} (out of {total_pages})")
        print(f"  Sample chunk: {chunks[0]['chunk_id']}")
        print(f"  Sample text:  {chunks[0]['text'][:120]}...")
        print()
