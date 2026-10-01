"""
Document extraction for ASTRA INTEL.
Supports PDF, TXT and Markdown files.
"""
from pathlib import Path
import fitz

TEXT_EXTENSIONS = {".txt", ".md"}

def extract_pdf_pages(pdf_path):
    path = Path(pdf_path)
    pages = []
    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc, start=1):
            pages.append({"doc_name": path.name, "page": page_num, "text": page.get_text()})
    return pages

def extract_text_pages(file_path):
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return [{"doc_name": path.name, "page": 1, "text": ""}]
    step = 2500
    return [
        {"doc_name": path.name, "page": (i // step) + 1, "text": text[i:i + step]}
        for i in range(0, len(text), step)
    ]

def extract_document(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(path)
    if suffix in TEXT_EXTENSIONS:
        return extract_text_pages(path)
    raise ValueError(f"Unsupported document type: {path.suffix}")

if __name__ == "__main__":
    from astra.config import ALL_DOCS_DIRS
    for directory in ALL_DOCS_DIRS:
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
                print(f"{path.name}: {len(extract_document(path))} page/section(s)")
