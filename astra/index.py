"""
Vector index for all ASTRA documents.

Official challenge documents and user-added documents are indexed together.
User uploads live in sample-documents/additional/.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

from astra.embed import embed_texts
from astra.extract import extract_document
from astra.chunk import chunk_document
from astra.config import (
    INDEX_PATH, ALL_DOCS_DIRS, SUPPORTED_EXTENSIONS, TOP_K, PER_DOC_K, PROJECT_ROOT
)

class VectorIndex:
    def __init__(self, chunks, vectors, fingerprint=None):
        self.chunks = chunks
        self.vectors = vectors
        self.fingerprint = fingerprint

    @classmethod
    def build(cls, chunks, fingerprint=None):
        vectors = embed_texts([c["text"] for c in chunks]) if chunks else np.empty((0, 384), dtype="float32")
        return cls(chunks, vectors, fingerprint)

    def _top(self, q_vec, top_k, doc_name=None):
        if not len(self.chunks):
            return []
        scores = self.vectors @ q_vec
        if doc_name is not None:
            mask = np.array([c["doc_name"] == doc_name for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:top_k]
        return [{**self.chunks[i], "score": float(scores[i])}
                for i in order if np.isfinite(scores[i])]

    def search(self, query, top_k=TOP_K, doc_name=None):
        if not self.chunks:
            return []
        return self._top(embed_texts([query])[0], top_k, doc_name)

    def search_balanced(self, query, per_doc=PER_DOC_K):
        if not self.chunks:
            return []
        q_vec = embed_texts([query])[0]
        merged = []
        for name in sorted({c["doc_name"] for c in self.chunks}):
            merged.extend(self._top(q_vec, per_doc, doc_name=name))
        merged.sort(key=lambda r: -r["score"])
        return merged

    def save(self, path=INDEX_PATH):
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "vectors": self.vectors, "fingerprint": self.fingerprint}, f)

    @classmethod
    def load(cls, path=INDEX_PATH):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["chunks"], d["vectors"], d["fingerprint"])

def get_document_paths():
    paths = []
    for directory in ALL_DOCS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        paths.extend(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return sorted(paths, key=lambda p: p.name.lower())

def _fingerprint(paths):
    return [
        (str(p.relative_to(PROJECT_ROOT)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in paths
    ]

def get_or_build_index():
    paths = get_document_paths()
    fp = _fingerprint(paths)

    if INDEX_PATH.exists():
        try:
            idx = VectorIndex.load()
            if idx.fingerprint == fp:
                return idx
        except Exception:
            pass

    chunks = []
    for doc in paths:
        try:
            chunks.extend(chunk_document(extract_document(doc)))
        except Exception as exc:
            print(f"[index] skipped {doc.name}: {exc}")

    idx = VectorIndex.build(chunks, fp)
    idx.save()
    return idx

def document_summary(index):
    summary = {}
    for c in index.chunks:
        summary[c["doc_name"]] = summary.get(c["doc_name"], 0) + 1
    return summary

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Which sections discuss sense-and-avoid capabilities?"
    idx = get_or_build_index()
    print(f"Documents: {len(document_summary(idx))}")
    print(f"Chunks: {len(idx.chunks)}")
    for r in idx.search_balanced(question):
        print(f"{r['score']:.3f} | {r['doc_name']} | p{r['page']}")
