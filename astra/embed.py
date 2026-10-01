"""
Embeddings: turns text into vectors (lists of numbers that capture meaning).
Uses a small pretrained model that runs locally on CPU - no API key needed.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from astra.config import EMBEDDING_MODEL

_model = None  # loaded once, then reused (loading takes a few seconds)


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts, batch_size=32):
    """
    texts: list of strings
    returns: numpy array of shape (len(texts), 384), each row normalized to length 1
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # so dot product == cosine similarity
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


if __name__ == "__main__":
    # Manual test: python -m astra.embed
    import time
    from astra.extract import extract_pdf_pages
    from astra.chunk import chunk_document
    from astra.config import OFFICIAL_DOCS_DIR

    all_chunks = []
    for pdf in sorted(OFFICIAL_DOCS_DIR.glob("*.pdf")):
        all_chunks.extend(chunk_document(extract_pdf_pages(pdf)))

    print(f"Total chunks to embed: {len(all_chunks)}")
    start = time.time()
    vectors = embed_texts([c["text"] for c in all_chunks])
    print(f"Embedding matrix shape: {vectors.shape}")
    print(f"Time taken: {time.time() - start:.1f} seconds")
    print(f"Length of first vector (should be ~1.0): {np.linalg.norm(vectors[0]):.3f}")
