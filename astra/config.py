"""
Central place for ASTRA paths, constants, and settings.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OFFICIAL_DOCS_DIR = PROJECT_ROOT / "sample-documents" / "official"
ADDITIONAL_DOCS_DIR = PROJECT_ROOT / "sample-documents" / "additional"
ALL_DOCS_DIRS = (OFFICIAL_DOCS_DIR, ADDITIONAL_DOCS_DIR)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

COLOR_CHARCOAL = "#1B1F23"
COLOR_SIGNAL_GOLD = "#B8860B"
COLOR_WHITE = "#FFFFFF"

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_PATH = PROJECT_ROOT / "astra_index.pkl"
TOP_K = 5
PER_DOC_K = 6
MIN_CHUNK_SCORE = 0.15
MIN_BEST_SCORE = 0.25
MAX_UPLOAD_MB = 50
