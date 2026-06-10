import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_INDEX     = BASE_DIR / "data" / "index"

# ── Chunking ──────────────────────────────────────────────
MAX_CHARS = 600
OVERLAP   = 80

# ── Retrieval ─────────────────────────────────────────────
TOP_K          = 5
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "bm25")   # "tfidf" | "bm25"
MIN_SCORE      = 0.05   # порог для TF-IDF (cosine 0-1)
BM25_MIN_SCORE = 25.0    # порог для BM25 (абсолютный, зависит от корпуса)

# ── LLM (опционально) ─────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
USE_LLM = bool(ANTHROPIC_API_KEY)
MODEL   = "claude-sonnet-4-20250514"

# ── Пути к файлам ─────────────────────────────────────────
DATASETS_FILE   = DATA_RAW       / "datasets.json"
DOCUMENTS_FILE  = DATA_PROCESSED / "documents.jsonl"
CHUNKS_FILE     = DATA_PROCESSED / "chunks.jsonl"
INDEX_CHUNKS    = DATA_INDEX     / "chunks.jsonl"
VECTORIZER_FILE = DATA_INDEX     / "vectorizer.pkl"
MATRIX_FILE     = DATA_INDEX     / "matrix.npz"
BM25_FILE       = DATA_INDEX     / "bm25.pkl"
