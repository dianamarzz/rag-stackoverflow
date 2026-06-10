"""
Retriever: загружает индекс и ищет top-k чанков по запросу.
Поддерживает два режима: tfidf и bm25 (управляется через RETRIEVAL_MODE).
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from app.config import (
    TOP_K, MIN_SCORE, BM25_MIN_SCORE, RETRIEVAL_MODE,
    INDEX_CHUNKS, VECTORIZER_FILE, MATRIX_FILE, BM25_FILE,
)


def _load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


class TFIDFRetriever:
    def __init__(self):
        with open(VECTORIZER_FILE, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.matrix = sp.load_npz(MATRIX_FILE)
        self.chunks = _load_chunks(INDEX_CHUNKS)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in indices:
            score = float(scores[idx])
            chunk = dict(self.chunks[idx])
            chunk["score"] = score
            results.append(chunk)
        return results


class BM25Retriever:
    def __init__(self):
        with open(BM25_FILE, "rb") as f:
            self.bm25 = pickle.load(f)
        self.chunks = _load_chunks(INDEX_CHUNKS)

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


def load_retriever(mode: str = RETRIEVAL_MODE):
    """Фабрика: возвращает нужный ретривер по режиму."""
    if mode == "bm25":
        return BM25Retriever()
    return TFIDFRetriever()


def is_relevant(results: list[dict], min_score: float | None = None) -> bool:
    """
    True, если топ-результат превышает порог релевантности.
    Для BM25 используется BM25_MIN_SCORE, для TF-IDF — MIN_SCORE.
    """
    if not results:
        return False
    if min_score is None:
        # Определяем по режиму: BM25 scores абсолютные (> 1),
        # TF-IDF cosine — в диапазоне 0-1
        top_score = results[0]["score"]
        threshold = BM25_MIN_SCORE if top_score > 1.0 else MIN_SCORE
    else:
        threshold = min_score
    return results[0]["score"] >= threshold
