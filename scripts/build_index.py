"""
build_index.py
--------------
Полный pipeline: ingest → chunk → TF-IDF + BM25 fit → сохранение артефактов.

Запуск:
  uv run python scripts/build_index.py
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (
    DATA_INDEX, DATA_PROCESSED,
    DOCUMENTS_FILE, CHUNKS_FILE,
    VECTORIZER_FILE, MATRIX_FILE, BM25_FILE, INDEX_CHUNKS,
    MAX_CHARS, OVERLAP,
)
from app.chunker import chunk_document
from scripts.ingest import ingest


def load_documents():
    if not DOCUMENTS_FILE.exists():
        print("[build] documents.jsonl не найден — запускаем ingest …")
        ingest()
    docs = []
    with open(DOCUMENTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build_chunks(docs: list[dict]) -> list[dict]:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    chunks = []
    with open(CHUNKS_FILE, "w", encoding="utf-8") as out:
        for doc in docs:
            for chunk in chunk_document(doc, max_chars=MAX_CHARS, overlap=OVERLAP):
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunks.append(chunk)
    print(f"[chunk]  Всего чанков: {len(chunks):,}")
    return chunks


def build_tfidf(chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(texts)
    print(f"[tfidf]  Матрица: {matrix.shape[0]} чанков × {matrix.shape[1]} признаков")
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(MATRIX_FILE, matrix)
    print(f"[tfidf]  Сохранено → {VECTORIZER_FILE.name}, {MATRIX_FILE.name}")


def build_bm25(chunks: list[dict]):
    from rank_bm25 import BM25Okapi
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[bm25]   Сохранено → {BM25_FILE.name}")


def save_index_chunks(chunks: list[dict]):
    DATA_INDEX.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CHUNKS, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[index]  chunks.jsonl → {INDEX_CHUNKS}")


def main():
    print("=" * 50)
    print("  RAG StackOverflow — Build Index")
    print("=" * 50)

    docs = load_documents()
    print(f"[load]   Документов: {len(docs):,}")

    chunks = build_chunks(docs)

    DATA_INDEX.mkdir(parents=True, exist_ok=True)
    save_index_chunks(chunks)
    build_tfidf(chunks)
    build_bm25(chunks)

    print()
    print("✅ Индекс успешно построен!")
    print(f"   Документов : {len(docs):,}")
    print(f"   Чанков     : {len(chunks):,}")


if __name__ == "__main__":
    main()
