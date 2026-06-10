"""
Тесты для app/retriever.py
Используют мок-данные — не требуют реального индекса на диске.
"""
import sys
import json
import pickle
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

from app.retriever import TFIDFRetriever, BM25Retriever, is_relevant


# ── Фикстура: временный индекс ────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {"chunk_id": "so_1__c0", "doc_id": "so_1", "name": "Empty list check",
     "url": "https://stackoverflow.com/questions/1",
     "text": "How do I check if a list is empty in Python? Use 'if not my_list'."},
    {"chunk_id": "so_2__c0", "doc_id": "so_2", "name": "Sort dict by value",
     "url": "https://stackoverflow.com/questions/2",
     "text": "Sort a dictionary by value using sorted() with key=lambda x: x[1]."},
    {"chunk_id": "so_3__c0", "doc_id": "so_3", "name": "String formatting",
     "url": "https://stackoverflow.com/questions/3",
     "text": "Use f-strings for string formatting in Python 3.6+: f'{variable}'."},
    {"chunk_id": "so_4__c0", "doc_id": "so_4", "name": "List comprehension",
     "url": "https://stackoverflow.com/questions/4",
     "text": "List comprehension syntax: [expr for item in iterable if condition]."},
    {"chunk_id": "so_5__c0", "doc_id": "so_5", "name": "Exception handling",
     "url": "https://stackoverflow.com/questions/5",
     "text": "Use try/except to handle exceptions: try: ... except ValueError as e: ..."},
]


@pytest.fixture()
def tfidf_index(tmp_path):
    """Создаёт временный TF-IDF индекс и патчит пути конфига."""
    texts = [c["text"] for c in SAMPLE_CHUNKS]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    chunks_file    = tmp_path / "chunks.jsonl"
    vectorizer_file = tmp_path / "vectorizer.pkl"
    matrix_file    = tmp_path / "matrix.npz"

    with open(chunks_file, "w") as f:
        for c in SAMPLE_CHUNKS:
            f.write(json.dumps(c) + "\n")

    with open(vectorizer_file, "wb") as f:
        pickle.dump(vectorizer, f)

    sp.save_npz(str(matrix_file), matrix)

    # Патчим пути в модуле retriever
    import app.retriever as ret_mod
    orig_ic = ret_mod.INDEX_CHUNKS
    orig_vf = ret_mod.VECTORIZER_FILE
    orig_mf = ret_mod.MATRIX_FILE

    ret_mod.INDEX_CHUNKS    = chunks_file
    ret_mod.VECTORIZER_FILE = vectorizer_file
    ret_mod.MATRIX_FILE     = matrix_file

    yield TFIDFRetriever()

    ret_mod.INDEX_CHUNKS    = orig_ic
    ret_mod.VECTORIZER_FILE = orig_vf
    ret_mod.MATRIX_FILE     = orig_mf


@pytest.fixture()
def bm25_index(tmp_path):
    """Создаёт временный BM25 индекс."""
    tokenized = [c["text"].lower().split() for c in SAMPLE_CHUNKS]
    bm25 = BM25Okapi(tokenized)

    chunks_file = tmp_path / "chunks.jsonl"
    bm25_file   = tmp_path / "bm25.pkl"

    with open(chunks_file, "w") as f:
        for c in SAMPLE_CHUNKS:
            f.write(json.dumps(c) + "\n")

    with open(bm25_file, "wb") as f:
        pickle.dump(bm25, f)

    import app.retriever as ret_mod
    orig_ic  = ret_mod.INDEX_CHUNKS
    orig_bm = ret_mod.BM25_FILE

    ret_mod.INDEX_CHUNKS = chunks_file
    ret_mod.BM25_FILE    = bm25_file

    yield BM25Retriever()

    ret_mod.INDEX_CHUNKS = orig_ic
    ret_mod.BM25_FILE    = orig_bm


# ── TF-IDF тесты ──────────────────────────────────────────────────────────────

def test_tfidf_returns_results(tfidf_index):
    """TF-IDF возвращает непустой список результатов."""
    results = tfidf_index.search("how to check empty list python", top_k=3)
    assert len(results) > 0


def test_tfidf_results_have_score(tfidf_index):
    """Каждый результат содержит числовое поле score."""
    results = tfidf_index.search("list comprehension", top_k=3)
    for r in results:
        assert "score" in r
        assert isinstance(r["score"], float)


def test_tfidf_top_k_respected(tfidf_index):
    """Количество результатов не превышает top_k."""
    for k in [1, 2, 3]:
        results = tfidf_index.search("python string format", top_k=k)
        assert len(results) <= k


def test_tfidf_relevant_query_has_positive_score(tfidf_index):
    """Релевантный запрос даёт score > 0."""
    results = tfidf_index.search("empty list check python", top_k=1)
    assert results[0]["score"] > 0


# ── BM25 тесты ────────────────────────────────────────────────────────────────

def test_bm25_returns_results(bm25_index):
    """BM25 возвращает непустой список результатов."""
    results = bm25_index.search("sort dictionary value python", top_k=3)
    assert len(results) > 0


def test_bm25_top_k_respected(bm25_index):
    """BM25 не возвращает больше top_k результатов."""
    results = bm25_index.search("exception handling try except", top_k=2)
    assert len(results) <= 2


# ── is_relevant тест ──────────────────────────────────────────────────────────

def test_is_relevant_returns_false_for_zero_score():
    """is_relevant = False при score ниже любого порога."""
    chunks = [{"score": 0.0}, {"score": 0.0}]
    assert is_relevant(chunks, min_score=0.05) is False


def test_is_relevant_returns_true_for_high_score():
    """is_relevant = True при score выше явного порога."""
    chunks = [{"score": 0.5}]
    assert is_relevant(chunks, min_score=0.05) is True


# ── Дополнительный тест: качество ранжирования ───────────────────────────────

def test_tfidf_unrelated_query_scores_lower_than_related(tfidf_index):
    """
    Кастомный тест: нерелевантный запрос ('chocolate cake recipe')
    должен давать меньший max score, чем релевантный ('python list empty').
    Проверяет качество ранжирования.
    """
    relevant_results   = tfidf_index.search("python list empty check", top_k=1)
    irrelevant_results = tfidf_index.search("chocolate cake recipe baking", top_k=1)

    rel_score   = relevant_results[0]["score"]   if relevant_results   else 0.0
    irrel_score = irrelevant_results[0]["score"] if irrelevant_results else 0.0

    assert rel_score >= irrel_score, (
        f"Ожидали rel({rel_score:.4f}) >= irrel({irrel_score:.4f})"
    )
