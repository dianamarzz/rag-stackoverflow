"""Тесты для app/chunker.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.chunker import chunk_text, chunk_document


# ── chunk_text ────────────────────────────────────────────────────────────────

def test_empty_string_returns_empty():
    """Пустой текст → пустой список чанков."""
    assert chunk_text("") == []


def test_whitespace_only_returns_empty():
    """Только пробелы → пустой список."""
    assert chunk_text("   \n\n   ") == []


def test_short_text_single_chunk():
    """Текст короче max_chars → ровно один чанк."""
    text = "How do I sort a list?"
    chunks = chunk_text(text, max_chars=200)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_splits_into_multiple_chunks():
    """Длинный текст разбивается на несколько чанков."""
    text = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
    chunks = chunk_text(text, max_chars=60, overlap=10)
    assert len(chunks) > 1


def test_chunks_do_not_exceed_max_chars():
    """Ни один чанк не превышает max_chars."""
    text = " ".join([f"word{i}" for i in range(500)])
    max_chars = 100
    for chunk in chunk_text(text, max_chars=max_chars, overlap=0):
        assert len(chunk) <= max_chars, f"Чанк длиной {len(chunk)} > {max_chars}"


def test_no_chunk_is_empty():
    """Все чанки непустые."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    for chunk in chunk_text(text, max_chars=50):
        assert chunk.strip() != ""


# ── chunk_document ────────────────────────────────────────────────────────────

def test_chunk_document_inherits_metadata():
    """Каждый чанк наследует doc_id, name, url."""
    doc = {
        "doc_id": "so_42",
        "name":   "Test Question",
        "url":    "https://stackoverflow.com/questions/42",
        "text":   "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
    }
    chunks = list(chunk_document(doc, max_chars=30, overlap=5))
    assert len(chunks) >= 1
    for c in chunks:
        assert c["doc_id"] == "so_42"
        assert c["name"]   == "Test Question"
        assert c["url"]    == "https://stackoverflow.com/questions/42"


def test_chunk_document_unique_chunk_ids():
    """У каждого чанка уникальный chunk_id."""
    doc = {
        "doc_id": "so_99",
        "text": "\n\n".join([f"Paragraph {i} with some content here." for i in range(10)]),
    }
    chunks = list(chunk_document(doc, max_chars=50, overlap=10))
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Дублирующиеся chunk_id"
