"""
Chunker: нарезает текст документа на перекрывающиеся чанки по абзацам.
"""
from __future__ import annotations
import re
from typing import Iterator
from app.config import MAX_CHARS, OVERLAP


def _split_paragraphs(text: str) -> list[str]:
    """Делит текст на абзацы по двойному переносу или одиночному \\n."""
    parts = re.split(r"\n{2,}", text.strip())
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            result.append(part)
    return result


def chunk_text(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
) -> list[str]:
    """
    Нарезает текст на чанки размером ≤ max_chars символов.
    Соседние чанки перекрываются на `overlap` символов.
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # Если абзац сам по себе длиннее max_chars — нарежем его принудительно
        if len(para) > max_chars:
            if current:
                chunks.append(current.strip())
                current = current[-overlap:] if overlap else ""
            for start in range(0, len(para), max_chars - overlap):
                piece = para[start : start + max_chars]
                chunks.append(piece.strip())
            current = para[-overlap:] if overlap else ""
            continue

        # Иначе набираем абзацы, пока не переполним буфер
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # Берём хвост предыдущего чанка как контекст
            tail = current[-overlap:].strip() if overlap and current else ""
            current = (tail + "\n\n" + para).strip() if tail else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_document(doc: dict, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> Iterator[dict]:
    """
    Принимает документ (dict с полем 'text') и возвращает чанки.
    Каждый чанк наследует метаданные документа.
    """
    text = doc.get("text", "")
    pieces = chunk_text(text, max_chars=max_chars, overlap=overlap)
    doc_id = doc.get("doc_id", doc.get("id", "unknown"))

    for i, piece in enumerate(pieces):
        yield {
            "chunk_id": f"{doc_id}__c{i}",
            "doc_id":   doc_id,
            "name":     doc.get("name", ""),
            "url":      doc.get("url", ""),
            "tags":     doc.get("tags", []),
            "text":     piece,
        }
