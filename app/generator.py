"""
Generator: формирует ответ из найденных чанков.
Базовый режим — шаблон.
Улучшенный режим — вызов Claude API (если задан ANTHROPIC_API_KEY).
"""
from __future__ import annotations

from app.config import USE_LLM, ANTHROPIC_API_KEY, MODEL
from app.prompts import (
    SYSTEM_PROMPT, ANSWER_TEMPLATE, REFUSAL_MESSAGE, build_user_prompt
)
from app.retriever import is_relevant


def _format_sources(chunks: list[dict]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for c in chunks:
        url = c.get("url", "")
        name = c.get("name", c.get("doc_id", ""))
        score = c.get("score", 0.0)
        key = url or name
        if key and key not in seen:
            seen.add(key)
            lines.append(f"• [{name}]({url})  (score: {score:.3f})")
    return "\n".join(lines) if lines else "источники не указаны"


def _template_answer(chunks: list[dict]) -> str:
    """Простой шаблонный ответ — конкатенация найденных фрагментов."""
    body = "\n\n".join(c["text"] for c in chunks)
    sources = _format_sources(chunks)
    return ANSWER_TEMPLATE.format(answer=body, sources=sources)


def _llm_answer(query: str, chunks: list[dict]) -> str:
    """Ответ через Claude API (improvement #2 — LLM generation)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        user_msg = build_user_prompt(query, chunks)
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        answer_text = response.content[0].text
        sources = _format_sources(chunks)
        return ANSWER_TEMPLATE.format(answer=answer_text, sources=sources)
    except Exception as exc:
        # Fallback к шаблону при любой ошибке API
        return _template_answer(chunks) + f"\n\n⚠️ LLM недоступен: {exc}"


def generate(query: str, chunks: list[dict]) -> str:
    """
    Главная точка входа генератора.
    Возвращает REFUSAL_MESSAGE, если нет релевантных результатов.
    """
    if not is_relevant(chunks):
        return REFUSAL_MESSAGE

    if USE_LLM:
        return _llm_answer(query, chunks)
    return _template_answer(chunks)
