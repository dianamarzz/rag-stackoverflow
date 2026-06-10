SYSTEM_PROMPT = """Ты — ассистент-эксперт по Python, который отвечает ТОЛЬКО на основе предоставленного контекста из Stack Overflow.

Правила:
1. Используй ТОЛЬКО информацию из блока <context>. Не придумывай и не добавляй знания извне.
2. Если контекст не содержит достаточной информации для ответа — честно скажи об этом.
3. Приводи примеры кода из контекста.
4. В конце ответа всегда указывай источники (поле source из контекста).
5. Отвечай на языке вопроса (русский или английский).
"""

ANSWER_TEMPLATE = """\
На основе найденных фрагментов Stack Overflow:

{answer}

---
📚 Источники:
{sources}
"""

REFUSAL_MESSAGE = (
    "❌ В базе знаний не найдено релевантной информации по вашему запросу.\n\n"
    "Попробуйте переформулировать вопрос или убедитесь, что он касается Python-программирования."
)

def build_user_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] {c.get('name','')}\nURL: {c.get('url','')}\n\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"<context>\n{context}\n</context>\n\nВопрос: {query}"
