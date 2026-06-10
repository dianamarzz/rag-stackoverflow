"""
check_generator.py
------------------
Проверяет полный RAG pipeline: retrieval + generation.
Запуск:
  uv run python scripts/check_generator.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever import load_retriever
from app.generator import generate
from app.config import USE_LLM, RETRIEVAL_MODE

TESTS = [
    {
        "label": "DEMO-1",
        "query": "How do I check if a list is empty in Python?",
        "expect_refusal": False,
    },
    {
        "label": "DEMO-2",
        "query": "How to sort a dictionary by value in Python?",
        "expect_refusal": False,
    },
    {
        "label": "DEMO-3",
        "query": "What is the difference between @classmethod and @staticmethod?",
        "expect_refusal": False,
    },
    {
        "label": "NEGATIVE",
        "query": "What is the best recipe for a chocolate cake?",
        "expect_refusal": True,
    },
]


def main():
    mode = f"{RETRIEVAL_MODE} | LLM={'on' if USE_LLM else 'off (template)'}"
    print(f"[check_generator] Режим: {mode}\n")

    try:
        retriever = load_retriever()
    except FileNotFoundError:
        print("Индекс не найден. Запустите сначала build_index.py")
        sys.exit(1)

    passed = 0
    for t in TESTS:
        print(f"{'═'*60}")
        print(f"  {t['label']}: {t['query']}")
        chunks  = retriever.search(t["query"])
        answer  = generate(t["query"], chunks)
        refused = "❌" in answer or "не найдено" in answer.lower()

        status = "✅ PASS" if refused == t["expect_refusal"] else "❌ FAIL"
        if refused == t["expect_refusal"]:
            passed += 1

        print(f"  {status}  (ожидали {'отказ' if t['expect_refusal'] else 'ответ'})")
        print()
        print(answer[:500])
        print()

    print(f"{'═'*60}")
    print(f"Итого: {passed}/{len(TESTS)} тестов прошли")


if __name__ == "__main__":
    main()
