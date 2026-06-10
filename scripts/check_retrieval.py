"""
check_retrieval.py
------------------
Проверяет работу retrieval на 3 demo-вопросах и 1 negative-вопросе.
Запуск:
  uv run python scripts/check_retrieval.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.retriever import load_retriever, is_relevant
from app.config import RETRIEVAL_MODE

DEMO_QUESTIONS = [
    ("DEMO-1 [релевантный]",
     "How do I check if a list is empty in Python?"),
    ("DEMO-2 [релевантный]",
     "How to sort a dictionary by value in Python?"),
    ("DEMO-3 [релевантный]",
     "What is the difference between @classmethod and @staticmethod?"),
    ("DEMO-4 [negative — нерелевантный]",
     "What is the best recipe for a chocolate cake?"),
]


def main():
    print(f"[check_retrieval] Режим: {RETRIEVAL_MODE}\n")
    try:
        retriever = load_retriever()
    except FileNotFoundError:
        print("Индекс не найден. Сначала запустите:")
        print("  uv run python scripts/build_index.py")
        sys.exit(1)

    for label, query in DEMO_QUESTIONS:
        print(f"{'─'*60}")
        print(f"  {label}")
        print(f"  Запрос: {query}")
        results = retriever.search(query, top_k=3)
        relevant = is_relevant(results)
        print(f"  Релевантен: {'✅ да' if relevant else '❌ нет (отказ)'}")
        for i, r in enumerate(results[:3], 1):
            print(f"    [{i}] score={r['score']:.4f}  |  {r.get('name','')[:60]}")
            print(f"         {r['text'][:120].replace(chr(10),' ')} …")
        print()

    print("✅ check_retrieval завершён")


if __name__ == "__main__":
    main()
