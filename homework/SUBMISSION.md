# SUBMISSION.md

## Ссылка на репозиторий

**Repo URL:** `https://github.com/dianamarzz/rag-stackoverflow`

## Автор

**ФИО / ник:** Marzaganova Diana (dianamarzz)

## Комментарий

**Данные:** датасет [stackoverflow/stacksample](https://www.kaggle.com/datasets/stackoverflow/stacksample) (Kaggle, CC BY-SA 3.0) — 1 000 Python-вопросов с принятыми ответами (фильтр: тег `python`, score ≥ 5). Вопрос и принятый ответ объединяются в одно текстовое поле и нарезаются на более 4 000 чанков.

**Pipeline:** `prepare_datasets.py → ingest.py → build_index.py (chunking + TF-IDF + BM25) → retrieval → generation → Streamlit UI`.

**Улучшения:**
- **BM25** (реализовано) — заменяет TF-IDF как основной метод retrieval; точнее ранжирует технические термины, учитывает длину документа. По умолчанию включён через `RETRIEVAL_MODE=bm25`.
- **Генерация через Claude API** (реализовано) — при наличии `ANTHROPIC_API_KEY` генератор формирует связный ответ строго по найденному контексту вместо сырой конкатенации чанков.

## Demo-вопросы

| № | Вопрос | Результат |
|---|---|---|
| 1 | How do I check if a list is empty in Python? | ✅ Ответ с источниками |
| 2 | How to sort a dictionary by value in Python? | ✅ Ответ с источниками |
| 3 | What is the difference between @classmethod and @staticmethod? | ✅ Ответ с источниками |
| N | What is the best recipe for a chocolate cake? | ❌ Отказ — нет релевантных данных |

## Запуск

```bash
git clone https://github.com/dianamarzz/rag-stackoverflow
cd rag-stackoverflow
uv sync
uv run python scripts/build_index.py
uv run streamlit run app/main.py
```
