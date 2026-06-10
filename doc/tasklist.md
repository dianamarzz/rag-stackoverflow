# Tasklist — Итерационный план реализации

## Таблица прогресса

| Итерация | Название | Статус | Проверка |
|---|---|---|---|
| 00 | Scaffold | ✅ Готово | `uv run python -c "import app.config"` |
| 01 | Demo Data | ✅ Готово | `python -c "import json; d=json.load(open('data/raw/datasets.json')); print(len(d['datasets']))"` |
| 02 | Ingestion | ✅ Готово | `uv run python scripts/ingest.py` |
| 03 | Chunking | ✅ Готово | `uv run pytest tests/test_chunking.py -v` |
| 04 | TF-IDF Index | ✅ Готово | `uv run python scripts/build_index.py` |
| 05 | Retrieval | ✅ Готово | `uv run python scripts/check_retrieval.py` |
| 06 | Demo Answer | ✅ Готово | `uv run python scripts/check_generator.py` |
| 07 | Streamlit UI | ✅ Готово | `uv run streamlit run app/main.py` |
| 08 | Tests + README | ✅ Готово | `uv run pytest tests/ -v` |

## Детали итераций

### Iter 00 — Scaffold
**Задачи:** pyproject.toml, .gitignore, app/config.py, папки.
**Проверка:** `uv venv && uv sync` без ошибок; `import app.config` работает.

### Iter 01 — Data
**Задачи:** `scripts/prepare_datasets.py` (Kaggle → datasets.json, 1 000 записей).
**Проверка:** JSON читается, в `datasets` ≥ 1 000 записей.

### Iter 02 — Ingestion
**Задачи:** `scripts/ingest.py` (datasets.json → documents.jsonl + метаданные).
**Проверка:** Число строк в documents.jsonl = числу записей в datasets.json.

### Iter 03 — Chunking
**Задачи:** `app/chunker.py` (paragraph-based, max_chars, overlap).
**Проверка:** `pytest tests/test_chunking.py` — 8 тестов зелёные.

### Iter 04 — Index
**Задачи:** `scripts/build_index.py` (ingest + chunk + TF-IDF fit + BM25 fit).
**Проверка:** В `data/index/` есть vectorizer.pkl, matrix.npz, bm25.pkl, chunks.jsonl.

### Iter 05 — Retrieval
**Задачи:** `app/retriever.py` (TFIDFRetriever, BM25Retriever, is_relevant).
**Проверка:** `check_retrieval.py` выводит score > 0 для Python-вопросов.

### Iter 06 — Generation
**Задачи:** `app/prompts.py`, `app/generator.py` (шаблон + опциональный LLM).
**Проверка:** `check_generator.py` — 4/4 тестов PASS, negative → отказ.

### Iter 07 — Streamlit UI
**Задачи:** `app/main.py` с кнопками demo, колонками, expander для чанков.
**Проверка:** Браузер открывается, demo-кнопки работают, показываются doc_id и score.

### Iter 08 — Tests + README
**Задачи:** test_chunking.py (8 тестов), test_retrieval.py (кастомный тест качества).
**Проверка:** `pytest tests/ -v` — все зелёные.

## Текущая итерация и готовность MVP

Все итерации завершены.

## Критерии завершения MVP

- [x] `uv sync` без ошибок
- [x] `build_index.py` строит индекс из 1 000+ записей
- [x] 3 demo-вопроса дают ответ с источниками
- [x] Negative-вопрос даёт явный отказ
- [x] `pytest tests/ -v` — все тесты зелёные
- [x] Streamlit UI показывает doc_id, score, url
