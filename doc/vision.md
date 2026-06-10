# Vision — Техническое видение MVP

## Технологии

| Слой | Инструмент |
|---|---|
| Язык | Python 3.10+ |
| Пакетный менеджер | `uv` |
| UI | Streamlit |
| Retrieval (base) | TF-IDF + cosine similarity (scikit-learn) |
| Retrieval (улучшение) | BM25 (rank-bm25) — реализовано |
| Generation (base) | Шаблонная конкатенация чанков |
| Generation (улучшение) | Claude API через `anthropic` — реализовано |
| Тесты | pytest |

## Как строится индекс

1. `prepare_datasets.py` → загружает StackSample с Kaggle, фильтрует Python-вопросы с принятым ответом, объединяет Q+A в одно текстовое поле, сохраняет `data/raw/datasets.json` (1 000+ записей).
2. `ingest.py` → нормализует JSON в `documents.jsonl` (добавляет `doc_id`, `source_file`).
3. `build_index.py` → нарезает документы на чанки (paragraph-based), обучает TF-IDF, строит BM25, сохраняет артефакты в `data/index/`.

## Как работает поиск

- Запрос трансформируется тем же vectorizer-ом, что использовался при fit.
- Считается cosine similarity (TF-IDF) или BM25 score по всем чанкам.
- Возвращается top-K чанков.
- Если max score < `MIN_SCORE` → система возвращает отказ.

## Что НЕ входит в MVP

- Векторные базы данных (ChromaDB, Qdrant, FAISS).
- Embeddings (sentence-transformers) — описаны в IMPROVEMENTS.md.
- Аутентификация пользователей.
- Асинхронная обработка запросов.
- Обновление индекса в реальном времени.

## Как запускать проект

```bash
# 1. Установить зависимости
uv sync

# 2. Построить индекс (данные уже в репозитории)
uv run python scripts/build_index.py

# 3. Запустить UI
uv run streamlit run app/main.py
```

Для пересборки датасета из источника:
```bash
uv run python scripts/prepare_datasets.py
```
