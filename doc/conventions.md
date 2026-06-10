# Conventions — Соглашения о разработке

## Границы изменений

- **Минимальный diff**: каждое изменение затрагивает ровно один слой (chunker / retriever / generator / ui).
- **KISS**: никакой абстракции поверх абстракции — если задача решается в 5 строках, не пишем класс.
- Новые зависимости добавляются только в `pyproject.toml`, не устанавливаются вручную.

## Архитектура модулей

```
app/
  config.py     — все настройки, пути, константы (единственный источник правды)
  chunker.py    — только нарезка текста, без I/O
  retriever.py  — только поиск, загружает индекс при создании
  prompts.py    — только строки промптов и шаблоны
  generator.py  — генерация ответа, вызывает retriever + prompts
  main.py       — только Streamlit UI, оркестрирует retriever + generator

scripts/
  prepare_datasets.py  — I/O: Kaggle → datasets.json
  ingest.py            — I/O: datasets.json → documents.jsonl
  build_index.py       — I/O: documents.jsonl → data/index/*
  check_retrieval.py   — smoke-test retrieval
  check_generator.py   — smoke-test full pipeline
```

## Правила ответа

- Система отвечает **только** на основе найденных чанков.
- Если `max_score < MIN_SCORE` → возвращаем `REFUSAL_MESSAGE`, не придумываем.
- Источник (url, doc_id, score) обязательно присутствует в каждом ответе.

## Тестовые требования

- Все тесты в `tests/`, запускаются через `uv run pytest tests/ -v`.
- Тесты не обращаются к интернету и не требуют реального индекса (используют фикстуры).
- Минимум 8 тестов: модульные тесты в `tests/`, включая минимум 1 кастомный тест качества поиска.

## Правила по данным

- Сырые данные (`data/raw/datasets.json`) в `.gitignore` — не коммитить.
- Артефакты индекса (`data/index/`, `data/processed/`) в `.gitignore`.
- `data/raw/` содержит только `sample.json` (10 записей) для быстрого теста без Kaggle.

## Правила по зависимостям

- Нет прямых `import` из `scripts/` в `app/` (только наоборот).
- Конфигурация только через `app/config.py` и переменные окружения.
