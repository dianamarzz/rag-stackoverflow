# 🐍 RAG — Stack Overflow Python Q&A

RAG-система для поиска по реальным вопросам и принятым ответам Stack Overflow (Python).  
Задайте вопрос на естественном языке — система найдёт релевантные фрагменты и сформирует ответ **только** из них, с указанием источников.

---

## Требования

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — менеджер пакетов

---

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/dianamarzz/rag-stackoverflow
cd rag-stackoverflow
uv sync
```

### 2. Построить индекс

В репозитории уже есть `data/raw/datasets.json` — 1 000 Python-вопросов с принятыми ответами из Stack Overflow.

```bash
uv run python scripts/build_index.py
```

```
==================================================
  RAG StackOverflow — Build Index
==================================================
[load]   Документов: 1,000
[chunk]  Всего чанков: 4,265
[tfidf]  Матрица: 4,265 чанков × 47918 признаков
[tfidf]  Сохранено → vectorizer.pkl, matrix.npz
[bm25]   Сохранено → bm25.pkl

✅ Индекс успешно построен!
   Документов : 1,000
   Чанков     : 4,265
```

### 3. Запустить интерфейс

```bash
uv run streamlit run app/main.py
```

Откроется браузер

---

## Примеры запросов

| Вопрос | Результат |
|---|---|
| How do I check if a list is empty in Python? | Ответ с примером кода и ссылкой на SO |
| How to sort a dictionary by value in Python? | Ответ с примером кода и ссылкой на SO |
| What is the difference between @classmethod and @staticmethod? | Ответ с примером кода и ссылкой на SO |
| What is the best recipe for a chocolate cake? | Отказ — вопрос не относится к Python |

---

## Демо-ответы

<img width="1250" height="705" alt="Screenshot 2026-06-10 at 10 49 39 PM" src="https://github.com/user-attachments/assets/641cc777-965b-47b0-8cd0-e87791ea204b" />

<img width="1239" height="659" alt="Screenshot 2026-06-10 at 10 50 21 PM" src="https://github.com/user-attachments/assets/23ff6aa9-f1f6-4ba2-9ad9-bd1507bfb659" />

<img width="1231" height="672" alt="Screenshot 2026-06-10 at 10 50 37 PM" src="https://github.com/user-attachments/assets/9516f123-f13e-430a-8344-5c813dde68b1" />

<img width="1244" height="643" alt="Screenshot 2026-06-10 at 11 01 07 PM" src="https://github.com/user-attachments/assets/8b4976e4-00a1-4ea4-b1c3-88c24b21ae00" />


---

## Тесты

```bash
uv run pytest tests/ -v
```

```
tests/test_chunking.py::test_empty_string_returns_empty                       PASSED
tests/test_chunking.py::test_whitespace_only_returns_empty                    PASSED
tests/test_chunking.py::test_short_text_single_chunk                          PASSED
tests/test_chunking.py::test_long_text_splits_into_multiple_chunks            PASSED
tests/test_chunking.py::test_chunks_do_not_exceed_max_chars                   PASSED
tests/test_chunking.py::test_no_chunk_is_empty                                PASSED
tests/test_chunking.py::test_chunk_document_inherits_metadata                 PASSED
tests/test_chunking.py::test_chunk_document_unique_chunk_ids                  PASSED
tests/test_retrieval.py::test_tfidf_returns_results                           PASSED
tests/test_retrieval.py::test_tfidf_results_have_score                        PASSED
tests/test_retrieval.py::test_tfidf_top_k_respected                           PASSED
tests/test_retrieval.py::test_tfidf_relevant_query_has_positive_score         PASSED
tests/test_retrieval.py::test_bm25_returns_results                            PASSED
tests/test_retrieval.py::test_bm25_top_k_respected                            PASSED
tests/test_retrieval.py::test_is_relevant_returns_false_for_zero_score        PASSED
tests/test_retrieval.py::test_is_relevant_returns_true_for_high_score         PASSED
tests/test_retrieval.py::test_tfidf_unrelated_query_scores_lower_than_related PASSED

17 passed in 3.03s
```

---

## Реализованные улучшения

### BM25 вместо TF-IDF

По умолчанию используется BM25 (Best Match 25) — стандарт information retrieval для технических текстов. В отличие от TF-IDF, BM25 учитывает длину документа и насыщение частоты термина, что даёт более точное ранжирование для коротких точных запросов.

**Сравнение на запросе** *"how to sort dictionary by value"*:

| Режим | Top-1 результат | Score |
|---|---|---|
| TF-IDF | so_613183 — "How do I sort a dictionary by value?" | 0.52 |
| BM25 | so_613183 — "How do I sort a dictionary by value?" | 8.20 |

Оба режима находят правильный документ, но BM25 даёт более выраженный разрыв между релевантным и нерелевантным результатом, что позволяет точнее работать порогу отказа.

Переключить режим:
```bash
RETRIEVAL_MODE=tfidf uv run streamlit run app/main.py   # TF-IDF
RETRIEVAL_MODE=bm25  uv run streamlit run app/main.py   # BM25 (по умолчанию)
```

### Генерация через LLM (опционально)

При наличии `ANTHROPIC_API_KEY` генератор вместо сырой конкатенации чанков вызывает Claude, который формирует связный ответ строго по найденному контексту:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
uv run streamlit run app/main.py
```

Подробнее: [IMPROVEMENTS.md](IMPROVEMENTS.md)


## Архитектура

```
ingest → chunking → index (TF-IDF + BM25) → retrieval → generation → UI
```

```
rag-stackoverflow/
├── app/
│   ├── config.py          — настройки и пути
│   ├── chunker.py         — нарезка текста по абзацам
│   ├── retriever.py       — TF-IDF и BM25 поиск
│   ├── prompts.py         — шаблоны промптов
│   ├── generator.py       — генерация ответа
│   └── main.py            — Streamlit UI
├── scripts/
│   ├── prepare_datasets.py  — пересборка датасета с Kaggle
│   ├── ingest.py            — нормализация в documents.jsonl
│   ├── build_index.py       — сборка индекса
│   ├── check_retrieval.py   — проверка поиска
│   └── check_generator.py   — проверка полного pipeline
├── data/raw/datasets.json   — исходные данные (1 000 записей)
├── tests/                   — pytest-тесты
├── doc/                     — документация проекта
├── pyproject.toml
└── IMPROVEMENTS.md
```

---

## Данные

| Параметр | Значение |
|---|---|
| Источник | [stackoverflow/stacksample](https://www.kaggle.com/datasets/stackoverflow/stacksample) |
| Лицензия | CC BY-SA 3.0 |
| Фильтр | Тег `python`, score ≥ 5, есть принятый ответ |
| Размер | 1 000 записей |
| Чанков после нарезки | ~3 000–5 000 |

Подробнее о данных: [doc/DATA.md](doc/DATA.md)  
Описание улучшений: [IMPROVEMENTS.md](IMPROVEMENTS.md)
