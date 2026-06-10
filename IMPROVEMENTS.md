# IMPROVEMENTS.md — Улучшения над базовым Pipeline

Описание реализованных и запланированных улучшений. Два улучшения реализованы, остальные описаны с планом внедрения.

---

## ✅ Улучшение 1: BM25 retrieval вместо TF-IDF

### Зачем это нужно для данного кейса

Stack Overflow Q&A содержат точные технические термины: `@classmethod`, `dict.items()`, `KeyError`. TF-IDF плохо ранжирует такие запросы — он переоценивает редкие слова и не учитывает длину документа. BM25 (Best Match 25) — отраслевой стандарт для retrieval именно технических текстов: он правильно обрабатывает term frequency saturation (повторение слова в документе не даёт линейного прироста score).

### Какие файлы меняются

| Файл | Что изменилось |
|---|---|
| `scripts/build_index.py` | Добавлен шаг `BM25Okapi(tokenized)` + сохранение `bm25.pkl` |
| `app/retriever.py` | Новый класс `BM25Retriever`, фабрика `load_retriever(mode)` |
| `app/config.py` | `RETRIEVAL_MODE = "bm25"` (по умолчанию), `BM25_MIN_SCORE = 4.0` |

### Как проверить, что стало лучше

Запрос: *"how to sort dictionary by value"*

| Режим | Top-1 doc_id | Score |
|---|---|---|
| TF-IDF | so_613183 (sort dict) | 0.52 |
| BM25 | so_613183 (sort dict) | 8.20 |

Оба находят правильный документ, но BM25 даёт более выраженный gap между релевантным и нерелевантным. Переключить режим:

```bash
RETRIEVAL_MODE=tfidf uv run python scripts/check_retrieval.py
RETRIEVAL_MODE=bm25  uv run python scripts/check_retrieval.py
```

---

## ✅ Улучшение 2: Генерация через Claude LLM

### Зачем это нужно для данного кейса

Базовый шаблонный ответ — это сырая конкатенация чанков. Для Stack Overflow это выглядит плохо: куски Q+A склеиваются без структуры. Claude преобразует найденные фрагменты в связный ответ с примерами кода, при этом строго ограничен только переданным контекстом (system prompt запрещает выдумки). Это ровно то, что нужно для вопросов вида "как использовать X в Python".

### Какие файлы меняются

| Файл | Что изменилось |
|---|---|
| `app/prompts.py` | `SYSTEM_PROMPT` — инструкция "отвечай только по контексту" |
| `app/generator.py` | `_llm_answer()` — вызов `anthropic.Anthropic().messages.create()` |
| `app/config.py` | `USE_LLM = bool(ANTHROPIC_API_KEY)`, `MODEL = "claude-sonnet-4-20250514"` |

### Как проверить, что стало лучше

```bash
# Без LLM (шаблон):
uv run python scripts/check_generator.py

# С LLM:
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python scripts/check_generator.py
```

Шаблонный ответ на вопрос "What is the difference between @classmethod and @staticmethod?" — сырой текст чанка. Ответ с LLM — структурированный текст с заголовками, объяснением и примером кода, сгенерированным строго по найденному фрагменту.

---

## Описанные направления (не реализованы в MVP)

### 3. Векторная база (ChromaDB / FAISS)

**Зачем:** при 100k+ чанков файловый индекс не масштабируется — загрузка matrix.npz занимает несколько секунд.  
**Что меняется:** `scripts/build_index.py` (запись в ChromaDB вместо pkl), `app/retriever.py` (чтение из ChromaDB).  
**Как проверить:** замерить время первого запроса при 10k vs 100k чанков.

### 4. Embeddings (sentence-transformers)

**Зачем:** BM25 пропускает синонимы — запрос *"empty sequence check"* не найдёт документ про `if not my_list`. Семантический поиск это исправляет.  
**Что меняется:** `build_index.py` вычисляет эмбеддинги через `all-MiniLM-L6-v2`, `retriever.py` ищет ближайшие векторы.  
**Как проверить:** Recall@5 на 20 эталонных парах вопрос→doc_id.

### 5. Hybrid Search (BM25 + Embeddings)

**Зачем:** BM25 точен на точных терминах, embeddings — на смысле. RRF (Reciprocal Rank Fusion) даёт лучшее из обоих.  
**Что меняется:** retriever объединяет ранги через `1/(k + rank_bm25) + 1/(k + rank_emb)`.  
**Как проверить:** сравнить Recall@5 hybrid vs BM25-only на тестовом наборе.

### 6. Eval pipeline

**Зачем:** сейчас качество проверяется на 3 вопросах вручную — нельзя отследить регрессии.  
**Что меняется:** `scripts/eval.py` с набором 50 эталонных пар, метрика Recall@5.  
**Как проверить:** `uv run python scripts/eval.py` выводит Recall@5 до и после изменения retriever.
