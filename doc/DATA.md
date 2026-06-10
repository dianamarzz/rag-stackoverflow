# DATA.md — Описание данных

## Источник

**Датасет:** [stackoverflow/stacksample](https://www.kaggle.com/datasets/stackoverflow/stacksample)
**Платформа:** Kaggle
**Лицензия:** CC BY-SA 3.0 (Stack Overflow данные)

StackSample содержит 10% вопросов и ответов Stack Overflow — около 1.26 млн вопросов и 1.97 млн ответов.

## Что индексируется

Из всего датасета выбираем **Python-вопросы** со следующими критериями:

| Критерий | Значение |
|---|---|
| Тег | `python` |
| Минимальный score вопроса | ≥ 5 |
| Наличие принятого ответа | обязательно |
| Количество итоговых записей | ≥ 1 000 (топ по score) |

## Структура datasets.json

```json
{
  "datasets": [
    {
      "id":    "so_11227809",
      "title": "How do I check if a list is empty?",
      "text":  "Q: How do I check if a list is empty?\n\n<вопрос>\n\nA: <принятый ответ>",
      "tags":  ["python"],
      "score": 10234,
      "url":   "https://stackoverflow.com/questions/11227809"
    },
    ...
  ]
}
```

## Предобработка

1. **HTML → plain text:** тела вопросов и ответов хранятся в HTML. `BeautifulSoup` убирает теги, `<code>` оборачивается в backticks, `<pre>` — в code blocks.
2. **Объединение Q+A:** вопрос и принятый ответ конкатенируются в одно поле `text` для более богатого контекста при retrieval.
3. **Фильтрация:** отбрасываются записи с пустым текстом после очистки.

## Масштаб после обработки

| Этап | Количество |
|---|---|
| Сырые Python-вопросы с тегом `python` | ~250 000 |
| После фильтра (score≥5, есть ответ) | ~50 000 |
| Взято топ по score | **1 000** |
| Чанков после нарезки (max_chars=600) | **~3 000–5 000** |

## Воспроизводимость

Данные уже включены в репозиторий (`data/raw/datasets.json`). Для пересборки датасета из источника используется скрипт `scripts/prepare_datasets.py` (требует Kaggle API key и принятия лицензии датасета).

```bash
# Пересобрать индекс из существующих данных:
uv run python scripts/build_index.py
```
