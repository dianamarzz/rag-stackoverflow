"""
prepare_datasets.py
-------------------
Скачивает датасет 'stackoverflow/stacksample' с Kaggle,
объединяет Questions.csv + Answers.csv, фильтрует Python-вопросы
с принятым ответом и сохраняет data/raw/datasets.json (1000+ записей).

Требования перед запуском:
  1. pip install kaggle   (или уже в зависимостях через uv sync)
  2. Положить ~/.kaggle/kaggle.json с вашим API-ключом
     (скачать на https://www.kaggle.com/settings → API → Create New Token)
  3. Принять правила датасета: https://www.kaggle.com/datasets/stackoverflow/stacksample

Запуск:
  uv run python scripts/prepare_datasets.py
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ── Пути ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
DOWNLOAD_DIR = RAW_DIR / "kaggle_raw"
OUT_FILE = RAW_DIR / "datasets.json"

KAGGLE_DATASET = "stackoverflow/stacksample"
TARGET_TAG     = "python"
MIN_SCORE      = 5        # минимальный рейтинг вопроса
TARGET_RECORDS = 1000     # сколько записей хотим в итоге


# ── Утилиты ───────────────────────────────────────────────────────────────────

def clean_html(html_text: str) -> str:
    """Убирает HTML-теги из тела вопроса/ответа."""
    if not isinstance(html_text, str):
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    # Заменяем <code> блоки на обычный текст с сохранением контента
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")
    for pre in soup.find_all("pre"):
        pre.replace_with("\n```\n" + pre.get_text() + "\n```\n")
    return soup.get_text(separator="\n").strip()


def so_url(question_id: int) -> str:
    return f"https://stackoverflow.com/questions/{question_id}"


# ── Шаг 1: скачать датасет ────────────────────────────────────────────────────

def download_dataset():
    if DOWNLOAD_DIR.exists() and any(DOWNLOAD_DIR.glob("*.csv")):
        print(f"[skip] Данные уже скачаны в {DOWNLOAD_DIR}")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Скачиваем {KAGGLE_DATASET} …")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(DOWNLOAD_DIR),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("ОШИБКА при скачивании:\n", result.stderr)
        sys.exit(1)

    # Распаковываем zip
    for zf_path in DOWNLOAD_DIR.glob("*.zip"):
        print(f"[unzip] {zf_path.name}")
        with zipfile.ZipFile(zf_path) as zf:
            zf.extractall(DOWNLOAD_DIR)
        zf_path.unlink()

    print("[ok] Датасет скачан и распакован.")


# ── Шаг 2: обработать CSV ─────────────────────────────────────────────────────

def build_datasets_json():
    print("[load] Читаем Questions.csv …")
    q_path = DOWNLOAD_DIR / "Questions.csv"
    a_path = DOWNLOAD_DIR / "Answers.csv"
    t_path = DOWNLOAD_DIR / "Tags.csv"

    if not q_path.exists():
        # Иногда файлы лежат во вложенной папке
        candidates = list(DOWNLOAD_DIR.rglob("Questions.csv"))
        if not candidates:
            print("Не найден Questions.csv. Проверьте содержимое:", list(DOWNLOAD_DIR.iterdir()))
            sys.exit(1)
        q_path = candidates[0]
        a_path = q_path.parent / "Answers.csv"
        t_path = q_path.parent / "Tags.csv"

    # Читаем с обработкой ошибок кодировки
    questions = pd.read_csv(q_path, encoding="latin-1", low_memory=False)
    answers   = pd.read_csv(a_path, encoding="latin-1", low_memory=False)
    tags      = pd.read_csv(t_path, encoding="latin-1", low_memory=False)

    print(f"  Questions: {len(questions):,}  |  Answers: {len(answers):,}  |  Tags: {len(tags):,}")

    # Фильтруем Python-вопросы
    python_ids = set(tags.loc[tags["Tag"] == TARGET_TAG, "Id"].unique())
    print(f"[filter] Python-вопросов: {len(python_ids):,}")

    # Фильтруем по score (AcceptedAnswerId в StackSample отсутствует)
    q = questions[
        (questions["Id"].isin(python_ids)) &
        (questions["Score"] >= MIN_SCORE)
    ].copy()
    print(f"[filter] После фильтрации (score≥{MIN_SCORE}): {len(q):,}")

    # Берём лучший ответ для каждого вопроса по Score
    # В Answers.csv колонка ParentId = Id вопроса
    best_answers = (
        answers[answers["ParentId"].isin(q["Id"])]
        .sort_values("Score", ascending=False)
        .drop_duplicates("ParentId")
        [["ParentId", "Body", "Score"]]
        .rename(columns={"ParentId": "Id", "Body": "AnswerBody", "Score": "AnswerScore"})
    )
    merged = q.merge(best_answers, on="Id", how="inner")
    print(f"[merge] После джойна с ответами: {len(merged):,}")

    # Сортируем по рейтингу, берём топ TARGET_RECORDS
    merged = merged.sort_values("Score", ascending=False).head(TARGET_RECORDS)

    # Строим список записей
    records = []
    for _, row in merged.iterrows():
        q_text  = clean_html(str(row.get("Body", "")))
        a_text  = clean_html(str(row.get("AnswerBody", "")))
        title   = str(row.get("Title", "")).strip()
        qid     = int(row["Id"])

        # Объединяем вопрос и ответ в один текстовый документ
        full_text = f"Q: {title}\n\n{q_text}\n\nA: {a_text}"

        records.append({
            "id":     f"so_{qid}",
            "title":  title,
            "text":   full_text,
            "tags":   [TARGET_TAG],
            "score":  int(row.get("Score", 0)),
            "url":    so_url(qid),
        })

    print(f"[build] Итого записей: {len(records):,}")

    # Сохраняем
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"datasets": records}
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[done] Сохранено → {OUT_FILE}")
    print(f"        Записей: {len(records):,}  (цель «отлично»: {TARGET_RECORDS}+)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_dataset()
    build_datasets_json()
