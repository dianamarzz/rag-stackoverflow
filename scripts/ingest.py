"""
ingest.py
---------
Читает data/raw/datasets.json и преобразует в data/processed/documents.jsonl.
Добавляет поле doc_id и нормализует структуру.

Запуск:
  uv run python scripts/ingest.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DATASETS_FILE, DOCUMENTS_FILE, DATA_PROCESSED


def ingest():
    if not DATASETS_FILE.exists():
        print(f"[error] Файл не найден: {DATASETS_FILE}")
        print("  → Сначала запустите: uv run python scripts/prepare_datasets.py")
        sys.exit(1)

    with open(DATASETS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("datasets", data) if isinstance(data, dict) else data
    print(f"[ingest] Загружено записей: {len(records):,}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            doc = {
                "doc_id":      rec.get("id", f"doc_{i}"),
                "name":        rec.get("title", rec.get("name", f"Document {i}")),
                "text":        rec.get("text", ""),
                "url":         rec.get("url", ""),
                "tags":        rec.get("tags", []),
                "score":       rec.get("score", 0),
                "source_file": str(DATASETS_FILE.name),
            }
            if not doc["text"].strip():
                continue
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

    print(f"[ingest] Записано документов: {written:,}  →  {DOCUMENTS_FILE}")


if __name__ == "__main__":
    ingest()
