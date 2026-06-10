"""
main.py — Streamlit UI для RAG StackOverflow Python Q&A
Запуск: uv run streamlit run app/main.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from app.config import TOP_K, RETRIEVAL_MODE, USE_LLM, INDEX_CHUNKS, VECTORIZER_FILE
from app.retriever import load_retriever, is_relevant
from app.generator import generate

# ── Страница ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SO Python RAG",
    page_icon="🐍",
    layout="wide",
)

st.title("🐍 Stack Overflow Python Q&A — RAG")
st.caption(
    f"Режим поиска: **{RETRIEVAL_MODE.upper()}** | "
    f"Генерация: **{'Claude LLM' if USE_LLM else 'шаблон (без LLM)'}** | "
    f"Top-K: **{TOP_K}**"
)

# ── Проверка индекса ──────────────────────────────────────────────────────────
index_ready = VECTORIZER_FILE.exists() and INDEX_CHUNKS.exists()

if not index_ready:
    st.error(
        "⚠️ Индекс не построен. Выполните в терминале:\n\n"
        "```\n"
        "uv run python scripts/prepare_datasets.py\n"
        "uv run python scripts/build_index.py\n"
        "```"
    )
    st.stop()

# ── Загрузка ретривера (кешируем) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Загружаем индекс …")
def get_retriever():
    return load_retriever()

retriever = get_retriever()

# ── Demo-вопросы ──────────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "How do I check if a list is empty in Python?",
    "How to sort a dictionary by value in Python?",
    "What is the difference between @classmethod and @staticmethod?",
    "What is the best recipe for a chocolate cake?",   # negative
]

st.sidebar.header("📋 Demo-вопросы")
st.sidebar.markdown("Нажмите, чтобы подставить вопрос:")
for q in DEMO_QUESTIONS:
    if st.sidebar.button(q, key=q):
        st.session_state["query_input"] = q

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Данные:** [stackoverflow/stacksample](https://www.kaggle.com/datasets/stackoverflow/stacksample) "
    "на Kaggle — 10% вопросов и ответов SO."
)

# ── Ввод вопроса ──────────────────────────────────────────────────────────────
query = st.text_input(
    "Введите вопрос на Python:",
    value=st.session_state.get("query_input", ""),
    placeholder="How do I reverse a list in Python?",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    search_btn = st.button("🔍 Найти", type="primary")
with col2:
    top_k_ui = st.slider("Top-K фрагментов", 1, 10, TOP_K)

# ── Поиск и генерация ─────────────────────────────────────────────────────────
if search_btn and query.strip():
    with st.spinner("Ищем релевантные фрагменты …"):
        results = retriever.search(query, top_k=top_k_ui)

    relevant = is_relevant(results)

    # Колонки: ответ слева, фрагменты справа
    col_ans, col_chunks = st.columns([3, 2])

    with col_ans:
        st.subheader("💬 Ответ")
        with st.spinner("Генерируем ответ …"):
            answer = generate(query, results)
        st.markdown(answer)

    with col_chunks:
        st.subheader(f"📄 Найденные фрагменты ({'релевантны' if relevant else 'не релевантны'})")
        if not results:
            st.info("Ничего не найдено.")
        for i, chunk in enumerate(results, 1):
            score = chunk.get("score", 0.0)
            name  = chunk.get("name", chunk.get("doc_id", ""))
            url   = chunk.get("url", "")
            text  = chunk.get("text", "")

            color = "green" if score > 0.1 else ("orange" if score > 0.01 else "red")
            badge = f":{color}[score: {score:.4f}]"

            with st.expander(f"[{i}] {name[:55]}  {badge}", expanded=(i == 1)):
                if url:
                    st.markdown(f"🔗 [{url}]({url})")
                st.markdown(f"`doc_id`: `{chunk.get('doc_id','')}`")
                st.text(text[:800])
elif search_btn:
    st.warning("Введите вопрос перед поиском.")
