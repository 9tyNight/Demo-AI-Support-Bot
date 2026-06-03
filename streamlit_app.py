from __future__ import annotations

import streamlit as st

from rag_backend import SupportBotRAG


st.set_page_config(page_title="AI Support Bot Demo", page_icon="AI", layout="wide")


@st.cache_resource
def get_bot() -> SupportBotRAG:
    bot = SupportBotRAG()
    bot.build_index()
    return bot


st.markdown(
    """
    <style>
    .main-header {
        padding: 1rem 0 0.5rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .source-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.7rem;
        background: #fbfbfd;
    }
    .source-id {
        font-weight: 700;
        color: #2454ff;
    }
    .metric-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .pill {
        border: 1px solid #d1d5db;
        border-radius: 999px;
        padding: 0.15rem 0.55rem;
        font-size: 0.78rem;
        color: #374151;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="main-header">
        <h1>AI Support Bot Demo</h1>
        <p>Grounded support answers from mock KB articles and historical ticket exports, with source citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

bot = get_bot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me about password resets, billing holds, webhook delays, CSV imports, invitation emails, API keys, analytics, or mobile crashes.",
        }
    ]
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

chat_col, source_col = st.columns([0.64, 0.36], gap="large")

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a support question")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving grounded sources..."):
                result = bot.answer(prompt)
            st.markdown(result["answer"])

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.session_state.last_sources = result["sources"]
        st.rerun()

with source_col:
    st.subheader("Sources Used")
    st.caption("Shown from the latest response. Lower distance means a closer match.")

    if not st.session_state.last_sources:
        st.info("Ask a question to see retrieved tickets and KB articles.")
    else:
        for source in st.session_state.last_sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-id">{source["source_id"]}</div>
                    <div>{source["title"]}</div>
                    <div class="metric-row">
                        <span class="pill">{source["source_type"]}</span>
                        <span class="pill">{source["category"]}</span>
                        <span class="pill">distance {source["distance"]:.3f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"View source text: {source['source_id']}"):
                st.write(source["text"])
