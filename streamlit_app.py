from __future__ import annotations

import html

import streamlit as st

from rag_backend import SupportBotRAG, load_sample_records, source_confidence


st.set_page_config(page_title="AI Support Desk", page_icon="AI", layout="wide")

SAMPLE_QUESTIONS = [
    "How should we help a customer whose password reset link expires immediately?",
    "Why are webhook events delayed and how should we fix it?",
    "Can you help with my coffee machine warranty?",
]


@st.cache_resource(show_spinner=False)
def get_bot() -> SupportBotRAG:
    bot = SupportBotRAG()
    bot.build_index()
    return bot


@st.cache_data(show_spinner=False)
def get_samples() -> list[dict[str, str]]:
    return load_sample_records(limit=3)


def init_state() -> None:
    st.session_state.setdefault(
        "messages",
        [
            {
                "role": "assistant",
                "content": (
                    "Hi, I can answer from the ingested KB and ticket history. "
                    "Try asking about password resets, billing holds, webhook delays, "
                    "CSV imports, invitation emails, API keys, analytics, or mobile crashes."
                ),
            }
        ],
    )
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_confidence", 0.0)
    st.session_state.setdefault("last_status", "Ready")
    st.session_state.setdefault("solved", 0)
    st.session_state.setdefault("escalated", 0)


def render_pill(label: str, tone: str = "neutral") -> str:
    return f'<span class="pill pill-{tone}">{html.escape(label)}</span>'


def render_source_card(source: dict) -> None:
    confidence = source_confidence(float(source["distance"]))
    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-topline">
                <strong>{html.escape(source["source_id"])}</strong>
                {render_pill(f"{confidence:.0%} match", "good" if confidence >= 0.5 else "warn")}
            </div>
            <div class="source-title">{html.escape(source["title"])}</div>
            <div class="pill-row">
                {render_pill(source["source_type"])}
                {render_pill(source["category"])}
                {render_pill(f'distance {float(source["distance"]):.3f}')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"Evidence text: {source['source_id']}"):
        st.write(source["text"])


def answer_prompt(bot: SupportBotRAG, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving KB + historical tickets..."):
            result = bot.answer(prompt)
        st.markdown(result["answer"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.last_sources = result["sources"]
    st.session_state.last_confidence = float(result["confidence"])
    st.session_state.last_status = result["status"]
    if result["escalated"]:
        st.session_state.escalated += 1
    else:
        st.session_state.solved += 1
    st.rerun()


st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #657187;
        --line: #dbe3ef;
        --panel: #ffffff;
        --soft: #f5f7fb;
        --blue: #2454ff;
        --green: #18794e;
        --amber: #946200;
        --red: #c0362c;
    }
    .stApp {
        background: #edf1f7;
        color: var(--ink);
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer {
        display: none;
    }
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3, p {
        letter-spacing: 0;
    }
    .hero {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 1rem;
        align-items: center;
    }
    .hero h1 {
        font-size: clamp(1.8rem, 4vw, 2.7rem);
        margin: 0 0 0.25rem;
    }
    .hero p {
        margin: 0;
        color: var(--muted);
        line-height: 1.5;
    }
    .hero .mock-note {
        margin-top: 0.45rem;
        color: #7b8494;
        font-size: 0.82rem;
    }
    .flow {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.55rem;
        margin-top: 1rem;
    }
    .flow-step {
        border: 1px solid var(--line);
        background: var(--soft);
        border-radius: 8px;
        min-height: 86px;
        padding: 0.7rem;
    }
    .flow-step strong {
        display: block;
        font-size: 0.92rem;
        margin-bottom: 0.25rem;
    }
    .flow-step span {
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.35;
    }
    .sample-card, .source-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.82rem;
        margin-bottom: 0.72rem;
        background: #fbfcff;
    }
    .sample-card strong, .source-card strong {
        color: #1738ad;
    }
    .sample-card p, .source-card p {
        color: var(--muted);
        margin: 0.35rem 0 0;
        line-height: 1.42;
    }
    .source-topline {
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        align-items: center;
    }
    .source-title {
        margin: 0.3rem 0 0.48rem;
        line-height: 1.35;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid #d5dde8;
        border-radius: 999px;
        background: #fff;
        color: #344256;
        font-size: 0.75rem;
        line-height: 1;
        padding: 0.3rem 0.48rem;
        white-space: nowrap;
    }
    .pill-good {
        color: var(--green);
        border-color: #b8ddc9;
        background: #effaf3;
    }
    .pill-warn {
        color: var(--amber);
        border-color: #f0d690;
        background: #fff8e8;
    }
    .pill-danger {
        color: var(--red);
        border-color: #efb8b2;
        background: #fff1f0;
    }
    .status-band {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    div[data-testid="stMetric"] {
        background: #fbfcff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem;
    }
    div[data-testid="stMetric"] * {
        color: var(--ink);
    }
    div[data-testid="stChatMessage"] {
        border-radius: 8px;
        border: 1px solid var(--line);
        background: #ffffff;
    }
    div[data-testid="stChatMessage"] * {
        color: var(--ink) !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #eef4ff;
        border-color: #cddcff;
    }
    .stChatInputContainer {
        border-top: 1px solid var(--line);
    }
    div[data-testid="stButton"] button {
        width: 100%;
        min-height: 2.6rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fbfcff;
        color: var(--ink);
        font-size: 0.86rem;
        text-align: left;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #b7c5da;
        background: #f3f7ff;
    }
    @media (max-width: 900px) {
        .hero-grid, .flow {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


init_state()
bot = get_bot()

st.markdown(
    """
    <section class="hero">
        <div class="hero-grid">
            <div>
                <h1>AI Support Desk</h1>
                <p>
                    A retrieval-first support bot that ingests KB articles and historical tickets,
                    answers with citations, and hands uncertain cases to a human queue.
                </p>
                <p class="mock-note">Demo uses mock support data.</p>
            </div>
            <div class="status-band">
                <span class="pill pill-good">KB + tickets indexed</span>
                <span class="pill">Grounded answers</span>
                <span class="pill pill-warn">Low confidence handoff</span>
            </div>
        </div>
        <div class="flow" aria-label="Ingestion flow">
            <div class="flow-step"><strong>KB articles</strong><span>Policy, how-to, and troubleshooting records.</span></div>
            <div class="flow-step"><strong>Historical tickets</strong><span>Past issues mapped to agent resolutions.</span></div>
            <div class="flow-step"><strong>Clean + chunk</strong><span>Normalize records into searchable source blocks.</span></div>
            <div class="flow-step"><strong>Retrieve</strong><span>Find the most relevant ticket and KB evidence.</span></div>
            <div class="flow-step"><strong>Answer or handoff</strong><span>Respond when confident, escalate when not.</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

chat_col, insight_col = st.columns([0.63, 0.37], gap="large")

with insight_col:
    with st.container(border=True):
        st.subheader("Analytics")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Solved", st.session_state.solved)
        metric_cols[1].metric("Escalated", st.session_state.escalated)
        metric_cols[2].metric("Confidence", f"{st.session_state.last_confidence:.0%}")
        status_tone = "danger" if st.session_state.last_status == "Escalated" else "good"
        st.markdown(
            f"""
            <div class="status-band">
                {render_pill(f'Last status: {st.session_state.last_status}', status_tone)}
                {render_pill('Handoff trigger: confidence < 38%', 'warn')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.subheader("Sample Ingested Records")
        st.caption("Three tickets and three KB articles are indexed for the demo.")
        for sample in get_samples():
            st.markdown(
                f"""
                <div class="sample-card">
                    <strong>{html.escape(sample["source_id"])}</strong>
                    {render_pill(sample["source_type"])}
                    {render_pill(sample["category"])}
                    <p><b>{html.escape(sample["title"])}</b></p>
                    <p>{html.escape(sample["body"])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with chat_col:
    with st.container(border=True):
        st.subheader("Support Conversation")
        st.caption("Demo uses mock support data.")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.caption("Try a sample question")
        sample_cols = st.columns(3)
        for index, sample_question in enumerate(SAMPLE_QUESTIONS):
            if sample_cols[index].button(sample_question, key=f"sample-question-{index}"):
                answer_prompt(bot, sample_question)

        prompt = st.chat_input("Ask a support question")
        if prompt:
            answer_prompt(bot, prompt)

    with st.container(border=True):
        st.subheader("Retrieved Evidence")
        st.caption("Latest answer context. Lower distance means a closer semantic match.")
        if not st.session_state.last_sources:
            st.info("Ask a question to see the KB articles and historical tickets used.")
        else:
            for source in st.session_state.last_sources:
                render_source_card(source)
