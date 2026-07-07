# AI Support Bot RAG Demo

This is a lightweight proof-of-concept for an AI support assistant that ingests two source types:

- structured and unstructured knowledge base articles
- historical support ticket exports that are cleaned, structured, chunked, embedded, and retrieved

The app answers support questions with grounded citations such as `[KB-003]` or `[TCK-1003]`, then displays the retrieved sources in a side panel.

## 1. Demo Architecture

Recommended 48-hour prototype stack:

| Layer | Choice | Why |
| --- | --- | --- |
| Frontend | Flask/HTML on Vercel, Streamlit locally | Fast public demo link plus a richer local chat UI |
| LLM | Gemini API or OpenAI Responses API | Optional API-backed answer synthesis with a provider switch |
| Embeddings | Local hashed vectors, optional `text-embedding-3-small` | Lightweight retrieval that works without external services |
| Retrieval | In-memory cosine search | Keeps the Vercel demo small and reliable |
| Data cleaning | Python CSV utilities | No heavy data dependency needed for the mock support exports |
| Deployment | Vercel | Quick client demo link |

For a production MVP, this can evolve to managed Pinecone/Qdrant/Chroma, scheduled ingestion, role-based access control, analytics, ticket deflection tracking, and human handoff.

## 2. Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-streamlit.txt
python generate_sample_data.py
python rag_backend.py
streamlit run streamlit_app.py
```

Copy `.env.example` to `.env` and set either `GEMINI_API_KEY` or `OPENAI_API_KEY` for LLM-generated answers. Without an API key, the demo still runs with local deterministic embeddings and an extractive fallback answer.

```bash
copy .env.example .env
```

To use OpenAI embeddings instead of local demo embeddings, set:

```env
USE_OPENAI_EMBEDDINGS=true
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5
```

To use Gemini for answer generation, set:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

## Vercel Deployment

This repo includes `app.py`, a Flask entrypoint for Vercel. Vercel's Python runtime expects a Flask `app` instance in an entrypoint such as `app.py`; the Streamlit app remains available for local demos.

Live demo: https://demo-ai-support-bot.vercel.app

The Vercel deployment uses the Flask UI and the same lightweight RAG backend. Streamlit is kept in `requirements-streamlit.txt` so the serverless bundle stays small.

In Vercel, add these environment variables:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Then redeploy from the `main` branch. The Vercel version uses the same RAG backend and mock data, but serves a small Flask/HTML chat UI instead of Streamlit.

## 3. What The Demo Proves

- It ingests both KB articles and historical ticket exports, normalizes each into citation-ready source records, chunks them, embeds them, and retrieves them with in-memory vector search.
- It performs RAG retrieval over both source types and prompts the LLM to answer only from retrieved context with mandatory source IDs.
- It provides a clean chat interface where the answer and the underlying sources are visible together, making hallucination control tangible for the client.

## 4. Demo Walkthrough Script

- "I built a working mini-version of your AI Support Bot using mock KB articles and historical tickets, because the hard part is not just chat; it is cleaning and structuring messy historical support data so retrieval works."
- "Each response is grounded in retrieved sources and cites the exact KB article or ticket ID, which is how we reduce hallucinations and make the bot usable for both customers and internal support agents."
- "The MVP path is straightforward: replace the mock CSVs with your exports, add scheduled ingestion, run evaluation on historical tickets, and deploy the chat widget behind your existing support workflow."

## 5. File Map

- `generate_sample_data.py`: creates mock support ticket and KB CSVs
- `rag_backend.py`: chunks, embeds, indexes, retrieves, and answers with citations
- `streamlit_app.py`: chat UI with source panel and expandable evidence
- `app.py`: Flask/Vercel chat UI for the public demo
- `.env.example`: configuration template
