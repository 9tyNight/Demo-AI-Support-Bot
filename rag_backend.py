from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


load_dotenv()

DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")
COLLECTION_NAME = "support_bot_sources"
LOCAL_EMBED_DIM = 2048


@dataclass(frozen=True)
class Source:
    source_id: str
    source_type: str
    category: str
    title: str
    text: str
    distance: float


class Embedder:
    def __init__(self) -> None:
        self.use_openai = (
            os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
            and bool(os.getenv("OPENAI_API_KEY"))
        )
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = OpenAI() if self.use_openai else None
        self.local_vectorizer = HashingVectorizer(
            n_features=LOCAL_EMBED_DIM,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            stop_words="english",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.use_openai and self.client:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]

        matrix = self.local_vectorizer.transform(texts)
        matrix = normalize(matrix, norm="l2", copy=False)
        return matrix.astype(np.float32).toarray().tolist()


def ensure_sample_data() -> None:
    if (DATA_DIR / "support_tickets.csv").exists() and (DATA_DIR / "kb_articles.csv").exists():
        return

    from generate_sample_data import main as generate_data

    generate_data()


def chunk_text(text: str, max_chars: int = 900) -> list[str]:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in text.split(". "):
        candidate = f"{current}. {sentence}".strip(". ") if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def load_documents() -> list[dict[str, Any]]:
    ensure_sample_data()
    tickets = pd.read_csv(DATA_DIR / "support_tickets.csv")
    articles = pd.read_csv(DATA_DIR / "kb_articles.csv")

    documents: list[dict[str, Any]] = []
    for _, row in tickets.iterrows():
        source_id = row["Ticket_ID"]
        text = (
            f"Historical Support Ticket {source_id}\n"
            f"Category: {row['Category']}\n"
            f"Customer Issue: {row['Customer_Issue']}\n"
            f"Resolution: {row['Resolution']}"
        )
        for idx, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f"{source_id}-{idx}",
                    "text": chunk,
                    "metadata": {
                        "source_id": source_id,
                        "source_type": "ticket",
                        "category": row["Category"],
                        "title": f"Ticket {source_id}",
                    },
                }
            )

    for _, row in articles.iterrows():
        source_id = row["Article_ID"]
        text = (
            f"Knowledge Base Article {source_id}: {row['Title']}\n"
            f"Category: {row['Category']}\n"
            f"Body: {row['Body']}"
        )
        for idx, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f"{source_id}-{idx}",
                    "text": chunk,
                    "metadata": {
                        "source_id": source_id,
                        "source_type": "kb_article",
                        "category": row["Category"],
                        "title": row["Title"],
                    },
                }
            )

    return documents


class SupportBotRAG:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.client = self._new_client()
        self.collection = self._get_collection()

    def _new_client(self):
        return chromadb.PersistentClient(path=str(DB_DIR))

    def _get_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return self.collection

    def _is_missing_collection_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return (
            error.__class__.__name__ == "NotFoundError"
            or "does not exist" in message
            or "not found" in message
        )

    def _reconnect_collection(self):
        self.client = self._new_client()
        return self._get_collection()

    def build_index(self, reset: bool = False) -> int:
        if reset:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except ValueError:
                pass
            self.collection = self._get_collection()

        try:
            existing = self.collection.count()
        except Exception as error:
            if not self._is_missing_collection_error(error):
                raise
            self.collection = self._reconnect_collection()
            existing = self.collection.count()
        if existing:
            return existing

        docs = load_documents()
        texts = [doc["text"] for doc in docs]
        embeddings = self.embedder.embed(texts)
        self.collection.add(
            ids=[doc["id"] for doc in docs],
            documents=texts,
            metadatas=[doc["metadata"] for doc in docs],
            embeddings=embeddings,
        )
        return len(docs)

    def retrieve(self, query: str, top_k: int = 5) -> list[Source]:
        self.build_index()
        query_embedding = self.embedder.embed([query])[0]
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            if not self._is_missing_collection_error(error):
                raise
            self.collection = self._reconnect_collection()
            self.build_index()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        sources: list[Source] = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sources.append(
                Source(
                    source_id=str(metadata["source_id"]),
                    source_type=str(metadata["source_type"]),
                    category=str(metadata["category"]),
                    title=str(metadata["title"]),
                    text=text,
                    distance=float(distance),
                )
            )
        return sources

    def answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        sources = self.retrieve(query, top_k=top_k)
        provider = os.getenv("LLM_PROVIDER", "auto").lower()
        if provider in {"auto", "gemini"} and os.getenv("GEMINI_API_KEY"):
            answer = self._answer_with_gemini(query, sources)
        elif provider in {"auto", "openai"} and os.getenv("OPENAI_API_KEY"):
            answer = self._answer_with_openai(query, sources)
        else:
            answer = self._answer_without_llm(query, sources)

        return {"answer": answer, "sources": [source.__dict__ for source in sources]}

    def _build_grounded_prompt(self, query: str, sources: list[Source]) -> str:
        context = "\n\n".join(
            f"[{source.source_id}] {source.title} ({source.source_type}, {source.category})\n{source.text}"
            for source in sources
        )
        prompt = f"""
        You are an AI support assistant using retrieval-augmented generation.

        Rules:
        - Answer only from the provided context.
        - Every factual claim must cite at least one source ID in square brackets, such as [TCK-1007] or [KB-006].
        - If the context does not answer the question, say what is missing and recommend escalation.
        - Prefer concise, support-agent-ready language.

        Customer or internal question:
        {query}

        Retrieved context:
        {context}
        """
        return textwrap.dedent(prompt).strip()

    def _answer_with_openai(self, query: str, sources: list[Source]) -> str:
        client = OpenAI()
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5")
        response = client.responses.create(
            model=model,
            input=self._build_grounded_prompt(query, sources),
        )
        return response.output_text

    def _answer_with_gemini(self, query: str, sources: list[Source]) -> str:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        response = client.models.generate_content(
            model=model,
            contents=self._build_grounded_prompt(query, sources),
        )
        return response.text or "Gemini returned an empty response. Please retry or escalate."

    def _answer_without_llm(self, query: str, sources: list[Source]) -> str:
        if not sources:
            return "I could not find a grounded answer in the indexed KB or ticket history. Please escalate to support operations."

        selected: list[Source] = []
        for source_type in ("ticket", "kb_article"):
            match = next((source for source in sources if source.source_type == source_type), None)
            if match:
                selected.append(match)
        for source in sources:
            if source not in selected:
                selected.append(source)
            if len(selected) == 3:
                break

        bullets = []
        for source in selected[:3]:
            summary = source.text.split("Resolution:")[-1].strip() if "Resolution:" in source.text else source.text
            bullets.append(f"- {summary} [{source.source_id}]")

        return (
            "Demo answer generated without an LLM because OPENAI_API_KEY is not set. "
            "The highest-similarity sources suggest:\n\n"
            + "\n".join(bullets)
        )


def main() -> None:
    bot = SupportBotRAG()
    count = bot.build_index(reset=True)
    print(f"Indexed {count} chunks.")
    result = bot.answer("Why are webhook events delayed and how should we fix it?", top_k=4)
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['source_id']}: {source['title']} ({source['category']})")


if __name__ == "__main__":
    main()
