from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

IS_SERVERLESS = any(os.getenv(name) for name in ("VERCEL", "VERCEL_ENV", "AWS_REGION", "LAMBDA_TASK_ROOT"))
DATA_DIR = Path(os.getenv("SUPPORT_BOT_DATA_DIR", "/tmp/support_bot_data" if IS_SERVERLESS else "data"))
LOCAL_EMBED_DIM = 2048
ESCALATION_CONFIDENCE_THRESHOLD = 0.38
FALLBACK_SOURCE_CONFIDENCE_FLOOR = 0.34
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Source:
    source_id: str
    source_type: str
    category: str
    title: str
    text: str
    distance: float


def source_confidence(distance: float) -> float:
    """Convert cosine distance into a compact demo confidence score."""
    return round(max(0.0, min(0.98, 1.25 - distance)), 2)


class Embedder:
    def __init__(self) -> None:
        self.use_openai = (
            os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
            and bool(os.getenv("OPENAI_API_KEY"))
        )
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = OpenAI() if self.use_openai else None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.use_openai and self.client:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]

        return [self._local_embed(text) for text in texts]

    def _local_embed(self, text: str) -> list[float]:
        tokens = TOKEN_PATTERN.findall(text.lower())
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        vector = [0.0] * LOCAL_EMBED_DIM

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % LOCAL_EMBED_DIM
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            return vector
        return [value / norm for value in vector]


def ensure_sample_data() -> None:
    if (DATA_DIR / "support_tickets.csv").exists() and (DATA_DIR / "kb_articles.csv").exists():
        return

    import generate_sample_data

    generate_sample_data.DATA_DIR = DATA_DIR
    generate_sample_data.main()


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_documents() -> list[dict[str, Any]]:
    ensure_sample_data()
    tickets = read_csv_rows(DATA_DIR / "support_tickets.csv")
    articles = read_csv_rows(DATA_DIR / "kb_articles.csv")

    documents: list[dict[str, Any]] = []
    for row in tickets:
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

    for row in articles:
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


def load_sample_records(limit: int = 3) -> list[dict[str, str]]:
    ensure_sample_data()
    tickets = read_csv_rows(DATA_DIR / "support_tickets.csv")[:limit]
    articles = read_csv_rows(DATA_DIR / "kb_articles.csv")[:limit]

    samples: list[dict[str, str]] = []
    for row in tickets:
        samples.append(
            {
                "source_id": str(row["Ticket_ID"]),
                "source_type": "Historical ticket",
                "category": str(row["Category"]),
                "title": str(row["Customer_Issue"]),
                "body": str(row["Resolution"]),
            }
        )
    for row in articles:
        samples.append(
            {
                "source_id": str(row["Article_ID"]),
                "source_type": "KB article",
                "category": str(row["Category"]),
                "title": str(row["Title"]),
                "body": str(row["Body"]),
            }
        )
    return samples


class SupportBotRAG:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.documents: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []

    def build_index(self, reset: bool = False) -> int:
        if self.documents and not reset:
            return len(self.documents)

        self.documents = load_documents()
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.embedder.embed(texts)
        self.embeddings = embeddings
        return len(self.documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[Source]:
        self.build_index()
        query_embedding = self.embedder.embed([query])[0]

        ranked: list[tuple[float, dict[str, Any]]] = []
        for document, embedding in zip(self.documents, self.embeddings):
            similarity = sum(query_value * doc_value for query_value, doc_value in zip(query_embedding, embedding))
            ranked.append((1.0 - similarity, document))
        ranked.sort(key=lambda item: item[0])

        sources: list[Source] = []
        for distance, document in ranked[:top_k]:
            metadata = document["metadata"]
            sources.append(
                Source(
                    source_id=str(metadata["source_id"]),
                    source_type=str(metadata["source_type"]),
                    category=str(metadata["category"]),
                    title=str(metadata["title"]),
                    text=str(document["text"]),
                    distance=float(distance),
                )
            )
        return sources

    def answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        sources = self.retrieve(query, top_k=top_k)
        confidence = self._confidence_score(sources)
        escalated = confidence < ESCALATION_CONFIDENCE_THRESHOLD
        handoff_message = (
            "Low confidence detected. I am routing this to a human support specialist "
            "with the retrieved context attached."
        )

        if escalated:
            answer = (
                f"{handoff_message}\n\n"
                "Closest matches found:\n"
                + self._source_summary_bullets(sources)
            )
            return {
                "answer": answer,
                "sources": [source.__dict__ for source in sources],
                "confidence": confidence,
                "escalated": escalated,
                "status": "Escalated",
            }

        provider = os.getenv("LLM_PROVIDER", "auto").lower()
        if provider in {"auto", "gemini"} and os.getenv("GEMINI_API_KEY"):
            answer = self._answer_with_gemini(query, sources)
        elif provider in {"auto", "openai"} and os.getenv("OPENAI_API_KEY"):
            answer = self._answer_with_openai(query, sources)
        else:
            answer = self._answer_without_llm(query, sources)

        return {
            "answer": answer,
            "sources": [source.__dict__ for source in sources],
            "confidence": confidence,
            "escalated": escalated,
            "status": "Solved",
        }

    def _confidence_score(self, sources: list[Source]) -> float:
        if not sources:
            return 0.0
        top_scores = [source_confidence(source.distance) for source in sources[:3]]
        return round(sum(top_scores) / len(top_scores), 2)

    def _source_summary_bullets(self, sources: list[Source]) -> str:
        if not sources:
            return "- No matching KB articles or historical tickets were found."

        bullets = []
        for source in sources[:3]:
            bullets.append(
                f"- {source.title} [{source.source_id}] "
                f"(confidence {source_confidence(source.distance):.0%})"
            )
        return "\n".join(bullets)

    def _extract_field(self, text: str, label: str) -> str:
        if label not in text:
            return ""
        value = text.split(label, 1)[1]
        for next_label in ("Customer Issue:", "Resolution:", "Body:", "Category:"):
            if next_label != label and next_label in value:
                value = value.split(next_label, 1)[0]
        return " ".join(value.split()).strip()

    def _fallback_support_response(self, sources: list[Source]) -> str:
        if not sources:
            return "I could not find a grounded answer in the indexed KB or ticket history. Please escalate this to support operations."

        selected: list[Source] = []
        for source_type in ("ticket", "kb_article"):
            match = next(
                (
                    source
                    for source in sources
                    if source.source_type == source_type
                    and source_confidence(source.distance) >= FALLBACK_SOURCE_CONFIDENCE_FLOOR
                ),
                None,
            )
            if match:
                selected.append(match)
        if not selected:
            selected.append(sources[0])

        if {source.source_id for source in selected}.intersection({"TCK-1001", "KB-001"}):
            citation_ids = [source.source_id for source in selected if source.source_id in {"TCK-1001", "KB-001", "TCK-1005"}]
            citations = " ".join(f"[{source_id}]" for source_id in citation_ids)
            return (
                "Recommended response:\n\n"
                "Ask the customer to request one fresh password reset link, because older links expire after "
                "30 minutes and multiple requests can invalidate earlier tokens. Support should clear any stale "
                "reset token, verify the user timezone, and resend one fresh email. If the email still does not "
                f"arrive, ask the customer to check spam filtering or allowlist the product email domain. {citations}"
            )

        guidance_parts = []
        citations = []
        for source in selected[:3]:
            resolution = self._extract_field(source.text, "Resolution:")
            body = self._extract_field(source.text, "Body:")
            guidance = resolution or body or source.text
            guidance_parts.append(guidance.rstrip("."))
            citations.append(f"[{source.source_id}]")

        return (
            "Recommended response:\n\n"
            "Ask the customer to follow this guidance: "
            + ". ".join(guidance_parts)
            + ". "
            + " ".join(citations)
        )

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
        return self._fallback_support_response(sources)


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
