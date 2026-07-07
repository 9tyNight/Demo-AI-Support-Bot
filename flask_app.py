from __future__ import annotations

from functools import lru_cache

from flask import Flask, jsonify, render_template_string, request

from rag_backend import SupportBotRAG


app = Flask(__name__)


@lru_cache(maxsize=1)
def get_bot() -> SupportBotRAG:
    bot = SupportBotRAG()
    bot.build_index()
    return bot


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Support Bot Demo</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #121826;
      --muted: #647084;
      --line: #dfe5ee;
      --panel: #ffffff;
      --soft: #f6f8fb;
      --accent: #2454ff;
      --accent-dark: #1738ad;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef2f7;
      color: var(--ink);
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      padding: 22px clamp(18px, 5vw, 56px);
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: clamp(24px, 4vw, 36px);
      letter-spacing: 0;
    }
    header p {
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 860px;
      line-height: 1.5;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(320px, 0.8fr);
      gap: 18px;
      padding: 18px clamp(18px, 5vw, 56px) 28px;
    }
    .chat, .sources {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 72vh;
    }
    .chat {
      display: grid;
      grid-template-rows: 1fr auto;
      overflow: hidden;
    }
    #messages {
      padding: 20px;
      overflow-y: auto;
    }
    .message {
      max-width: 860px;
      margin-bottom: 14px;
      padding: 13px 15px;
      border-radius: 8px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .assistant {
      background: var(--soft);
      border: 1px solid var(--line);
    }
    .user {
      margin-left: auto;
      background: #e8efff;
      border: 1px solid #cad8ff;
    }
    form {
      display: flex;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
      background: #fff;
    }
    input {
      flex: 1;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      font-size: 15px;
    }
    button {
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      background: #94a3b8;
      cursor: wait;
    }
    .sources {
      padding: 18px;
      overflow-y: auto;
    }
    .sources h2 {
      margin: 0;
      font-size: 20px;
    }
    .sources p {
      color: var(--muted);
      line-height: 1.45;
    }
    .source {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin: 12px 0;
      background: #fbfcff;
    }
    .source strong {
      color: var(--accent-dark);
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0;
    }
    .pill {
      font-size: 12px;
      color: #334155;
      border: 1px solid #d6dde8;
      background: #fff;
      border-radius: 999px;
      padding: 3px 8px;
    }
    details {
      margin-top: 8px;
      color: #334155;
      line-height: 1.45;
    }
    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
      }
      .chat, .sources {
        min-height: auto;
      }
      .chat {
        min-height: 68vh;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>AI Support Bot Demo</h1>
      <p>Ask support questions grounded in mock knowledge base articles and historical ticket exports. Responses cite exact ticket or KB IDs, and the evidence appears beside the chat.</p>
    </header>
    <main>
      <section class="chat" aria-label="Chat">
        <div id="messages">
          <div class="message assistant">Ask me about password resets, billing holds, webhook delays, CSV imports, invitation emails, API keys, analytics, or mobile crashes.</div>
        </div>
        <form id="chat-form">
          <input id="query" name="query" autocomplete="off" placeholder="Ask a support question" required>
          <button id="send" type="submit">Send</button>
        </form>
      </section>
      <aside class="sources" aria-label="Sources Used">
        <h2>Sources Used</h2>
        <p id="source-note">Ask a question to see retrieved tickets and KB articles.</p>
        <div id="sources"></div>
      </aside>
    </main>
  </div>
  <script>
    const form = document.getElementById("chat-form");
    const input = document.getElementById("query");
    const send = document.getElementById("send");
    const messages = document.getElementById("messages");
    const sources = document.getElementById("sources");
    const sourceNote = document.getElementById("source-note");

    function addMessage(role, text) {
      const div = document.createElement("div");
      div.className = `message ${role}`;
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }

    function renderSources(items) {
      sources.innerHTML = "";
      sourceNote.textContent = items.length ? "Retrieved from the latest response." : "No sources returned.";
      items.forEach((source) => {
        const card = document.createElement("div");
        card.className = "source";
        card.innerHTML = `
          <strong>${source.source_id}</strong>
          <div>${source.title}</div>
          <div class="meta">
            <span class="pill">${source.source_type}</span>
            <span class="pill">${source.category}</span>
            <span class="pill">distance ${Number(source.distance).toFixed(3)}</span>
          </div>
          <details>
            <summary>View source text</summary>
            <p>${source.text}</p>
          </details>
        `;
        sources.appendChild(card);
      });
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = input.value.trim();
      if (!query) return;

      addMessage("user", query);
      input.value = "";
      send.disabled = true;
      addMessage("assistant", "Retrieving grounded sources...");

      try {
        const response = await fetch("/api/answer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query })
        });
        const data = await response.json();
        messages.lastChild.textContent = data.answer || data.error || "No answer returned.";
        renderSources(data.sources || []);
      } catch (error) {
        messages.lastChild.textContent = "The demo could not reach the RAG backend. Please retry.";
      } finally {
        send.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/answer")
def answer():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"error": "Query is required.", "sources": []}), 400

    try:
        return jsonify(get_bot().answer(query))
    except Exception as error:
        get_bot.cache_clear()
        return jsonify({"error": f"Backend error: {error}", "sources": []}), 500
