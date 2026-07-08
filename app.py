"""
app.py -- QuickCrate Support RAG  |  Streamlit Frontend  (Phase 7B)
====================================================================

Conversation state design
--------------------------
The Streamlit tab owns ONLY:
  - The rendered chat history (for display).
  - The session_id returned by the backend on first contact.

All authoritative conversation history lives server-side (api.py).
When the user sends a new message, the tab sends only the raw query +
session_id to /chat; the backend resolves the full history itself.

This means:
  - If the user opens a new browser tab, they get a fresh session.
  - Refreshing the page clears the visual history but the backend session
    persists (until the server restarts, since history is in-memory for now).
  - The frontend has zero RAG logic -- it is a pure display layer.

Run with:
  streamlit run app.py
"""

import os
import time

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Set QC_API_URL in your shell or .env to point at the backend.
# Defaults to localhost during local development.
API_URL: str = os.getenv("QC_API_URL", "http://localhost:8000")
CHAT_ENDPOINT: str = f"{API_URL}/chat"
HEALTH_ENDPOINT: str = f"{API_URL}/health"

# Timeout for API calls (seconds).  Cross-encoder + LLM can take ~5-15 s on CPU.
REQUEST_TIMEOUT: float = 90.0


# ---------------------------------------------------------------------------
# Page config -- must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QuickCrate Support",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS -- subtle polish for the demo
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Soften the escalation warning box border */
    .escalation-box {
        border-left: 4px solid #f59e0b;
        background: #fffbeb;
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin-top: 0.5rem;
    }
    /* Make source pills look tidy */
    .source-pill {
        display: inline-block;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        color: #1d4ed8;
        margin: 2px 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
# st.session_state persists for the lifetime of the browser tab.
if "session_id" not in st.session_state:
    st.session_state.session_id = None        # assigned after first API call

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str,
    #              "sources": list, "escalated": bool}
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shopping-cart.png",
        width=64,
    )
    st.title("QuickCrate Support")
    st.caption("Powered by a multi-stage RAG pipeline")

    st.markdown("---")
    st.subheader("About this project")
    st.markdown(
        """
        This demo showcases a production-grade **Retrieval-Augmented Generation
        (RAG)** system built for QuickCrate, a fictional quick-commerce app.

        **Pipeline stages:**
        1. 🔍 **Hybrid retrieval** — dense vector search (BAAI/bge-large-en-v1.5
           via Qdrant) fused with BM25 using Reciprocal Rank Fusion (RRF).
        2. 🎯 **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` rescores
           the top-20 candidates with full query × document attention.
        3. 🚦 **Confidence gate** — if the top rerank score is below a tuned
           threshold, the query is escalated to human support instead of
           calling the LLM.
        4. 💬 **Grounded generation** — Gemini 2.5 Flash answers strictly from
           retrieved KB chunks and cites the source article(s).

        **Tech stack:** FastAPI · Qdrant · sentence-transformers ·
        google-genai · Streamlit
        """
    )

    st.markdown("---")

    # Health check widget
    if st.button("🔁 Check backend health", use_container_width=True):
        try:
            resp = httpx.get(HEALTH_ENDPOINT, timeout=5.0)
            h = resp.json()
            if h.get("status") == "ok":
                st.success(f"✅ Backend OK · Qdrant: {h.get('qdrant')}")
            else:
                st.warning(f"⚠️ Backend degraded · Qdrant: {h.get('qdrant')}")
        except Exception as exc:
            st.error(f"❌ Backend unreachable: {exc}")

    # Session info (useful during demos)
    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        # Clear the visual history and drop the session_id so the next
        # message starts a fresh server-side session.
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()


# ---------------------------------------------------------------------------
# Chat header
# ---------------------------------------------------------------------------
st.markdown("## 🛒 QuickCrate Customer Support")
st.markdown(
    "Ask me anything about orders, delivery, payments, returns, or your "
    "QuickCrate Plus subscription."
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Render existing conversation history
# ---------------------------------------------------------------------------
def _render_message(msg: dict) -> None:
    """Render one message bubble with sources / escalation decorations."""
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])
    escalated = msg.get("escalated", False)

    with st.chat_message(role, avatar="🛒" if role == "assistant" else None):
        if role == "assistant" and escalated:
            # Visually distinct escalation box so it's unmistakable in demos
            st.markdown(
                f"""<div class="escalation-box">
                ⚠️ <strong>Escalated to human support</strong><br><br>
                {content}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(content)

        # Collapsible sources section -- builds trust and demonstrates
        # the citation mechanism clearly during a walkthrough
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                for src in sources:
                    title = src.get("title", src) if isinstance(src, dict) else src
                    st.markdown(
                        f'<span class="source-pill">📄 {title}</span>',
                        unsafe_allow_html=True,
                    )


for msg in st.session_state.messages:
    _render_message(msg)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
def _call_api(query: str) -> dict:
    """
    POST to /chat and return the parsed JSON response.

    The frontend sends only the query + session_id.  Conversation history
    is resolved entirely server-side -- the frontend does NOT send history
    on every turn (that would violate the server-authoritative design).
    """
    payload: dict = {"query": query}
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id

    resp = httpx.post(CHAT_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


if prompt := st.chat_input("Type your question here…"):
    # 1. Show the user's message immediately
    user_msg = {"role": "user", "content": prompt, "sources": [], "escalated": False}
    st.session_state.messages.append(user_msg)
    _render_message(user_msg)

    # 2. Call the backend with a spinner
    with st.chat_message("assistant", avatar="🛒"):
        with st.spinner("Searching the knowledge base…"):
            try:
                t0 = time.perf_counter()
                data = _call_api(prompt)
                latency = time.perf_counter() - t0

                # Persist session_id returned by the backend
                st.session_state.session_id = data.get("session_id")

                answer = data.get("answer", "")
                sources = data.get("sources", [])
                escalated = data.get("escalated", False)

            except httpx.ConnectError:
                answer = (
                    "⚠️ Could not reach the backend API. "
                    f"Make sure it is running at `{API_URL}` "
                    "(`uvicorn api:app --port 8000`)."
                )
                sources = []
                escalated = True
                latency = 0.0
            except Exception as exc:
                answer = f"⚠️ Unexpected error: {exc}"
                sources = []
                escalated = True
                latency = 0.0

    # 3. Re-render (Streamlit re-runs after chat_input, so we append and rerun)
    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "escalated": escalated,
    }
    st.session_state.messages.append(assistant_msg)

    # Append latency note in dev mode so it's visible during demos/interviews
    if latency > 0:
        st.caption(f"⏱️ Response in {latency:.1f}s")

    st.rerun()
