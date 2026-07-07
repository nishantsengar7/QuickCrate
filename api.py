"""
api.py -- QuickCrate RAG  |  FastAPI Backend  (Phase 7A)
=========================================================

Endpoints
---------
  POST /chat    Accept a query + optional session_id, run the full RAG
                pipeline (generate.py), return the answer and metadata.
  GET  /health  Confirm Qdrant is reachable and models are loaded.

Server-side session management
-------------------------------
Conversation history is stored in a plain in-memory dict keyed by session_id.
This is intentional for a portfolio/single-instance deployment.

In a production system this store should be replaced with Redis (e.g.,
redis-py + redis.asyncio) or a database (e.g., Postgres via asyncpg) so that:
  - History survives server restarts.
  - Multiple API workers / replicas can share session state.
  - Sessions can be expired with a TTL.

Observability log
-----------------
Every request is appended as a JSON row to requests.log (local file) and
also inserted into requests.db (SQLite).  In a production system this data
could feed a real-time dashboard (e.g., Grafana, Metabase, or a custom
Streamlit analytics page) to track:
  - Escalation rate over time (signals KB coverage gaps).
  - Latency percentiles per query type.
  - rerank_score distribution (useful for re-tuning CONFIDENCE_THRESHOLD).

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env before importing generate.py so GEMINI_API_KEY is available.
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# RAG pipeline imports
from generate import AnswerResponse, answer_query
from rerank import load_cross_encoder
from retrieval import HybridRetriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "requests.log"      # one JSON object per line (JSONL)
DB_FILE  = BASE_DIR / "requests.db"       # SQLite for easy ad-hoc queries

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# Maps session_id (str) -> list of {"role": str, "content": str} dicts.
#
# NOTE: This is intentionally a plain dict.  For production, replace with
# Redis (redis-py) or a database so sessions survive restarts and scale
# across multiple workers.  A reasonable TTL for support chat is 30 minutes.
SESSION_STORE: dict[str, list[dict[str, str]]] = {}

# Maximum turns to keep per session to avoid unbounded memory growth and
# LLM context overflows.  Older turns are trimmed from the front.
MAX_HISTORY_TURNS: int = 20


# ---------------------------------------------------------------------------
# SQLite observability schema
# ---------------------------------------------------------------------------
def _init_db() -> None:
    """Create the requests table if it does not yet exist."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,   -- ISO-8601 UTC timestamp
            session_id  TEXT,
            query       TEXT    NOT NULL,
            latency_ms  REAL    NOT NULL,
            escalated   INTEGER NOT NULL,   -- 0 or 1
            rerank_score REAL   NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _log_request(
    session_id: str,
    query: str,
    latency_ms: float,
    escalated: bool,
    rerank_score: float,
) -> None:
    """
    Append one row to both the JSONL log file and the SQLite DB.

    This lightweight observability hook lets you:
    - grep the JSONL log for specific queries during debugging.
    - Run SQL against requests.db to compute escalation rates, p95 latency,
      and rerank_score distributions -- all useful inputs for tuning
      CONFIDENCE_THRESHOLD and identifying KB coverage gaps.

    In a production system this could be replaced with an OpenTelemetry
    trace or streamed to a data warehouse (BigQuery, Snowflake).
    """
    ts = datetime.now(timezone.utc).isoformat()
    row = {
        "ts": ts,
        "session_id": session_id,
        "query": query,
        "latency_ms": round(latency_ms, 1),
        "escalated": escalated,
        "rerank_score": round(rerank_score, 4),
    }
    # JSONL file
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    # SQLite
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            "INSERT INTO requests (ts, session_id, query, latency_ms, escalated, rerank_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, session_id, query, latency_ms, int(escalated), rerank_score),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        # Never let observability failures crash the request.
        logger.warning("SQLite log failed: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan: load models once at startup, share across all requests
# ---------------------------------------------------------------------------
# Model state is attached to app.state so it is accessible inside endpoints
# without using module-level globals (which break hot-reload in development).
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the retriever and cross-encoder once when the server starts."""
    logger.info("Loading RAG pipeline models...")
    _init_db()
    app.state.retriever = HybridRetriever()
    app.state.ce_model = load_cross_encoder()
    logger.info("Models loaded. Server ready.")
    yield
    # Shutdown: nothing to clean up for in-memory models.
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="QuickCrate Support RAG API",
    description=(
        "Hybrid retrieval (dense + BM25) → cross-encoder reranking → "
        "confidence-gated LLM generation for QuickCrate customer support."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow the Streamlit frontend (typically :8501) to call this API
# during local development.  In production, replace "*" with your actual
# frontend origin (e.g., "https://quickcrate-support.streamlit.app").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       description="The user's support question.")
    session_id: str | None = Field(
        default=None,
        description=(
            "Client-supplied session identifier. If omitted, a new UUID is "
            "generated and returned so the client can reuse it for follow-ups."
        ),
    )
    # conversation_history is accepted from the client but is informational
    # only — the server's SESSION_STORE is authoritative.  This allows a
    # stateless client to bootstrap a session on first contact.
    conversation_history: list[dict[str, str]] | None = Field(
        default=None,
        description="Ignored if session_id already exists server-side.",
    )


class SourceItem(BaseModel):
    article_id: str
    title: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    escalated: bool
    session_id: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    models_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse, summary="Ask a support question")
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """
    Run the full RAG pipeline and return a grounded answer (or escalation).

    Session management:
    - If session_id is absent or unknown, create a new session.
    - Retrieve the server-side history for the session.
    - Append the new turn after a successful response.
    - The client only needs to store the session_id; all history lives here.
    """
    # Resolve or create session
    sid = req.session_id or str(uuid.uuid4())
    if sid not in SESSION_STORE:
        # New session: optionally seed with client-supplied history
        SESSION_STORE[sid] = req.conversation_history or []

    history = SESSION_STORE[sid]

    t0 = time.perf_counter()
    try:
        result: AnswerResponse = answer_query(
            query=req.query,
            retriever=request.app.state.retriever,
            ce_model=request.app.state.ce_model,
            conversation_history=history if history else None,
        )
    except Exception as exc:
        # Graceful fallback: never expose a raw stack trace to the client.
        logger.exception("Pipeline error for query '%s': %s", req.query[:80], exc)
        latency_ms = (time.perf_counter() - t0) * 1000
        _log_request(sid, req.query, latency_ms, escalated=True, rerank_score=-99.0)
        return ChatResponse(
            answer=(
                "I'm sorry, something went wrong on our end while processing "
                "your request. Please try again in a moment, or contact our "
                f"support team directly at {_SUPPORT_CONTACT}."
            ),
            sources=[],
            escalated=True,
            session_id=sid,
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    # Update server-side history (trim to avoid unbounded growth)
    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": result.answer})
    if len(history) > MAX_HISTORY_TURNS * 2:
        # Drop oldest pair of turns
        SESSION_STORE[sid] = history[-(MAX_HISTORY_TURNS * 2):]

    # Observability
    _log_request(sid, req.query, latency_ms, result.escalated, result.rerank_score)
    logger.info(
        "session=%s  latency=%.0fms  escalated=%s  score=%.3f",
        sid[:8], latency_ms, result.escalated, result.rerank_score,
    )

    # Build structured sources from parsed title strings
    sources = [
        SourceItem(article_id="", title=t) for t in result.sources
    ]

    return ChatResponse(
        answer=result.answer,
        sources=sources,
        escalated=result.escalated,
        session_id=sid,
    )


@app.get("/health", response_model=HealthResponse, summary="Pipeline health check")
async def health(request: Request) -> HealthResponse:
    """
    Confirm that:
    - The retriever is loaded and Qdrant is reachable.
    - The cross-encoder model is loaded.

    A monitoring tool (e.g., UptimeRobot, Kubernetes liveness probe) can
    poll this endpoint to detect cold-start failures or Qdrant disconnects.
    """
    models_loaded = (
        hasattr(request.app.state, "retriever")
        and hasattr(request.app.state, "ce_model")
    )

    qdrant_status = "unreachable"
    if models_loaded:
        try:
            # A lightweight collections list call -- does not fetch any vectors.
            request.app.state.retriever.qdrant.get_collections()
            qdrant_status = "ok"
        except Exception as exc:
            qdrant_status = f"error: {exc}"

    overall = "ok" if (models_loaded and qdrant_status == "ok") else "degraded"
    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        models_loaded=models_loaded,
    )


# Expose the support contact for the fallback error message above
from generate import SUPPORT_CONTACT as _SUPPORT_CONTACT  # noqa: E402
