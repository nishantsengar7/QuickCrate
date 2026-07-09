from __future__ import annotations
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from generate import AnswerResponse, answer_query, QuotaExhaustedError
from rerank import load_cross_encoder
from retrieval import HybridRetriever
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('api')
BASE_DIR = Path(__file__).parent
try:
    _test_file = BASE_DIR / '.write_test'
    _test_file.touch()
    _test_file.unlink()
    BASE_WRITE_DIR = BASE_DIR
except (OSError, PermissionError):
    BASE_WRITE_DIR = Path('/tmp')
    logger.info('Workspace directory is read-only. Redirecting SQLite database and log file to %s', BASE_WRITE_DIR)
LOG_FILE = BASE_WRITE_DIR / 'requests.log'
DB_FILE = BASE_WRITE_DIR / 'requests.db'
SESSION_STORE: dict[str, list[dict[str, str]]] = {}
MAX_HISTORY_TURNS: int = 20

def _init_db() -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.execute('\n        CREATE TABLE IF NOT EXISTS requests (\n            id          INTEGER PRIMARY KEY AUTOINCREMENT,\n            ts          TEXT    NOT NULL,   -- ISO-8601 UTC timestamp\n            session_id  TEXT,\n            query       TEXT    NOT NULL,\n            latency_ms  REAL    NOT NULL,\n            escalated   INTEGER NOT NULL,   -- 0 or 1\n            rerank_score REAL   NOT NULL\n        )\n        ')
    conn.commit()
    conn.close()

def _log_request(session_id: str, query: str, latency_ms: float, escalated: bool, rerank_score: float, outcome: str | None = None) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    if outcome is None:
        outcome = 'escalated' if escalated else 'generation_success'
    row = {
        'ts': ts,
        'session_id': session_id,
        'query': query,
        'latency_ms': round(latency_ms, 1),
        'escalated': escalated,
        'rerank_score': round(rerank_score, 4),
        'outcome': outcome
    }
    with LOG_FILE.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('INSERT INTO requests (ts, session_id, query, latency_ms, escalated, rerank_score) VALUES (?, ?, ?, ?, ?, ?)', (ts, session_id, query, latency_ms, int(escalated), rerank_score))
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning('SQLite log failed: %s', exc)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Loading RAG pipeline models...')
    _init_db()
    app.state.retriever = HybridRetriever()
    app.state.ce_model = load_cross_encoder()
    logger.info('Models loaded. Server ready.')
    _collection = os.environ.get('QDRANT_COLLECTION', 'quickcrate_kb')
    _dummy_vector = [0.0] * 1024
    try:
        app.state.retriever.qdrant.query_points(collection_name=_collection, query=_dummy_vector, limit=1, with_payload=False)
        logger.info('Qdrant POST connection warmed up.')
    except Exception as _exc:
        logger.warning('Qdrant warm-up failed (non-fatal): %s', _exc)
    yield
    logger.info('Server shutting down.')
app = FastAPI(title='QuickCrate Support RAG API', description='Hybrid retrieval (dense + BM25) → cross-encoder reranking → confidence-gated LLM generation for QuickCrate customer support.', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET', 'POST'], allow_headers=['*'])

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's support question.")
    session_id: str | None = Field(default=None, description='Client-supplied session identifier. If omitted, a new UUID is generated and returned so the client can reuse it for follow-ups.')
    conversation_history: list[dict[str, str]] | None = Field(default=None, description='Ignored if session_id already exists server-side.')

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

@app.post('/chat', response_model=ChatResponse, summary='Ask a support question')
async def chat(req: ChatRequest, request: Request) -> Any:
    sid = req.session_id or str(uuid.uuid4())
    if sid not in SESSION_STORE:
        SESSION_STORE[sid] = req.conversation_history or []
    history = SESSION_STORE[sid]
    t0 = time.perf_counter()
    try:
        result: AnswerResponse = answer_query(query=req.query, retriever=request.app.state.retriever, ce_model=request.app.state.ce_model, conversation_history=history if history else None)
    except QuotaExhaustedError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("Outcome: generation_failed | error=QuotaExhaustedError | session=%s  latency=%.0fms", sid[:8], latency_ms)
        _log_request(sid, req.query, latency_ms, escalated=False, rerank_score=-99.0, outcome='generation_failed')
        return JSONResponse(
            status_code=503,
            content={
                "error": "generation_unavailable",
                "message": "Our AI service is temporarily busy due to daily quota limits. Please try again in a moment."
            }
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("Outcome: generation_failed | error=%s | session=%s  latency=%.0fms", str(exc), sid[:8], latency_ms)
        _log_request(sid, req.query, latency_ms, escalated=False, rerank_score=-99.0, outcome='generation_failed')
        return JSONResponse(
            status_code=503,
            content={
                "error": "generation_unavailable",
                "message": "Our AI service is temporarily busy, please try again in a moment."
            }
        )
    latency_ms = (time.perf_counter() - t0) * 1000
    history.append({'role': 'user', 'content': req.query})
    history.append({'role': 'assistant', 'content': result.answer})
    if len(history) > MAX_HISTORY_TURNS * 2:
        SESSION_STORE[sid] = history[-(MAX_HISTORY_TURNS * 2):]
    
    outcome = 'escalated' if result.escalated else 'generation_success'
    _log_request(sid, req.query, latency_ms, result.escalated, result.rerank_score, outcome=outcome)
    logger.info("Outcome: %s | session=%s  latency=%.0fms  score=%.3f", outcome, sid[:8], latency_ms, result.rerank_score)
    sources = [SourceItem(article_id='', title=t) for t in result.sources]
    return ChatResponse(answer=result.answer, sources=sources, escalated=result.escalated, session_id=sid)

@app.get('/health', response_model=HealthResponse, summary='Pipeline health check')
async def health(request: Request) -> HealthResponse:
    models_loaded = hasattr(request.app.state, 'retriever') and hasattr(request.app.state, 'ce_model')
    qdrant_status = 'unreachable'
    if models_loaded:
        try:
            request.app.state.retriever.qdrant.get_collections()
            qdrant_status = 'ok'
        except Exception as exc:
            qdrant_status = f'error: {exc}'
    overall = 'ok' if models_loaded and qdrant_status == 'ok' else 'degraded'
    return HealthResponse(status=overall, qdrant=qdrant_status, models_loaded=models_loaded)
from generate import SUPPORT_CONTACT as _SUPPORT_CONTACT