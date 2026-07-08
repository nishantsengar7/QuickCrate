---
title: QuickCrate Customer Support RAG
emoji: 🛒
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# QuickCrate Customer Support RAG

This is a multi-stage, production-ready Retrieval-Augmented Generation (RAG) system built for **QuickCrate**, an ultra-fast grocery delivery app. The system provides a highly accurate, grounded customer support assistant with safe guardrails.

## Pipeline Architecture

The system operates as a four-stage retrieval and routing pipeline:

1. **Query Expansion & Rewriting**: Every query is rewritten into a standalone question to resolve references and expand keywords (such as `"payment option"`), ensuring optimal downstream search performance.
2. **Hybrid Retrieval**: Combines semantic vector search (via Qdrant Cloud using `BAAI/bge-large-en-v1.5`) and lexical search (via BM25), fused together using Reciprocal Rank Fusion (RRF).
3. **Cross-Encoder Reranking**: Reranks the top retrieved passages using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to capture deep joint-attention relevance.
4. **Confidence Gate**: Applies a calibrated score threshold (`1.6`). Queries scoring below this threshold are safely routed to human support (`support.quickcrate.in`), preventing hallucinated answers on out-of-scope requests.
5. **Grounded Generation**: Feeds the top-scoring passages into a Gemini LLM (configured to `gemini-3.5-flash` with robust exponential backoff retry logic) for grounded, warm customer support answers.

## Repository Layout
- `api.py`: FastAPI server exposing `/chat` and `/health` endpoints.
- `app.py`: Streamlit chat UI for customer interactions.
- `generate.py`: Standalone query expansion, answer generation, and confidence gate checking.
- `retrieval.py`: Hybrid search fusing lexical BM25 and semantic Qdrant indexes.
- `rerank.py`: Cross-Encoder reranking using joint attention.
- `ingest.py`: Parses the local knowledge base, computes embeddings, and builds indexes.
- `migrate_qdrant.py`: Database migration script to sync vectors with Qdrant Cloud.
- `smoke_test.py`: Observability and deployment drift validation suite.

## Environment Variables
Ensure the following variables are configured in `.env` or the hosting environment:
- `GEMINI_API_KEY`: API key for Gemini.
- `QC_GEMINI_MODEL`: Model identifier (e.g., `gemini-3.5-flash` for high-quota free tier access).
- `QC_CONFIDENCE_THRESHOLD`: Reranking threshold (calibrated to `1.6`).
- `QDRANT_URL`: Qdrant Cloud instance URL.
- `QDRANT_API_KEY`: Qdrant Cloud API key.

## Local Deployment & Verification

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start FastAPI Backend**:
   ```bash
   python -m uvicorn api:app --host 0.0.0.0 --port 8000
   ```

3. **Run Validation Tests**:
   ```bash
   python smoke_test.py http://localhost:8000
   ```
