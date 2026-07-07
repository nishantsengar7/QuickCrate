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

This is a multi-stage, production-ready Retrieval-Augmented Generation (RAG) system built for **QuickCrate**, a fictional quick-commerce grocery delivery app.

It combines hybrid keyword/dense search, cross-encoder reranking, a custom confidence gate, and Gemini 2.5 Flash for grounded support generation.

## Repository Layout
- `api.py`: FastAPI backend exposing `/chat` and `/health` endpoints.
- `app.py`: Streamlit frontend providing the chat interface.
- `ingest.py`: Knowledge base parser, BGE embedder, Qdrant upsert client, and BM25 builder.
- `retrieval.py`: Hybrid search engine fusing Qdrant & BM25 with Reciprocal Rank Fusion (RRF).
- `rerank.py`: Cross-encoder reranking layer utilizing `ms-marco-MiniLM-L-6-v2`.
- `generate.py`: Generation logic and confidence gate routing.
- `migrate_qdrant.py`: Utility script to migrate local Qdrant vectors to Qdrant Cloud.
- `smoke_test.py`: Observability and deployment drift validation suite.

## How to Deploy to Hugging Face Spaces
See details in the documentation below for environment variables, secrets, and verification steps.
