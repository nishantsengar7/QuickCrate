"""
migrate_qdrant.py -- QuickCrate local-to-cloud Qdrant Migration Script (Phase 8)
================================================================================

Why migrate from local Qdrant to Qdrant Cloud?
----------------------------------------------
During local development, running Qdrant as a local container (localhost:6333)
with a mounted directory is ideal: it is fast, free, requires no internet connection,
and is simple to set up.

However, when deploying to stateless, containerized environments like Hugging Face Spaces:
  1. Ephemeral storage: HF Spaces containers are ephemeral. Any data written to
     the container's disk (such as local Qdrant db files) is wiped on every
     rebuild, restart, or scaling event.
  2. Multi-process overhead: Running Qdrant inside the same container as the
     FastAPI/Streamlit app via a sidecar increases CPU/memory usage, potentially
     exceeding HF Spaces' free tier limits.
  3. Decoupled scaling: In production, the vector store (database) should be decoupled
     from the application (stateless compute). This allows scaling compute and
     database storage independently.

Migrating to a hosted vector database (like Qdrant Cloud's free tier) ensures:
  - Persistence: The indexed FAQ vectors persist indefinitely across container rebuilds.
  - Performance: The container runs only the API server and Streamlit frontend,
    saving RAM and startup time (we don't need to rebuild Qdrant or re-embed).
  - Scalability: Production traffic can hit decoupled instances.

This script loads the local KB articles, chunks them, embeds them using BGE-large,
and upserts them directly to the specified Qdrant Cloud collection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (e.g. from local .env)
load_dotenv(Path(__file__).parent / ".env")

import ingest
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate_qdrant")


def main() -> None:
    # Read Qdrant Cloud connection details from the environment
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    collection_name = os.environ.get("QDRANT_COLLECTION", ingest.QDRANT_COLLECTION)

    if not qdrant_url or not qdrant_api_key:
        logger.error("Missing QDRANT_URL or QDRANT_API_KEY in environment variables.")
        print("\nERROR: Please set QDRANT_URL and QDRANT_API_KEY environment variables.")
        print("You can set them in your .env file or run:")
        print("  $env:QDRANT_URL = 'https://your-cluster-url.aws.qdrant.io'")
        print("  $env:QDRANT_API_KEY = 'your-api-key'")
        return

    logger.info("Starting migration to Qdrant Cloud...")
    logger.info("Target Cluster: %s", qdrant_url)
    logger.info("Target Collection: %s", collection_name)

    # 1. Load articles
    logger.info("Step 1: Loading articles from disk...")
    articles = ingest.load_articles()

    # 2. Chunk articles
    logger.info("Step 2: Chunking articles...")
    chunks = ingest.chunk_articles(articles)

    # 3. Embed chunks using BGE-large
    logger.info("Step 3: Loading embedding model and embedding chunks (this might take a while on CPU)...")
    model = ingest.load_embedding_model()
    embeddings = ingest.embed_chunks(chunks, model)

    # 4. Connect to Qdrant Cloud and upsert
    logger.info("Step 4: Connecting to Qdrant Cloud cluster...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    logger.info("Step 5: Recreating collection and upserting points...")
    ingest.upsert_to_qdrant(client, chunks, embeddings, collection_name=collection_name)

    logger.info("Migration to Qdrant Cloud completed successfully!")


if __name__ == "__main__":
    main()
