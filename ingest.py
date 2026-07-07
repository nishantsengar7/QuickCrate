"""
ingest.py — QuickCrate Customer Support RAG Ingestion Pipeline
==============================================================

Design overview
---------------
This module performs the full ingestion pipeline for the QuickCrate knowledge
base: it loads raw FAQ articles, decides whether each article needs splitting,
embeds the resulting chunks, writes them to a local Qdrant vector store, and
separately builds a BM25 sparse-retrieval index.

The pipeline is intentionally separated into small, composable functions so
that retrieval.py can import the Qdrant client and the BM25 index without
re-running the full ingest.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — single source of truth so retrieval.py can import these
# ---------------------------------------------------------------------------

#: Directory containing one JSON file per KB category.
KB_ARTICLES_DIR: Path = Path("data/kb_articles")

#: Topic manifest (used for metadata / validation, not for ingestion directly).
KB_TOPICS_FILE: Path = Path("data/kb_topics.json")

#: BAAI/bge-large-en-v1.5 produces 1024-dimensional embeddings.
#: We use the large variant (vs. base) for its superior retrieval quality on
#: domain-specific corpora; the size trade-off is acceptable for a local setup.
EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

#: BGE models are trained with a special instruction prefix for *passage*
#: (document) embedding vs. *query* embedding:
#:   - Passage prefix : "Represent this sentence for searching relevant passages: "
#:   - Query prefix   : "Represent this question for searching relevant passages: "
#: Using the WRONG prefix degrades retrieval quality because the model's
#: attention heads were fine-tuned to distinguish these roles.  At query time
#: (retrieval.py), use the query prefix; at ingest time, use the passage prefix.
BGE_PASSAGE_PREFIX: str = "Represent this sentence for searching relevant passages: "

#: Qdrant connection settings.  Using gRPC is faster for bulk upserts, but HTTP
#: is simpler for local dev; switch prefer_grpc=True if throughput becomes an
#: issue with larger corpora.
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "quickcrate_kb")

#: Token budget per chunk before we apply recursive splitting.  We chose
#: article-level chunking as the primary strategy because each FAQ article is
#: already a self-contained, semantically cohesive unit (150-400 words).
#: Splitting within an article risks separating the question context from its
#: answer, harming retrieval precision.  The 400-word guard rail below exists
#: only for abnormally long articles that could overflow the BGE context window
#: (512 sub-word tokens for bge-large-en-v1.5).
MAX_WORDS_BEFORE_SPLIT: int = 400

#: Overlap in *words* when recursive splitting kicks in.
#: ~50 tokens ≈ 50 words for English text (GPT-style BPE tokenizer averages
#: ~1.3 words/token, but for simple word-level splitting a word≈token is close
#: enough given the model's 512-token window).
SPLIT_OVERLAP_WORDS: int = 50

#: Chunk size in words for recursive splitting.
SPLIT_CHUNK_WORDS: int = 300

#: Path where the pickled BM25 index is saved.  retrieval.py loads this file.
BM25_INDEX_PATH: Path = Path("bm25_index.pkl")

# ---------------------------------------------------------------------------
# Step 1: Load & merge KB articles
# ---------------------------------------------------------------------------


def load_articles(kb_dir: Path = KB_ARTICLES_DIR) -> list[dict[str, Any]]:
    """
    Load every *.json file in *kb_dir* and merge them into a single list.

    Each file is expected to be a JSON array of article objects with at least
    the fields: id, category, title, body, tags.

    Why merge at load time?
    -----------------------
    The pipeline treats the full corpus as a flat list of articles.  Merging
    early means every downstream function has a uniform interface regardless of
    how the source data is partitioned across files.

    Parameters
    ----------
    kb_dir:
        Directory containing per-category JSON files.

    Returns
    -------
    list[dict]
        Merged list of article dicts, preserving insertion order (alphabetical
        by filename, so results are reproducible across runs).
    """
    all_articles: list[dict[str, Any]] = []
    json_files = sorted(kb_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No .json files found in {kb_dir.resolve()}")

    for path in json_files:
        with path.open(encoding="utf-8-sig") as fh:  # utf-8-sig strips Windows BOM
            articles = json.load(fh)
        logger.info("Loaded %d articles from %s", len(articles), path.name)
        all_articles.extend(articles)

    logger.info("Total articles loaded: %d", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# Step 2: Chunk articles
# ---------------------------------------------------------------------------


def _split_text_recursive(
    text: str,
    chunk_words: int = SPLIT_CHUNK_WORDS,
    overlap_words: int = SPLIT_OVERLAP_WORDS,
) -> list[str]:
    """
    Recursively split *text* into overlapping word-window chunks.

    Why word-level (not token-level) splitting here?
    ------------------------------------------------
    A pure token-level split requires running a tokenizer on every article,
    adding latency.  For English FAQ text in the 300-500 word range, the
    word-to-token ratio is stable enough that a 300-word window stays safely
    within BGE's 512-token context window with overhead to spare.  If the
    corpus were multilingual or code-heavy, a proper BPE tokenizer would be
    warranted.

    Parameters
    ----------
    text:
        Raw article body text.
    chunk_words:
        Target number of words per chunk.
    overlap_words:
        Number of words to repeat between adjacent chunks (ensures continuity
        of context at chunk boundaries).

    Returns
    -------
    list[str]
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words  # slide with overlap

    return chunks


def chunk_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Produce a flat list of *chunk* dicts from the article list.

    Chunking strategy
    -----------------
    Primary strategy — article-level chunking:
        Each FAQ article is already a self-contained answer to a single
        question (150-400 words).  Keeping the article whole preserves the
        full semantic context, avoids dangling sentence fragments, and ensures
        the retriever can rank the best article rather than the best sentence.
        This is preferred over fixed-size chunking for short, structured FAQ
        corpora.

    Fallback — recursive word-window splitting:
        If a rare article exceeds MAX_WORDS_BEFORE_SPLIT words, we split it
        with a sliding window and generous overlap so that every fact in the
        article appears in at least one complete chunk.  Chunks inherit the
        parent article's metadata (id, category, title, tags) so the retriever
        can surface source attribution for any chunk.

    Parameters
    ----------
    articles:
        List of article dicts as returned by :func:`load_articles`.

    Returns
    -------
    list[dict]
        Each dict contains:
        - chunk_id   : unique identifier, e.g. "orders-001#0"
        - article_id : parent article id
        - category   : article category
        - title      : article title
        - tags       : article tags
        - text       : the chunk text (title + body for richer embedding signal)
    """
    chunks: list[dict[str, Any]] = []

    for article in articles:
        body: str = article["body"]
        word_count = len(body.split())

        if word_count <= MAX_WORDS_BEFORE_SPLIT:
            # --- Primary path: single-chunk per article ---
            # We prepend the title to the body before embedding because BGE was
            # trained on title+passage pairs; the title acts as a semantic
            # anchor that improves topical precision.
            text = f"{article['title']}\n\n{body}"
            chunks.append(
                {
                    "chunk_id": f"{article['id']}#0",
                    "article_id": article["id"],
                    "category": article["category"],
                    "title": article["title"],
                    "tags": article["tags"],
                    "text": text,
                }
            )
        else:
            # --- Fallback path: recursive splitting for long articles ---
            logger.warning(
                "Article %s has %d words (> %d) — applying recursive split.",
                article["id"],
                word_count,
                MAX_WORDS_BEFORE_SPLIT,
            )
            body_chunks = _split_text_recursive(body)
            for idx, body_chunk in enumerate(body_chunks):
                text = f"{article['title']}\n\n{body_chunk}"
                chunks.append(
                    {
                        "chunk_id": f"{article['id']}#{idx}",
                        "article_id": article["id"],
                        "category": article["category"],
                        "title": article["title"],
                        "tags": article["tags"],
                        "text": text,
                    }
                )

    logger.info("Total chunks produced: %d", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Embed chunks
# ---------------------------------------------------------------------------


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Load (or download) the BGE embedding model.

    Model choice rationale
    ----------------------
    BAAI/bge-large-en-v1.5 was chosen because:
    1. It achieves top-tier performance on MTEB Retrieval benchmarks for
       English text, consistently outperforming OpenAI ada-002 on many tasks.
    2. It supports the passage/query instruction prefix protocol (INSTRUCTOR-
       style) which lets us optimise embeddings for asymmetric search (short
       query vs. long passage) — critical for FAQ retrieval.
    3. Running locally avoids API rate limits and keeps PII on-premises.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or local path.

    Returns
    -------
    SentenceTransformer
        Loaded model ready for inference.
    """
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info(
        "Model loaded. Embedding dimension: %d", model.get_embedding_dimension()
    )
    return model


def embed_chunks(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Embed each chunk's text using the BGE passage instruction prefix.

    Why the passage prefix matters
    --------------------------------
    BAAI/bge-large-en-v1.5 was fine-tuned with contrastive learning on pairs
    of (query, passage).  During training, queries were prefixed with a "query"
    instruction and passages with a "passage" instruction.  At inference time:
    - Passage embeddings (this function): use BGE_PASSAGE_PREFIX
    - Query embeddings  (retrieval.py)  : use a query-specific prefix
    Mixing them up produces embeddings from different learned sub-spaces,
    degrading cosine similarity scores and retrieval quality.

    Parameters
    ----------
    chunks:
        List of chunk dicts; each must have a "text" key.
    model:
        Loaded SentenceTransformer model.
    batch_size:
        Number of chunks to encode per forward pass.  32 is a safe default
        that fits in CPU memory for bge-large.  Increase to 64+ with a GPU.

    Returns
    -------
    list[list[float]]
        Parallel list of embedding vectors (one per chunk).
    """
    texts = [BGE_PASSAGE_PREFIX + chunk["text"] for chunk in chunks]
    logger.info("Embedding %d chunks (batch_size=%d) …", len(texts), batch_size)

    # show_progress_bar=True gives a tqdm bar in the terminal during a long run.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity ≡ dot product after L2-norm
        convert_to_numpy=True,
    )
    logger.info("Embedding complete. Shape: %s", embeddings.shape)
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Step 4: Upsert to Qdrant
# ---------------------------------------------------------------------------


def get_qdrant_client(url: str = QDRANT_URL) -> QdrantClient:
    """
    Return a QdrantClient connected to a local or cloud Qdrant instance.

    This function is intentionally thin so that retrieval.py can import it
    and reuse the same client without duplicating connection logic.

    Parameters
    ----------
    url:
        HTTP/HTTPS URL of the Qdrant server.

    Returns
    -------
    QdrantClient
    """
    api_key = os.environ.get("QDRANT_API_KEY")
    if api_key:
        logger.info("Connecting to Qdrant Cloud (secure client) at %s", url)
        return QdrantClient(url=url, api_key=api_key)
    else:
        logger.info("Connecting to Qdrant at %s", url)
        return QdrantClient(url=url)


def upsert_to_qdrant(
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    collection_name: str = QDRANT_COLLECTION,
) -> None:
    """
    Create (or recreate) the Qdrant collection and upsert all chunk vectors.

    Why upsert (not insert)?
    ------------------------
    Upsert is idempotent — re-running ingest.py after adding new articles
    updates existing points and inserts new ones without duplicating data.
    The collection is recreated from scratch on each ingest run to avoid
    stale vectors from deleted articles; for incremental updates on large
    corpora, consider using a filter-based delete + upsert pattern instead.

    Parameters
    ----------
    client:
        Connected QdrantClient.
    chunks:
        List of chunk dicts (must match order of *embeddings*).
    embeddings:
        Parallel list of float vectors.
    collection_name:
        Qdrant collection name.
    """
    vector_dim = len(embeddings[0])

    # Recreate the collection to ensure a clean slate.
    # In a production incremental pipeline, replace this with a check-and-create
    # pattern so existing vectors are preserved between runs.
    if client.collection_exists(collection_name):
        logger.info("Dropping existing collection '%s' for clean re-ingest.", collection_name)
        client.delete_collection(collection_name)

    logger.info(
        "Creating collection '%s' with vector_size=%d, distance=COSINE.",
        collection_name,
        vector_dim,
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    # Build PointStructs — Qdrant requires integer or UUID point IDs.
    # We use enumerate to generate stable integer IDs; the original chunk_id
    # string is stored in the payload so retrieval.py can surface it.
    points = [
        PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "chunk_id": chunk["chunk_id"],
                "article_id": chunk["article_id"],
                "category": chunk["category"],
                "title": chunk["title"],
                "tags": chunk["tags"],
                "text": chunk["text"],
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # Batch upsert in groups of 256 to avoid large request payloads.
    batch_size = 256
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info("Upserted points %d–%d", i, i + len(batch) - 1)

    logger.info(
        "Qdrant upsert complete. Collection '%s' now holds %d points.",
        collection_name,
        len(points),
    )


# ---------------------------------------------------------------------------
# Step 5: Build & persist BM25 index
# ---------------------------------------------------------------------------


def build_bm25_index(chunks: list[dict[str, Any]]) -> BM25Okapi:
    """
    Build a BM25Okapi index over the chunk corpus.

    Why BM25 alongside dense vectors?
    -----------------------------------
    Dense (neural) retrieval excels at semantic similarity but can miss exact
    keyword matches — e.g., a customer asking about "GSTIN" or "CoFT" where
    those exact terms appear in only a few articles.  BM25 is a proven sparse
    retriever that handles exact-match and rare-term queries well.  Combining
    both retrievers via Reciprocal Rank Fusion (RRF) in retrieval.py gives
    hybrid retrieval that is more robust than either alone.

    Tokenisation: simple whitespace split + lowercase.  This is sufficient for
    English FAQ text; for a production system, add lemmatisation (NLTK/spaCy)
    and stop-word removal to improve IDF weighting.

    Parameters
    ----------
    chunks:
        List of chunk dicts; each must have a "text" key.

    Returns
    -------
    BM25Okapi
        Fitted BM25 index.
    """
    corpus_tokens = [chunk["text"].lower().split() for chunk in chunks]
    logger.info("Building BM25Okapi index over %d documents …", len(corpus_tokens))
    bm25 = BM25Okapi(corpus_tokens)
    logger.info("BM25 index built. Vocabulary size: %d", len(bm25.idf))
    return bm25


def save_bm25_index(
    bm25: BM25Okapi,
    chunks: list[dict[str, Any]],
    path: Path = BM25_INDEX_PATH,
) -> None:
    """
    Pickle the BM25 index *and* the parallel chunk list to *path*.

    We bundle the chunk list with the index so that retrieval.py can map a
    BM25 rank back to the corresponding chunk payload without querying Qdrant.
    The two are stored together in a single dict under a versioned key to make
    future format migrations easier.

    Parameters
    ----------
    bm25:
        Fitted BM25Okapi instance.
    chunks:
        Chunk list in the same order that was used to build the index.
    path:
        Output pickle file path.
    """
    payload = {
        "version": 1,
        "bm25": bm25,
        "chunks": chunks,  # parallel to bm25 corpus — do not reorder
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("BM25 index pickled to %s (%.1f KB).", path, path.stat().st_size / 1024)


def load_bm25_index(path: Path = BM25_INDEX_PATH) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """
    Load and return the BM25 index and parallel chunk list from *path*.

    This function is the companion to :func:`save_bm25_index` and is intended
    to be called from retrieval.py.

    Parameters
    ----------
    path:
        Path to the pickled BM25 bundle.

    Returns
    -------
    tuple[BM25Okapi, list[dict]]
        ``(bm25_index, chunks)`` — the index and the parallel chunk metadata.

    Raises
    ------
    FileNotFoundError
        If the index has not been built yet (run ingest.py first).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {path}. Run ingest.py first."
        )
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return payload["bm25"], payload["chunks"]


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------


def run_ingest_pipeline() -> dict[str, Any]:
    """
    Execute the full ingest pipeline end-to-end and return a summary dict.

    Sequence
    --------
    1. Load all KB articles from disk.
    2. Chunk each article (article-level with long-article fallback).
    3. Load the BGE embedding model.
    4. Embed all chunks using the passage instruction prefix.
    5. Upsert chunks + vectors to Qdrant.
    6. Build BM25 index and pickle to disk.

    Returns
    -------
    dict
        Summary statistics for logging / testing:
        - total_articles
        - total_chunks
        - embedding_dim
        - bm25_vocab_size
        - qdrant_collection
    """
    # 1. Load
    articles = load_articles()

    # 2. Chunk
    chunks = chunk_articles(articles)

    # 3. Embed
    model = load_embedding_model()
    embedding_dim: int = model.get_embedding_dimension()
    embeddings = embed_chunks(chunks, model)

    # 4. Upsert to Qdrant
    client = get_qdrant_client()
    upsert_to_qdrant(client, chunks, embeddings)

    # 5. BM25
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25, chunks)

    return {
        "total_articles": len(articles),
        "total_chunks": len(chunks),
        "embedding_dim": embedding_dim,
        "bm25_vocab_size": len(bm25.idf),
        "qdrant_collection": QDRANT_COLLECTION,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = run_ingest_pipeline()

    print("\n" + "=" * 60)
    print("  QuickCrate KB Ingest — Summary")
    print("=" * 60)
    print(f"  Total articles   : {summary['total_articles']}")
    print(f"  Total chunks     : {summary['total_chunks']}")
    print(f"  Embedding dim    : {summary['embedding_dim']}")
    print(f"  BM25 vocab size  : {summary['bm25_vocab_size']}")
    print(f"  Qdrant collection: {summary['qdrant_collection']}")
    print("=" * 60)
    print("\nIngest complete. Run retrieval.py to test queries.\n")
