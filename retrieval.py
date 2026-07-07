"""
retrieval.py — QuickCrate Hybrid Retrieval Engine
==================================================

This module provides three retrieval modes over the QuickCrate FAQ knowledge
base that was ingested by ingest.py:

  1. Dense retrieval   — cosine similarity in BAAI/bge-large-en-v1.5 vector space
                         (Qdrant backend)
  2. Sparse retrieval  — BM25Okapi term frequency scoring
                         (rank_bm25, loaded from disk)
  3. Hybrid retrieval  — Reciprocal Rank Fusion (RRF) over both ranked lists

Run ``python ingest.py`` once to populate Qdrant and build the BM25 index,
then use this module for all retrieval needs.

Public interface
----------------
The :class:`HybridRetriever` class exposes three independent methods:

  * ``dense_search(query, top_k=20)``   -> list of (chunk_id, score)
  * ``sparse_search(query, top_k=20)``  -> list of (chunk_id, score)
  * ``hybrid_search(query, top_k=10)``  -> list of enriched result dicts

All three are callable on their own so callers can compare or ablate modes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Re-use ingest.py as the single source of truth for connection settings,
# model name, and index paths.  This ensures retrieval always targets the
# same collection and model that was used during ingestion.
# ---------------------------------------------------------------------------
from ingest import (
    BGE_PASSAGE_PREFIX,   # noqa: F401 — re-exported for callers that need it
    BM25_INDEX_PATH,
    EMBEDDING_MODEL_NAME,
    QDRANT_COLLECTION,
    QDRANT_URL,
    get_qdrant_client,
    load_bm25_index,
    load_embedding_model,
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
# Query-time instruction prefix
#
# BGE models are trained with *asymmetric* instruction prefixes:
#   - Passage prefix (used at INGEST time in ingest.py):
#       "Represent this sentence for searching relevant passages: "
#   - Query prefix   (used at RETRIEVAL time, defined here):
#       "Represent this question for searching relevant passages: "
#
# Why must they differ?
# During contrastive fine-tuning, BGE saw (query_prefix + question,
# passage_prefix + answer) pairs as positives.  The model learned to map
# query-prefixed text and passage-prefixed text into *aligned* sub-spaces
# so that cosine distance between them reflects semantic relevance.
# If you embed a query with the passage prefix (or vice-versa), you land
# in the wrong learned sub-space and cosine scores degrade significantly —
# independent evaluations show a 3-8 point NDCG@10 drop on BEIR benchmarks.
# ---------------------------------------------------------------------------
BGE_QUERY_PREFIX: str = "Represent this question for searching relevant passages: "

# Standard RRF constant (Cormack, Clarke, & Buettcher, SIGIR 2009).
# At k=60, rank-1 contributes 1/61 ~= 0.016 and rank-100 contributes 1/160
# ~= 0.006 — the constant dampens the outsized advantage of a single top-1
# hit, making the fused ranking more robust to outlier results from one leg.
RRF_K: int = 60


class HybridRetriever:
    """
    Hybrid retriever combining dense (Qdrant cosine) and sparse (BM25) search
    via Reciprocal Rank Fusion.

    Parameters
    ----------
    qdrant_url : str
        HTTP URL of the local Qdrant instance.
    collection : str
        Qdrant collection name to search.
    model_name : str
        HuggingFace model identifier for the embedding model.
    bm25_path : Path
        Path to the pickled BM25 bundle written by ingest.py.

    Examples
    --------
    >>> retriever = HybridRetriever()
    >>> results = retriever.hybrid_search("COD limit for cash orders", top_k=5)
    >>> for r in results:
    ...     print(r["title"], r["rrf_score"])
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection: str = QDRANT_COLLECTION,
        model_name: str = EMBEDDING_MODEL_NAME,
        bm25_path: Path = BM25_INDEX_PATH,
    ) -> None:
        self.collection = collection

        # --- Dense retrieval components ---
        self.qdrant: QdrantClient = get_qdrant_client(qdrant_url)
        self.model: SentenceTransformer = load_embedding_model(model_name)

        # --- Sparse retrieval components ---
        self.bm25: BM25Okapi
        self.bm25_chunks: list[dict[str, Any]]
        if bm25_path.exists():
            self.bm25, self.bm25_chunks = load_bm25_index(bm25_path)
        else:
            logger.info("BM25 index file not found at %s. Rebuilding dynamically from raw KB articles...", bm25_path)
            from ingest import load_articles, chunk_articles, build_bm25_index
            articles = load_articles()
            self.bm25_chunks = chunk_articles(articles)
            self.bm25 = build_bm25_index(self.bm25_chunks)
            logger.info("Dynamic BM25 index built successfully over %d chunks.", len(self.bm25_chunks))

        # Build a chunk_id -> metadata dict for O(1) payload lookups when
        # enriching hybrid results without a second Qdrant round-trip.
        self._chunk_meta: dict[str, dict[str, Any]] = {
            c["chunk_id"]: c for c in self.bm25_chunks
        }
        logger.info(
            "HybridRetriever ready. Collection='%s', chunks=%d",
            collection,
            len(self.bm25_chunks),
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed *query* with the BGE **query** instruction prefix.

        IMPORTANT: this must use BGE_QUERY_PREFIX (not BGE_PASSAGE_PREFIX).
        See the module-level comment on why the two prefixes must differ.
        """
        vec = self.model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,  # L2-norm -> cosine sim == dot product
            convert_to_numpy=True,
        )
        return vec.tolist()

    # -----------------------------------------------------------------------
    # Public retrieval methods
    # -----------------------------------------------------------------------

    def dense_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Embed *query* and return the top-k nearest chunks from Qdrant.

        Why the query prefix MUST differ from the passage prefix
        ---------------------------------------------------------
        BAAI/bge-large-en-v1.5 was contrastively trained on asymmetric pairs:
        queries were embedded with a *query* instruction and documents with a
        *passage* instruction.  At inference time we must honour this split:
          - Passages indexed at ingest time -> BGE_PASSAGE_PREFIX (in ingest.py)
          - Queries at retrieval time       -> BGE_QUERY_PREFIX   (this method)
        Swapping or unifying the prefixes puts the query into the passage
        embedding sub-space, breaking the learned alignment and degrading
        cosine similarity scores.

        Parameters
        ----------
        query : str
            Raw user query string.
        top_k : int
            Number of results to return from Qdrant.

        Returns
        -------
        list[tuple[str, float]]
            Ordered list of (chunk_id, cosine_score) pairs, highest score
            first.  chunk_id matches the payload field set during ingestion.
        """
        query_vector = self._embed_query(query)
        hits = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        return [(hit.payload["chunk_id"], float(hit.score)) for hit in hits]

    def sparse_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Tokenise *query* and return the top-k BM25-ranked chunks.

        Tokenisation mirrors the ingest-time strategy (whitespace split +
        lowercase) so that query and document token spaces are identical.
        Using a different tokeniser at retrieval time would shift IDF weights
        and hurt recall.

        Parameters
        ----------
        query : str
            Raw user query string.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            Ordered list of (chunk_id, bm25_score) pairs, highest score first.
            Chunks with a zero score are excluded.
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Pair each score with its parallel chunk_id, sort descending.
        ranked = sorted(
            enumerate(scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        results: list[tuple[str, float]] = []
        for idx, score in ranked[:top_k]:
            if score == 0.0:
                break   # BM25 scores are non-negative; stop at zero hits
            results.append((self.bm25_chunks[idx]["chunk_id"], float(score)))
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        dense_k: int = 20,
        sparse_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Fuse dense and sparse rankings via Reciprocal Rank Fusion (RRF).

        Why RRF instead of weighted score fusion?
        -----------------------------------------
        Weighted score fusion (e.g., ``alpha * dense_score + (1-alpha) *
        bm25_score``) requires the two score distributions to be calibrated
        on the same scale.  In practice:
          - Cosine similarity scores from Qdrant live in [-1, 1].
          - BM25 scores are unbounded positive floats whose magnitude scales
            with corpus size and query length.
        Choosing ``alpha`` without this calibration is fragile: a long query
        inflates BM25 scores and drowns out dense signals regardless of alpha.
        Re-normalising both distributions (e.g., min-max scaling per query)
        is computationally cheap but adds another hyperparameter and still
        collapses to score fusion's core fragility.

        RRF sidesteps all of this by working purely on *ranks* rather than
        raw scores.  The RRF formula is:

            RRF(d) = sum_i  1 / (k + rank_i(d))

        where rank_i(d) is document d's position (1-indexed) in ranked list i.
        Key properties:
          - Scale-invariant: multiplying all BM25 scores by 1000 has zero
            effect on ranks and therefore zero effect on the fused order.
          - Robust to rank gaps: a result at rank 1 in one list and rank 50 in
            the other still gets a competitive fused score.
          - No hyperparameter tuning: k=60 (Cormack et al. 2009) works well
            across domains without per-dataset calibration.
          - Documents that appear in only one list are still scored (they
            receive 1/(k + rank) from that list alone), so neither retriever's
            unique hits are silenced.

        Parameters
        ----------
        query : str
            Raw user query string.
        top_k : int
            Number of final results after fusion.
        dense_k : int
            Candidate pool size from the dense leg before fusion.
        sparse_k : int
            Candidate pool size from the sparse leg before fusion.

        Returns
        -------
        list[dict]
            Top-k fused results, each dict containing:
            rrf_score, chunk_id, article_id, category, title, tags, text.
        """
        # 1. Retrieve candidates from each leg.
        dense_hits: list[tuple[str, float]] = self.dense_search(query, dense_k)
        sparse_hits: list[tuple[str, float]] = self.sparse_search(query, sparse_k)

        # 2. Accumulate RRF scores keyed by chunk_id.
        rrf_scores: dict[str, float] = {}

        for rank, (chunk_id, _score) in enumerate(dense_hits, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        for rank, (chunk_id, _score) in enumerate(sparse_hits, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        # 3. Sort by fused score descending and enrich with metadata.
        ranked_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        results: list[dict[str, Any]] = []
        for chunk_id in ranked_ids[:top_k]:
            meta = self._chunk_meta.get(chunk_id, {})
            results.append(
                {
                    "rrf_score": round(rrf_scores[chunk_id], 6),
                    "chunk_id": chunk_id,
                    "article_id": meta.get("article_id", ""),
                    "category": meta.get("category", ""),
                    "title": meta.get("title", ""),
                    "tags": meta.get("tags", []),
                    "text": meta.get("text", ""),
                }
            )
        return results

    # -----------------------------------------------------------------------
    # Convenience wrapper for side-by-side comparison
    # -----------------------------------------------------------------------

    def compare(
        self,
        query: str,
        top_k: int = 3,
        dense_k: int = 20,
        sparse_k: int = 20,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Run all three retrieval modes and return them under labelled keys.

        Returns
        -------
        dict with keys "dense", "sparse", "hybrid", each holding a list of
        result dicts with keys: title, category, chunk_id, score.
        """
        # --- Dense only ---
        dense_hits = self.dense_search(query, top_k)
        dense_results = [
            {
                "title": self._chunk_meta.get(cid, {}).get("title", cid),
                "category": self._chunk_meta.get(cid, {}).get("category", ""),
                "chunk_id": cid,
                "score": score,
            }
            for cid, score in dense_hits
        ]

        # --- Sparse only ---
        sparse_hits = self.sparse_search(query, top_k)
        sparse_results = [
            {
                "title": self._chunk_meta.get(cid, {}).get("title", cid),
                "category": self._chunk_meta.get(cid, {}).get("category", ""),
                "chunk_id": cid,
                "score": score,
            }
            for cid, score in sparse_hits
        ]

        # --- Hybrid ---
        hybrid_results = [
            {
                "title": r["title"],
                "category": r["category"],
                "chunk_id": r["chunk_id"],
                "score": r["rrf_score"],
            }
            for r in self.hybrid_search(query, top_k, dense_k, sparse_k)
        ]

        return {"dense": dense_results, "sparse": sparse_results, "hybrid": hybrid_results}


# ---------------------------------------------------------------------------
# Pretty-printer helper
# ---------------------------------------------------------------------------

def _print_comparison(
    query: str,
    scenario_label: str,
    results: dict[str, list[dict[str, Any]]],
) -> None:
    """
    Print dense / sparse / hybrid top results side-by-side in a readable table.

    Fixed-width columns let a human reviewer scan across retrieval modes at a
    glance to spot where hybrid outperforms either leg alone.
    """
    col_w = 52
    modes = ["dense", "sparse", "hybrid"]
    header_sep = "+" + (("-" * col_w + "+") * 3)

    print("\n" + "=" * (col_w * 3 + 4))
    print(f"  SCENARIO : {scenario_label}")
    print(f"  QUERY    : {query}")
    print("=" * (col_w * 3 + 4))
    print(header_sep)

    mode_labels = [
        f"  {'DENSE':^{col_w - 4}}  ",
        f"  {'SPARSE (BM25)':^{col_w - 4}}  ",
        f"  {'HYBRID (RRF)':^{col_w - 4}}  ",
    ]
    print("|" + "|".join(mode_labels) + "|")
    print(header_sep)

    max_rows = max(len(results[m]) for m in modes)
    padded = {m: results[m] + [None] * (max_rows - len(results[m])) for m in modes}

    for row_idx in range(max_rows):
        cells = []
        for mode in modes:
            item = padded[mode][row_idx]
            if item is None:
                cells.append(" " * col_w)
            else:
                rank = row_idx + 1
                score_label = "rrf" if mode == "hybrid" else "scr"
                cat = item["category"][:6] if item["category"] else "?"
                title_max = col_w - 24
                title = item["title"]
                if len(title) > title_max:
                    title = title[:title_max - 1] + "+"
                cell = f"  {rank}. [{cat:<6}] {title:<{title_max}} {score_label}={item['score']:.4f}  "
                cells.append(cell[:col_w])
        print("|" + "|".join(cells) + "|")

    print(header_sep)


# ---------------------------------------------------------------------------
# Entry point — three diagnostic scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = HybridRetriever()

    # -----------------------------------------------------------------------
    # Scenario A — Exact-term query
    # -----------------------------------------------------------------------
    # "COD limit" is a verbatim phrase from payments-001: "COD is available
    # for orders up to Rs.2,000".  BM25 should rank this near the top because
    # the exact tokens {"cod", "limit"} are rare and discriminative.  Dense
    # retrieval may also score it well via semantic similarity, but BM25's
    # precision for rare exact terms is typically unmatched.  Hybrid should
    # at minimum match the best of the two legs.
    # -----------------------------------------------------------------------
    scenario_a_query = "What is the COD limit for cash payment orders?"
    results_a = retriever.compare(scenario_a_query, top_k=3)
    _print_comparison(
        query=scenario_a_query,
        scenario_label="A — Exact rare-term: 'COD limit'  (expect BM25 to excel)",
        results=results_a,
    )

    # -----------------------------------------------------------------------
    # Scenario B — Paraphrased query (no shared vocabulary with target)
    # -----------------------------------------------------------------------
    # Target: payments-002 ("My payment failed but money was deducted").
    # The query below uses zero words from that title or body — it describes
    # the situation in plain paraphrase.  BM25 will almost certainly fail
    # because none of the query tokens appear at high frequency in the article.
    # Dense (semantic) retrieval should still find it because BGE understands
    # the equivalence between "funds disappeared from my bank" and "money was
    # deducted".  Hybrid should match or beat dense alone without being harmed
    # by BM25's miss.
    # -----------------------------------------------------------------------
    scenario_b_query = (
        "I bought something and the funds disappeared from my bank account "
        "but the purchase never went through on the app"
    )
    results_b = retriever.compare(scenario_b_query, top_k=3)
    _print_comparison(
        query=scenario_b_query,
        scenario_label="B — Paraphrased, zero shared vocab  (expect dense to excel, BM25 may miss)",
        results=results_b,
    )

    # -----------------------------------------------------------------------
    # Scenario C — Ambiguous query spanning multiple articles
    # -----------------------------------------------------------------------
    # "cancel and get money back" is intentionally vague: it could match
    # orders-002 (cancelling an order), returns-001 (refund policy), or
    # payments-005 (refund timelines).  This tests whether hybrid surfaces
    # the most broadly useful articles and orders them better than either
    # individual leg, which may over-commit to one interpretation.
    # -----------------------------------------------------------------------
    scenario_c_query = "I want to cancel my order and get my money back"
    results_c = retriever.compare(scenario_c_query, top_k=3)
    _print_comparison(
        query=scenario_c_query,
        scenario_label="C — Ambiguous: spans orders-002, returns-001, payments-005",
        results=results_c,
    )

    print(
        "\nDone. Compare columns to validate hybrid outperforms either leg alone.\n"
    )
