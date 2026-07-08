from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any
from qdrant_client import QdrantClient
from ingest import BGE_PASSAGE_PREFIX, BM25_INDEX_PATH, EMBEDDING_MODEL_NAME, QDRANT_COLLECTION, QDRANT_URL, get_qdrant_client, load_bm25_index, load_embedding_model
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
BGE_QUERY_PREFIX: str = 'Represent this question for searching relevant passages: '
RRF_K: int = 60

class HybridRetriever:

    def __init__(self, qdrant_url: str=QDRANT_URL, collection: str=QDRANT_COLLECTION, model_name: str=EMBEDDING_MODEL_NAME, bm25_path: Path=BM25_INDEX_PATH) -> None:
        self.collection = collection
        self._qdrant_url = qdrant_url
        self.qdrant: QdrantClient = get_qdrant_client(qdrant_url)
        self.model: SentenceTransformer = load_embedding_model(model_name)
        self.bm25: BM25Okapi
        self.bm25_chunks: list[dict[str, Any]]
        if bm25_path.exists():
            self.bm25, self.bm25_chunks = load_bm25_index(bm25_path)
        else:
            logger.info('BM25 index file not found at %s. Rebuilding dynamically from raw KB articles...', bm25_path)
            from ingest import load_articles, chunk_articles, build_bm25_index
            articles = load_articles()
            self.bm25_chunks = chunk_articles(articles)
            self.bm25 = build_bm25_index(self.bm25_chunks)
            logger.info('Dynamic BM25 index built successfully over %d chunks.', len(self.bm25_chunks))
        self._chunk_meta: dict[str, dict[str, Any]] = {c['chunk_id']: c for c in self.bm25_chunks}
        logger.info("HybridRetriever ready. Collection='%s', chunks=%d", collection, len(self.bm25_chunks))

    def _embed_query(self, query: str) -> list[float]:
        vec = self.model.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True)
        return vec.tolist()

    def dense_search(self, query: str, top_k: int=20) -> list[tuple[str, float]]:
        query_vector = self._embed_query(query)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                hits = self.qdrant.query_points(collection_name=self.collection, query=query_vector, limit=top_k, with_payload=True).points
                return [(hit.payload['chunk_id'], float(hit.score)) for hit in hits]
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                wait = 2 ** (attempt - 1)
                logger.warning('Qdrant query_points attempt %d/%d failed (%s). Recreating client and retrying in %ds…', attempt, max_attempts, exc, wait)
                try:
                    self.qdrant.close()
                except Exception:
                    pass
                self.qdrant = get_qdrant_client(self._qdrant_url)
                time.sleep(wait)
        return []

    def sparse_search(self, query: str, top_k: int=20) -> list[tuple[str, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        results: list[tuple[str, float]] = []
        for idx, score in ranked[:top_k]:
            if score == 0.0:
                break
            results.append((self.bm25_chunks[idx]['chunk_id'], float(score)))
        return results

    def hybrid_search(self, query: str, top_k: int=10, dense_k: int=20, sparse_k: int=20) -> list[dict[str, Any]]:
        dense_hits: list[tuple[str, float]] = self.dense_search(query, dense_k)
        sparse_hits: list[tuple[str, float]] = self.sparse_search(query, sparse_k)
        rrf_scores: dict[str, float] = {}
        for rank, (chunk_id, _score) in enumerate(dense_hits, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (chunk_id, _score) in enumerate(sparse_hits, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        ranked_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        results: list[dict[str, Any]] = []
        for chunk_id in ranked_ids[:top_k]:
            meta = self._chunk_meta.get(chunk_id, {})
            results.append({'rrf_score': round(rrf_scores[chunk_id], 6), 'chunk_id': chunk_id, 'article_id': meta.get('article_id', ''), 'category': meta.get('category', ''), 'title': meta.get('title', ''), 'tags': meta.get('tags', []), 'text': meta.get('text', '')})
        return results

    def compare(self, query: str, top_k: int=3, dense_k: int=20, sparse_k: int=20) -> dict[str, list[dict[str, Any]]]:
        dense_hits = self.dense_search(query, top_k)
        dense_results = [{'title': self._chunk_meta.get(cid, {}).get('title', cid), 'category': self._chunk_meta.get(cid, {}).get('category', ''), 'chunk_id': cid, 'score': score} for cid, score in dense_hits]
        sparse_hits = self.sparse_search(query, top_k)
        sparse_results = [{'title': self._chunk_meta.get(cid, {}).get('title', cid), 'category': self._chunk_meta.get(cid, {}).get('category', ''), 'chunk_id': cid, 'score': score} for cid, score in sparse_hits]
        hybrid_results = [{'title': r['title'], 'category': r['category'], 'chunk_id': r['chunk_id'], 'score': r['rrf_score']} for r in self.hybrid_search(query, top_k, dense_k, sparse_k)]
        return {'dense': dense_results, 'sparse': sparse_results, 'hybrid': hybrid_results}

def _print_comparison(query: str, scenario_label: str, results: dict[str, list[dict[str, Any]]]) -> None:
    col_w = 52
    modes = ['dense', 'sparse', 'hybrid']
    header_sep = '+' + ('-' * col_w + '+') * 3
    print('\n' + '=' * (col_w * 3 + 4))
    print(f'  SCENARIO : {scenario_label}')
    print(f'  QUERY    : {query}')
    print('=' * (col_w * 3 + 4))
    print(header_sep)
    mode_labels = [f"  {'DENSE':^{col_w - 4}}  ", f"  {'SPARSE (BM25)':^{col_w - 4}}  ", f"  {'HYBRID (RRF)':^{col_w - 4}}  "]
    print('|' + '|'.join(mode_labels) + '|')
    print(header_sep)
    max_rows = max((len(results[m]) for m in modes))
    padded = {m: results[m] + [None] * (max_rows - len(results[m])) for m in modes}
    for row_idx in range(max_rows):
        cells = []
        for mode in modes:
            item = padded[mode][row_idx]
            if item is None:
                cells.append(' ' * col_w)
            else:
                rank = row_idx + 1
                score_label = 'rrf' if mode == 'hybrid' else 'scr'
                cat = item['category'][:6] if item['category'] else '?'
                title_max = col_w - 24
                title = item['title']
                if len(title) > title_max:
                    title = title[:title_max - 1] + '+'
                cell = f"  {rank}. [{cat:<6}] {title:<{title_max}} {score_label}={item['score']:.4f}  "
                cells.append(cell[:col_w])
        print('|' + '|'.join(cells) + '|')
    print(header_sep)
if __name__ == '__main__':
    retriever = HybridRetriever()
    scenario_a_query = 'What is the COD limit for cash payment orders?'
    results_a = retriever.compare(scenario_a_query, top_k=3)
    _print_comparison(query=scenario_a_query, scenario_label="A — Exact rare-term: 'COD limit'  (expect BM25 to excel)", results=results_a)
    scenario_b_query = 'I bought something and the funds disappeared from my bank account but the purchase never went through on the app'
    results_b = retriever.compare(scenario_b_query, top_k=3)
    _print_comparison(query=scenario_b_query, scenario_label='B — Paraphrased, zero shared vocab  (expect dense to excel, BM25 may miss)', results=results_b)
    scenario_c_query = 'I want to cancel my order and get my money back'
    results_c = retriever.compare(scenario_c_query, top_k=3)
    _print_comparison(query=scenario_c_query, scenario_label='C — Ambiguous: spans orders-002, returns-001, payments-005', results=results_c)
    print('\nDone. Compare columns to validate hybrid outperforms either leg alone.\n')