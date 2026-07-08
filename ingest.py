from __future__ import annotations
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
KB_ARTICLES_DIR: Path = Path('data/kb_articles')
KB_TOPICS_FILE: Path = Path('data/kb_topics.json')
EMBEDDING_MODEL_NAME: str = 'BAAI/bge-large-en-v1.5'
BGE_PASSAGE_PREFIX: str = 'Represent this sentence for searching relevant passages: '
QDRANT_URL: str = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_COLLECTION: str = os.getenv('QDRANT_COLLECTION', 'quickcrate_kb')
MAX_WORDS_BEFORE_SPLIT: int = 400
SPLIT_OVERLAP_WORDS: int = 50
SPLIT_CHUNK_WORDS: int = 300
BM25_INDEX_PATH: Path = Path('bm25_index.pkl')

def load_articles(kb_dir: Path=KB_ARTICLES_DIR) -> list[dict[str, Any]]:
    all_articles: list[dict[str, Any]] = []
    json_files = sorted(kb_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f'No .json files found in {kb_dir.resolve()}')
    for path in json_files:
        with path.open(encoding='utf-8-sig') as fh:
            articles = json.load(fh)
        logger.info('Loaded %d articles from %s', len(articles), path.name)
        all_articles.extend(articles)
    logger.info('Total articles loaded: %d', len(all_articles))
    return all_articles

def _split_text_recursive(text: str, chunk_words: int=SPLIT_CHUNK_WORDS, overlap_words: int=SPLIT_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks

def chunk_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for article in articles:
        body: str = article['body']
        word_count = len(body.split())
        if word_count <= MAX_WORDS_BEFORE_SPLIT:
            text = f"{article['title']}\n\n{body}"
            chunks.append({'chunk_id': f"{article['id']}#0", 'article_id': article['id'], 'category': article['category'], 'title': article['title'], 'tags': article['tags'], 'text': text})
        else:
            logger.warning('Article %s has %d words (> %d) — applying recursive split.', article['id'], word_count, MAX_WORDS_BEFORE_SPLIT)
            body_chunks = _split_text_recursive(body)
            for idx, body_chunk in enumerate(body_chunks):
                text = f"{article['title']}\n\n{body_chunk}"
                chunks.append({'chunk_id': f"{article['id']}#{idx}", 'article_id': article['id'], 'category': article['category'], 'title': article['title'], 'tags': article['tags'], 'text': text})
    logger.info('Total chunks produced: %d', len(chunks))
    return chunks

def load_embedding_model(model_name: str=EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    logger.info('Loading embedding model: %s', model_name)
    model = SentenceTransformer(model_name)
    logger.info('Model loaded. Embedding dimension: %d', model.get_embedding_dimension())
    return model

def embed_chunks(chunks: list[dict[str, Any]], model: SentenceTransformer, batch_size: int=32) -> list[list[float]]:
    texts = [BGE_PASSAGE_PREFIX + chunk['text'] for chunk in chunks]
    logger.info('Embedding %d chunks (batch_size=%d) …', len(texts), batch_size)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    logger.info('Embedding complete. Shape: %s', embeddings.shape)
    return embeddings.tolist()

def get_qdrant_client(url: str=QDRANT_URL) -> QdrantClient:
    api_key = os.environ.get('QDRANT_API_KEY')
    if api_key:
        logger.info('Connecting to Qdrant Cloud (secure client) at %s', url)
        return QdrantClient(url=url, api_key=api_key, timeout=90.0)
    else:
        logger.info('Connecting to Qdrant at %s', url)
        return QdrantClient(url=url, timeout=90.0)

def upsert_to_qdrant(client: QdrantClient, chunks: list[dict[str, Any]], embeddings: list[list[float]], collection_name: str=QDRANT_COLLECTION) -> None:
    vector_dim = len(embeddings[0])
    if client.collection_exists(collection_name):
        logger.info("Dropping existing collection '%s' for clean re-ingest.", collection_name)
        client.delete_collection(collection_name)
    logger.info("Creating collection '%s' with vector_size=%d, distance=COSINE.", collection_name, vector_dim)
    client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE))
    points = [PointStruct(id=idx, vector=embedding, payload={'chunk_id': chunk['chunk_id'], 'article_id': chunk['article_id'], 'category': chunk['category'], 'title': chunk['title'], 'tags': chunk['tags'], 'text': chunk['text']}) for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
    batch_size = 256
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        logger.info('Upserted points %d–%d', i, i + len(batch) - 1)
    logger.info("Qdrant upsert complete. Collection '%s' now holds %d points.", collection_name, len(points))

def build_bm25_index(chunks: list[dict[str, Any]]) -> BM25Okapi:
    corpus_tokens = [chunk['text'].lower().split() for chunk in chunks]
    logger.info('Building BM25Okapi index over %d documents …', len(corpus_tokens))
    bm25 = BM25Okapi(corpus_tokens)
    logger.info('BM25 index built. Vocabulary size: %d', len(bm25.idf))
    return bm25

def save_bm25_index(bm25: BM25Okapi, chunks: list[dict[str, Any]], path: Path=BM25_INDEX_PATH) -> None:
    payload = {'version': 1, 'bm25': bm25, 'chunks': chunks}
    with path.open('wb') as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('BM25 index pickled to %s (%.1f KB).', path, path.stat().st_size / 1024)

def load_bm25_index(path: Path=BM25_INDEX_PATH) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f'BM25 index not found at {path}. Run ingest.py first.')
    with path.open('rb') as fh:
        payload = pickle.load(fh)
    return (payload['bm25'], payload['chunks'])

def run_ingest_pipeline() -> dict[str, Any]:
    articles = load_articles()
    chunks = chunk_articles(articles)
    model = load_embedding_model()
    embedding_dim: int = model.get_embedding_dimension()
    embeddings = embed_chunks(chunks, model)
    client = get_qdrant_client()
    upsert_to_qdrant(client, chunks, embeddings)
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25, chunks)
    return {'total_articles': len(articles), 'total_chunks': len(chunks), 'embedding_dim': embedding_dim, 'bm25_vocab_size': len(bm25.idf), 'qdrant_collection': QDRANT_COLLECTION}
if __name__ == '__main__':
    summary = run_ingest_pipeline()
    print('\n' + '=' * 60)
    print('  QuickCrate KB Ingest — Summary')
    print('=' * 60)
    print(f"  Total articles   : {summary['total_articles']}")
    print(f"  Total chunks     : {summary['total_chunks']}")
    print(f"  Embedding dim    : {summary['embedding_dim']}")
    print(f"  BM25 vocab size  : {summary['bm25_vocab_size']}")
    print(f"  Qdrant collection: {summary['qdrant_collection']}")
    print('=' * 60)
    print('\nIngest complete. Run retrieval.py to test queries.\n')