from __future__ import annotations
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')
import ingest
from qdrant_client import QdrantClient
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('migrate_qdrant')

def main() -> None:
    qdrant_url = os.environ.get('QDRANT_URL')
    qdrant_api_key = os.environ.get('QDRANT_API_KEY')
    collection_name = os.environ.get('QDRANT_COLLECTION', ingest.QDRANT_COLLECTION)
    if not qdrant_url or not qdrant_api_key:
        logger.error('Missing QDRANT_URL or QDRANT_API_KEY in environment variables.')
        print('\nERROR: Please set QDRANT_URL and QDRANT_API_KEY environment variables.')
        print('You can set them in your .env file or run:')
        print("  $env:QDRANT_URL = 'https://your-cluster-url.aws.qdrant.io'")
        print("  $env:QDRANT_API_KEY = 'your-api-key'")
        return
    logger.info('Starting migration to Qdrant Cloud...')
    logger.info('Target Cluster: %s', qdrant_url)
    logger.info('Target Collection: %s', collection_name)
    logger.info('Step 1: Loading articles from disk...')
    articles = ingest.load_articles()
    logger.info('Step 2: Chunking articles...')
    chunks = ingest.chunk_articles(articles)
    logger.info('Step 3: Loading embedding model and embedding chunks (this might take a while on CPU)...')
    model = ingest.load_embedding_model()
    embeddings = ingest.embed_chunks(chunks, model)
    logger.info('Step 4: Connecting to Qdrant Cloud cluster...')
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    logger.info('Step 5: Recreating collection and upserting points...')
    ingest.upsert_to_qdrant(client, chunks, embeddings, collection_name=collection_name)
    logger.info('Migration to Qdrant Cloud completed successfully!')
if __name__ == '__main__':
    main()