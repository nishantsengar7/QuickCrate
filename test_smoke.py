import sys
sys.path.insert(0, '.')
from ingest import load_articles, chunk_articles
articles = load_articles()
chunks = chunk_articles(articles)
long_articles = [a for a in articles if len(a['body'].split()) > 400]
print(f'Articles loaded      : {len(articles)}')
print(f'Chunks produced      : {len(chunks)}')
print(f'Long articles (>400w): {len(long_articles)}')
for a in long_articles:
    print(f"  - {a['id']} ({len(a['body'].split())} words)")
c = chunks[0]
print(f'\nSample chunk[0]:')
print(f"  chunk_id  : {c['chunk_id']}")
print(f"  article_id: {c['article_id']}")
print(f"  category  : {c['category']}")
print(f"  title     : {c['title']}")
print(f"  tags      : {c['tags']}")
print(f"  text[:120]: {c['text'][:120]}...")
print('\nAll checks passed.')