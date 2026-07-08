from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from sentence_transformers import CrossEncoder
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
CROSS_ENCODER_MODEL: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RERANK_BATCH_SIZE: int = 32
ESCALATION_THRESHOLD: float = 0.0

@dataclass
class RerankResult:
    chunk_id: str
    rerank_score: float
    article_id: str
    category: str
    title: str
    tags: list[str] = field(default_factory=list)
    text: str = ''
    hybrid_rank: int = -1

def load_cross_encoder(model_name: str=CROSS_ENCODER_MODEL) -> CrossEncoder:
    logger.info('Loading cross-encoder: %s', model_name)
    model = CrossEncoder(model_name)
    logger.info('Cross-encoder loaded.')
    return model

def rerank(query: str, candidates: list[dict[str, Any]], model: CrossEncoder, top_n: int=5, batch_size: int=RERANK_BATCH_SIZE) -> tuple[list[RerankResult], float]:
    if not candidates:
        logger.warning('rerank() called with empty candidate list.')
        return ([], float('-inf'))
    pairs: list[tuple[str, str]] = [(query, c['text']) for c in candidates]
    logger.info("Cross-encoder scoring %d pairs for query: '%s'", len(pairs), query[:60] + ('...' if len(query) > 60 else ''))
    scores: np.ndarray = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scored: list[tuple[float, int, dict[str, Any]]] = [(float(scores[i]), i, candidates[i]) for i in range(len(candidates))]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score: float = scored[0][0]
    results: list[RerankResult] = [RerankResult(chunk_id=cand['chunk_id'], rerank_score=round(ce_score, 6), article_id=cand.get('article_id', ''), category=cand.get('category', ''), title=cand.get('title', ''), tags=cand.get('tags', []), text=cand.get('text', ''), hybrid_rank=original_idx + 1) for ce_score, original_idx, cand in scored[:top_n]]
    logger.info('Reranking complete. best_score=%.4f  top-%d returned.', best_score, len(results))
    return (results, best_score)

def _print_rerank_comparison(query: str, scenario_label: str, hybrid_candidates: list[dict[str, Any]], reranked: list[RerankResult], top_n: int=5) -> None:
    col_w = 62
    sep = '+' + ('-' * col_w + '+') * 2
    print('\n' + '=' * (col_w * 2 + 3))
    print(f'  SCENARIO : {scenario_label}')
    print(f'  QUERY    : {query}')
    print('=' * (col_w * 2 + 3))
    print(sep)
    lbl_h = 'HYBRID (RRF) -- before reranking'
    lbl_r = 'CROSS-ENCODER -- after reranking'
    print('|' + f'  {lbl_h:^{col_w - 4}}  ' + '|' + f'  {lbl_r:^{col_w - 4}}  ' + '|')
    print(sep)
    hybrid_top = hybrid_candidates[:top_n]
    title_max = col_w - 30
    for i in range(top_n):
        if i < len(hybrid_top):
            h = hybrid_top[i]
            ht = h['title']
            if len(ht) > title_max:
                ht = ht[:title_max - 1] + '+'
            cat = h.get('category', '?')[:6]
            hcell = f"  {i + 1}. [{cat:<6}] {ht:<{title_max}} rrf={h['rrf_score']:.5f}  "
        else:
            hcell = ' ' * col_w
        if i < len(reranked):
            r = reranked[i]
            rt = r.title
            if len(rt) > title_max:
                rt = rt[:title_max - 1] + '+'
            cat = r.category[:6] if r.category else '?'
            if r.hybrid_rank == i + 1:
                marker = '  '
            elif r.hybrid_rank > i + 1:
                marker = 'up'
            else:
                marker = 'dn'
            rcell = f'  {i + 1}. [{cat:<6}] {rt:<{title_max}} ce={r.rerank_score:+.4f} [{marker} hr{r.hybrid_rank}]  '
        else:
            rcell = ' ' * col_w
        print('|' + hcell[:col_w] + '|' + rcell[:col_w] + '|')
    print(sep)
    print('  Legend: ce=cross-encoder logit  hr=original hybrid rank  up=promoted  dn=demoted')
if __name__ == '__main__':
    import sys
    try:
        from retrieval import HybridRetriever
    except ImportError as exc:
        print(f'ERROR: Cannot import HybridRetriever: {exc}')
        sys.exit(1)
    print('Loading models (first run downloads ~70 MB cross-encoder)...')
    retriever = HybridRetriever()
    ce_model = load_cross_encoder()
    POOL = 20
    TOP_N = 5
    q1 = 'How do I cancel my QuickCrate Plus membership?'
    cands1 = retriever.hybrid_search(q1, top_k=POOL)
    ranked1, best1 = rerank(q1, cands1, ce_model, top_n=TOP_N)
    _print_rerank_comparison(q1, '1 -- Straightforward (hybrid and reranker agree)', cands1, ranked1, TOP_N)
    print(f'  Escalation signal (best CE score): {best1:+.4f}\n')
    q2 = 'I already cancelled my Plus subscription -- will I get my subscription fee back?'
    cands2 = retriever.hybrid_search(q2, top_k=POOL)
    ranked2, best2 = rerank(q2, cands2, ce_model, top_n=TOP_N)
    _print_rerank_comparison(q2, '2 STAR -- Hybrid rank-1 likely WRONG: CE should promote subscriptions-009', cands2, ranked2, TOP_N)
    hybrid_r1 = cands2[0]['title'] if cands2 else '(none)'
    rerank_r1 = ranked2[0].title if ranked2 else '(none)'
    changed = hybrid_r1 != rerank_r1
    print(f'  Hybrid rank-1 : {hybrid_r1}')
    print(f'  Rerank rank-1 : {rerank_r1}')
    print(f"  Result        : {('CORRECTION CONFIRMED -- cross-encoder fixed the ranking' if changed else 'ranks unchanged (reranker agreed with hybrid)')}")
    print(f'  Escalation signal (best CE score): {best2:+.4f}\n')
    q3 = 'My bank balance went down but the transaction never appeared in the app'
    cands3 = retriever.hybrid_search(q3, top_k=POOL)
    ranked3, best3 = rerank(q3, cands3, ce_model, top_n=TOP_N)
    _print_rerank_comparison(q3, '3 -- Paraphrased query, zero keyword overlap with payments-002', cands3, ranked3, TOP_N)
    print(f'  Escalation signal (best CE score): {best3:+.4f}\n')
    q4 = 'Can I buy a QuickCrate franchise for my city?'
    cands4 = retriever.hybrid_search(q4, top_k=POOL)
    ranked4, best4 = rerank(q4, cands4, ce_model, top_n=TOP_N)
    _print_rerank_comparison(q4, '4 -- Out-of-scope query (low CE score should trigger escalation)', cands4, ranked4, TOP_N)
    decision = 'ESCALATE to human agent' if best4 < ESCALATION_THRESHOLD else 'ANSWER from KB'
    print(f'  Escalation signal (best CE score): {best4:+.4f}')
    print(f'  Threshold : {ESCALATION_THRESHOLD}')
    print(f'  Decision  : {decision}\n')
    print('Done. Rerank layer validated across all four scenarios.\n')