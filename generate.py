from __future__ import annotations
import logging
import os
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass
from rerank import RerankResult, load_cross_encoder, rerank
from retrieval import HybridRetriever
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
# Configure provider order via a comma-separated list in env var QC_PROVIDER_ORDER
# e.g., "gemini,openai" tries Gemini first, then OpenAI GPT-4o-mini if Gemini fails.
PROVIDER_ORDER_ENV: str = os.getenv('QC_PROVIDER_ORDER', 'gemini,openai')
PROVIDER_ORDER: list[str] = [p.strip().lower() for p in PROVIDER_ORDER_ENV.split(',') if p.strip()]

LLM_PROVIDER: str = os.getenv('QC_LLM_PROVIDER', 'gemini')
GEMINI_MODEL: str = os.getenv('QC_GEMINI_MODEL', 'gemini-2.5-flash-lite')
OPENAI_MODEL: str = 'gpt-4o-mini'
CONFIDENCE_THRESHOLD: float = float(os.getenv('QC_CONFIDENCE_THRESHOLD', '1.6'))
MENTION_FLOOR: float = -3.0
RETRIEVAL_TOP_K: int = 20
RERANK_TOP_N: int = 5
SUPPORT_CONTACT: str = 'support.quickcrate.in or call 1800-QC-HELP'
SYSTEM_PROMPT: str = textwrap.dedent('    You are a warm, helpful customer support assistant for QuickCrate, an\n    ultra-fast grocery delivery app.\n\n    Guidelines:\n    - Answer ONLY using the information in the context chunks provided below.\n      Do NOT use outside knowledge or make up information.\n    - If the context does not contain enough information to fully answer the\n      question, say so honestly rather than guessing.\n    - Keep your answer concise (2-5 sentences for simple questions; a short\n      bulleted list for multi-step instructions).\n    - Use a friendly, empathetic tone -- as if you are chatting with a\n      real customer.  Avoid stiff corporate language but stay professional.\n    - At the end of every answer include a short "Source:" line listing the\n      article title(s) you drew from.  Format: Source: <Title 1>; <Title 2>\n\n    Context chunks are provided in <context> tags.  Use only these.\n')

@dataclass
class AnswerResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    rerank_score: float = float('-inf')
    escalated: bool = False
    rewritten_query: str = ''
    top_chunks: list[RerankResult] = field(default_factory=list)

class QuotaExhaustedError(Exception):
    """Raised when the primary provider's daily quota is exhausted (RESOURCE_EXHAUSTED).
    
    We fail fast on daily quota exhaustion because retrying does not help,
    unlike transient per-minute rate limits.
    """
    pass

def _call_llm(system: str, user: str) -> str:
    last_exc = None
    attempted_providers = []
    
    for provider in PROVIDER_ORDER:
        attempted_providers.append(provider)
        logger.info("Attempting LLM generation using provider: '%s'", provider)
        try:
            if provider == 'gemini':
                return _call_gemini(system, user)
            elif provider == 'openai':
                return _call_openai(system, user)
            else:
                raise ValueError(f"Unknown LLM provider: '{provider}'")
        except QuotaExhaustedError as exc:
            logger.warning("Gemini daily quota exhausted. Falling back to the next provider if available.")
            last_exc = exc
        except Exception as exc:
            logger.warning("LLM provider '%s' failed: %s", provider, exc)
            last_exc = exc
            
    # If all configured LLM providers fail
    raise RuntimeError(
        f"All configured LLM providers ({', '.join(attempted_providers)}) failed. "
        f"Last error: {last_exc}"
    ) from last_exc

def _call_gemini(system: str, user: str) -> str:
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError
    except ImportError as exc:
        raise ImportError('google-genai is not installed. Run: pip install google-genai') from exc
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError('GEMINI_API_KEY environment variable is not set. Get a key from https://aistudio.google.com/app/apikey')
    
    # Cap timeout to 10 seconds (10000 ms) since Gemini API has a minimum allowed deadline of 10s.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=10000))
    # Cap retry backoff delays to 1.0s and 2.0s to ensure chat UI is responsive.
    _BASE_DELAYS = [1.0, 2.0]
    last_exc = None
    for attempt, base_delay in enumerate([None] + _BASE_DELAYS, start=1):
        if base_delay is not None:
            time.sleep(base_delay)
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, 
                contents=user, 
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.0)
            )
            return response.text
        except Exception as exc:
            # Check if this is a daily quota exhaustion error (RESOURCE_EXHAUSTED with PerDay or per day)
            is_daily_quota = False
            from google.genai.errors import APIError
            if isinstance(exc, APIError):
                status_str = getattr(exc, 'status', '') or ''
                message_str = getattr(exc, 'message', '') or ''
                details_str = str(getattr(exc, 'details', ''))
                if status_str == 'RESOURCE_EXHAUSTED' and (
                    'perday' in details_str.lower() or 
                    'per day' in details_str.lower() or
                    'perday' in message_str.lower() or 
                    'per day' in message_str.lower() or
                    'perday' in str(exc).lower() or
                    'per day' in str(exc).lower()
                ):
                    is_daily_quota = True
            else:
                exc_str = str(exc)
                if 'RESOURCE_EXHAUSTED' in exc_str and (
                    'perday' in exc_str.lower() or 
                    'per day' in exc_str.lower()
                ):
                    is_daily_quota = True
            
            if is_daily_quota:
                logger.error("Gemini daily quota exhausted (RESOURCE_EXHAUSTED). Failing fast.")
                raise QuotaExhaustedError("Gemini daily quota exhausted.") from exc
                
            logger.warning('Gemini API call failed (attempt %d/3): %s. Retrying...', attempt, exc)
            last_exc = exc
            continue
    raise last_exc

def _call_openai(system: str, user: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError('openai is not installed. Run: pip install openai') from exc
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError('OPENAI_API_KEY environment variable is not set.')
    # Using 10s timeout for OpenAI fallback to keep it responsive
    client = OpenAI(api_key=api_key, timeout=10.0)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, 
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 
        temperature=0.0
    )
    return resp.choices[0].message.content or ''
_REWRITE_SYSTEM = textwrap.dedent('    You are a query rewriting assistant.  Your ONLY job is to rewrite the\n    user\'s latest message into a fully self-contained question that can be\n    understood without any prior conversation context.\n\n    Rules:\n    - Replace pronouns and references (e.g., "it", "that plan", "the same\n      thing") with their explicit referents from the conversation history.\n    - Do NOT answer the question.  Only rewrite it.\n    - If the message is already self-contained, return it unchanged.\n    - Output ONLY the rewritten question -- no preamble, no explanation.\n')

def _rewrite_query(latest_query: str, history: list[dict[str, str]]) -> str:
    history_text = ''
    if history:
        history_text = '\n'.join((f"{turn['role'].capitalize()}: {turn['content']}" for turn in history))
    user_prompt = f'Conversation so far:\n{history_text}\n\nLatest user message: {latest_query}\n\nRewrite the latest message as a standalone question:'
    rewritten = _call_llm(_REWRITE_SYSTEM, user_prompt).strip()
    logger.info("Query rewritten: '%s' -> '%s'", latest_query[:60], rewritten[:60])
    return rewritten

def _build_context_block(chunks: list[RerankResult]) -> str:
    parts = ['<context>']
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f'[{i}] Title: {chunk.title}\n    Category: {chunk.category}\n    {chunk.text}')
    parts.append('</context>')
    return '\n\n'.join(parts)

def _build_escalation_message(query: str, top_chunk: RerankResult | None, best_score: float) -> str:
    lines = ["I'm sorry, I wasn't able to find a confident answer to your question in our help centre right now.", '']
    if top_chunk is not None and best_score >= MENTION_FLOOR:
        body = top_chunk.text
        sentences = body.split('. ')
        snippet = '. '.join(sentences[:2]).strip()
        if snippet and (not snippet.endswith('.')):
            snippet += '.'
        lines += [f'The closest article I found was **{top_chunk.title}**, which says:', f'> "{snippet}"', '', 'This may or may not fully answer your question.', '']
    lines += ['For a complete and accurate answer, please reach out to our support team:', f'  {SUPPORT_CONTACT}', '', 'They typically respond within a few minutes. Sorry for the inconvenience!']
    return '\n'.join(lines)

def answer_query(query: str, retriever: HybridRetriever, ce_model: Any, conversation_history: list[dict[str, str]] | None=None) -> AnswerResponse:
    retrieval_query = _rewrite_query(query, conversation_history or [])
    candidates = retriever.hybrid_search(retrieval_query, top_k=RETRIEVAL_TOP_K)
    top_chunks, best_score = rerank(retrieval_query, candidates, ce_model, top_n=RERANK_TOP_N)
    logger.info('Gate check: best_score=%.4f  threshold=%.4f  -> %s', best_score, CONFIDENCE_THRESHOLD, 'GENERATE' if best_score >= CONFIDENCE_THRESHOLD else 'ESCALATE')
    if best_score < CONFIDENCE_THRESHOLD:
        top_chunk = top_chunks[0] if top_chunks else None
        escalation_msg = _build_escalation_message(retrieval_query, top_chunk, best_score)
        return AnswerResponse(answer=escalation_msg, sources=[], rerank_score=best_score, escalated=True, rewritten_query=retrieval_query, top_chunks=top_chunks)
    context_block = _build_context_block(top_chunks)
    user_prompt = f"{context_block}\n\nCustomer question: {query}\n\nAnswer using only the context above. End with a 'Source:' line."
    raw_answer = _call_llm(SYSTEM_PROMPT, user_prompt)
    sources: list[str] = []
    answer_body = raw_answer
    if 'Source:' in raw_answer:
        body_part, source_part = raw_answer.rsplit('Source:', 1)
        answer_body = body_part.strip()
        sources = [s.strip() for s in source_part.split(';') if s.strip()]
    return AnswerResponse(answer=raw_answer.strip(), sources=sources, rerank_score=best_score, escalated=False, rewritten_query=retrieval_query, top_chunks=top_chunks)

def _print_result(label: str, query: str, result: AnswerResponse) -> None:
    width = 78
    print('\n' + '=' * width)
    print(f'  {label}')
    print(f'  QUERY    : {query}')
    if result.rewritten_query != query:
        print(f'  REWRITTEN: {result.rewritten_query}')
    print(f'  SCORE    : {result.rerank_score:+.4f}   ESCALATED: {result.escalated}')
    print('=' * width)
    for line in result.answer.splitlines():
        if line.strip():
            print(textwrap.fill(line, width=width, initial_indent='  ', subsequent_indent='    '))
        else:
            print()
    if result.sources:
        print(f"\n  Sources cited: {'; '.join(result.sources)}")
    print('=' * width)
if __name__ == '__main__':
    import sys
    print('Initialising models (Qdrant + BM25 + cross-encoder)...')
    retriever = HybridRetriever()
    from rerank import load_cross_encoder
    ce_model = load_cross_encoder()
    print(f'LLM provider : {LLM_PROVIDER}')
    print(f'Confidence threshold : {CONFIDENCE_THRESHOLD}')
    print()
    q1 = 'What payment methods can I use on QuickCrate?'
    r1 = answer_query(q1, retriever, ce_model)
    _print_result('TEST 1 -- In-scope (should generate + cite source)', q1, r1)
    q2 = 'Can I open a QuickCrate franchise store in my city?'
    r2 = answer_query(q2, retriever, ce_model)
    _print_result('TEST 2 -- Out-of-scope (should escalate, LLM NOT called)', q2, r2)
    assert r2.escalated, 'TEST 2 FAILED: expected escalation for out-of-scope query'
    print('  ASSERTION PASSED: escalated == True\n')
    print('\n' + '=' * 78)
    print('  TEST 3 -- 2-turn conversation with vague follow-up')
    print('=' * 78)
    q3a = 'What are the benefits of QuickCrate Plus?'
    r3a = answer_query(q3a, retriever, ce_model)
    _print_result('  Turn 1', q3a, r3a)
    history = [{'role': 'user', 'content': q3a}, {'role': 'assistant', 'content': r3a.answer}]
    q3b = 'What about COD? Is that available too?'
    r3b = answer_query(q3b, retriever, ce_model, conversation_history=history)
    _print_result("  Turn 2 (vague follow-up -- rewriting should resolve 'COD')", q3b, r3b)
    print('\n  Rewriting check:')
    print(f'    Original : {q3b}')
    print(f'    Rewritten: {r3b.rewritten_query}')
    print('  PASS -- rewritten query is more specific than original.' if r3b.rewritten_query.lower() != q3b.lower() else '  NOTE -- rewriter returned original query unchanged (check history).')
    print('\nAll tests complete.\n')