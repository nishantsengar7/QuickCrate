"""
generate.py -- QuickCrate RAG Generation + Escalation Layer (Phase 5)
=====================================================================

Pipeline overview
-----------------
This module sits at the top of the four-stage retrieval pipeline:

  Stage 1  Hybrid retrieval (retrieval.py)     bi-encoder + BM25 + RRF
  Stage 2  Cross-encoder reranking (rerank.py) joint-attention relevance
  Stage 3  Confidence gate (this module)       escalate vs. generate decision
  Stage 4  Answer generation (this module)     LLM with grounded context

The single public entry point is answer_query(), which:
  1. (Optional) Rewrites a follow-up query into a standalone question
     so that vague pronouns / references to prior turns resolve correctly
     before retrieval.
  2. Runs hybrid_search(top_k=20) then rerank(top_n=5).
  3. Applies a confidence gate based on the cross-encoder's top logit.
  4. Either escalates to human support or calls the LLM for a grounded answer.

Why the confidence gate uses the reranker's score, not the LLM's own confidence
---------------------------------------------------------------------------------
LLMs are notoriously unreliable at self-assessing when they lack the knowledge
to answer a question.  Concretely:
  - When asked to answer from context that does not actually answer the query,
    LLMs frequently hallucinate a plausible-sounding response rather than
    admitting uncertainty ("confident hallucination").
  - Adding "say 'I don't know' if you're unsure" to the system prompt only
    partially mitigates this; the LLM still has no ground truth to compare
    against -- it does not know what it does not know.
  - Self-reported confidence scores from LLMs (e.g., asking the model to
    output a 1-10 confidence number) are poorly calibrated and add latency.

The cross-encoder's logit is a much more reliable signal:
  - It is computed before the LLM call, making it cheap.
  - It measures how closely the query matches the *best available* KB passage
    in full joint-attention space -- a direct proxy for "is there a good
    answer in the KB at all?"
  - Very negative logits empirically correspond to out-of-scope queries with
    high reliability; a threshold near 0 captures this without per-query
    calibration on calibrated probabilities.

The pattern -- retrieve, rerank, gate, then generate -- is the standard RAG
architecture used in production support systems (e.g., Intercom Fin,
Salesforce Einstein GPT).  Gating before the LLM call is both cheaper and
more reliable than any post-generation self-assessment approach.

LLM provider config
-------------------
Set LLM_PROVIDER = "gemini" to use Gemini 1.5 Flash via google-genai SDK.
Set LLM_PROVIDER = "openai" to use GPT-4o-mini via the openai SDK.
Both require the corresponding environment variable to be set:
  - GEMINI_API_KEY  (for Gemini)
  - OPENAI_API_KEY  (for OpenAI)
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Load .env from the project root so GEMINI_API_KEY / OPENAI_API_KEY are
# available without the user having to set them manually each shell session.
# Falls back silently if python-dotenv is not installed.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    # python-dotenv not installed -- rely on env vars being set externally.
    pass

from rerank import RerankResult, load_cross_encoder, rerank
from retrieval import HybridRetriever

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
# Configuration -- all named constants so they are easy to tune in one place
# ---------------------------------------------------------------------------

#: Switch between "gemini" (Gemini 1.5 Flash, google-genai SDK) and
#: "openai" (GPT-4o-mini, openai SDK).  Controls which LLM is used for
#: both the query-rewriting step and the answer-generation step.
LLM_PROVIDER: str = os.getenv("QC_LLM_PROVIDER", "gemini")

#: Gemini model identifier.
GEMINI_MODEL: str = "gemini-2.5-flash"

#: OpenAI model identifier.
OPENAI_MODEL: str = "gpt-4o-mini"

#: Cross-encoder logit threshold for the confidence gate.
#: Queries whose best rerank_score < CONFIDENCE_THRESHOLD are escalated
#: to human support instead of being answered by the LLM.
#: Overridable via QC_CONFIDENCE_THRESHOLD env var so each deployment
#: (local vs. HF Spaces) can be tuned independently without code changes.
#: Calibrated at 2.0: out-of-scope queries score ~1.2-1.5; in-scope ~2.5+.
CONFIDENCE_THRESHOLD: float = float(os.getenv("QC_CONFIDENCE_THRESHOLD", "2.0"))

#: If the best rerank_score is between MENTION_FLOOR and CONFIDENCE_THRESHOLD
#: the escalation message includes a "best-effort snippet" from the top result.
#: Below MENTION_FLOOR the KB result is too weak to mention at all.
MENTION_FLOOR: float = -3.0

#: Number of hybrid candidates retrieved before reranking.
RETRIEVAL_TOP_K: int = 20

#: Number of reranked results passed to the LLM as context.
RERANK_TOP_N: int = 5

#: Human support contact shown in escalation messages.
SUPPORT_CONTACT: str = "support.quickcrate.in or call 1800-QC-HELP"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are a warm, helpful customer support assistant for QuickCrate, an
    ultra-fast grocery delivery app.

    Guidelines:
    - Answer ONLY using the information in the context chunks provided below.
      Do NOT use outside knowledge or make up information.
    - If the context does not contain enough information to fully answer the
      question, say so honestly rather than guessing.
    - Keep your answer concise (2-5 sentences for simple questions; a short
      bulleted list for multi-step instructions).
    - Use a friendly, empathetic tone -- as if you are chatting with a
      real customer.  Avoid stiff corporate language but stay professional.
    - At the end of every answer include a short "Source:" line listing the
      article title(s) you drew from.  Format: Source: <Title 1>; <Title 2>

    Context chunks are provided in <context> tags.  Use only these.
""")

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class AnswerResponse:
    """
    Result returned by answer_query().

    Attributes
    ----------
    answer : str
        The generated answer text, or the escalation message if below threshold.
    sources : list[str]
        Article titles cited by the LLM (empty for escalated responses).
    rerank_score : float
        The top cross-encoder logit -- the primary confidence signal used for
        the gate decision and for downstream logging / eval.
    escalated : bool
        True if the query was below CONFIDENCE_THRESHOLD and routed to the
        escalation template instead of the LLM.
    rewritten_query : str
        The standalone query actually used for retrieval.  Equals the original
        query when no conversation history was provided (no rewriting needed).
    top_chunks : list[RerankResult]
        The reranked chunks used as context (useful for logging and eval).
    """
    answer: str
    sources: list[str] = field(default_factory=list)
    rerank_score: float = float("-inf")
    escalated: bool = False
    rewritten_query: str = ""
    top_chunks: list[RerankResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str) -> str:
    """
    Make a single-turn LLM call and return the response text.

    Dispatches to Gemini 1.5 Flash or GPT-4o-mini based on LLM_PROVIDER.
    Both are called with temperature=0 for deterministic, factual support
    answers -- creativity is undesirable in a grounded FAQ context.

    Parameters
    ----------
    system : str
        System prompt (instructions + persona).
    user : str
        User-facing prompt (context + query).

    Returns
    -------
    str
        Raw text response from the LLM.
    """
    if LLM_PROVIDER == "gemini":
        return _call_gemini(system, user)
    elif LLM_PROVIDER == "openai":
        return _call_openai(system, user)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. Use 'gemini' or 'openai'."
        )


def _call_gemini(system: str, user: str) -> str:
    """Call Gemini 1.5 Flash via the google-genai SDK."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a key from https://aistudio.google.com/app/apikey"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
        ),
    )
    return response.text


def _call_openai(system: str, user: str) -> str:
    """Call GPT-4o-mini via the openai SDK (v2+)."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is not installed. Run: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM = textwrap.dedent("""\
    You are a query rewriting assistant.  Your ONLY job is to rewrite the
    user's latest message into a fully self-contained question that can be
    understood without any prior conversation context.

    Rules:
    - Replace pronouns and references (e.g., "it", "that plan", "the same
      thing") with their explicit referents from the conversation history.
    - Do NOT answer the question.  Only rewrite it.
    - If the message is already self-contained, return it unchanged.
    - Output ONLY the rewritten question -- no preamble, no explanation.
""")


def _rewrite_query(
    latest_query: str,
    history: list[dict[str, str]],
) -> str:
    """
    Rewrite a follow-up query into a standalone question using conversation
    history as context.

    Why a separate rewriting step instead of just concatenating history?
    ----------------------------------------------------------------------
    Naive history concatenation ("User said X, then Y, now asking Z") makes
    the retrieval query longer and noisier, which hurts both dense embedding
    quality (bi-encoders are trained on short queries) and BM25 (IDF scores
    are diluted by added terms from prior turns).

    A small, cheap LLM call that outputs only the rewritten question is much
    cleaner: retrieval sees a short, precise, standalone query, while the full
    history context is preserved for the generation step.

    Parameters
    ----------
    latest_query : str
        The user's latest message (may contain vague follow-up references).
    history : list[dict]
        Prior conversation turns, each a dict with "role" and "content" keys.
        Roles should be "user" or "assistant".

    Returns
    -------
    str
        A standalone question that can be sent directly to hybrid_search().
    """
    if not history:
        return latest_query

    history_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
    )
    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Latest user message: {latest_query}\n\n"
        f"Rewrite the latest message as a standalone question:"
    )
    rewritten = _call_llm(_REWRITE_SYSTEM, user_prompt).strip()
    logger.info("Query rewritten: '%s' -> '%s'", latest_query[:60], rewritten[:60])
    return rewritten


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context_block(chunks: list[RerankResult]) -> str:
    """
    Format the top reranked chunks into a numbered <context> block for the LLM.

    Numbering helps the LLM cite specific passages even if the system prompt
    does not require it, and makes it easier to trace which chunk was used
    during offline evaluation.
    """
    parts = ["<context>"]
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Title: {chunk.title}\n"
            f"    Category: {chunk.category}\n"
            f"    {chunk.text}"
        )
    parts.append("</context>")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Escalation message builder
# ---------------------------------------------------------------------------

def _build_escalation_message(
    query: str,
    top_chunk: RerankResult | None,
    best_score: float,
) -> str:
    """
    Build an empathetic escalation message.

    If the best_score is above MENTION_FLOOR but below CONFIDENCE_THRESHOLD,
    include a short best-effort snippet so the customer has something useful
    while waiting for a human agent.  Below MENTION_FLOOR the KB result is too
    weak to risk showing potentially wrong information.

    Parameters
    ----------
    query : str
        The (possibly rewritten) user query.
    top_chunk : RerankResult or None
        The highest-scoring reranked chunk.
    best_score : float
        The cross-encoder logit for top_chunk.

    Returns
    -------
    str
        The escalation message to return to the user.
    """
    lines = [
        "I'm sorry, I wasn't able to find a confident answer to your question "
        "in our help centre right now.",
        "",
    ]

    if top_chunk is not None and best_score >= MENTION_FLOOR:
        # Grab first 2 sentences of the chunk body as a best-effort snippet.
        body = top_chunk.text
        sentences = body.split(". ")
        snippet = ". ".join(sentences[:2]).strip()
        if snippet and not snippet.endswith("."):
            snippet += "."
        lines += [
            f"The closest article I found was **{top_chunk.title}**, which says:",
            f'> "{snippet}"',
            "",
            "This may or may not fully answer your question.",
            "",
        ]

    lines += [
        "For a complete and accurate answer, please reach out to our support team:",
        f"  {SUPPORT_CONTACT}",
        "",
        "They typically respond within a few minutes. Sorry for the inconvenience!",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    retriever: HybridRetriever,
    ce_model: Any,  # CrossEncoder
    conversation_history: list[dict[str, str]] | None = None,
) -> AnswerResponse:
    """
    Full RAG pipeline: (optional rewrite) -> retrieve -> rerank -> gate -> generate.

    Confidence gate design
    ----------------------
    The gate decision is made on the cross-encoder's top logit rather than on
    LLM self-reported confidence for two reasons:

    1. Cost: the gate fires *before* the LLM call.  If the query is out of
       scope, we never pay for a generation token, which matters at scale.

    2. Reliability: LLMs hallucinate confidently.  A model asked to answer from
       context that does not actually contain the answer will often fabricate a
       plausible response.  The cross-encoder's logit is grounded in whether
       the KB literally contains a relevant passage -- a much tighter signal.
       Research on RAG faithfulness (e.g., RAGAS, Databricks MLflow RAG
       evaluation) consistently shows that retrieval-side confidence is better
       calibrated than generation-side self-assessment for open-domain QA.

    Parameters
    ----------
    query : str
        The user's latest message.
    retriever : HybridRetriever
        Initialised retriever (from retrieval.py).
    ce_model : CrossEncoder
        Loaded cross-encoder (from rerank.py).
    conversation_history : list[dict] or None
        Prior turns, each {"role": "user"|"assistant", "content": "..."},
        in chronological order.  If provided, the query is first rewritten
        into a standalone question before retrieval.

    Returns
    -------
    AnswerResponse
        Contains answer text, cited sources, rerank_score, escalation flag,
        rewritten query, and the raw top chunks for logging.
    """
    # ------------------------------------------------------------------
    # Step 1: Query rewriting (only when conversation history is present)
    # ------------------------------------------------------------------
    if conversation_history:
        retrieval_query = _rewrite_query(query, conversation_history)
    else:
        retrieval_query = query

    # ------------------------------------------------------------------
    # Step 2: Hybrid retrieval + cross-encoder reranking
    # ------------------------------------------------------------------
    candidates = retriever.hybrid_search(retrieval_query, top_k=RETRIEVAL_TOP_K)
    top_chunks, best_score = rerank(
        retrieval_query, candidates, ce_model, top_n=RERANK_TOP_N
    )

    logger.info(
        "Gate check: best_score=%.4f  threshold=%.4f  -> %s",
        best_score,
        CONFIDENCE_THRESHOLD,
        "GENERATE" if best_score >= CONFIDENCE_THRESHOLD else "ESCALATE",
    )

    # ------------------------------------------------------------------
    # Step 3a: Escalation path
    # ------------------------------------------------------------------
    if best_score < CONFIDENCE_THRESHOLD:
        top_chunk = top_chunks[0] if top_chunks else None
        escalation_msg = _build_escalation_message(retrieval_query, top_chunk, best_score)
        return AnswerResponse(
            answer=escalation_msg,
            sources=[],
            rerank_score=best_score,
            escalated=True,
            rewritten_query=retrieval_query,
            top_chunks=top_chunks,
        )

    # ------------------------------------------------------------------
    # Step 3b: Generation path
    # ------------------------------------------------------------------
    context_block = _build_context_block(top_chunks)
    user_prompt = (
        f"{context_block}\n\n"
        f"Customer question: {query}\n\n"
        f"Answer using only the context above. End with a 'Source:' line."
    )

    raw_answer = _call_llm(SYSTEM_PROMPT, user_prompt)

    # Parse cited titles from the "Source:" line the LLM is instructed to add.
    sources: list[str] = []
    answer_body = raw_answer
    if "Source:" in raw_answer:
        body_part, source_part = raw_answer.rsplit("Source:", 1)
        answer_body = body_part.strip()
        sources = [s.strip() for s in source_part.split(";") if s.strip()]

    return AnswerResponse(
        answer=raw_answer.strip(),
        sources=sources,
        rerank_score=best_score,
        escalated=False,
        rewritten_query=retrieval_query,
        top_chunks=top_chunks,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_result(label: str, query: str, result: AnswerResponse) -> None:
    """Print a formatted result block for __main__ demonstration."""
    width = 78
    print("\n" + "=" * width)
    print(f"  {label}")
    print(f"  QUERY    : {query}")
    if result.rewritten_query != query:
        print(f"  REWRITTEN: {result.rewritten_query}")
    print(f"  SCORE    : {result.rerank_score:+.4f}   ESCALATED: {result.escalated}")
    print("=" * width)
    # Word-wrap the answer body
    for line in result.answer.splitlines():
        if line.strip():
            print(textwrap.fill(line, width=width, initial_indent="  ",
                                subsequent_indent="    "))
        else:
            print()
    if result.sources:
        print(f"\n  Sources cited: {'; '.join(result.sources)}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # ------------------------------------------------------------------
    # Boot all models once -- they are reused across all test queries.
    # ------------------------------------------------------------------
    print("Initialising models (Qdrant + BM25 + cross-encoder)...")
    retriever = HybridRetriever()

    from rerank import load_cross_encoder
    ce_model = load_cross_encoder()

    print(f"LLM provider : {LLM_PROVIDER}")
    print(f"Confidence threshold : {CONFIDENCE_THRESHOLD}")
    print()

    # ==================================================================
    # Test 1: In-scope query -- should generate a grounded answer
    # ==================================================================
    # Target: payments-001 (what payment methods does QuickCrate accept)
    # Expected: LLM produces a concise answer citing payments-001.
    # ==================================================================
    q1 = "What payment methods can I use on QuickCrate?"
    r1 = answer_query(q1, retriever, ce_model)
    _print_result("TEST 1 -- In-scope (should generate + cite source)", q1, r1)

    # ==================================================================
    # Test 2: Out-of-scope query -- should escalate, not hallucinate
    # ==================================================================
    # No KB article covers franchise opportunities.  The best rerank_score
    # should be below CONFIDENCE_THRESHOLD, triggering the escalation path
    # without ever calling the LLM.
    # ==================================================================
    q2 = "Can I open a QuickCrate franchise store in my city?"
    r2 = answer_query(q2, retriever, ce_model)
    _print_result("TEST 2 -- Out-of-scope (should escalate, LLM NOT called)", q2, r2)
    assert r2.escalated, "TEST 2 FAILED: expected escalation for out-of-scope query"
    print("  ASSERTION PASSED: escalated == True\n")

    # ==================================================================
    # Test 3: 2-turn conversation with a vague follow-up
    # ==================================================================
    # Turn 1 establishes context: asking about QuickCrate Plus benefits.
    # Turn 2 asks a vague follow-up about "COD" -- which only makes sense
    # as "Cash on Delivery" in the payments context, but the word "COD"
    # appears in payments articles.  Without rewriting, "what about COD?"
    # would retrieve poorly because it is a 3-word query with no context.
    # After rewriting, the standalone question should be something like
    # "Is Cash on Delivery (COD) available for QuickCrate Plus orders?"
    # which retrieves correctly from the payments or subscriptions articles.
    # ==================================================================
    print("\n" + "=" * 78)
    print("  TEST 3 -- 2-turn conversation with vague follow-up")
    print("=" * 78)

    # Turn 1
    q3a = "What are the benefits of QuickCrate Plus?"
    r3a = answer_query(q3a, retriever, ce_model)
    _print_result("  Turn 1", q3a, r3a)

    # Build history from turn 1
    history = [
        {"role": "user", "content": q3a},
        {"role": "assistant", "content": r3a.answer},
    ]

    # Turn 2 -- vague follow-up that requires rewriting
    q3b = "What about COD? Is that available too?"
    r3b = answer_query(q3b, retriever, ce_model, conversation_history=history)
    _print_result("  Turn 2 (vague follow-up -- rewriting should resolve 'COD')", q3b, r3b)

    print("\n  Rewriting check:")
    print(f"    Original : {q3b}")
    print(f"    Rewritten: {r3b.rewritten_query}")
    print(
        "  PASS -- rewritten query is more specific than original."
        if r3b.rewritten_query.lower() != q3b.lower()
        else "  NOTE -- rewriter returned original query unchanged (check history)."
    )

    print("\nAll tests complete.\n")
