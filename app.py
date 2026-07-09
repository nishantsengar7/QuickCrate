import os
import time
import httpx
import streamlit as st
API_URL: str = os.getenv('QC_API_URL', 'http://localhost:8000')
CHAT_ENDPOINT: str = f'{API_URL}/chat'
HEALTH_ENDPOINT: str = f'{API_URL}/health'
REQUEST_TIMEOUT: float = 90.0
st.set_page_config(page_title='QuickCrate Support', page_icon='🛒', layout='centered', initial_sidebar_state='expanded')
st.markdown('\n    <style>\n    /* Soften the escalation warning box border */\n    .escalation-box {\n        border-left: 4px solid #f59e0b;\n        background: #fffbeb;\n        padding: 0.75rem 1rem;\n        border-radius: 0.375rem;\n        margin-top: 0.5rem;\n    }\n    /* Distinct style for infrastructure/generation errors */\n    .error-box {\n        border-left: 4px solid #ef4444;\n        background: #fef2f2;\n        padding: 0.75rem 1rem;\n        border-radius: 0.375rem;\n        margin-top: 0.5rem;\n    }\n    /* Make source pills look tidy */\n    .source-pill {\n        display: inline-block;\n        background: #eff6ff;\n        border: 1px solid #bfdbfe;\n        border-radius: 999px;\n        padding: 2px 10px;\n        font-size: 0.78rem;\n        color: #1d4ed8;\n        margin: 2px 3px;\n    }\n    </style>\n    ', unsafe_allow_html=True)
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
with st.sidebar:
    st.image('https://img.icons8.com/fluency/96/shopping-cart.png', width=64)
    st.title('QuickCrate Support')
    st.caption('Powered by a multi-stage RAG pipeline')
    st.markdown('---')
    st.subheader('About this project')
    st.markdown('\n        This demo showcases a production-grade **Retrieval-Augmented Generation\n        (RAG)** system built for QuickCrate, a fictional quick-commerce app.\n\n        **Pipeline stages:**\n        1. 🔍 **Hybrid retrieval** — dense vector search (BAAI/bge-large-en-v1.5\n           via Qdrant) fused with BM25 using Reciprocal Rank Fusion (RRF).\n        2. 🎯 **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` rescores\n           the top-20 candidates with full query × document attention.\n        3. 🚦 **Confidence gate** — if the top rerank score is below a tuned\n           threshold, the query is escalated to human support instead of\n           calling the LLM.\n        4. 💬 **Grounded generation** — Gemini 2.5 Flash answers strictly from\n           retrieved KB chunks and cites the source article(s).\n\n        **Tech stack:** FastAPI · Qdrant · sentence-transformers ·\n        google-genai · Streamlit\n        ')
    st.markdown('---')
    if st.button('🔁 Check backend health', use_container_width=True):
        try:
            resp = httpx.get(HEALTH_ENDPOINT, timeout=5.0)
            h = resp.json()
            if h.get('status') == 'ok':
                st.success(f"✅ Backend OK · Qdrant: {h.get('qdrant')}")
            else:
                st.warning(f"⚠️ Backend degraded · Qdrant: {h.get('qdrant')}")
        except Exception as exc:
            st.error(f'❌ Backend unreachable: {exc}')
    if st.session_state.session_id:
        st.caption(f'Session: `{st.session_state.session_id[:8]}…`')
    if st.button('🗑️ Clear conversation', use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()
st.markdown('## 🛒 QuickCrate Customer Support')
st.markdown('Ask me anything about orders, delivery, payments, returns, or your QuickCrate Plus subscription.')
st.markdown('---')

def _render_message(msg: dict) -> None:
    role = msg['role']
    content = msg['content']
    sources = msg.get('sources', [])
    escalated = msg.get('escalated', False)
    error = msg.get('error', False)
    with st.chat_message(role, avatar='🛒' if role == 'assistant' else None):
        if role == 'assistant' and error:
            st.markdown(f'<div class="error-box">\n                🚨 <strong>Service Error</strong><br><br>\n                {content}\n                </div>', unsafe_allow_html=True)
        elif role == 'assistant' and escalated:
            st.markdown(f'<div class="escalation-box">\n                ⚠️ <strong>Escalated to human support</strong><br><br>\n                {content}\n                </div>', unsafe_allow_html=True)
        else:
            st.markdown(content)
        if sources:
            with st.expander(f'📚 Sources ({len(sources)})', expanded=False):
                for src in sources:
                    title = src.get('title', src) if isinstance(src, dict) else src
                    st.markdown(f'<span class="source-pill">📄 {title}</span>', unsafe_allow_html=True)
for msg in st.session_state.messages:
    _render_message(msg)

def _call_api(query: str) -> dict:
    payload: dict = {'query': query}
    if st.session_state.session_id:
        payload['session_id'] = st.session_state.session_id
    resp = httpx.post(CHAT_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()
if (prompt := st.chat_input('Type your question here…')):
    user_msg = {'role': 'user', 'content': prompt, 'sources': [], 'escalated': False}
    st.session_state.messages.append(user_msg)
    _render_message(user_msg)
    with st.chat_message('assistant', avatar='🛒'):
        with st.spinner('Searching the knowledge base…'):
            error_state = False
            try:
                t0 = time.perf_counter()
                data = _call_api(prompt)
                latency = time.perf_counter() - t0
                st.session_state.session_id = data.get('session_id')
                answer = data.get('answer', '')
                sources = data.get('sources', [])
                escalated = data.get('escalated', False)
            except httpx.HTTPStatusError as exc:
                latency = 0.0
                error_state = True
                try:
                    err_json = exc.response.json()
                    err_msg = err_json.get('message', 'Our AI service is temporarily busy, please try again in a moment.')
                except Exception:
                    err_msg = 'Our AI service is temporarily busy, please try again in a moment.'
                answer = err_msg
                sources = []
                escalated = False
            except httpx.ConnectError:
                answer = f'Could not reach the backend API. Make sure it is running at `{API_URL}`.'
                sources = []
                escalated = False
                error_state = True
                latency = 0.0
            except Exception as exc:
                answer = f'Unexpected error: {exc}'
                sources = []
                escalated = False
                error_state = True
                latency = 0.0
    assistant_msg = {
        'role': 'assistant',
        'content': answer,
        'sources': sources,
        'escalated': escalated,
        'error': error_state
    }
    st.session_state.messages.append(assistant_msg)
    if latency > 0:
        st.caption(f'⏱️ Response in {latency:.1f}s')
    st.rerun()