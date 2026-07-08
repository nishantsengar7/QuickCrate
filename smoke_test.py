from __future__ import annotations
import argparse
import sys
import io
import httpx
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass
TEST_CASES = [{'id': 'TC-1-IN-SCOPE', 'description': 'Clearly in-scope query (payment methods)', 'query': 'What payment methods can I use on QuickCrate?', 'expected_escalated': False, 'expected_source_keywords': ['payment', 'methods', 'pay']}, {'id': 'TC-2-OUT-OF-SCOPE', 'description': 'Clearly out-of-scope query (franchise opportunities)', 'query': 'Can I open a QuickCrate franchise store in my city?', 'expected_escalated': True, 'expected_source_keywords': []}]

def test_single_turns(api_url: str) -> bool:
    print('\n--- Running Single-Turn API Tests ---')
    chat_url = f'{api_url}/chat'
    all_passed = True
    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['description']}")
        print(f"  Query: '{tc['query']}'")
        payload = {'query': tc['query']}
        try:
            resp = httpx.post(chat_url, json=payload, timeout=90.0)
            if resp.status_code != 200:
                print(f'  ❌ Failed: Server returned status code {resp.status_code}')
                print(f'     Body: {resp.text}')
                all_passed = False
                continue
            data = resp.json()
            answer = data.get('answer', '')
            escalated = data.get('escalated', False)
            sources = data.get('sources', [])
            session_id = data.get('session_id', '')
            print(f'  Response: {answer[:120]}...')
            print(f"  Escalated: {escalated} (Expected: {tc['expected_escalated']})")
            print(f'  Sources  : {sources}')
            print(f'  Session  : {session_id}')
            if escalated != tc['expected_escalated']:
                print(f"  ❌ FAILED: escalated flag mismatch (got {escalated}, expected {tc['expected_escalated']})")
                all_passed = False
                continue
            if not escalated:
                if not sources:
                    print('  ❌ FAILED: In-scope query returned empty sources.')
                    all_passed = False
                    continue
                kw_found = any((any((kw in src.get('title', '').lower() for kw in tc['expected_source_keywords'])) for src in sources))
                if not kw_found and tc['expected_source_keywords']:
                    print(f"  ⚠️ Warning: Sources returned {sources} but did not contain expected keywords {tc['expected_source_keywords']}.")
            elif sources:
                print('  ❌ FAILED: Escalated query returned sources.')
                all_passed = False
                continue
            print('  ✅ TC Passed.')
        except Exception as exc:
            print(f'  ❌ Error contacting API: {exc}')
            all_passed = False
    return all_passed

def test_two_turn_chat(api_url: str) -> bool:
    print('\n--- Running Multi-Turn Chat (Query Rewriting) Test ---')
    chat_url = f'{api_url}/chat'
    q1 = 'What are the benefits of QuickCrate Plus?'
    print(f"\nTurn 1 Query: '{q1}'")
    try:
        r1 = httpx.post(chat_url, json={'query': q1}, timeout=90.0)
        r1.raise_for_status()
        data1 = r1.json()
        session_id = data1.get('session_id')
        print(f'  Session ID assigned: {session_id}')
        print(f"  Escalated: {data1.get('escalated')}")
        q2 = 'What about COD? Is that available too?'
        print(f"\nTurn 2 Query: '{q2}' (using Session ID: {session_id})")
        r2 = httpx.post(chat_url, json={'query': q2, 'session_id': session_id}, timeout=90.0)
        r2.raise_for_status()
        data2 = r2.json()
        print(f"  Response: {data2.get('answer')[:120]}...")
        print(f"  Escalated: {data2.get('escalated')} (Expected: False)")
        print(f"  Sources: {data2.get('sources')}")
        if data2.get('escalated', True):
            print('  ❌ FAILED: Turn 2 was escalated (rewriter likely failed to resolve COD in history).')
            return False
        print('  ✅ Multi-turn query rewriting verified successfully.')
        return True
    except Exception as exc:
        print(f'  ❌ Multi-turn test failed due to request error: {exc}')
        return False

def main() -> None:
    parser = argparse.ArgumentParser(description='QuickCrate RAG API Smoke Tester')
    parser.add_argument('api_url', nargs='?', default='http://localhost:8001', help='FastAPI root URL (default: http://localhost:8001)')
    args = parser.parse_args()
    api_url = args.api_url.rstrip('/')
    print('==================================================')
    print(f'QuickCrate RAG Live Smoke Test')
    print(f'Target URL: {api_url}')
    print('==================================================')
    try:
        health_resp = httpx.get(f'{api_url}/health', timeout=10.0)
        print(f'\n[Health Check] Status code: {health_resp.status_code}')
        hdata = health_resp.json()
        print(f"  Overall Status: {hdata.get('status')}")
        print(f"  Qdrant connection: {hdata.get('qdrant')}")
        print(f"  Models loaded: {hdata.get('models_loaded')}")
        if hdata.get('status') != 'ok':
            print('❌ Health check returned degraded status. Aborting.')
            sys.exit(1)
    except Exception as exc:
        print(f'❌ Failed to reach health endpoint: {exc}. Aborting.')
        sys.exit(1)
    st_ok = test_single_turns(api_url)
    mt_ok = test_two_turn_chat(api_url)
    print('\n' + '=' * 50)
    if st_ok and mt_ok:
        print('🎉 SUCCESS: All smoke tests passed successfully!')
        sys.exit(0)
    else:
        print('❌ FAILURE: One or more test suites failed.')
        sys.exit(1)
if __name__ == '__main__':
    main()