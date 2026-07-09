#!/bin/bash
# start_services.sh - Startup script for HF Spaces Docker container
# This script ensures Streamlit is fully ready before Nginx starts reverse-proxying,
# preventing the initial websocket handshake from failing due to port binding race conditions.
set -e

echo "=== Starting FastAPI backend ==="
# Start FastAPI with Host 127.0.0.1 and Port 8001
python -m uvicorn api:app --host 127.0.0.1 --port 8001 > /tmp/fastapi.log 2>&1 &

echo "=== Starting Streamlit frontend ==="
# Start Streamlit bound to 127.0.0.1 on Port 8501 with CORS & XSRF disabled
python -m streamlit run app.py \
    --server.port=8501 \
    --server.address=127.0.0.1 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false > /tmp/streamlit.log 2>&1 &

echo "=== Polling Streamlit health endpoint ==="
MAX_ATTEMPTS=60
ATTEMPT=1
HEALTH_URL="http://127.0.0.1:8501/_stcore/health"

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    # Perform health check using python (highly compatible, doesn't require curl/wget)
    if python -c "import urllib.request; urllib.request.urlopen('$HEALTH_URL', timeout=1)" >/dev/null 2>&1; then
        echo "Streamlit health check passed on attempt $ATTEMPT!"
        break
    else
        echo "Streamlit not yet responding. (Attempt $ATTEMPT/$MAX_ATTEMPTS)"
        sleep 1
        ATTEMPT=$((ATTEMPT + 1))
    fi
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
    echo "ERROR: Streamlit did not become healthy within $MAX_ATTEMPTS seconds. Exiting startup."
    echo "=== Streamlit Logs ==="
    cat /tmp/streamlit.log || true
    echo "=== FastAPI Logs ==="
    cat /tmp/fastapi.log || true
    exit 1
fi

echo "=== Streamlit is healthy. Launching Nginx in the foreground ==="
exec nginx -g "daemon off;"
