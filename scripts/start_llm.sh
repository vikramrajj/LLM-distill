#!/usr/bin/env bash
# Start/stop the local LLM server (llama.cpp)

set -euo pipefail

LLAMA_BIN="/tmp/llama.cpp/build/bin/llama-server"
MODEL="${LLM_MODEL:-$HOME/models/phi-3.5-mini-instruct-Q4_K_M.gguf}"
HOST="${LLM_HOST:-127.0.0.1}"
PORT="${LLM_PORT:-8080}"
CTX="${LLM_CTX:-4096}"
THREADS="${LLM_THREADS:-12}"
LOG_FILE="${HOME}/models/llama-server.log"

PIDFILE="/tmp/llama-server.pid"

start() {
    if is_running; then
        echo "Already running (PID: $(cat "$PIDFILE"))"
        return 0
    fi

    echo "Starting llama-server..."
    echo "  Model:   $MODEL"
    echo "  Context: $CTX tokens"
    echo "  Threads: $THREADS"
    echo "  API:     http://${HOST}:${PORT}/v1"

    nohup "$LLAMA_BIN" \
        -m "$MODEL" \
        -ngl 0 \
        -c "$CTX" \
        -t "$THREADS" \
        --host "$HOST" \
        --port "$PORT" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PIDFILE"
    sleep 2

    if is_running; then
        echo "✅ Server started (PID: $(cat "$PIDFILE"))"
        echo "   Log: $LOG_FILE"
    else
        echo "❌ Failed to start. Check log: $LOG_FILE"
        exit 1
    fi
}

stop() {
    if is_running; then
        kill "$(cat "$PIDFILE")" 2>/dev/null || true
        rm -f "$PIDFILE"
        echo "Stopped"
    else
        echo "Not running"
    fi
}

status() {
    if is_running; then
        echo "Running (PID: $(cat "$PIDFILE"))"
        curl -s "http://${HOST}:${PORT}/v1/models" | \
            python3 -c "import sys,json; print('Model:', json.load(sys.stdin)['models'][0]['name'])" 2>/dev/null || true
    else
        echo "Not running"
    fi
}

test_llm() {
    echo "Testing LLM..."
    curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Say hi in 3 words."}],"max_tokens":30}' | \
        python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

is_running() {
    [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "${1:-help}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    status)  status ;;
    test)    test_llm ;;
    help|*)
        echo "Usage: $0 {start|stop|restart|status|test}"
        ;;
esac
