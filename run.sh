#!/usr/bin/env bash
# Build the C agent host, then start the orchestrator and one local agent host.
# Open http://localhost:8000 in a browser to watch the 3D world.
set -e
cd "$(dirname "$0")"

echo "[run] checking python deps (numpy, fastapi, uvicorn, websockets)…"
python3 - <<'PY'
import importlib, sys
missing = [m for m in ("numpy","fastapi","uvicorn","websockets") if importlib.util.find_spec(m) is None]
if missing:
    print("  missing:", ", ".join(missing))
    print("  install with:  pip install fastapi 'uvicorn[standard]' numpy")
    sys.exit(1)
print("  ok")
PY

echo "[run] building C agent host…"
make -C agent_host >/dev/null

echo "[run] starting orchestrator (web UI :8000, agent socket :9000)…"
( cd orchestrator && python3 server.py ) &
ORCH=$!
trap 'echo; echo "[run] shutting down…"; kill $ORCH $HOST 2>/dev/null; exit 0' INT TERM

# wait for the agent port to open
for _ in $(seq 1 40); do
  (echo > /dev/tcp/127.0.0.1/9000) >/dev/null 2>&1 && break
  sleep 0.25
done

echo "[run] starting one local agent host…"
./agent_host/agent_host 127.0.0.1 9000 host-local &
HOST=$!

echo
echo "  ┌────────────────────────────────────────────────┐"
echo "  │  ALIFE PoC running.                              │"
echo "  │  Open  http://localhost:8000  in your browser.   │"
echo "  │  Start more agent hosts (even on other machines):│"
echo "  │     ./agent_host/agent_host <orch-ip> 9000 name  │"
echo "  │  Ctrl-C to stop.                                 │"
echo "  └────────────────────────────────────────────────┘"
echo
wait $ORCH
