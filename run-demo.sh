#!/usr/bin/env bash
# One-command launcher for the Public Speaking dimension demo.
#   ./run-demo.sh          → install API deps, boot server on :8400, open browser
#   PORT=9000 ./run-demo.sh → use a different port
# Local-first: no API keys. Heavy scoring stays in the Colab notebook; this serves
# the thin wrapper + 'My Learning Path' widget against already-extracted scores.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PORT="${PORT:-8400}"
PY="${PYTHON:-python3}"

echo "==> Personalised LMS — Public Speaking dimension demo"
echo "    root: $ROOT"
echo "    port: $PORT"

# 1. Deps — install only if FastAPI is missing (keeps re-runs fast).
if ! "$PY" -c "import fastapi, uvicorn, httpx" >/dev/null 2>&1; then
  echo "==> Installing API deps (fastapi, uvicorn, httpx)…"
  "$PY" -m pip install -q -r requirements-api.txt
else
  echo "==> API deps already present — skipping install."
fi

# 2. Sanity: the project must be importable from here.
if [ ! -f "utils/taxonomy.py" ]; then
  echo "ERROR: run this from the personalised-lms-service/ root." >&2
  exit 1
fi

# 3. Quick smoke test before serving (fail fast if something is broken).
echo "==> Running API tests…"
"$PY" -m pytest tests/test_api.py -q

# 4. Open the browser shortly after the server comes up (best-effort, non-fatal).
URL="http://localhost:${PORT}"
(
  sleep 2
  if command -v open >/dev/null 2>&1; then open "$URL"          # macOS
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" # Linux
  fi
) >/dev/null 2>&1 &

# 5. Serve. Ctrl+C stops it.
echo "==> Serving at ${URL}   (docs: ${URL}/docs)   — Ctrl+C to stop"
exec "$PY" -m uvicorn api.main:app --reload --port "$PORT"
