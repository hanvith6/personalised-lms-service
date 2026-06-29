#!/usr/bin/env bash
# One-command sync: commit everything and push to GitHub so Colab's next
# "Run All" pulls the latest. Usage:
#   ./push.sh                 → auto commit message with timestamp
#   ./push.sh "fix: pacing"   → custom commit message
# Local-first / no secrets: .gitignore already excludes caches, media, lms.zip.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Nothing to do? Say so and exit cleanly.
if [ -z "$(git status --porcelain)" ]; then
  echo "==> Nothing to commit — working tree clean. Pushing any unpushed commits…"
  git push --quiet origin "$(git branch --show-current)" && echo "✅ up to date" || true
  exit 0
fi

# Safety: never push a stray secret/.env even if .gitignore missed it.
if git status --porcelain | grep -qiE '\.env($|[^.])|secret|credential|\.pem$|id_rsa'; then
  echo "⛔ Refusing to push — a possible secret is staged:" >&2
  git status --porcelain | grep -iE '\.env|secret|credential|\.pem$|id_rsa' >&2
  echo "   Add it to .gitignore or remove it, then re-run." >&2
  exit 1
fi

MSG="${1:-sync: $(date '+%Y-%m-%d %H:%M')}"
BRANCH="$(git branch --show-current)"

echo "==> Committing on '$BRANCH': $MSG"
git add -A
git commit -q -m "$MSG"

echo "==> Pushing to origin/$BRANCH…"
git push --quiet origin "$BRANCH"

echo "✅ Pushed. On Colab: just Run All — cell 2 will git pull the latest."
git log --oneline -1
