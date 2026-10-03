#!/usr/bin/env bash
# Refresh the Google Trends snapshots for the config.yaml watchlist from the owner's Mac
# and push them (Google Trends has no keyless API and blocks CI IPs, so the daily
# GitHub Actions refresh can't fetch it). The next daily refresh / Pages deploy then
# recomputes the sample verdicts with the new Trends data.
#
# Only data/samples/google_trends/ is staged, committed and pushed; nothing else in the
# working tree is touched. Exits 0 without committing when nothing changed.
#
#   scripts/refresh_trends_local.sh            # all watchlist topics
#   scripts/refresh_trends_local.sh --dry-run  # fetch, but don't commit or push
#
# Optional daily schedule via launchd: ops/macos/com.signalcheck.refresh-trends.plist
# (template; not installed). See README, "Google Trends from your Mac".
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
BRANCH="${SIGNALCHECK_BRANCH:-main}"
TRENDS_DIR="data/samples/google_trends"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }

if [[ ! -x .venv/bin/python ]]; then
  log "error: $REPO/.venv not found; create it first (see README, Local development)." >&2
  exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate

current="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current" != "$BRANCH" ]]; then
  log "error: on branch '$current', expected '$BRANCH' (set SIGNALCHECK_BRANCH to override)." >&2
  exit 2
fi

log "pulling $BRANCH"
git pull -q --rebase --autostash origin "$BRANCH"

log "refreshing Google Trends for the watchlist"
python -m scripts.refresh_samples --sources google_trends

git add -A -- "$TRENDS_DIR"
if git diff --cached --quiet -- "$TRENDS_DIR"; then
  log "no Google Trends changes"
  exit 0
fi
git --no-pager diff --cached --stat -- "$TRENDS_DIR"
if [[ "$DRY_RUN" == 1 ]]; then
  git reset -q -- "$TRENDS_DIR"
  log "dry run: changes left unstaged, nothing committed"
  exit 0
fi

git commit -q -m "Refresh Google Trends snapshots $(date -u +%F)" -- "$TRENDS_DIR"
git push -q origin "HEAD:$BRANCH"
log "pushed $(git rev-parse --short HEAD)"
