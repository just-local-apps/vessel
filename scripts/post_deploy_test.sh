#!/usr/bin/env bash
# Master post-deploy verification.
#
# Runs every "live" pytest suite against the deployed Vessel instance,
# in the order they take to fail-fast on the cheapest checks first:
#
#   1. test_fly_live.py       – /health, MCP auth, get_state, apply_instruction
#   2. test_phoenix_live.py   – MCP apply_instruction → span lands in Phoenix
#   3. test_pwa_ui_live.py    – Playwright UI smoke (slowest, optional)
#
# Required env (from .env or shell):
#   VESSEL_FLY_URL              e.g. https://vessel-ravi.fly.dev
#   VESSEL_FLY_TOKEN            same value as VESSEL_AUTH_TOKEN secret on Fly
#   PHOENIX_API_KEY             Phoenix Cloud API key
#   PHOENIX_COLLECTOR_ENDPOINT  e.g. https://app.phoenix.arize.com/s/<workspace>
#
# Optional:
#   PHOENIX_PROJECT             defaults to "vessel"
#   VESSEL_LIVE_URL             defaults to VESSEL_FLY_URL
#   VESSEL_AUTH_TOKEN           defaults to VESSEL_FLY_TOKEN
#
# The UI suite always runs — it's part of post-deploy verification.
#
# Usage:
#   ./scripts/post_deploy_test.sh
#
# Exit code: 0 only if every non-skipped test passes.
set -euo pipefail

cd "$(dirname "$0")/.."

# Pull defaults from .env so callers don't have to re-export everything.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Fall back to Vessel's own auth vars if Fly-specific aliases are missing.
: "${VESSEL_FLY_URL:=${VESSEL_LIVE_URL:-}}"
: "${VESSEL_FLY_TOKEN:=${VESSEL_AUTH_TOKEN:-}}"
: "${VESSEL_LIVE_URL:=${VESSEL_FLY_URL:-}}"
: "${VESSEL_AUTH_TOKEN:=${VESSEL_FLY_TOKEN:-}}"
export VESSEL_FLY_URL VESSEL_FLY_TOKEN VESSEL_LIVE_URL VESSEL_AUTH_TOKEN

missing=()
for v in VESSEL_FLY_URL VESSEL_FLY_TOKEN PHOENIX_API_KEY PHOENIX_COLLECTOR_ENDPOINT; do
  [[ -z "${!v:-}" ]] && missing+=("$v")
done
if (( ${#missing[@]} > 0 )); then
  echo "ERROR: missing required env vars: ${missing[*]}" >&2
  echo "Set them in .env or export before running." >&2
  exit 2
fi

echo "==> Target:        $VESSEL_FLY_URL"
echo "==> Phoenix:       $PHOENIX_COLLECTOR_ENDPOINT (project ${PHOENIX_PROJECT:-vessel})"
echo

# Confirm the deploy is even up before spending time on slow suites.
echo "==> Pre-flight: GET /health"
status_json=$(curl -fsS --max-time 10 "$VESSEL_FLY_URL/health")
echo "$status_json" | python -m json.tool
configured=$(echo "$status_json" | python -c "import sys,json; print(json.load(sys.stdin).get('tracing',{}).get('configured'))")
if [[ "$configured" != "True" ]]; then
  echo "ERROR: deployed app reports tracing.configured=$configured — Phoenix env vars are not set on Fly." >&2
  echo "Run: fly secrets set PHOENIX_API_KEY=... PHOENIX_COLLECTOR_ENDPOINT=... PHOENIX_PROJECT=vessel" >&2
  exit 3
fi
echo

run_suite() {
  local name="$1"
  shift
  echo "==> $name"
  if uv run --extra dev "$@"; then
    echo "    ✔ $name passed"
  else
    echo "    ✘ $name FAILED" >&2
    exit 1
  fi
  echo
}

run_suite "MCP smoke (test_fly_live.py)" \
  pytest -q tests/test_fly_live.py

run_suite "Phoenix trace verification (test_phoenix_live.py)" \
  pytest -q tests/test_phoenix_live.py -s

echo "==> PWA UI smoke (test_pwa_ui_live.py)"
if uv run --extra dev --extra ui-test pytest -q tests/test_pwa_ui_live.py; then
  echo "    ✔ PWA UI smoke passed"
else
  echo "    ✘ PWA UI smoke FAILED" >&2
  exit 1
fi
echo

echo
echo "All post-deploy checks passed."
