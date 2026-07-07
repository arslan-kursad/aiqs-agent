#!/usr/bin/env bash
# Phase-3 serving demo: ONE continuous flow — clean PASS -> escalation triggers the VLM
# second-look -> the VLM abstains -> the graph PAUSES for a human -> resume to a final
# decision. Run against a live `aiqs-serve` (mock VLM backend by default, no API key).
#
# Usage (two shells):
#   shell 1:  make serve RUN=<run_id>
#   shell 2:  bash scripts/demo_requests.sh <run_id>
#
# LOW/HIGH anchor scores are read straight from that run's own image_scores.csv (the min
# and max raw scores), so the flow is real data, not invented numbers. Tuned against the
# committed capsules headline run — at the server's default target_prevalence (2%), the
# min score PASSes and the max score ESCALATEs (see CLAUDE.md's substrate notes). If your
# run's detector separates differently, the escalation step may resolve immediately
# instead of pausing — the script still prints exactly what happened.
#
# Requires: curl, uv (runs python IN THE PROJECT VENV, not the system python3 -- pandas
# lives there). jq is used for pretty-printing if present, else falls back to `PY -m json.tool`.

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE="http://${HOST}:${PORT}"
RUN="${1:?Usage: $0 <run_id>  (the same run_id passed to: make serve RUN=<run_id>)}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SCORES_CSV="${RESULTS_DIR}/runs/${RUN}/image_scores.csv"
PY="uv run python3"

if [[ ! -f "$SCORES_CSV" ]]; then
  echo "error: $SCORES_CSV not found -- is RUN correct?" >&2
  exit 1
fi

read -r LOW HIGH <<EOF
$($PY -c "
import pandas as pd
df = pd.read_csv('${SCORES_CSV}')
col = 'score' if 'score' in df.columns else 'anomaly_score'
print(df[col].min(), df[col].max())
")
EOF

pp() { if command -v jq >/dev/null 2>&1; then jq .; else $PY -m json.tool; fi; }
field() { $PY -c "import json,sys; print(json.load(sys.stdin).get('$1'))"; }

echo "=== 0. GET /health ==="
curl -sf "${BASE}/health" | pp
echo

echo "=== 1. Clean PASS -- low anomaly score (${LOW}) ==="
curl -sf -X POST "${BASE}/adjudicate" -H 'content-type: application/json' \
  -d "{\"anomaly_score\": ${LOW}, \"item_id\": \"demo-clean\"}" | tee /tmp/aiqs_demo_1.json | pp
echo "  -> decision: $(field decision < /tmp/aiqs_demo_1.json)"
echo

echo "=== 2. Escalation + VLM second-look -- high anomaly score (${HIGH}), with an image ==="
IMG_B64=$(printf 'not a real image -- the mock backend never decodes bytes' | base64)
curl -sf -X POST "${BASE}/adjudicate" -H 'content-type: application/json' \
  -d "{\"anomaly_score\": ${HIGH}, \"item_id\": \"demo-escalate\", \"image_b64\": \"${IMG_B64}\"}" \
  | tee /tmp/aiqs_demo_2.json | pp
DECISION_2="$(field decision < /tmp/aiqs_demo_2.json)"
echo "  -> tier1_decision: $(field tier1_decision < /tmp/aiqs_demo_2.json), decision: ${DECISION_2}"
echo

if [[ "$DECISION_2" != "pending_human" ]]; then
  echo "The VLM auto-resolved this item (decision=${DECISION_2}) -- no human step needed"
  echo "for THIS score on THIS run. That is a valid outcome, just not the pause/resume leg."
  exit 0
fi

echo "=== 3. GET /decisions/demo-escalate (pending human review) ==="
curl -sf "${BASE}/decisions/demo-escalate" | pp
echo

echo "=== 4. POST /human-verdict/demo-escalate -- resume with a human verdict ==="
curl -sf -X POST "${BASE}/human-verdict/demo-escalate" -H 'content-type: application/json' \
  -d '{"decision": "fail", "reviewer": "kursad", "note": "confirmed on human review"}' | pp
echo

echo "=== 5. GET /decisions/demo-escalate (finalized) ==="
curl -sf "${BASE}/decisions/demo-escalate" | pp
echo

echo "Done. Re-running this script will 409 on demo-clean/demo-escalate (already"
echo "adjudicated) -- restart the server for a fresh checkpoint, or edit the item_ids."
