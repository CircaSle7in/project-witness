#!/usr/bin/env bash
set -euo pipefail

# Discord Webhook Notifier for Claude-to-Codex Review Loop
# Called by submit-for-review.sh when human gate is required.
# Usage: notify-discord.sh <run_id> <reason>

RUN_ID="${1:?Usage: notify-discord.sh <run_id> <reason>}"
REASON="${2:-human_gate}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RUN_DIR="${REPO_ROOT}/.coordination/reviews/${RUN_ID}"
WEBHOOK_URL_FILE="$HOME/.config/project-witness/discord-webhook-url"

# Read webhook URL
if [[ ! -f "$WEBHOOK_URL_FILE" ]]; then
    echo "Warning: Discord webhook URL not found at ${WEBHOOK_URL_FILE}" >&2
    exit 0  # Don't fail the review loop over a missing webhook
fi
WEBHOOK_URL="$(cat "$WEBHOOK_URL_FILE" | tr -d '\n')"

# Read report data
REPORT_FILE="${RUN_DIR}/report.json"
if [[ -f "$REPORT_FILE" ]]; then
    VERDICT=$(python3.12 -c "import json; print(json.load(open('${REPORT_FILE}')).get('verdict','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    MAX_SEV=$(python3.12 -c "import json; print(json.load(open('${REPORT_FILE}')).get('max_severity','unknown'))" 2>/dev/null || echo "unknown")
    SUMMARY=$(python3.12 -c "import json; print(json.load(open('${REPORT_FILE}')).get('summary','No summary available')[:300])" 2>/dev/null || echo "No summary")
    FINDING_COUNT=$(python3.12 -c "import json; print(len(json.load(open('${REPORT_FILE}')).get('findings',[])))" 2>/dev/null || echo "?")
    TOP_FINDINGS=$(python3.12 -c "
import json
report = json.load(open('${REPORT_FILE}'))
findings = report.get('findings', [])[:3]
lines = []
for f in findings:
    sev = f.get('severity', '?')
    title = f.get('title', 'untitled')
    file = f.get('file', '?')
    lines.append(f'  [{sev}] {file}: {title}')
print('\n'.join(lines) if lines else '  (none)')
" 2>/dev/null || echo "  (could not parse findings)")
else
    VERDICT="UNKNOWN"
    MAX_SEV="unknown"
    SUMMARY="Report file not found"
    FINDING_COUNT="?"
    TOP_FINDINGS="  (no report)"
fi

# Build reason label
case "$REASON" in
    human_gate)  REASON_LABEL="Human review required" ;;
    timeout)     REASON_LABEL="Codex timed out" ;;
    codex_failure) REASON_LABEL="Codex failed" ;;
    *)           REASON_LABEL="$REASON" ;;
esac

# Choose emoji
case "$REASON" in
    human_gate)    EMOJI="🔶" ;;
    timeout)       EMOJI="⏱️" ;;
    codex_failure) EMOJI="❌" ;;
    *)             EMOJI="📋" ;;
esac

# Send webhook
MESSAGE="${EMOJI} **Review Gate: ${REASON_LABEL}**

**Run ID:** \`${RUN_ID}\`
**Verdict:** ${VERDICT} | **Severity:** ${MAX_SEV} | **Findings:** ${FINDING_COUNT}

**Summary:** ${SUMMARY}

**Top findings:**
${TOP_FINDINGS}

**Report:** \`${RUN_DIR}/report.md\`
**Resume:** Tell Claude \`address review run ${RUN_ID}\`"

curl -s -o /dev/null -w "" \
    -H "Content-Type: application/json" \
    -d "$(python3.12 -c "import json; print(json.dumps({'content': $(python3.12 -c "import json; print(json.dumps('''${MESSAGE}'''))")}))")" \
    "$WEBHOOK_URL" 2>/dev/null || \
curl -s -o /dev/null \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"${EMOJI} Review gate (${REASON_LABEL}) for run ${RUN_ID}. Check .coordination/reviews/${RUN_ID}/report.md\"}" \
    "$WEBHOOK_URL" 2>/dev/null || true
