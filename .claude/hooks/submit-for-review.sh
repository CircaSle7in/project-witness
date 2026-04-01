#!/usr/bin/env bash
set -euo pipefail

# Claude-to-Codex Review Loop: Stop Hook Orchestrator
# Triggered by Claude Code's Stop hook. Creates a review request,
# runs Codex CLI for testing/review, and decides whether to allow
# Claude to stop or block for auto-fix / human gate.

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
COORD_DIR="${REPO_ROOT}/.coordination"
CONFIG="${COORD_DIR}/review-config.yaml"
REVIEWS_DIR="${COORD_DIR}/reviews"
LOCK_FILE="${COORD_DIR}/.review-lock"
NOTIFY_SCRIPT="${REPO_ROOT}/.claude/hooks/notify-discord.sh"

# Prevent reentrancy
if [[ "${stop_hook_active:-}" == "true" ]]; then
    exit 0
fi

if [[ -f "$LOCK_FILE" ]]; then
    exit 0
fi

# Check for actual code changes (not just .coordination/ changes)
CHANGED_FILES=$(git diff --name-only HEAD 2>/dev/null || git diff --name-only --cached 2>/dev/null || echo "")
if [[ -z "$CHANGED_FILES" ]]; then
    CHANGED_FILES=$(git status --porcelain 2>/dev/null | grep -E '^\s*[MADRCU?]' | awk '{print $NF}' || echo "")
fi

# Filter to only review-worthy paths
REVIEW_FILES=""
for f in $CHANGED_FILES; do
    case "$f" in
        src/*|tests/*|configs/*) REVIEW_FILES="${REVIEW_FILES} ${f}" ;;
    esac
done

# Nothing to review
if [[ -z "${REVIEW_FILES// /}" ]]; then
    exit 0
fi

# Create run ID and directories
RUN_ID="$(date +%Y%m%d-%H%M%S)-$(openssl rand -hex 3)"
RUN_DIR="${REVIEWS_DIR}/${RUN_ID}"
mkdir -p "$RUN_DIR"

# Acquire lock
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# Check retry count from previous runs in this session
SESSION_ID="${CLAUDE_SESSION_ID:-unknown}"
RETRY_COUNT=0
for prev_dir in "${REVIEWS_DIR}"/*/; do
    if [[ -f "${prev_dir}request.json" ]]; then
        prev_session=$(python3.12 -c "import json; print(json.load(open('${prev_dir}request.json')).get('session_id',''))" 2>/dev/null || echo "")
        if [[ "$prev_session" == "$SESSION_ID" ]]; then
            RETRY_COUNT=$((RETRY_COUNT + 1))
        fi
    fi
done

MAX_RETRIES=2

# Write request.json
DIFF_CONTENT=$(git diff HEAD 2>/dev/null || git diff --cached 2>/dev/null || echo "no diff available")
cat > "${RUN_DIR}/request.json" <<REQEOF
{
    "repo_root": "${REPO_ROOT}",
    "changed_files": [$(echo "$REVIEW_FILES" | xargs -n1 | sort -u | sed 's/.*/"&"/' | paste -sd, -)],
    "run_id": "${RUN_ID}",
    "session_id": "${SESSION_ID}",
    "retry_count": ${RETRY_COUNT},
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
REQEOF

# Write diff to file (keep request.json small)
echo "$DIFF_CONTENT" > "${RUN_DIR}/diff.patch"

# Write status
echo '{"status": "running"}' > "${RUN_DIR}/status.json"

# Build Codex prompt
CODEX_PROMPT="You are a code reviewer for Project Witness. Review ONLY the changed files listed below.

Changed files:
${REVIEW_FILES}

Your tasks:
1. Run: python -m pytest tests/ -v --tb=short
2. Run: python -m ruff check src/ tests/
3. Review the changed files for: logic errors, type mismatches, missing error handling at system boundaries, security issues, and adherence to project conventions (Pydantic v2, type hints, no em dashes).
4. Do NOT edit any tracked files. Only report findings.

Write your review as JSON to stdout with this exact schema:
{
    \"verdict\": \"PASS\" or \"NEEDS_CHANGES\",
    \"max_severity\": \"low\" or \"medium\" or \"high\" or \"critical\",
    \"summary\": \"one paragraph summary\",
    \"findings\": [
        {
            \"severity\": \"low|medium|high|critical\",
            \"kind\": \"test|logic|lint|security|architecture|migration\",
            \"file\": \"path/to/file.py\",
            \"line\": 42,
            \"title\": \"short title\",
            \"body\": \"explanation\",
            \"suggested_test\": \"optional test suggestion\"
        }
    ],
    \"tests\": [
        {
            \"command\": \"pytest ...\",
            \"status\": \"pass|fail|error|skip\",
            \"details\": \"summary of output\"
        }
    ],
    \"auto_fixable\": true or false
}"

# Run Codex with timeout
CODEX_TIMEOUT=120
CODEX_OUTPUT="${RUN_DIR}/codex-output.txt"
CODEX_LAST_MSG="${RUN_DIR}/codex-last-message.txt"

export stop_hook_active=true

if timeout "${CODEX_TIMEOUT}" codex exec \
    --full-auto \
    --json \
    --output-last-message "$CODEX_LAST_MSG" \
    -C "$REPO_ROOT" \
    "$CODEX_PROMPT" \
    > "$CODEX_OUTPUT" 2>&1; then
    CODEX_EXIT=0
else
    CODEX_EXIT=$?
fi

unset stop_hook_active

# Parse Codex output
if [[ $CODEX_EXIT -eq 124 ]]; then
    # Timeout
    echo '{"status": "timeout"}' > "${RUN_DIR}/status.json"
    echo "# Review Timed Out\n\nCodex did not complete within ${CODEX_TIMEOUT}s.\nRun ID: ${RUN_ID}" > "${RUN_DIR}/report.md"
    echo "{\"verdict\":\"TIMEOUT\",\"max_severity\":\"unknown\",\"summary\":\"Codex timed out\",\"findings\":[],\"tests\":[],\"auto_fixable\":false}" > "${RUN_DIR}/report.json"

    if [[ -x "$NOTIFY_SCRIPT" ]]; then
        bash "$NOTIFY_SCRIPT" "$RUN_ID" "timeout"
    fi
    exit 0  # Allow Claude to stop
fi

if [[ $CODEX_EXIT -ne 0 ]]; then
    # Codex failure
    echo '{"status": "failed"}' > "${RUN_DIR}/status.json"
    echo "# Review Failed\n\nCodex exited with code ${CODEX_EXIT}.\nRun ID: ${RUN_ID}" > "${RUN_DIR}/report.md"
    echo "{\"verdict\":\"ERROR\",\"max_severity\":\"unknown\",\"summary\":\"Codex failed with exit ${CODEX_EXIT}\",\"findings\":[],\"tests\":[],\"auto_fixable\":false}" > "${RUN_DIR}/report.json"

    if [[ -x "$NOTIFY_SCRIPT" ]]; then
        bash "$NOTIFY_SCRIPT" "$RUN_ID" "codex_failure"
    fi
    exit 0  # Allow Claude to stop
fi

# Extract report from Codex last message
if [[ -f "$CODEX_LAST_MSG" ]]; then
    cp "$CODEX_LAST_MSG" "${RUN_DIR}/report.md"

    # Try to extract JSON from the output
    python3.12 -c "
import json, sys, re

with open('${CODEX_LAST_MSG}') as f:
    content = f.read()

# Try to find JSON block
match = re.search(r'\{.*\"verdict\".*\}', content, re.DOTALL)
if match:
    try:
        report = json.loads(match.group())
        with open('${RUN_DIR}/report.json', 'w') as out:
            json.dump(report, out, indent=2)
        sys.exit(0)
    except json.JSONDecodeError:
        pass

# Fallback: create minimal report
report = {
    'verdict': 'NEEDS_CHANGES' if 'fail' in content.lower() or 'error' in content.lower() else 'PASS',
    'max_severity': 'medium',
    'summary': content[:500],
    'findings': [],
    'tests': [],
    'auto_fixable': False
}
with open('${RUN_DIR}/report.json', 'w') as out:
    json.dump(report, out, indent=2)
" 2>/dev/null || echo '{"verdict":"UNKNOWN","max_severity":"unknown","summary":"Could not parse Codex output","findings":[],"tests":[],"auto_fixable":false}' > "${RUN_DIR}/report.json"
fi

# Read verdict and decide
VERDICT=$(python3.12 -c "import json; print(json.load(open('${RUN_DIR}/report.json')).get('verdict','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
MAX_SEV=$(python3.12 -c "import json; print(json.load(open('${RUN_DIR}/report.json')).get('max_severity','unknown'))" 2>/dev/null || echo "unknown")
AUTO_FIX=$(python3.12 -c "import json; print(json.load(open('${RUN_DIR}/report.json')).get('auto_fixable', False))" 2>/dev/null || echo "False")

# Check for human-gated finding kinds
HAS_GATED=$(python3.12 -c "
import json
report = json.load(open('${RUN_DIR}/report.json'))
gated_kinds = {'security', 'architecture', 'migration'}
for f in report.get('findings', []):
    if f.get('kind') in gated_kinds or f.get('severity') in ('high', 'critical'):
        print('true')
        break
else:
    print('false')
" 2>/dev/null || echo "false")

if [[ "$VERDICT" == "PASS" ]]; then
    echo '{"status": "done", "verdict": "PASS"}' > "${RUN_DIR}/status.json"
    exit 0  # Allow Claude to stop

elif [[ "$HAS_GATED" == "true" ]] || [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
    # Human gate required
    echo '{"status": "done", "verdict": "NEEDS_CHANGES", "waiting_for_human": true}' > "${RUN_DIR}/status.json"
    touch "${RUN_DIR}/WAITING_FOR_HUMAN"

    if [[ -x "$NOTIFY_SCRIPT" ]]; then
        bash "$NOTIFY_SCRIPT" "$RUN_ID" "human_gate"
    fi
    exit 0  # Allow Claude to stop, human will resume

elif [[ "$AUTO_FIX" == "True" ]] && [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; then
    # Auto-fixable: block Claude stop so it reads the report and fixes
    echo '{"status": "done", "verdict": "NEEDS_CHANGES", "auto_fix": true}' > "${RUN_DIR}/status.json"
    echo "Review found auto-fixable issues. Read ${RUN_DIR}/report.md and address the findings."
    exit 1  # Block Claude from stopping

else
    # Default: human gate
    echo '{"status": "done", "verdict": "NEEDS_CHANGES", "waiting_for_human": true}' > "${RUN_DIR}/status.json"
    touch "${RUN_DIR}/WAITING_FOR_HUMAN"

    if [[ -x "$NOTIFY_SCRIPT" ]]; then
        bash "$NOTIFY_SCRIPT" "$RUN_ID" "human_gate"
    fi
    exit 0
fi
