#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# governance_shim.sh — Governance observation layer for Claude Code hooks
#
# Emits structured JSONL events to a SWARM-readable sink, providing
# a governance instrumentation surface without modifying Superpowers
# or any Claude Code internals.
#
# Hook events handled:
#   SessionStart   → session init with governance regime config
#   PreToolUse     → tool invocation logging + Byline provenance
#   PostToolUse    → outcome logging + plan deviation detection
#   SubagentStop   → task completion checkpoint + review verdict
#
# Sink: $GOVERNANCE_SHIM_SINK (default: runs/governance_shim.jsonl)
# All events are append-only JSONL for generic consumers (not guaranteed to be
# directly compatible with swarm.logging.event_log Event.from_dict()).
#
# Usage in settings.json:
#   "SessionStart": [{ "hooks": [{ "command": "bash .claude/hooks/governance_shim.sh session_start" }] }]
#   "PreToolUse":   [{ "hooks": [{ "command": "bash .claude/hooks/governance_shim.sh pre_tool_use" }] }]
#   "PostToolUse":  [{ "hooks": [{ "command": "bash .claude/hooks/governance_shim.sh post_tool_use" }] }]
#   "SubagentStop": [{ "hooks": [{ "command": "bash .claude/hooks/governance_shim.sh subagent_stop" }] }]
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ──
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")"
SINK="${GOVERNANCE_SHIM_SINK:-${REPO_ROOT}/runs/governance_shim.jsonl}"
HOOK_EVENT="${1:-unknown}"
SESSION="${SESSION_ID:-${AO_SESSION:-anonymous}}"
WORKTREE="${WORKTREE_ID:-}"
BRANCH="${SESSION_BRANCH:-$(git branch --show-current 2>/dev/null || echo "unknown")}"

# Ensure sink directory exists
mkdir -p "$(dirname "$SINK")"

# ── Concurrent-session heartbeats (beads oldj) ──
# Every live session on this checkout maintains a heartbeat file keyed by its
# Claude process pid. Session-start and pre-commit use these to detect the
# shared-checkout race that has repeatedly clobbered work (exhibits in bead
# oldj; 2026-07-22: two sessions fought over the index mid-7ge5-commit).
HEARTBEAT_DIR="${REPO_ROOT}/.claude/session-heartbeats"
HEARTBEAT_FRESH_SECONDS=900
SESSION_PID="${PPID:-$$}"

update_heartbeat() {
  mkdir -p "$HEARTBEAT_DIR" 2>/dev/null || return 0
  printf '{"pid": %s, "worktree": "%s", "ts": "%s"}\n'     "$SESSION_PID" "${IS_SESSION_WORKTREE:-false}"     "$(date -u +%Y-%m-%dT%H:%M:%SZ)"     > "${HEARTBEAT_DIR}/${SESSION_PID}.json" 2>/dev/null || true
}

# Prints the pids of OTHER live main-checkout sessions (fresh heartbeat +
# process alive), one per line. Empty output = no concurrency.
detect_concurrent_sessions() {
  [ -d "$HEARTBEAT_DIR" ] || return 0
  local now f pid age mtime
  now=$(date +%s)
  for f in "$HEARTBEAT_DIR"/*.json; do
    [ -e "$f" ] || continue
    pid="$(basename "$f" .json)"
    [ "$pid" = "$SESSION_PID" ] && continue
    # GNU (Linux) spells mtime `-c %Y`; BSD/macOS spells it `-f %m`. These must
    # be separate assignments, not `$(a || b)`: on Linux, GNU stat reads `-f` as
    # --file-system and prints a block starting with `File: ...` to stdout
    # before failing, and command substitution concatenates that with the
    # fallback's output. `mtime` then held `File: ...`, and the arithmetic below
    # evaluated the bare word `File` as a variable name -- an unbound-variable
    # abort under `set -u`.
    mtime=$(stat -c %Y "$f" 2>/dev/null) \
      || mtime=$(stat -f %m "$f" 2>/dev/null) \
      || mtime=0
    case "$mtime" in ''|*[!0-9]*) mtime=0 ;; esac
    age=$(( now - mtime ))
    if [ "$age" -gt "$HEARTBEAT_FRESH_SECONDS" ] || ! ps -p "$pid" > /dev/null 2>&1; then
      rm -f "$f" 2>/dev/null || true   # reap dead/stale heartbeats
      continue
    fi
    # Sessions isolated in their own worktree don't count as contention.
    grep -q '"worktree": "true"' "$f" 2>/dev/null && continue
    echo "$pid"
  done
}

# ── Helpers ──

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%S.%3NZ" 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# Deterministic event ID: sha256 of (hook_event + session + timestamp) truncated to 12 hex chars
make_event_id() {
  local ts="$1"
  local input="${HOOK_EVENT}:${SESSION}:${ts}"

  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$input" | sha256sum | cut -c1-12
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s' "$input" | shasum -a 256 | cut -c1-12
  elif command -v openssl >/dev/null 2>&1; then
    printf '%s' "$input" | openssl dgst -sha256 | awk '{print $NF}' | cut -c1-12
  else
    # Last-resort fallback: cksum (weaker, but avoids failure)
    printf '%s' "$input" | cksum | awk '{print $1}' | cut -c1-12
  fi
}

# Read stdin once and cache (hooks receive context JSON on stdin)
read_stdin() {
  if [ -z "${_STDIN_CACHE+x}" ]; then
    _STDIN_CACHE="$(cat)"
    if [ -z "$_STDIN_CACHE" ]; then
      _STDIN_CACHE="${ARGUMENTS:-}"
    fi
  fi
  echo "$_STDIN_CACHE"
}

# Safe jq extraction with fallback for environments without jq
jq_or_fallback() {
  local json="$1"
  local path="$2"
  local default="${3:-}"

  if command -v jq &>/dev/null; then
    local result
    result="$(echo "$json" | jq -r "$path // empty" 2>/dev/null)" || true
    if [ -z "$result" ]; then
      echo "$default"
    else
      echo "$result"
    fi
  else
    echo "$default"
  fi
}

# Emit a single JSONL event to the sink
emit_event() {
  local event_type="$1"
  local payload="$2"
  local ts
  ts="$(timestamp)"
  local event_id
  event_id="$(make_event_id "$ts")"

  local event
  if command -v jq &>/dev/null; then
    event=$(jq -nc \
      --arg eid "$event_id" \
      --arg ts "$ts" \
      --arg etype "$event_type" \
      --arg sid "$SESSION" \
      --arg wt "$WORKTREE" \
      --arg br "$BRANCH" \
      --argjson payload "$payload" \
      '{
        event_id: $eid,
        timestamp: $ts,
        event_type: $etype,
        agent_id: $sid,
        payload: ($payload + {session_id: $sid, worktree_id: $wt, branch: $br})
      }')
  else
    # Fallback: use python3 for proper JSON escaping
    event=$(python3 -c "
import json, sys
print(json.dumps({
  'event_id': sys.argv[1],
  'timestamp': sys.argv[2],
  'event_type': sys.argv[3],
  'agent_id': sys.argv[4],
  'payload': json.loads(sys.argv[5])
}))" "$event_id" "$ts" "$event_type" "$SESSION" "$payload" 2>/dev/null) || \
    event="{\"event_id\":\"${event_id}\",\"timestamp\":\"${ts}\",\"event_type\":\"${event_type}\",\"agent_id\":\"${SESSION}\",\"payload\":${payload}}"
  fi

  echo "$event" >> "$SINK"
}

# ── Hook Handlers ──

handle_session_start() {
  update_heartbeat
  if [ "${IS_SESSION_WORKTREE:-false}" != "true" ]; then
    local others
    others="$(detect_concurrent_sessions | tr '\n' ' ')"
    if [ -n "${others// /}" ]; then
      # stderr, NOT stdout: session_start stdout is a JSON contract
      # (test_governance_shim_bash), and this warning corrupted it whenever
      # concurrent sessions were actually present.
      cat >&2 <<EOWARN
[worktree-discipline] WARNING: other live Claude session(s) (pid ${others}) are
working in THIS shared checkout. Concurrent sessions on one working tree race
each other's index, revert each other's files, and corrupt shared tool caches
(see beads oldj for exhibits). Strongly prefer an isolated worktree:
  ./scripts/claude-tmux.sh N     # isolated sessions
or coordinate via the agent_messages table before mutating git state.
EOWARN
    fi
  fi

  # Load governance regime from scenario config or env
  local regime="${GOVERNANCE_REGIME:-default}"
  local knobs_file="${REPO_ROOT}/.letta/memory/project/governance-knobs.md"
  local knobs_summary="none"

  if [ -f "$knobs_file" ]; then
    # Extract knob names for the event payload
    knobs_summary=$(grep -E '^## ' "$knobs_file" | sed 's/^## //' | tr '\n' ',' | sed 's/,$//' || echo "none")
  fi

  emit_event "governance.session_init" "$(jq_build_or_raw \
    "regime" "$regime" \
    "governance_knobs" "$knobs_summary" \
    "hook_version" "1.0.0")"

  # Coordination channel read-path (bead vw8g): surface unacked agent_messages
  # so reading the channel is a side effect of session start, not a virtue.
  # Fail-open: any missing dependency degrades to the empty-JSON no-op.
  local msg_db="${REPO_ROOT}/runs/runs.db" unacked=""
  if [ -f "$msg_db" ] && command -v sqlite3 >/dev/null 2>&1; then
    unacked=$(sqlite3 -separator ' :: ' "$msg_db" \
      "SELECT id, from_agent, substr(body,1,300) FROM agent_messages WHERE acked=0 ORDER BY ts LIMIT 12;" \
      2>/dev/null || true)
  fi
  if [ -n "$unacked" ] && command -v jq >/dev/null 2>&1; then
    printf '%s\n' "$unacked" | jq -Rs '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: ("UNACKED agent_messages (runs/runs.db) — act on ASSIGN rows for your role per AGENTS.md \u00a7 Session Start, then ack: sqlite3 runs/runs.db \"UPDATE agent_messages SET acked=1 WHERE id=<id>\"\n" + .)}}'
    return
  fi

  # Return empty JSON (no override)
  echo '{}'
}

handle_pre_tool_use() {
  update_heartbeat
  local input
  input="$(read_stdin)"

  local tool_name
  tool_name="$(jq_or_fallback "$input" '.tool_name' 'unknown')"
  local tool_input
  tool_input="$(jq_or_fallback "$input" '.tool_input | tostring' '{}')"

  # Tag with Byline provenance frame if available
  local byline="${BYLINE_AGENT:-${SESSION}}"

  emit_event "governance.tool_invocation" "$(jq_build_or_raw \
    "tool_name" "$tool_name" \
    "byline_agent" "$byline" \
    "phase" "pre")"

  # Return empty JSON — no approve/block decision by default.
  # To enable blocking, set GOVERNANCE_SHIM_ENFORCE=1 and
  # add policy logic here that returns {"decision": "block", "reason": "..."}.
  echo '{}'
}

handle_post_tool_use() {
  local input
  input="$(read_stdin)"

  local tool_name
  tool_name="$(jq_or_fallback "$input" '.tool_name' 'unknown')"
  local exit_code
  exit_code="$(jq_or_fallback "$input" '.exit_code' '0')"
  local tool_response_preview
  tool_response_preview="$(jq_or_fallback "$input" '.tool_response | tostring | .[0:200]' '')"

  # Destructive operation detection: always flag dangerous commands.
  # When GOVERNANCE_TASK_SPEC is set, this also serves as plan deviation detection.
  local deviation="none"
  if [ "$tool_name" = "Bash" ]; then
    local command
    command="$(jq_or_fallback "$input" '.tool_input.command' '')"
    case "$command" in
      *"rm -rf"*|*"git reset --hard"*|*"git push --force"*|*"DROP TABLE"*)
        deviation="destructive_operation"
        ;;
    esac
  fi

  emit_event "governance.tool_outcome" "$(jq_build_or_raw \
    "tool_name" "$tool_name" \
    "exit_code" "$exit_code" \
    "deviation" "$deviation" \
    "phase" "post")"

  echo '{}'
}

handle_subagent_stop() {
  local input
  input="$(read_stdin)"

  local subagent_id
  subagent_id="$(jq_or_fallback "$input" '.agent_id // .subagent_id' 'unknown')"
  local reason
  reason="$(jq_or_fallback "$input" '.reason // .stop_reason' 'completed')"

  # Review verdict: did the subagent complete its task?
  local verdict="pass"
  if [ "$reason" = "error" ] || [ "$reason" = "timeout" ]; then
    verdict="fail"
  fi

  emit_event "governance.task_checkpoint" "$(jq_build_or_raw \
    "subagent_id" "$subagent_id" \
    "stop_reason" "$reason" \
    "review_verdict" "$verdict" \
    "phase" "subagent_stop")"

  echo '{}'
}

# ── JSON builder helper ──
# Builds a JSON object from key-value pairs (requires jq, falls back to raw)
jq_build_or_raw() {
  if command -v jq &>/dev/null; then
    local args=()
    local filter="{"
    local first=true
    while [ $# -ge 2 ]; do
      local key="$1"
      local val="$2"
      shift 2
      args+=(--arg "$key" "$val")
      if [ "$first" = true ]; then
        filter+="$key: \$$key"
        first=false
      else
        filter+=", $key: \$$key"
      fi
    done
    filter+="}"
    jq -nc "${args[@]}" "$filter" 2>/dev/null || echo '{}'
  else
    # Fallback: use python3 for proper JSON escaping
    local py_args=()
    while [ $# -ge 2 ]; do
      py_args+=("$1" "$2")
      shift 2
    done
    python3 -c "
import json, sys
args = sys.argv[1:]
d = {}
for i in range(0, len(args), 2):
    d[args[i]] = args[i+1]
print(json.dumps(d))
" "${py_args[@]}" 2>/dev/null || echo '{}'
  fi
}

# ── Dispatch ──
case "$HOOK_EVENT" in
  session_start)   handle_session_start ;;
  pre_tool_use)    handle_pre_tool_use ;;
  post_tool_use)   handle_post_tool_use ;;
  subagent_stop)   handle_subagent_stop ;;
  *)
    # Return valid JSON on stdout to satisfy hook contract
    echo '{}'
    exit 0
    ;;
esac
