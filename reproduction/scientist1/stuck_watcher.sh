#!/usr/bin/env bash
# Report every cc_* tmux session stuck waiting for a USER decision.
# All interactive prompts are surfaced here — nothing is auto-answered.
#
# Detected states:
#   * "Auto mode classifier requires confirmation" — auto-mode-classifier
#     deny prompt (Yes / Yes-allow / No menu).
#   * "Enter to select · ↑/↓ to navigate" — an AskUserQuestion menu
#     (Data access, Sandbox override, method choice, ...).
#   * "Do you want to proceed?" — any other confirmation prompt.
#   * "Interrupted · What should Claude do instead?" — Claude was
#     cancelled and is now waiting for guidance.
#
# Emits a report to /data/zhenqian/Reproduction1/cc/logs/stuck_watcher.log
# Each stuck session is logged at most once per 60 s window (SHA-based dedup
# + min-realert throttle) so navigating a menu does not spam the log.
#
# Run detached:
#   nohup /data/zhenqian/Reproduction1/cc/stuck_watcher.sh \
#         > /data/zhenqian/Reproduction1/cc/logs/stuck_watcher.out 2>&1 &
#   disown
#
# Stop:
#   pkill -f 'stuck_watcher.sh'

set -uo pipefail

POLL="${POLL:-15}"
LOG_DIR="/data/zhenqian/Reproduction1/cc/logs"
LOG="${LOG_DIR}/stuck_watcher.log"

# Waiting-for-user markers. Any one being present on the visible pane
# is enough evidence that the session is blocked.
INTERACTIVE='Enter to select|Do you want to proceed|Interrupted · What should Claude do instead|What should Claude do instead'

# Auto-mode-classifier deny prompt marker (used only to classify the kind).
AUTO_DENY='Auto mode classifier requires confirmation'

mkdir -p "$LOG_DIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] stuck-watcher started (poll=${POLL}s)" | tee -a "$LOG"

declare -A LAST_HASH
declare -A LAST_ALERT
MIN_REALERT_SEC=60

classify() {
  # Categorize the stuck state into a short tag for the report.
  local visible="$1"
  local context="$2"
  if echo "$context" | grep -qE "$AUTO_DENY"; then
    echo "auto-mode-classifier deny prompt (Yes / Yes-allow / No)"
  elif echo "$visible" | grep -qE 'Interrupted · What should Claude do instead|What should Claude do instead'; then
    echo "interrupted / needs-guidance"
  elif echo "$visible" | grep -qE 'Enter to select'; then
    echo "AskUserQuestion menu (Data access / Sandbox / method-choice ...)"
  elif echo "$visible" | grep -qE 'Do you want to proceed'; then
    echo "confirmation prompt"
  else
    echo "unknown"
  fi
}

while :; do
  sessions=$(tmux ls 2>/dev/null | awk -F: '/^cc_/ {print $1}' || true)
  if [[ -z "$sessions" ]]; then
    sleep "$POLL"
    continue
  fi

  for s in $sessions; do
    tmux has-session -t "$s" 2>/dev/null || continue
    visible=$(tmux capture-pane -p -t "$s" 2>/dev/null || true)
    context=$(tmux capture-pane -p -t "$s" -S -50 2>/dev/null || true)
    [[ -z "$visible" ]] && continue

    # Not a stuck-on-user state? Skip.
    if ! echo "$visible" | grep -qE "$INTERACTIVE"; then
      continue
    fi

    hash=$(printf '%s' "$visible" | sha256sum | cut -c1-16)
    if [[ "${LAST_HASH[$s]:-}" == "$hash" ]]; then
      continue  # same stuck state; already reported
    fi
    # Throttle re-alerts (e.g. user navigating menu changes hash rapidly).
    now=$(date +%s)
    prev="${LAST_ALERT[$s]:-0}"
    if (( now - prev < MIN_REALERT_SEC )); then
      LAST_HASH[$s]="$hash"
      continue
    fi

    kind=$(classify "$visible" "$context")
    workdir=$(tmux display-message -t "$s" -p '#{pane_current_path}' 2>/dev/null || echo unknown)
    snippet=$(echo "$visible" | tail -25)

    {
      echo "──────────────────────────────────────────────────────"
      echo "[$(ts)] [$s] STUCK — needs manual decision"
      echo "  kind:    $kind"
      echo "  workdir: $workdir"
      echo "  attach:  tmux attach -t $s"
      echo "  ---- last 25 visible lines ----"
      echo "$snippet" | sed 's/^/  | /'
      echo
    } >> "$LOG"

    printf '[%s] [%s] STUCK — %s (attach: tmux attach -t %s)\n' \
      "$(ts)" "$s" "$kind" "$s"

    LAST_HASH[$s]="$hash"
    LAST_ALERT[$s]="$now"
  done

  sleep "$POLL"
done
