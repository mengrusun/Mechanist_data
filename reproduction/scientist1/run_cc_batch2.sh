#!/usr/bin/env bash
# Batch launcher for cc reproduction tasks rows 8, 9, 10, 11, 12, 15, 17
# (indexes in /data/zhenqian/Reproduction1/resource/resource_status.md).
#
# Behavior mirrors run_cc_batch.sh:
#   * For each experiment: open a detached tmux session cc_<subfolder>,
#     cd into /data/zhenqian/Reproduction1/cc/<category>/<subfolder>,
#     start `claude`, submit the Chinese verification prompt.
#   * A per-session _claude_watcher.sh is attached, appending to
#     /data/zhenqian/Reproduction1/cc/logs/anomalies.log.
#   * stuck_watcher.sh (single global process) is started if not already
#     running, appending to /data/zhenqian/Reproduction1/cc/logs/stuck_watcher.log.
#   * Existing sessions with the same name are killed then relaunched.

set -euo pipefail

# Rows 8, 9, 10, 11, 12, 15, 17 as "<category>/<subfolder>" under CC_ROOT.
EXPERIMENTS=(
  "multi-agent_safety/multi_agent"
  "multilingual/lasa_safety"
  "multilingual/multi_lingual_reasoning"
  "multimodal/alignet_visual"
  "multimodal/universal_steering"
  "safety/circuit_breakers"
  "science/esmfold_mechanism"
)

# Prompt sent to claude after the TUI is ready.
USER_PROMPT='做实验验证task.md中的claim'

# Command used to boot claude inside the tmux pane.
CLAUDE_CMD='claude'

CC_ROOT="/data/zhenqian/Reproduction1/cc"
LOG_DIR="${CC_ROOT}/logs"
ANOMALY_LOG="${LOG_DIR}/anomalies.log"
STUCK_LOG="${LOG_DIR}/stuck_watcher.log"
STUCK_OUT="${LOG_DIR}/stuck_watcher.out"
WATCHER="/data/zhenqian/Reproduction1/script/_claude_watcher.sh"
STUCK_WATCHER="${CC_ROOT}/stuck_watcher.sh"

CLAUDE_BOOT_WAIT=15   # seconds to wait for the claude TUI to be ready before typing the prompt
INTER_EXP_DELAY=5     # seconds between successive experiment launches
SUBMIT_VERIFY_WAIT=3  # seconds to wait before checking that the prompt was actually submitted

for cmd in tmux claude; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd not in PATH" >&2; exit 1; }
done
if [[ ! -x "${WATCHER}" ]]; then
  echo "ERROR: watcher helper not executable: ${WATCHER}" >&2
  exit 1
fi
if [[ ! -x "${STUCK_WATCHER}" ]]; then
  echo "ERROR: stuck watcher not executable: ${STUCK_WATCHER}" >&2
  exit 1
fi
mkdir -p "${LOG_DIR}"

# Ensure the global stuck_watcher is running (append to stuck_watcher.log).
if pgrep -f "stuck_watcher.sh" >/dev/null 2>&1; then
  echo "stuck_watcher already running — reusing (logs append to ${STUCK_LOG})"
else
  echo "starting stuck_watcher (logs append to ${STUCK_LOG})"
  nohup "${STUCK_WATCHER}" >> "${STUCK_OUT}" 2>&1 &
  disown
fi

launch_one() {
  local rel="$1"                       # e.g. "multilingual/lasa_safety"
  local subfolder="${rel##*/}"         # leaf name for session id
  local workdir="${CC_ROOT}/${rel}"
  local session="cc_${subfolder}"

  printf '\n[%s]\n' "$session"

  if [[ ! -d "$workdir" ]]; then
    echo "       SKIP — workdir not found: $workdir"
    return 0
  fi
  if [[ ! -f "$workdir/task.md" ]]; then
    echo "       SKIP — task.md not found in: $workdir"
    return 0
  fi

  # Clobber any pre-existing session with the same name.
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "       killing existing tmux session $session"
    tmux kill-session -t "$session" 2>/dev/null || true
    sleep 1
  fi

  tmux new-session -d -s "$session" -c "$workdir"
  tmux send-keys -t "$session" "$CLAUDE_CMD" Enter
  sleep "$CLAUDE_BOOT_WAIT"
  # Type the prompt as literal characters, THEN send Enter as a separate
  # keystroke after a short pause. Claude Code's TUI treats a bursty
  # text+Enter payload as a bracketed paste — the Enter inside a paste is
  # interpreted as a newline in the input buffer, NOT as submit.
  tmux send-keys -t "$session" -l "$USER_PROMPT"
  sleep 1
  tmux send-keys -t "$session" Enter
  sleep "$SUBMIT_VERIFY_WAIT"
  # Verify submission: look for signals that Claude actually started
  # thinking. Do NOT include 'auto' — the "⏵⏵ auto mode on" banner is a
  # persistent false positive.
  if ! tmux capture-pane -p -t "$session" -S -100 | grep -qE '(Cogitating|esc to interrupt|✻)'; then
    echo "       first Enter appears dropped — re-sending Enter"
    tmux send-keys -t "$session" Enter
    sleep 2
  fi
  nohup "$WATCHER" "$session" "$ANOMALY_LOG" 30 >/dev/null 2>&1 &
  disown

  echo "       launched in $workdir"
}

echo "Batch launcher (cc reproduction tasks — rows 8,9,10,11,12,15,17) — ${#EXPERIMENTS[@]} experiments queued"
echo "Working root:  ${CC_ROOT}"
echo "Anomaly log:   ${ANOMALY_LOG}"
echo "Stuck log:     ${STUCK_LOG}"
echo "Watcher:       ${WATCHER}"
echo "Stuck watcher: ${STUCK_WATCHER}"
echo "Claude cmd:    ${CLAUDE_CMD}"
echo "User prompt:   ${USER_PROMPT}"

for rel in "${EXPERIMENTS[@]}"; do
  launch_one "$rel"
  sleep "$INTER_EXP_DELAY"
done

echo
echo "Done. Running cc_* tmux sessions:"
tmux ls 2>/dev/null | grep '^cc_' || echo "  (none)"
echo
echo "Watch anomalies:   tail -F ${ANOMALY_LOG}"
echo "Watch stuck:       tail -F ${STUCK_LOG}"
echo "Attach a session:  tmux attach -t cc_<subfolder>"
echo "Kill a session:    tmux kill-session -t cc_<subfolder>"
