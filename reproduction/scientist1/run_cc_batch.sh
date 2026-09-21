#!/usr/bin/env bash
# Batch launcher for the first 7 cc reproduction tasks
# (rows 1-7 in /data/zhenqian/Reproduction1/resource/resource_status.md).
#
# For each experiment: open a detached tmux session named cc_<subfolder>,
# cd into /data/zhenqian/Reproduction1/cc/<category>/<subfolder>,
# start `claude`, submit the Chinese verification prompt, and attach a
# background anomaly watcher.
#
# Unlike the mechanica launcher, an existing session with the same name is
# KILLED first (not skipped) — the user explicitly asked for clobber-on-rerun.

set -euo pipefail

# First-7 experiments as "<category>/<subfolder>" under CC_ROOT.
# Ordering follows resource_status.md rows 1-7.
EXPERIMENTS=(
  "belief/closing_gap_belief"
  "belief/verbal_confidence"
  "belief/llm_social_decision"
  "emotion/emotion_circuit"
  "emotion/emotion_prompts"
  "feature_description/multi_modal_feature_description"
  "feature_description/sae_agentic_explainer"
)

# Prompt sent to claude after the TUI is ready.
USER_PROMPT='做实验验证task.md中的claim'

# Command used to boot claude inside the tmux pane.
CLAUDE_CMD='claude'

CC_ROOT="/data/zhenqian/Reproduction1/cc"
LOG_DIR="${CC_ROOT}/logs"
ANOMALY_LOG="${LOG_DIR}/anomalies.log"
WATCHER="/data/zhenqian/Reproduction1/script/_claude_watcher.sh"

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
mkdir -p "${LOG_DIR}"

launch_one() {
  local rel="$1"                       # e.g. "belief/closing_gap_belief"
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

echo "Batch launcher (cc reproduction tasks — first 7) — ${#EXPERIMENTS[@]} experiments queued"
echo "Working root:  ${CC_ROOT}"
echo "Anomaly log:   ${ANOMALY_LOG}"
echo "Watcher:       ${WATCHER}"
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
echo "Attach a session:  tmux attach -t cc_<subfolder>"
echo "Kill a session:    tmux kill-session -t cc_<subfolder>"
