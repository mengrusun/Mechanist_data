#!/usr/bin/env python3
"""PreToolUse guard: hard-block any tool call that touches a forbidden path.

Registered in .claude/settings.json. Receives the hook payload on stdin:
  {"tool_name": ..., "tool_input": {...}, "cwd": ..., ...}

Exit 0  -> allow
Exit 2  -> block; stderr is fed back to the model as the reason.

This runs outside the model's control, so it holds even if the model is
confused, jailbroken, or running as a subagent.

Experiment isolation: everything is derived from this file's own location, so
the guard keeps working if the experiment folder is renamed, and every *sibling*
experiment folder under the experiment root is forbidden automatically. Only
this experiment's own directory is readable.
"""
import json
import os
import re
import sys

# .../<experiment_dir>/.claude/hooks/guard_paths.py  ->  .../<experiment_dir>
SELF_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
EXPERIMENT_ROOT = os.path.dirname(SELF_DIR)  # /data/wanghaoxiong/Mechanist-DNA-experiment

# Other users' / other projects' trees, plus the ~/.claude backup archives
# (a tarball there contains every other experiment's transcripts).
FORBIDDEN_DIRS = [
    "/data/zhenqian",
    "/data/wmr",
    "/data/wanghaoxiong/Mechanist-DNA",
    "/data/wanghaoxiong/mechanist_frontend",
    "/data/wanghaoxiong/claude-backups",
]


# Every other experiment folder next to this one (prior designs and results).
def sibling_experiment_dirs():
    if not os.path.isdir(EXPERIMENT_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(EXPERIMENT_ROOT)):
        path = os.path.join(EXPERIMENT_ROOT, name)
        if name.startswith("."):
            continue
        if not os.path.isdir(path):
            continue
        if os.path.realpath(path) == SELF_DIR:
            continue  # this experiment — allowed
        out.append(path)
    return out


SIBLINGS = sibling_experiment_dirs()
FORBIDDEN_DIRS = FORBIDDEN_DIRS + SIBLINGS

# Claude session transcripts / history for *other* projects.
CLAUDE_HOME = os.path.expanduser("~/.claude")
PROJECTS_DIR = os.path.join(CLAUDE_HOME, "projects")
ALLOWED_PROJECT_DIRS = {
    SELF_DIR.replace("/", "-").replace("_", "-"),  # this project's own transcripts
    "-data",                                       # memory dir lives here
}
HISTORY_FILES = [
    os.path.join(CLAUDE_HOME, "history.jsonl"),
    os.path.join(CLAUDE_HOME, "file-history"),
    os.path.join(CLAUDE_HOME, "backups"),
    os.path.join(CLAUDE_HOME, "debug"),
]

# Directories that merely *contain* forbidden roots: reading a file inside an
# allowed sub-dir is fine, but pointing a recursive tool (grep -r, Glob) at the
# container itself would sweep up every other experiment / project's files.
CONTAINER_DIRS = [PROJECTS_DIR, CLAUDE_HOME, EXPERIMENT_ROOT]

# Tool-input keys that carry a filesystem path.
PATH_KEYS = ("file_path", "notebook_path", "path", "pattern")


def forbidden_roots():
    roots = [d for d in FORBIDDEN_DIRS if os.path.exists(d)]
    if os.path.isdir(PROJECTS_DIR):
        for name in os.listdir(PROJECTS_DIR):
            if name not in ALLOWED_PROJECT_DIRS:
                roots.append(os.path.join(PROJECTS_DIR, name))
    roots.extend(HISTORY_FILES)
    return roots


ROOTS = forbidden_roots()
# Resolve symlinks once so a symlink-based bypass still lands on a real root.
REAL_ROOTS = [os.path.realpath(r) for r in ROOTS]
REAL_CONTAINERS = [os.path.realpath(d) for d in CONTAINER_DIRS]


def is_forbidden(raw_path, cwd):
    if not raw_path or not isinstance(raw_path, str):
        return None
    p = os.path.expanduser(raw_path)
    if not os.path.isabs(p):
        p = os.path.join(cwd, p)
    for cand in (os.path.normpath(p), os.path.realpath(p)):
        for root in REAL_ROOTS:
            if cand == root or cand.startswith(root + os.sep):
                return root
        for container in REAL_CONTAINERS:
            if cand == container:
                return container
    return None


# Bash is a string, not a path. Match forbidden roots (and the distinctive
# leaf names) anywhere in the command, plus obvious obfuscation vectors.
LEAF_NAMES = [
    "zhenqian", "wmr", "mechanist_frontend", "Mechanist-DNA/", "claude-backups",
] + [
    # sibling experiment folder names, e.g. "simple_20260815" — distinctive
    # enough to match relative references like `../simple_20260815/results`.
    os.path.basename(p) for p in SIBLINGS if len(os.path.basename(p)) >= 8
]
OBFUSCATION = re.compile(
    r"base64\s+(-d|--decode)|"           # base64 -d | sh
    r"\bxxd\s+-r|\becho\s+-e\s+.*\\x|"   # hex-encoded paths
    r"\$\(\s*printf",                    # printf-assembled paths
)


def check_bash(cmd, cwd):
    for root in ROOTS:
        # Boundary-aware: "/data/wanghaoxiong/Mechanist-DNA" must not match the
        # sibling path ".../Mechanist-DNA-experiment/<this experiment>/...".
        if re.search(re.escape(root) + r"(?![\w.+-])", cmd):
            return root
    for leaf in LEAF_NAMES:
        if leaf in cmd:
            return leaf
    # Path-looking tokens get the full realpath treatment (catches symlinks,
    # ../ traversal, and $HOME expansion of the .claude tree).
    for tok in re.findall(r"[~/][\w./~+-]*", cmd):
        hit = is_forbidden(tok, cwd)
        if hit:
            return hit
    if OBFUSCATION.search(cmd):
        return "encoded/obfuscated command (blocked as unverifiable)"
    return None


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # malformed payload: do not wedge the session

    tool = payload.get("tool_name", "")
    ti = payload.get("tool_input") or {}
    cwd = payload.get("cwd") or os.getcwd()

    hit = None
    if tool == "Bash":
        hit = check_bash(ti.get("command", "") or "", cwd)
    else:
        for key in PATH_KEYS:
            hit = is_forbidden(ti.get(key), cwd)
            if hit:
                break
        # Grep/Glob take a search root separately from the pattern.
        if not hit:
            hit = is_forbidden(ti.get("glob"), cwd)

    if hit:
        sys.stderr.write(
            f"BLOCKED by project policy (.claude/hooks/guard_paths.py): this {tool} "
            f"call touches a forbidden location ({hit}).\n"
            "task.md forbids reading prior experiment designs/results and other "
            "users' directories. Do not attempt a workaround — proceed using only "
            "this project's own files.\n"
        )
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
