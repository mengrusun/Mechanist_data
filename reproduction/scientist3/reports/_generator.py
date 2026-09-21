#!/usr/bin/env python3
"""
Generate English Markdown reproduction reports for scientist3.

Iterates over /data/zhenqian/Reproduction/scientist3/<category>/<task_id>/
(instead of the timestamped experiments/ tree), reuses task.md from the
AI-Scientist-v2 repro tasks for claim/resources extraction, and writes:

  <SCIENTIST3>/reports/<task_id>.md
  <SCIENTIST3>/reports/INDEX.md

For tasks without complete completion markers (token_tracker.json /
auto_plot_aggregator.py), pipeline_status still shows what stages/nodes were
actually reached, so the reports handle the killed / stalled runs gracefully.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- allow importing ai_scientist repro helpers ---
AI_SCI_ROOT = Path("/data/zhenqian/exp/AI-Scientist-v2-main")
if str(AI_SCI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SCI_ROOT))

from repro_task_paths import discover_task_mds, task_id_from_md  # noqa: E402

# --- scientist3 layout ---
SCIENTIST3_ROOT = Path("/data/zhenqian/Reproduction/scientist3")
REPORTS_DIR = SCIENTIST3_ROOT / "reports"

# Same stage vocabulary as the original generator
STAGE_ORDER = [
    ("stage_1", "Stage 1 — Initial implementation"),
    ("stage_2", "Stage 2 — Baseline tuning"),
    ("stage_3", "Stage 3 — Creative research"),
    ("stage_4", "Stage 4 — Ablation studies"),
]
SUMMARY_FILES = {
    "draft_summary.json": "Overall draft summary (best starting point for what was run)",
    "research_summary.json": "Research-stage summary with best-node metrics",
    "baseline_summary.json": "Baseline tuning summary",
    "ablation_summary.json": "Ablation studies summary",
}


@dataclass
class TaskMeta:
    task_id: str
    task_md_path: Path
    title: str
    claims: list[str]
    resources_text: str
    models: list[str]
    datasets: list[str]
    data_dir: str | None = None
    model_dir: str | None = None


@dataclass
class ExperimentRun:
    path: Path
    task_id: str
    category: str
    idea: dict
    bfts_config: dict
    stages: list[dict] = field(default_factory=list)
    summaries: dict[str, dict] = field(default_factory=dict)
    plot_count: int = 0
    pipeline_status: str = "unknown"
    has_token_tracker: bool = False
    has_aggregator: bool = False


# ---------------- parse task.md (verbatim from original) ----------------
def parse_task_md(md_path: Path) -> TaskMeta:
    text = md_path.read_text(encoding="utf-8")
    task_id = task_id_from_md(md_path)
    title_m = re.match(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else task_id

    claims: list[str] = []
    claim_m = re.search(r"^##\s+Claim\s*$", text, re.MULTILINE | re.IGNORECASE)
    if claim_m:
        rest = text[claim_m.end():]
        next_sec = re.search(r"^##\s+", rest, re.MULTILINE)
        claim_body = rest[: next_sec.start()] if next_sec else rest
        for line in claim_body.splitlines():
            s = line.strip()
            if s.startswith("- ") or s.startswith("* "):
                claims.append(s[2:].strip())

    resources_text = ""
    res_m = re.search(r"^##\s+Resources\s*$", text, re.MULTILINE | re.IGNORECASE)
    if res_m:
        rest = text[res_m.end():]
        next_sec = re.search(r"^##\s+", rest, re.MULTILINE)
        resources_text = (rest[: next_sec.start()] if next_sec else rest).strip()

    data_dir = None
    model_dir = None
    for pat, var in [
        (r"DATA_DIR\s*=\s*(\S+)", "data_dir"),
        (r"MODEL_DIR\s*=\s*(\S+)", "model_dir"),
    ]:
        m = re.search(pat, resources_text)
        if m:
            if var == "data_dir":
                data_dir = m.group(1)
            else:
                model_dir = m.group(1)

    models: list[str] = []
    datasets: list[str] = []
    in_models = in_datasets = False
    for line in resources_text.splitlines():
        if re.match(r"^-\s+\*\*models?\*\*", line, re.I) or re.match(r"^-\s+\*\*target llms?\*\*", line, re.I):
            in_models, in_datasets = True, False
            continue
        if re.match(r"^-\s+\*\*datasets?\*\*", line, re.I) or re.match(r"^-\s+\*\*probe datasets?\*\*", line, re.I):
            in_models, in_datasets = False, True
            continue
        if line.strip().startswith("- **") and not line.strip().startswith("  "):
            in_models = in_datasets = False
        s = line.strip()
        if s.startswith("- ") and (in_models or in_datasets):
            item = re.sub(r"^\-\s+", "", s)
            item = re.sub(r"\*\*", "", item).strip()
            if item:
                (models if in_models else datasets).append(item)

    return TaskMeta(
        task_id=task_id,
        task_md_path=md_path,
        title=title,
        claims=claims,
        resources_text=resources_text,
        models=models,
        datasets=datasets,
        data_dir=data_dir,
        model_dir=model_dir,
    )


def build_task_index() -> dict[str, TaskMeta]:
    return {parse_task_md(md).task_id: parse_task_md(md) for md in discover_task_mds()}


def load_yaml(p: Path):
    if not p.exists(): return None
    try:
        import yaml
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def load_json(p: Path):
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_stages(run_dir: Path) -> list[dict]:
    log0 = run_dir / "logs" / "0-run"
    if not log0.exists(): return []
    stages = []
    for st in sorted(log0.glob("stage_*")):
        if not st.is_dir(): continue
        progress = load_json(st / "notes" / "stage_progress.json") or {}
        journal = load_json(st / "journal.json")
        nc = 0
        if isinstance(journal, dict) and "nodes" in journal:
            nc = len(journal["nodes"])
        elif isinstance(journal, list):
            nc = len(journal)
        # compute clean/buggy from journal if progress didn't have it
        good = buggy = 0
        if isinstance(journal, dict):
            nodes = journal.get("nodes", [])
        elif isinstance(journal, list):
            nodes = journal
        else:
            nodes = []
        for n in nodes:
            if isinstance(n, dict):
                if n.get("is_buggy"): buggy += 1
                else: good += 1
        best_solutions = list(st.glob("best_solution*.py"))
        stages.append({
            "dir_name": st.name,
            "path": st.relative_to(SCIENTIST3_ROOT),
            "progress": progress,
            "node_count": nc,
            "good_nodes": good,
            "buggy_nodes": buggy,
            "best_solution_count": len(best_solutions),
        })
    return stages


def infer_pipeline_status(exp_dir: Path, stages: list[dict], summaries: dict) -> str:
    has_tk = (exp_dir / "token_tracker.json").exists()
    has_agg = (exp_dir / "auto_plot_aggregator.py").exists()
    stage_nums = set()
    for s in stages:
        m = re.match(r"stage_(\d+)", s["dir_name"])
        if m:
            stage_nums.add(int(m.group(1)))
    if not stage_nums:
        return "not_started (no stage_ dirs)"
    if has_tk and has_agg:
        if stage_nums >= {1, 2, 3, 4}:
            return "completed (stages 1–4 + finalization markers)"
        return f"finalized (stages {sorted(stage_nums)}, aggregate+tracker present)"
    if "draft_summary.json" in summaries:
        return f"partial (stages {sorted(stage_nums)}, summaries but no top-level markers)"
    return f"incomplete (stages {sorted(stage_nums)}, no finalization; likely killed mid-run)"


def extract_models_from_config(cfg: dict) -> dict[str, str]:
    agent = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
    out = {}
    for role in ("code", "feedback", "summary", "select_node", "vlm_feedback"):
        b = agent.get(role, {})
        if isinstance(b, dict) and b.get("model"):
            out[role] = b["model"]
    report = cfg.get("report", {}) if isinstance(cfg, dict) else {}
    if isinstance(report, dict) and report.get("model"):
        out["report"] = report["model"]
    return out


def heuristic_verdict(summaries: dict, pipeline_status: str, claims: list[str]) -> tuple[str, list[dict]]:
    text_parts = []
    for key in ("draft_summary.json", "research_summary.json", "baseline_summary.json"):
        s = summaries.get(key)
        if isinstance(s, dict):
            for field in ("Significance", "Experiment_description", "Description"):
                if s.get(field):
                    text_parts.append(str(s[field]))
    blob = " ".join(text_parts).lower()

    if "incomplete" in pipeline_status or "not_started" in pipeline_status:
        overall = ("**Not assessable via automation** — pipeline stopped before finalization; "
                   "reports below list what was actually produced (best_solutions, exp_results) "
                   "so a human can still eyeball claim support.")
    elif any(w in blob for w in ("inconclusive", "null result", "not statistically")):
        overall = "**Partially supported / inconclusive** — experiments ran but summaries indicate weak or mixed evidence relative to the original claims."
    elif any(w in blob for w in ("contradict", "failed to support", "does not support", "refute")):
        overall = "**Not supported** — agent-reported findings contradict or fail to support the target claims."
    elif any(w in blob for w in ("support", "confirm", "corroborate", "validates", "demonstrates", "establishes", "outperform")):
        overall = "**Partially or fully supported** — summaries suggest supportive evidence, but human review against the source paper is still required."
    else:
        overall = "**Requires human review** — automated scan could not classify support level from available summaries."

    per_claim = []
    for i, claim in enumerate(claims, 1):
        per_claim.append({
            "index": i,
            "claim": claim,
            "verdict": overall if len(claims) == 1 else f"See overall verdict (claim {i}/{len(claims)}).",
        })
    if not claims:
        per_claim.append({"index": 1, "claim": "(no Claim bullets in task.md)", "verdict": overall})
    return overall, per_claim


def render_report(run: ExperimentRun, task: TaskMeta) -> str:
    idea = run.idea
    rel_exp = run.path.relative_to(SCIENTIST3_ROOT)
    models_cfg = extract_models_from_config(run.bfts_config)
    overall, per_claim = heuristic_verdict(run.summaries, run.pipeline_status, task.claims)

    L: list[str] = []
    L.append(f"# Reproduction Report: `{run.task_id}`")
    L.append("")
    L.append(f"- **Category:** `{run.category}`")
    L.append(f"- **Experiment directory:** `scientist3/{rel_exp}`")
    L.append(f"- **Task definition:** `{task.task_md_path.name}` (see AI-Scientist-v2 reproduction_tasks)")
    L.append("")

    L.append("## 1. Global Reproduction Overview")
    L.append("")
    L.append(f"**Theme / hypothesis:** {task.title}")
    L.append("")
    L.append(f"**Pipeline status:** {run.pipeline_status}")
    if not run.has_token_tracker or not run.has_aggregator:
        L.append("")
        L.append("> ⚠ This run was terminated before natural completion (dmxapi hang / BFTS "
                 "internal stall / manual kill). Artifacts up to the last completed BFTS "
                 "node are preserved; final aggregate + token_tracker markers are missing.")
    L.append("")
    L.append("**BFTS stages observed:**")
    if not run.stages:
        L.append("- No stage directories found under `logs/0-run/`.")
    else:
        for s in run.stages:
            L.append(
                f"- `{s['dir_name']}` — nodes: {s['node_count']} "
                f"(good: {s['good_nodes']}, buggy: {s['buggy_nodes']}, "
                f"best_solutions: {s['best_solution_count']})"
            )
    L.append("")
    L.append("**Models used in this run (from `bfts_config.yaml`):**")
    if models_cfg:
        for role, m in sorted(models_cfg.items()):
            L.append(f"- {role}: `{m}`")
    else:
        L.append("- (config not found; default project config is `claude-opus-4-7` for agent roles)")
    L.append("")
    L.append("**Writeup / review flags:** launch used `--skip_writeup --skip_review` (no PDF; in-run bug review only).")
    L.append("")

    L.append("## 2. Verdict: Do Results Support the Claims?")
    L.append("")
    L.append("### Overall (automated heuristic)")
    L.append("")
    L.append(overall)
    L.append("")
    draft = run.summaries.get("draft_summary.json")
    if isinstance(draft, dict) and draft.get("Significance"):
        L.append("### Agent summary excerpt (`Significance` from draft_summary.json)")
        L.append("")
        L.append(f"> {draft['Significance'].strip()}")
        L.append("")
    L.append("### Per-claim verdict")
    L.append("")
    L.append("| # | Verdict |")
    L.append("|---|---------|")
    for row in per_claim:
        short = row["claim"][:80] + ("…" if len(row["claim"]) > 80 else "")
        L.append(f"| {row['index']} | {row['verdict']} — *{short}* |")
    L.append("")

    L.append("## 3. Where to Find Artifacts")
    L.append("")
    L.append("| What | Path |")
    L.append("|------|------|")
    L.append(f"| Idea snapshot | `scientist3/{rel_exp / 'idea.json'}` |")
    L.append(f"| Human-readable idea | `scientist3/{rel_exp / 'idea.md'}` |")
    L.append(f"| BFTS config copy | `scientist3/{rel_exp / 'bfts_config.yaml'}` |")
    if (run.path / "logs" / "0-run" / "unified_tree_viz.html").exists():
        L.append(f"| Tree visualization | `scientist3/{rel_exp / 'logs/0-run/unified_tree_viz.html'}` |")
    L.append(f"| Plots & run outputs ({run.plot_count} files) | `scientist3/{rel_exp / 'logs/0-run/experiment_results/'}` |")
    for fname, desc in SUMMARY_FILES.items():
        p = rel_exp / "logs" / "0-run" / fname
        mark = "✓" if fname in run.summaries else "—"
        L.append(f"| {desc} [{mark}] | `scientist3/{p}` |")
    L.append(f"| Completion markers | token_tracker.json: {'✓' if run.has_token_tracker else '—'}, "
             f"auto_plot_aggregator.py: {'✓' if run.has_aggregator else '—'} |")
    L.append("")

    L.append("## 4. Per-Claim Experimental Configuration")
    L.append("")
    if not task.claims:
        L.append("_No `## Claim` bullets found in task.md; using Short Hypothesis from idea.json._")
        claims = [idea.get("Short Hypothesis", "")]
    else:
        claims = task.claims

    draft_desc = ""
    if isinstance(draft, dict):
        draft_desc = draft.get("Experiment_description") or draft.get("Description") or ""

    for i, claim in enumerate(claims, 1):
        L.append(f"### Claim {i}")
        L.append("")
        L.append("**Original claim (verbatim from task.md):**")
        L.append("")
        L.append(f"> {claim}")
        L.append("")
        L.append(f"**Source:** `{task.task_md_path.name}` → section `## Claim`")
        L.append("")
        L.append("**Configured resources (from task.md `## Resources`):**")
        L.append("")
        if task.data_dir:
            L.append(f"- DATA_DIR: `{task.data_dir}`")
        if task.model_dir:
            L.append(f"- MODEL_DIR: `{task.model_dir}`")
        if task.models:
            L.append("- Models listed in task:")
            for m in task.models:
                L.append(f"  - {m}")
        if task.datasets:
            L.append("- Datasets listed in task:")
            for d in task.datasets:
                L.append(f"  - {d}")
        if not (task.models or task.datasets):
            L.append("- (see full Resources section in task.md)")
        L.append("")
        L.append("**What the agent actually ran (from draft_summary, if available):**")
        L.append("")
        if draft_desc:
            L.append(f"> {draft_desc.strip()}")
        else:
            L.append("> _No draft_summary.json — inspect stage journals and experiment_results._")
        L.append("")

    L.append("---")
    L.append("*Generated by `generate_scientist3_reports.py`. Heuristic verdicts are not peer review.*")
    L.append("")
    return "\n".join(L)


def load_run(exp_dir: Path, task_index: dict[str, TaskMeta]) -> ExperimentRun | None:
    idea_path = exp_dir / "idea.json"
    if not idea_path.exists():
        return None
    idea = json.loads(idea_path.read_text(encoding="utf-8"))
    if isinstance(idea, list):
        idea = idea[0]

    task_id = exp_dir.name  # scientist3 uses task_id directly as folder name
    category = exp_dir.parent.name

    bfts = load_yaml(exp_dir / "bfts_config.yaml") or {}

    log0 = exp_dir / "logs" / "0-run"
    summaries = {}
    for fname in SUMMARY_FILES:
        data = load_json(log0 / fname)
        if isinstance(data, dict):
            summaries[fname] = data

    stages = collect_stages(exp_dir)
    pipeline_status = infer_pipeline_status(exp_dir, stages, summaries)

    plot_count = 0
    er = log0 / "experiment_results"
    if er.exists():
        plot_count = sum(1 for _ in er.rglob("*") if _.is_file())

    return ExperimentRun(
        path=exp_dir,
        task_id=task_id,
        category=category,
        idea=idea,
        bfts_config=bfts,
        stages=stages,
        summaries=summaries,
        plot_count=plot_count,
        pipeline_status=pipeline_status,
        has_token_tracker=(exp_dir / "token_tracker.json").exists(),
        has_aggregator=(exp_dir / "auto_plot_aggregator.py").exists(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPORTS_DIR))
    args = parser.parse_args()

    task_index = build_task_index()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover scientist3/<cat>/<tid>/ dirs (must have idea.json)
    exp_dirs = []
    for cat in sorted(SCIENTIST3_ROOT.iterdir()):
        if not cat.is_dir() or cat.name in ("reports",) or cat.name.startswith("."):
            continue
        for tid in sorted(cat.iterdir()):
            if tid.is_dir() and (tid / "idea.json").exists():
                exp_dirs.append(tid)

    runs: list[ExperimentRun] = []
    for d in exp_dirs:
        r = load_run(d, task_index)
        if r:
            runs.append(r)
        else:
            print(f"[SKIP] {d}")

    idx = [
        "# scientist3 Reproduction Reports Index",
        "",
        f"Reports generated for **{len(runs)}** paper(s) under `scientist3/`.",
        "",
        "| Category | Task ID | Pipeline Status | Report |",
        "|---|---|---|---|",
    ]
    for run in sorted(runs, key=lambda r: (r.category, r.task_id)):
        task = task_index.get(run.task_id)
        if not task:
            print(f"[WARN] no task.md for task_id={run.task_id}")
            continue
        report = render_report(run, task)
        out_path = out_dir / f"{run.task_id}.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"[OK] {out_path.relative_to(SCIENTIST3_ROOT)}")
        idx.append(f"| `{run.category}` | `{run.task_id}` | {run.pipeline_status} | [report]({run.task_id}.md) |")

    (out_dir / "INDEX.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print(f"[OK] {(out_dir / 'INDEX.md').relative_to(SCIENTIST3_ROOT)}")


if __name__ == "__main__":
    main()
