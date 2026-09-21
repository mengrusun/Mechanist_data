#!/usr/bin/env python3
"""Aggregate human-judge review JSONs into a per-(experiment, scientist) score table.

Applies the SAME scoring strategy as llm_judge/_orchestrator/score_stats
(see scoring_rules/scoring_rules.md):

- dim1..dim8: each out of 5, take the `score`.
- dim9 = mean of the three orthogonal sub-dimensions (method / experiment / result),
  where each sub-dim is the mean over all claims (skipping n/a claims).
- score handling: number -> as is; 0 -> counts as 0; "n/a"/null -> dropped
  (removed from BOTH numerator and denominator; max_base drops accordingly).
- raw = Σ(applicable dim1..8) + dim9_final; max_base = 5*n_applicable(1..8) + (5 if dim9 applicable else 0).
- total(0-100) = raw / max_base * 100.
- multiple judge files for one (exp, scientist): compute each file, then average per metric.

Two papers were NOT actually evaluated (template files, all scores null):
  feature_description/sae_agentic_explainer, safety/encode_harmfulness_refusal.
For those, every dimension is filled with a PLACEHOLDER = the per-scientist mean of that
dimension across all evaluated experiments (so each scientist's mean is left undistorted).
Placeholder rows are flagged with placeholder=True and n=0.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("REPRODUCTION_ROOT", HERE.parents[3] / "reproduction"))
SCI_RE = re.compile(r"scientist([123])")

# category label per top-level dir (matches llm_scores.md "类别" column; dir name as-is)
DIMS_1_8 = list(range(1, 9))


def parse_score(v):
    """Return ('num', float) | ('na', None) for one raw score value."""
    if v is None:
        return ("na", None)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("n/a", "na", "n\\a", ""):
            return ("na", None)
        try:
            return ("num", float(s))
        except ValueError:
            return ("na", None)
    return ("na", None)


def dim_key(doc, i):
    for k in doc:
        if k.startswith(f"dimension_{i}_"):
            return k
    return None


def compute_file(doc):
    """Compute metrics for a single review JSON. Returns dict or None if all-null (template)."""
    d18 = {}
    n_num = 0
    for i in DIMS_1_8:
        k = dim_key(doc, i)
        kind, val = parse_score(doc.get(k, {}).get("score") if k else None)
        if kind == "num":
            d18[i] = val
            n_num += 1
        else:
            d18[i] = None

    # dim9
    dim9 = doc.get("dimension_9_reproduction_fidelity", {})
    per_claim = dim9.get("per_claim", {}) if isinstance(dim9, dict) else {}
    sub_vals = {"method": [], "experiment": [], "result": []}
    for _cid, claim in per_claim.items():
        if not isinstance(claim, dict):
            continue
        for sub in sub_vals:
            node = claim.get(sub)
            raw = node.get("score") if isinstance(node, dict) else node
            kind, val = parse_score(raw)
            if kind == "num":
                sub_vals[sub].append(val)
    sub_avg = {s: (sum(v) / len(v) if v else None) for s, v in sub_vals.items()}
    applicable_subs = [a for a in sub_avg.values() if a is not None]
    dim9_final = sum(applicable_subs) / len(applicable_subs) if applicable_subs else None

    # template detection: no numeric dim1..8 AND no dim9 sub scores at all
    if n_num == 0 and dim9_final is None:
        return None

    max_base = 5 * n_num + (5 if dim9_final is not None else 0)
    raw = sum(v for v in d18.values() if v is not None) + (dim9_final or 0.0)
    total = raw / max_base * 100 if max_base > 0 else None
    return {
        "d": d18,
        "sub_avg": sub_avg,
        "dim9_final": dim9_final,
        "raw": raw,
        "max_base": max_base,
        "total": total,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def avg_files(file_metrics):
    """Average a list of per-file metric dicts into one cell (rule §7)."""
    d = {i: mean([fm["d"][i] for fm in file_metrics]) for i in DIMS_1_8}
    sub = {s: mean([fm["sub_avg"][s] for fm in file_metrics]) for s in ("method", "experiment", "result")}
    return {
        "d": d,
        "sub_avg": sub,
        "dim9_final": mean([fm["dim9_final"] for fm in file_metrics]),
        "raw": mean([fm["raw"] for fm in file_metrics]),
        "max_base": mean([fm["max_base"] for fm in file_metrics]),
        "total": mean([fm["total"] for fm in file_metrics]),
    }


def main():
    # discover experiments: leaf dirs two levels below ROOT with json files
    exp_dirs = sorted(
        p for p in ROOT.glob("*/*") if p.is_dir() and list(p.glob("*.json"))
    )

    # cells[(cat, exp, sci)] = {"metrics": aggregated, "n": nfiles, "placeholder": bool}
    evaluated = {}   # keyed as above, only real
    unevaluated_exps = []  # (cat, exp)

    for ed in exp_dirs:
        cat, exp = ed.parent.name, ed.name
        by_sci = {1: [], 2: [], 3: []}
        for jf in sorted(ed.glob("*.json")):
            m = SCI_RE.search(jf.name)
            if not m:
                continue
            doc = json.loads(jf.read_text(encoding="utf-8"))
            fm = compute_file(doc)
            by_sci[int(m.group(1))].append(fm)
        # experiment is "unevaluated" if every file across all scientists is a template (None)
        all_files = [fm for lst in by_sci.values() for fm in lst]
        real_files = [fm for fm in all_files if fm is not None]
        if not real_files:
            unevaluated_exps.append((cat, exp))
            continue
        for sci, lst in by_sci.items():
            real = [fm for fm in lst if fm is not None]
            if not real:
                continue
            evaluated[(cat, exp, sci)] = {"metrics": avg_files(real), "n": len(real)}

    # placeholder = per-scientist mean of each dimension across evaluated cells
    def sci_dim_mean(sci, getter):
        return mean([getter(v["metrics"]) for (c, e, s), v in evaluated.items() if s == sci])

    placeholder_rows = []
    for cat, exp in unevaluated_exps:
        for sci in (1, 2, 3):
            d = {i: sci_dim_mean(sci, lambda m, i=i: m["d"][i]) for i in DIMS_1_8}
            sub = {s: sci_dim_mean(sci, lambda m, s=s: m["sub_avg"][s]) for s in ("method", "experiment", "result")}
            subs_ok = [a for a in sub.values() if a is not None]
            dim9_final = sum(subs_ok) / len(subs_ok) if subs_ok else None
            n_app = sum(1 for v in d.values() if v is not None)
            max_base = 5 * n_app + (5 if dim9_final is not None else 0)
            raw = sum(v for v in d.values() if v is not None) + (dim9_final or 0.0)
            total = raw / max_base * 100 if max_base else None
            placeholder_rows.append(
                {"cat": cat, "exp": exp, "sci": sci, "n": 0, "placeholder": True,
                 "metrics": {"d": d, "sub_avg": sub, "dim9_final": dim9_final,
                             "raw": raw, "max_base": max_base, "total": total}}
            )

    # assemble ordered rows: evaluated (in exp_dir order) then placeholders, both by sci
    rows = []
    seen_exp_order = []
    for ed in exp_dirs:
        cat, exp = ed.parent.name, ed.name
        if (cat, exp) in unevaluated_exps:
            continue
        seen_exp_order.append((cat, exp))
        for sci in (1, 2, 3):
            cell = evaluated.get((cat, exp, sci))
            if not cell:
                continue
            rows.append({"cat": cat, "exp": exp, "sci": sci, "n": cell["n"],
                         "placeholder": False, "metrics": cell["metrics"]})
    rows.extend(placeholder_rows)

    # ---- write CSV ----
    def r2(x):
        return "" if x is None else round(x, 2)

    header = ["类别", "实验", "sci", "n", "placeholder", "总分(0-100)", "原始总分", "满分基准",
              "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "前8小计",
              "dim9_method", "dim9_exp", "dim9_result", "dim9最终"]
    csv_path = HERE / "human_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            m = r["metrics"]
            d = m["d"]
            sub8 = sum(v for v in d.values() if v is not None)
            w.writerow([
                r["cat"], r["exp"], f"s{r['sci']}", r["n"], "yes" if r["placeholder"] else "",
                r2(m["total"]), r2(m["raw"]), r2(m["max_base"]),
                *[r2(d[i]) for i in DIMS_1_8], r2(sub8),
                r2(m["sub_avg"]["method"]), r2(m["sub_avg"]["experiment"]),
                r2(m["sub_avg"]["result"]), r2(m["dim9_final"]),
            ])

    # ---- write MD ----
    md_path = HERE / "human_scores.md"
    lines = []
    lines.append("# Human Judge 分数统计")
    lines.append("")
    lines.append("**计分规则**：与 `llm_judge/_orchestrator/score_stats` 完全一致（见 `scoring_rules/scoring_rules.md`）。"
                 "dim1–8 每维满分 5；dim9 = method/experiment/result 三个正交子维各自对所有 claim 取均分后再取三者平均。"
                 "原始总分 = Σ(dim1–8) + dim9最终；总分 = 原始总分 / 满分基准 × 100。")
    lines.append("**n/a**：剔除并重新归一化。 **score=0**：按 0 计入。 多专家格取平均（`n` 列为文件数）。")
    lines.append("")
    lines.append(f"**占位符**：以下 {len(unevaluated_exps)} 篇论文尚未评估（模板文件、打分全为 null），"
                 + "、".join(f"`{c}/{e}`" for c, e in unevaluated_exps)
                 + "。其每个维度暂填 **该 scientist 在所有已评估实验上的该维度均值** 作为占位符"
                 "（`placeholder=yes`、`n=0`；此填法不改变各 scientist 的维度均值）。")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        m = r["metrics"]
        d = m["d"]
        sub8 = sum(v for v in d.values() if v is not None)
        cells = [r["cat"], r["exp"], f"s{r['sci']}", str(r["n"]),
                 "yes" if r["placeholder"] else "",
                 str(r2(m["total"])), str(r2(m["raw"])), str(r2(m["max_base"])),
                 *[str(r2(d[i])) for i in DIMS_1_8], str(r2(sub8)),
                 str(r2(m["sub_avg"]["method"])), str(r2(m["sub_avg"]["experiment"])),
                 str(r2(m["sub_avg"]["result"])), str(r2(m["dim9_final"]))]
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- console summary ----
    print(f"experiments discovered : {len(exp_dirs)}")
    print(f"unevaluated (placeholder): {unevaluated_exps}")
    print(f"evaluated cells         : {len(evaluated)}")
    print(f"placeholder cells       : {len(placeholder_rows)}")
    print(f"total rows              : {len(rows)}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")
    # per-scientist mean total (evaluated only, for sanity)
    for sci in (1, 2, 3):
        tot = mean([v["metrics"]["total"] for (c, e, s), v in evaluated.items() if s == sci])
        print(f"  s{sci} mean total (evaluated) = {tot:.1f}" if tot else f"  s{sci}: n/a")


if __name__ == "__main__":
    main()
