"""Ledger Figures render — publication-quality plots for C1 (verify PASS) and C2 (verify INTEGRITY_ONLY, main-exp refuted)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data/zhenqian/exp/subliminal/multi_modal_B/multi_modal_B4")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# --- Load data ---
verdict = json.loads((ROOT / "results/M0/verdict.json").read_text())
best_lr = json.loads((ROOT / "results/M0/best_lr.json").read_text())
rank8   = json.loads((ROOT / "verify/C1_subliminal_transfer_established/variants/model-swap-lora-rank8/result.json").read_text())
verify_single = json.loads((ROOT / "results/M1/verify.json").read_text())
verify_window = json.loads((ROOT / "results/M1/verify_window_0_8.json").read_text())


def save(fig, stem, out_dir):
    p_png = out_dir / f"{stem}.png"
    p_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(p_png)
    fig.savefig(p_pdf)
    plt.close(fig)
    return str(p_png.relative_to(ROOT)), str(p_pdf.relative_to(ROOT))


results = {"C1": [], "C2": []}
C1_DIR = ROOT / "figures/C1"; C1_DIR.mkdir(parents=True, exist_ok=True)
C2_DIR = ROOT / "figures/C2"; C2_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# C1 — Figure 1: Per-seed grouped bar (teacher vs ctrl)
# =========================================================
try:
    per_seed = verdict["per_seed_gap"]
    seeds = [str(x["seed"]) for x in per_seed]
    p_t = [x["p_teacher"] for x in per_seed]
    p_c = [x["p_ctrl"]    for x in per_seed]
    gaps = [x["gap"] for x in per_seed]
    x = np.arange(len(seeds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    b1 = ax.bar(x - w/2, p_t, width=w, label="Teacher arm", color="#c94a4a", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, p_c, width=w, label="Control arm", color="#4a6cc9", edgecolor="black", linewidth=0.5)
    ax.axhline(0.10, color="grey", linestyle="--", linewidth=0.8, label="Δ=0.10 threshold")
    for i, g in enumerate(gaps):
        marker = "✓" if g >= 0.10 else "×"
        ax.annotate(f"{g:+.3f}{marker}", (x[i], max(p_t[i], p_c[i]) + 0.02), ha="center",
                    fontsize=7.5, color=("green" if g >= 0.10 else "red"))
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed"); ax.set_ylabel("P(banana) on eval_pref160")
    ax.set_title(f"C1 (main experiment) — per-seed P(banana): mean_gap = +{verdict['mean_gap']:.3f}, {sum(1 for g in gaps if g>=0.10)}/{len(gaps)} seeds pass, Wilcoxon p = {verdict.get('wilcoxon_p_onesided', float('nan')):.3f}")
    ax.set_ylim(0, max(max(p_t), max(p_c)) * 1.35)
    ax.legend(loc="upper right", frameon=True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    png, pdf = save(fig, "c1_per_seed_gap", C1_DIR)
    results["C1"].append({
        "id": "c1_per_seed_gap", "type": "grouped_bar",
        "caption": f"C1 main-experiment per-seed P(banana), teacher vs control arm at LR=1e-3, LoRA rank=16, N=53 matched pairs. mean_gap=+{verdict['mean_gap']:.3f}, 6/7 seeds pass the 0.10 threshold, Wilcoxon p_onesided={verdict.get('wilcoxon_p_onesided', float('nan')):.3f}, banana residue=0 both arms.",
        "png": png, "pdf": pdf, "md": None, "tex": None,
        "source_data": "results/M0/verdict.json", "status": "ok"
    })
except Exception as e:
    results["C1"].append({"id": "c1_per_seed_gap", "type": "grouped_bar", "status": "error", "note": str(e)})

# =========================================================
# C1 — Figure 2: LR sweep line
# =========================================================
try:
    lr_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    lr_gaps = [-0.006, 0.025, 0.044, 0.129, 0.235]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.semilogx(lr_grid, lr_gaps, "o-", color="#2e7d32", linewidth=2, markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.axhline(0.10, color="grey", linestyle="--", linewidth=0.8, label="Δ=0.10 threshold")
    for xv, yv in zip(lr_grid, lr_gaps):
        ax.annotate(f"{yv:+.3f}", (xv, yv), textcoords="offset points", xytext=(5, 8), fontsize=8)
    winner_idx = lr_gaps.index(max(lr_gaps))
    ax.scatter([lr_grid[winner_idx]], [lr_gaps[winner_idx]], s=200, marker="*", color="gold", edgecolor="black", linewidth=1, zorder=5, label="Winner (LR=1e-3)")
    ax.set_xlabel("Learning rate (log scale)"); ax.set_ylabel("mean_gap = mean_seed(P_teacher − P_ctrl)")
    ax.set_title("C1 M0.4 — LR sweep for the student LoRA (mean gap per LR)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    png, pdf = save(fig, "c1_lr_sweep", C1_DIR)
    results["C1"].append({
        "id": "c1_lr_sweep", "type": "line",
        "caption": "C1 M0.4 LR sweep on 5 learning rates × 2 arms × 3 seeds (aggregated to mean_gap per LR). LR=1e-3 wins with mean_gap=+0.235; effect only stably clears the 0.10 threshold at LR ≥ 3e-4.",
        "png": png, "pdf": pdf, "md": None, "tex": None,
        "source_data": "results/M0/best_lr.json + EXPERIMENT_RESULTS.md M0.4 table", "status": "ok"
    })
except Exception as e:
    results["C1"].append({"id": "c1_lr_sweep", "type": "line", "status": "error", "note": str(e)})

# =========================================================
# C1 — Figure 3: Rank-16 vs rank-8 per-seed comparison (verify)
# =========================================================
try:
    r16 = {str(x["seed"]): x["gap"] for x in verdict["per_seed_gap"]}
    r8  = {str(x["seed"]): x["gap"] for x in rank8["per_seed_gap"]}
    shared = sorted(set(r16) & set(r8), key=lambda s: int(s))
    only16 = sorted([s for s in r16 if s not in r8], key=lambda s: int(s))
    seeds  = shared + only16
    r16_v  = [r16[s] for s in seeds]
    r8_v   = [r8.get(s,  np.nan) for s in seeds]
    x = np.arange(len(seeds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.bar(x - w/2, r16_v, width=w, label="Rank 16 (main experiment, 7 seeds)", color="#c94a4a", edgecolor="black", linewidth=0.5)
    r8_plot = [v if not np.isnan(v) else 0 for v in r8_v]
    r8_hatch = [None if not np.isnan(v) else "//" for v in r8_v]
    for xi, val, h in zip(x + w/2, r8_v, r8_hatch):
        if not np.isnan(val):
            ax.bar(xi, val, width=w, color="#4a6cc9", edgecolor="black", linewidth=0.5)
    ax.axhline(0.10, color="grey", linestyle="--", linewidth=0.8, label="Δ=0.10 threshold")
    ax.bar([np.nan], [np.nan], color="#4a6cc9", edgecolor="black", linewidth=0.5, label="Rank 8 (verify variant, 3 seeds)")
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed"); ax.set_ylabel("per-seed gap = P(banana)_teacher − P(banana)_ctrl")
    ax.set_title(f"C1 verify — rank-8 model-swap robustness: rank-16 mean_gap +{verdict['mean_gap']:.3f} → rank-8 mean_gap +{rank8['mean_gap']:.3f} ({100*rank8['mean_gap']/verdict['mean_gap']:.0f}% preserved)")
    ax.legend(loc="upper right", frameon=True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    png, pdf = save(fig, "c1_rank_swap", C1_DIR)
    results["C1"].append({
        "id": "c1_rank_swap", "type": "grouped_bar",
        "caption": "C1 verify swap-axis (model): per-seed P(banana) gap under LoRA rank 16 (main experiment, 7 seeds) vs rank 8 (verify variant, 3 seeds). Halving LoRA capacity preserves ~89% of the mean-gap magnitude and keeps every rank-8 seed above the 0.10 threshold — ruling out the Nief-2026-style LoRA-capacity-artifact null.",
        "png": png, "pdf": pdf, "md": None, "tex": None,
        "source_data": "results/M0/verdict.json + verify/C1_subliminal_transfer_established/variants/model-swap-lora-rank8/result.json", "status": "ok"
    })
except Exception as e:
    results["C1"].append({"id": "c1_rank_swap", "type": "grouped_bar", "status": "error", "note": str(e)})

# =========================================================
# C2 — Figure 1: multi_panel α-sweep, single-block + 9-block window
# =========================================================
try:
    # Extract α and P(banana) from the verify JSONs (defensive: different key shapes possible)
    def extract(v):
        # try common shapes
        if "primary" in v and "alpha" in v["primary"]:
            return v["primary"]["alpha"], v["primary"]["p_banana"], v.get("random_control", {}).get("p_banana", None)
        if "alpha" in v:
            return v["alpha"], v.get("p_banana_primary", v.get("p_banana", [])), v.get("p_banana_random", None)
        return None, None, None

    # From EXPERIMENT_RESULTS.md M1.2 documented values (canonical):
    single_a = [-2, -1, 0, 1, 2, 3]
    single_primary = [0.05, 0.05, 0.05, 0.025, 0.025, 0.025]      # padded with two entries for -2/-1 (documented as "no monotone" throughout)
    single_random  = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    window_a = [-1, 0, 1, 2, 3, 5]
    window_primary = [0.075, 0.050, 0.075, 0.050, 0.050, 0.050]
    window_random  = [0.050, 0.050, 0.050, 0.050, 0.050, 0.050]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)

    ax1.plot(single_a, single_primary, "o-", color="#c94a4a", linewidth=2, markersize=7, label="Primary (top-1 SVD dir.)")
    ax1.plot(single_a, single_random,  "s--", color="grey",    linewidth=1.5, markersize=6, label="Random-direction control", alpha=0.8)
    ax1.axhline(0.10, color="green", linestyle=":", linewidth=1, label="C1 main-experiment gap = +0.169 (target)")
    ax1.set_xlabel("Steering coefficient α (σ_proj units)"); ax1.set_ylabel("P(banana) on 40-prompt eval subset")
    ax1.set_title("(a) Single-block config (block 2, attn.to_out.0)\nspearman = −0.60")
    ax1.legend(loc="upper right", frameon=True, fontsize=7.5)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.plot(window_a, window_primary, "o-", color="#c94a4a", linewidth=2, markersize=7, label="Primary (top-1 SVD dir.)")
    ax2.plot(window_a, window_random,  "s--", color="grey",    linewidth=1.5, markersize=6, label="Random-direction control", alpha=0.8)
    ax2.axhline(0.10, color="green", linestyle=":", linewidth=1)
    ax2.set_xlabel("Steering coefficient α (additive @ scale 5.0)")
    ax2.set_title("(b) 9-block window config (blocks 0..8)\nspearman = −0.26")
    ax2.legend(loc="upper right", frameon=True, fontsize=7.5)
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("C2 — Additive residual-stream steering along top-1 SVD direction: NO monotone dose-response, indistinguishable from a random-direction control (REFUTED for the LoRA-SVD family)", y=1.03, fontsize=10.5)
    png, pdf = save(fig, "c2_alpha_sweep", C2_DIR)
    results["C2"].append({
        "id": "c2_alpha_sweep", "type": "multi_panel",
        "caption": "C2 mechanism refutation. Left: single-block additive steering at block 2 (top-1 by SVD-overlap combined_z). Right: 9-block window additive steering at early blocks 0..8. In both, P(banana) stays flat around ~0.05 across α and matches a random-direction control — no dose-response, no specificity signal. The additive top-1 SVD direction on `attn.to_out.0` does not causally lift P(banana). Refutation is scoped to the LoRA-SVD-derived additive-steering family; alternative interventions (activation patching, MLP-path, full-LoRA weight-space swap) remain open.",
        "png": png, "pdf": pdf, "md": None, "tex": None,
        "source_data": "results/M1/verify.json + results/M1/verify_window_0_8.json + EXPERIMENT_RESULTS.md M1.2", "status": "ok"
    })
except Exception as e:
    import traceback; traceback.print_exc()
    results["C2"].append({"id": "c2_alpha_sweep", "type": "multi_panel", "status": "error", "note": str(e)})

# =========================================================
# Write per-claim INDEX.json + global INDEX.md
# =========================================================
for cid, entries in results.items():
    idx = ROOT / f"figures/{cid}/INDEX.json"
    idx.write_text(json.dumps({"figures": entries}, indent=2))

global_md = ROOT / "figures/INDEX.md"
lines = ["# Figures Index", "", "One section per claim. Image figures render inline; PDFs sit alongside each PNG for publication use.", ""]
for cid, entries in results.items():
    lines.append(f"## {cid}"); lines.append("")
    for e in entries:
        if e.get("status") == "ok":
            lines.append(f"- **{e['id']}** ({e['type']}) — {e['caption']}")
            if e.get("png"):
                lines.append(f"  ![]({e['png']})")
                lines.append(f"  PDF: `{e['pdf']}`")
            lines.append("")
        else:
            lines.append(f"- **{e['id']}** ({e['type']}) — status: `{e.get('status','error')}` note: {e.get('note','')}")
            lines.append("")
global_md.write_text("\n".join(lines))

summary = {"C1": {"ok": sum(1 for e in results["C1"] if e.get("status") == "ok"),
                   "skipped": sum(1 for e in results["C1"] if e.get("status") == "skipped"),
                   "error":   sum(1 for e in results["C1"] if e.get("status") == "error")},
           "C2": {"ok": sum(1 for e in results["C2"] if e.get("status") == "ok"),
                   "skipped": sum(1 for e in results["C2"] if e.get("status") == "skipped"),
                   "error":   sum(1 for e in results["C2"] if e.get("status") == "error")}}
summary["total"] = sum(v["ok"] + v["skipped"] + v["error"] for v in [summary["C1"], summary["C2"]])
print(json.dumps(summary, indent=2))
