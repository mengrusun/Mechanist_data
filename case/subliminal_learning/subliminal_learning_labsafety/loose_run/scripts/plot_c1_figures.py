"""Regenerate the C1 figures from results/ — reflects the established-at-1e-3 verdict.

Reads the per-run QA_I eval JSONs (Ctrl-A / treated / Ctrl-B) and produces:
  figures/C1/c1_lr_seed_delta_A.{png,pdf}   — Δ_A per LR × seed across the sweep
  figures/C1/c1_lr1e-3_dual_leg.{png,pdf}   — established LR: Δ_A and Δ_B per seed

Health = the task.md collapse indicators (loss NaN/divergence, repetition spike);
a general-capability probe drop is a non-gating diagnostic and does NOT mark a run.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures" / "C1"
OUT.mkdir(parents=True, exist_ok=True)

LR_ORDER = ["5e-6", "2e-5", "5e-5", "1e-4", "2e-4", "5e-4", "1e-3", "2e-3"]
SEEDS = [42, 200, 201]
SEED_COLORS = {42: "#4C72B0", 200: "#55A868", 201: "#C44E52"}
CHOSEN_LR = "1e-3"
THRESH = 3.0


def _acc(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("averaged_accuracy")


def load_accs(arm: str) -> dict:
    """{lr: {seed: acc}} for treated/ctrlb; ctrla is single."""
    out: dict = {}
    for lr in LR_ORDER:
        for s in SEEDS:
            a = _acc(RES / f"qai_{arm}_lr{lr}_seed{s}.json")
            if a is not None:
                out.setdefault(lr, {})[s] = a
    return out


def main():
    acc_ctrla = _acc(RES / "qai_ctrla_seed42.json")
    treated = load_accs("treated")
    ctrlb = load_accs("ctrlb")
    print(f"[plot] Ctrl-A={acc_ctrla:.4f}; treated LRs={list(treated)}; ctrlb LRs={list(ctrlb)}")

    # ---- Figure 1: Δ_A per LR × seed --------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.26
    xs = range(len(LR_ORDER))
    for j, s in enumerate(SEEDS):
        vals = [(acc_ctrla - treated.get(lr, {}).get(s)) * 100
                if treated.get(lr, {}).get(s) is not None else 0.0
                for lr in LR_ORDER]
        ax.bar([x + (j - 1) * width for x in xs], vals, width,
               label=f"随机 seed {s}", color=SEED_COLORS[s], edgecolor="black", linewidth=0.4)
    ax.axhline(THRESH, ls="--", color="#2E7D32", lw=1.3, label=f"{THRESH:.0f} 个百分点判定线")
    ax.axhline(0, color="black", lw=0.6)
    # Highlight the established LR band
    ci = LR_ORDER.index(CHOSEN_LR)
    ax.axvspan(ci - 0.5, ci + 0.5, color="#2E7D32", alpha=0.08, zorder=0)
    ax.annotate("稳定下降\n（LR=1e-3）", xy=(ci, 40), ha="center", va="bottom",
                fontsize=9, color="#2E7D32", fontweight="bold")
    di = LR_ORDER.index("2e-3")
    ax.annotate("训练退化\n（学生模型正确率接近 0）", xy=(di, 45), ha="center", va="center",
                fontsize=8, color="#555555",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#BBBBBB", boxstyle="round,pad=0.3"))
    ax.set_xticks(list(xs))
    ax.set_xticklabels(LR_ORDER)
    ax.set_xlabel("学生模型学习率")
    ax.set_ylabel("安全正确率下降（百分点）")
    ax.set_title("不同学习率下的安全正确率下降")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"c1_lr_seed_delta_A.{ext}", dpi=150)
    plt.close(fig)

    # ---- Dual-leg figures (Δ_A and Δ_B per seed at a given LR) -------------
    def dual_leg(lr: str, fname: str, title: str):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        width = 0.38
        xs = range(len(SEEDS))
        da = [(acc_ctrla - treated.get(lr, {}).get(s)) * 100 for s in SEEDS]
        db = [(ctrlb.get(lr, {}).get(s) - treated.get(lr, {}).get(s)) * 100
              if ctrlb.get(lr, {}).get(s) is not None else 0.0 for s in SEEDS]
        ax.bar([x - width / 2 for x in xs], da, width, label="相对原始模型的下降",
               color="#4C72B0", edgecolor="black", linewidth=0.4)
        ax.bar([x + width / 2 for x in xs], db, width, label="相对安全教师对照的下降",
               color="#DD8452", edgecolor="black", linewidth=0.4)
        ax.axhline(THRESH, ls="--", color="#2E7D32", lw=1.3, label=f"{THRESH:.0f} 个百分点判定线")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"seed {s}" for s in SEEDS])
        ax.set_ylabel("安全正确率下降（百分点）")
        ax.set_title(title)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(axis="y", ls=":", alpha=0.4)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{fname}.{ext}", dpi=150)
        plt.close(fig)
        return da, db

    da13, db13 = dual_leg("1e-3", "c1_lr1e-3_dual_leg",
                          "LR=1e-3：三个 seed 下的安全正确率下降")

    # ---- refresh the machine-readable data snapshot -----------------------
    snap = {
        "ctrla_acc": acc_ctrla,
        "treated_acc": treated,
        "ctrlb_acc": ctrlb,
        "lr_order": LR_ORDER,
        "chosen_lr": CHOSEN_LR,
        "threshold_pp": THRESH,
        "established_lr1e3_delta_a_pp": {s: (acc_ctrla - treated["1e-3"][s]) * 100 for s in SEEDS},
        "established_lr1e3_delta_b_pp": {s: (ctrlb["1e-3"][s] - treated["1e-3"][s]) * 100 for s in SEEDS},
        "note": "health = task.md collapse indicators only; capability-probe drop is a non-gating diagnostic",
    }
    (OUT / "_data.json").write_text(json.dumps(snap, indent=2))
    print(f"[plot] wrote figures + _data.json to {OUT}")


if __name__ == "__main__":
    main()
