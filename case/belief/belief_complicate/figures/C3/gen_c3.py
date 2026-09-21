"""Generate C3 figures: behavioral + causal trajectories on pythia-1b intermediate checkpoints."""
import json, os, sys, glob, re
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_plot_style import plt, COLORS, save_fig  # noqa: E402
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _extract_step(dirname, prefix):
    m = re.match(rf"{prefix}(\d+)", os.path.basename(dirname))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# M3.a — behavioral trajectory
# ---------------------------------------------------------------------------
m3a_dirs = sorted(
    glob.glob(os.path.join(PROJECT_ROOT, "runs/M3a_pythia-1b/step*")),
    key=lambda p: _extract_step(p, "step") or 0,
)
frames = ["world_knowledge", "personal_belief", "attributed_belief"]
frame_labels = {"world_knowledge": "world knowledge (control)",
                "personal_belief": "personal belief",
                "attributed_belief": "attributed belief"}

steps_beh = []
acc_beh = {fr: [] for fr in frames}
for d in m3a_dirs:
    step = _extract_step(d, "step")
    rj = os.path.join(d, "results.json")
    if step is None or not os.path.exists(rj):
        continue
    with open(rj) as f:
        r = json.load(f)
    steps_beh.append(step)
    for fr in frames:
        acc_beh[fr].append(r["frames"][fr]["accuracy"])

# Log-safe x-axis (replace step=0 with step=1 to avoid log(0))
steps_beh_arr = np.array(steps_beh, dtype=float)
steps_beh_arr[steps_beh_arr == 0] = 0.5

fig, ax = plt.subplots(figsize=(6.5, 3.5))
for j, fr in enumerate(frames):
    ax.plot(steps_beh_arr, acc_beh[fr], marker="o", markersize=3.5,
            color=COLORS[j], label=frame_labels[fr], linewidth=1.2)

ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel("pythia-1b training step (log scale)")
ax.set_ylabel("accuracy (full belief_core)")
ax.set_ylim(0.0, 1.05)
ax.set_xlim(0.4, max(steps_beh_arr) * 1.4)

# Behavioral-emergence markers
ax.axvline(2000, color=COLORS[2], linestyle=":", linewidth=0.7, alpha=0.7)
ax.text(2000, 1.02, "AB @2k", fontsize=7, color=COLORS[2], ha="center", va="bottom")
ax.axvline(33000, color=COLORS[1], linestyle=":", linewidth=0.7, alpha=0.7)
ax.text(33000, 1.02, "PB @33k", fontsize=7, color=COLORS[1], ha="center", va="bottom")

ax.legend(frameon=False, loc="lower right", fontsize=8)
save_fig(fig, "c3_behavioral_trajectory", formats=("pdf", "png"), out_dir=OUT_DIR)
plt.close(fig)


# ---------------------------------------------------------------------------
# M3.b — causal-intervention trajectory
# HeadSet_personal ablation → PB target_drop; HeadSet_attributed ablation → AB target_drop
# ---------------------------------------------------------------------------
m3b_dirs = sorted(glob.glob(os.path.join(PROJECT_ROOT, "runs/M3b_pythia-1b/step*")),
                  key=lambda p: (_extract_step(p, "step") or 0, os.path.basename(p)))

# Bucket by step + headset_frame
by_step = {}  # step -> {'PB': target_drop of PB when ablating HeadSet_personal, 'AB': ...}
for d in m3b_dirs:
    basename = os.path.basename(d)
    m = re.match(r"step(\d+)_(personal_belief|attributed_belief)", basename)
    if not m:
        continue
    step = int(m.group(1))
    hset_frame = m.group(2)
    rj = os.path.join(d, "results.json")
    if not os.path.exists(rj):
        continue
    with open(rj) as f:
        r = json.load(f)
    target = "PB" if hset_frame == "personal_belief" else "AB"
    target_frame = "personal_belief" if hset_frame == "personal_belief" else "attributed_belief"
    by_step.setdefault(step, {})[target] = r["frames"][target_frame]["drop"]

steps_c = sorted(by_step.keys())
target_drop_PB = [by_step[s].get("PB", np.nan) for s in steps_c]
target_drop_AB = [by_step[s].get("AB", np.nan) for s in steps_c]

steps_c_arr = np.array(steps_c, dtype=float)
steps_c_arr[steps_c_arr == 0] = 0.5

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(steps_c_arr, target_drop_PB, marker="s", markersize=3.5,
        color=COLORS[1], label="HeadSet_personal ablation → PB drop", linewidth=1.2)
ax.plot(steps_c_arr, target_drop_AB, marker="o", markersize=3.5,
        color=COLORS[2], label="HeadSet_attributed ablation → AB drop", linewidth=1.2)

ax.axhline(0.15, color="black", linestyle="--", linewidth=0.7, alpha=0.7)
ax.text(max(steps_c_arr) * 0.9, 0.16, "causal emergence (0.15)", fontsize=7,
        ha="right", va="bottom")
ax.axhline(0.30, color="black", linestyle=":", linewidth=0.7, alpha=0.7)
ax.text(max(steps_c_arr) * 0.9, 0.31, "consolidation (0.30)", fontsize=7,
        ha="right", va="bottom")
ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.5)

# Vertical AB behavioral-emergence marker (from M3.a)
ax.axvline(2000, color=COLORS[2], linestyle=":", linewidth=0.7, alpha=0.5)
ax.text(2000, -0.03, "AB behavioral\nemergence @2k", fontsize=6.5,
        color=COLORS[2], ha="center", va="top", alpha=0.9)

# Causal-emergence markers
ax.axvline(23000, color=COLORS[2], linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(23000, 0.53, "AB causal @23k", fontsize=7, color=COLORS[2], ha="left", va="center")
ax.axvline(33000, color=COLORS[1], linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(33000, 0.48, "PB causal @33k", fontsize=7, color=COLORS[1], ha="left", va="center")

# 21k-step decoupling annotation
ax.annotate("", xy=(23000, 0.62), xytext=(2000, 0.62),
            arrowprops=dict(arrowstyle="<->", color="tab:brown", lw=0.8))
ax.text(np.sqrt(2000 * 23000), 0.635, "21k-step behavioral–causal\ndecoupling for AB",
        ha="center", va="bottom", fontsize=7, color="tab:brown")

ax.set_xscale("log")
ax.set_xlabel("pythia-1b training step (log scale)")
ax.set_ylabel("target-frame accuracy drop under ablation")
ax.set_ylim(-0.08, 0.72)
ax.set_xlim(0.4, max(steps_c_arr) * 1.4)
ax.legend(frameon=False, loc="lower left", fontsize=8)

save_fig(fig, "c3_causal_trajectory", formats=("pdf", "png"), out_dir=OUT_DIR)
plt.close(fig)


# ---------------------------------------------------------------------------
# INDEX.json
# ---------------------------------------------------------------------------
index = {
    "claim_id": "C3",
    "claim_title": "Formation Window — distinct developmental trajectories of PB vs AB circuits on pythia-1b",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "figures": [
        {
            "id": "c3_causal_trajectory",
            "type": "line",
            "caption": (
                "Causal-intervention trajectory on pythia-1b: target-frame accuracy drop when zero-ablating each "
                "Claim-2 head set, across pythia-1b intermediate checkpoints. Attributed-belief causal circuit "
                "emerges at step 23000 (drop crosses 0.15) and consolidates at step 43000 (crosses 0.30); "
                "personal-belief circuit emerges at step 33000 and consolidates at step 63000 — a 10k-step gap. "
                "Attributed-belief shows a 21k-step behavioral-vs-causal decoupling (behavioral emergence at "
                "step 2000 but causal reliance on the final head set not until step 23000)."
            ),
            "png": "figures/C3/c3_causal_trajectory.png",
            "pdf": "figures/C3/c3_causal_trajectory.pdf",
            "md": None,
            "tex": None,
            "source_data": "runs/M3b_pythia-1b/step*/results.json",
            "status": "ok",
        },
        {
            "id": "c3_behavioral_trajectory",
            "type": "line",
            "caption": (
                "Behavioral trajectory on pythia-1b: world-knowledge / personal-belief / attributed-belief "
                "accuracy on full belief_core across intermediate checkpoints. Attributed-belief reaches 0.97 "
                "by step 2000 (behavioral emergence); personal-belief fluctuates and only reliably passes 0.60 "
                "from step 33000 onward."
            ),
            "png": "figures/C3/c3_behavioral_trajectory.png",
            "pdf": "figures/C3/c3_behavioral_trajectory.pdf",
            "md": None,
            "tex": None,
            "source_data": "runs/M3a_pythia-1b/step*/results.json",
            "status": "ok",
        },
    ],
    "skipped": [],
}
with open(os.path.join(OUT_DIR, "INDEX.json"), "w") as f:
    json.dump(index, f, indent=2)
print("C3: 2/2 figures generated")
