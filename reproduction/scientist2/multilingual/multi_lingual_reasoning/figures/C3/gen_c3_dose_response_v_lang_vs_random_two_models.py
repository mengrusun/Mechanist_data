"""C3 multi-panel: signed α-sweep on two models × two subspaces.
Panel 0: Qwen-3-4B-Thinking (main); Panel 1: DeepSeek-R1-Distill-LLaMA-8B (variant).
Each panel plots V_lang (solid) vs random-subspace (dashed) macro_acc vs α.

Main model data: refine-logs/EXPERIMENT_RESULTS.md#M3 (verbatim).
Variant data: verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/results/*_summary.json.
"""
import json
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, "..", ".."))

# Panel 0 — Qwen-3-4B-Thinking (main), from EXPERIMENT_RESULTS.md#M3 verbatim
alphas_main = [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]
v_lang_main = [0.053, 0.051, 0.085, 0.175, 0.744, 0.009, 0.000, 0.000, 0.000]
random_main = [0.745, 0.740, 0.747, 0.753, 0.744, 0.729, 0.438, 0.004, 0.000]

# Panel 1 — DeepSeek-R1-Distill-LLaMA-8B (variant), read from variant summary jsons.
VARIANT_DIR = os.path.join(PROJECT_ROOT, "verify", "C3_signed_dose_response",
                           "variants", "model_swap_deepseek_r1_llama8b", "results")

def _load_variant(kind):
    """kind ∈ {'vlang', 'random'}; return (alphas_sorted, macro_acc_sorted)."""
    out = {}
    for path in glob.glob(os.path.join(VARIANT_DIR, f"{kind}_alpha*_s42_summary.json")):
        with open(path) as f:
            d = json.load(f)
        # extract α from filename
        base = os.path.basename(path)
        # e.g. vlang_alpha-1.5_s42_summary.json
        alpha_str = base.replace(f"{kind}_alpha", "").split("_s42_")[0]
        alpha = float(alpha_str)
        # macro_acc lives under different keys depending on writer; try common ones
        macro = None
        for k in ("macro_accuracy", "macro_acc", "mean_accuracy", "accuracy"):
            if k in d:
                macro = d[k]; break
        if macro is None and "per_language" in d:
            vals = [v.get("accuracy") for v in d["per_language"].values() if isinstance(v, dict)]
            vals = [v for v in vals if v is not None]
            if vals:
                macro = sum(vals) / len(vals)
        if macro is None:
            # nested: iterate first-level dict for a numeric field
            for k, v in d.items():
                if isinstance(v, (int, float)) and "acc" in k.lower():
                    macro = v; break
        if macro is None:
            print(f"[warn] no macro_acc found in {path}; keys={list(d.keys())}")
            continue
        out[alpha] = float(macro)
    if not out:
        return [], []
    alphas_sorted = sorted(out.keys())
    return alphas_sorted, [out[a] for a in alphas_sorted]

alphas_var_vlang, v_lang_var = _load_variant("vlang")
alphas_var_rand, random_var = _load_variant("random")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

def _plot_panel(ax, alphas_v, v_vals, alphas_r, r_vals, title):
    if alphas_v:
        ax.plot(alphas_v, v_vals, marker="o", linewidth=1.8, color=COLORS[3],
                label=r"$V_{\mathrm{lang}}$ subspace")
    if alphas_r:
        ax.plot(alphas_r, r_vals, marker="s", linewidth=1.4, linestyle="--", color=COLORS[0],
                label="matched random subspace")
    ax.axhline(0.0, color="gray", linewidth=0.4, linestyle=":")
    ax.axvline(0.0, color="gray", linewidth=0.4, linestyle=":")
    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.02, 0.85)
    ax.set_title(title)

_plot_panel(axes[0], alphas_main, v_lang_main, alphas_main, random_main,
            "Qwen-3-4B-Thinking (main experiment)")
axes[0].set_ylabel("MGSM macro-accuracy (11 langs)")

_plot_panel(axes[1], alphas_var_vlang, v_lang_var, alphas_var_rand, random_var,
            "DeepSeek-R1-Distill-LLaMA-8B (verify variant)")

axes[0].legend(frameon=False, loc="upper right")
plt.tight_layout()

save_fig(fig, "c3_dose_response_v_lang_vs_random_two_models",
         formats=("pdf", "png"), out_dir=OUT_DIR)
print("wrote c3_dose_response_v_lang_vs_random_two_models.pdf and .png")
