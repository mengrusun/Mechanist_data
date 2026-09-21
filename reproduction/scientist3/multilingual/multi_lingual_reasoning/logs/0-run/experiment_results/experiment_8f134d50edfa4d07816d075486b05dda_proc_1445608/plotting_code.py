import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

mgsm = experiment_data.get("max_new_tokens", {}).get("mgsm", {})
mnt_grid = mgsm.get("hyperparam_values", [])
results_by_value = mgsm.get("results_by_value", {})
conds = ["baseline", "lang_suppression", "random_control"]

# Plot 1: Overall accuracy sweep across max_new_tokens
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(mnt_grid))
    for i, c in enumerate(conds):
        vals = [results_by_value[str(m)][c]["overall"] for m in mnt_grid]
        ax.bar(x + (i - 1) * width, vals, width, label=c)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mnt_grid])
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("Multilingual Reasoning Accuracy")
    ax.set_title(
        "MGSM Dataset: Overall Accuracy vs max_new_tokens\nComparing Baseline, Language Subspace Suppression, Random Control"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_overall_accuracy_sweep.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Per-language accuracy at best (largest) mnt
try:
    best_mnt = mnt_grid[-1]
    r = results_by_value[str(best_mnt)]
    langs = list(r["baseline"]["per_lang"].keys())
    x = np.arange(len(langs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, c in enumerate(conds):
        vals = [r[c]["per_lang"][l] for l in langs]
        ax.bar(x + (i - 1) * width, vals, width, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_xlabel("Language")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"MGSM Dataset: Per-Language Accuracy (max_new_tokens={best_mnt})\nBaseline vs Language Suppression vs Random Control"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_per_language_accuracy.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Per-language delta (suppression - baseline) at best mnt
try:
    best_mnt = mnt_grid[-1]
    r = results_by_value[str(best_mnt)]
    langs = list(r["baseline"]["per_lang"].keys())
    delta_supp = [
        r["lang_suppression"]["per_lang"][l] - r["baseline"]["per_lang"][l]
        for l in langs
    ]
    delta_rand = [
        r["random_control"]["per_lang"][l] - r["baseline"]["per_lang"][l] for l in langs
    ]
    x = np.arange(len(langs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        x - width / 2,
        delta_supp,
        width,
        label="lang_suppression - baseline",
        color="tab:orange",
    )
    ax.bar(
        x + width / 2,
        delta_rand,
        width,
        label="random_control - baseline",
        color="tab:green",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_xlabel("Language")
    ax.set_ylabel("Accuracy Delta vs Baseline")
    ax.set_title(
        f"MGSM Dataset: Accuracy Delta from Baseline (max_new_tokens={best_mnt})\nEffect of Language Subspace Suppression vs Random Control"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "mgsm_accuracy_delta_per_language.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Line plot of overall accuracy vs mnt for each condition
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in conds:
        vals = [results_by_value[str(m)][c]["overall"] for m in mnt_grid]
        ax.plot(mnt_grid, vals, marker="o", label=c)
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("Overall Accuracy")
    ax.set_title(
        "MGSM Dataset: Overall Accuracy Trend vs max_new_tokens\nAcross Three Intervention Conditions"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_accuracy_trend_line.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Heatmap of per-language accuracy for each condition at best mnt
try:
    best_mnt = mnt_grid[-1]
    r = results_by_value[str(best_mnt)]
    langs = list(r["baseline"]["per_lang"].keys())
    matrix = np.array([[r[c]["per_lang"][l] for l in langs] for c in conds])
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_yticks(np.arange(len(conds)))
    ax.set_yticklabels(conds)
    for i in range(len(conds)):
        for j in range(len(langs)):
            ax.text(
                j,
                i,
                f"{matrix[i,j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_title(
        f"MGSM Dataset: Per-Language Accuracy Heatmap (max_new_tokens={best_mnt})\nRows: Conditions, Columns: Languages"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_per_language_heatmap.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print summary metrics
try:
    print("=== Summary of Overall Accuracy ===")
    for m in mnt_grid:
        r = results_by_value[str(m)]
        print(f"mnt={m}: " + ", ".join(f"{c}={r[c]['overall']:.4f}" for c in conds))
except Exception as e:
    print(f"Error printing summary: {e}")
