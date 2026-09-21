import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_06-08-42_propositional_logic_circuit_attempt_1/logs/0-run/experiment_results/experiment_28eb91b6d82b4c089868071a20b65268_proc_4108744/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


def sem(arr, axis=0):
    arr = np.array(arr)
    n = arr.shape[axis]
    if n <= 1:
        return np.zeros_like(np.mean(arr, axis=axis))
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(n)


if len(all_experiment_data) > 0:
    # Extract common structure across runs
    roots = [ed["multi_dataset_circuit_generalization"] for ed in all_experiment_data]
    # Union of dataset names from first run's summary
    summary0 = roots[0].get("_summary", {})
    ds_names = summary0.get(
        "dataset_names", [k for k in roots[0].keys() if k != "_summary"]
    )
    n_runs = len(roots)
    print(f"Aggregating over {n_runs} run(s). Datasets: {ds_names}")

    # Plot 1: Aggregated head effect heatmaps (mean across runs)
    try:
        fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 5))
        if len(ds_names) == 1:
            axes = [axes]
        for idx, ds in enumerate(ds_names):
            stacks = []
            for r in roots:
                if ds in r:
                    stacks.append(np.array(r[ds]["head_effects"]))
            if len(stacks) == 0:
                continue
            mean_he = np.mean(np.stack(stacks, axis=0), axis=0)
            vmax = max(abs(mean_he).max(), 1e-6)
            im = axes[idx].imshow(
                mean_he, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )
            axes[idx].set_title(f"{ds} (mean over {len(stacks)} runs)")
            axes[idx].set_xlabel("Head")
            axes[idx].set_ylabel("Layer")
            plt.colorbar(im, ax=axes[idx])
        fig.suptitle(
            "Aggregated Attention Head Patching Effects - Logic Reasoning Datasets"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_head_effects_heatmap_agg.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating head effects plot: {e}")
        plt.close()

    # Plot 2: MLP effects per dataset with mean and SEM error bars
    try:
        fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 4))
        if len(ds_names) == 1:
            axes = [axes]
        for idx, ds in enumerate(ds_names):
            stacks = []
            for r in roots:
                if ds in r:
                    stacks.append(np.array(r[ds]["mlp_effects"]))
            if len(stacks) == 0:
                continue
            arr = np.stack(stacks, axis=0)
            mean_me = arr.mean(axis=0)
            sem_me = sem(arr, axis=0)
            x = np.arange(len(mean_me))
            axes[idx].bar(
                x,
                mean_me,
                yerr=sem_me,
                capsize=3,
                label=f"mean ± SEM (n={len(stacks)})",
            )
            axes[idx].set_title(f"{ds}")
            axes[idx].set_xlabel("Layer")
            axes[idx].set_ylabel("Recovery")
            axes[idx].legend(fontsize=8)
        fig.suptitle(
            "Aggregated MLP Patching Effects Per Layer - Logic Reasoning Datasets"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_mlp_effects_bars_agg.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating MLP effects plot: {e}")
        plt.close()

    # Plot 3: Faithfulness vs K per dataset with error bars
    try:
        plt.figure(figsize=(9, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(ds_names)))
        for cidx, ds in enumerate(ds_names):
            ks_union = set()
            for r in roots:
                if ds in r:
                    ks_union.update(r[ds]["faithfulness"].keys())
            ks = sorted(ks_union)
            match_runs = []
            faith_runs = []
            for r in roots:
                if ds not in r:
                    continue
                fd = r[ds]["faithfulness"]
                match_runs.append([fd[k]["match"] for k in ks if k in fd])
                faith_runs.append([fd[k]["faithfulness"] for k in ks if k in fd])
            match_arr = np.array(match_runs)
            faith_arr = np.array(faith_runs)
            if match_arr.size == 0:
                continue
            m_mean = match_arr.mean(axis=0)
            m_sem = sem(match_arr, axis=0)
            f_mean = faith_arr.mean(axis=0)
            f_sem = sem(faith_arr, axis=0)
            plt.errorbar(
                ks,
                m_mean,
                yerr=m_sem,
                marker="o",
                color=colors[cidx],
                label=f"{ds} match (mean±SEM)",
            )
            plt.errorbar(
                ks,
                f_mean,
                yerr=f_sem,
                marker="x",
                linestyle="--",
                color=colors[cidx],
                label=f"{ds} faith (mean±SEM)",
            )
        plt.xlabel("K (num components)")
        plt.ylabel("Score")
        plt.title(
            "Aggregated Circuit Faithfulness vs K - Logic Reasoning Datasets\n"
            "(solid: match, dashed: faithfulness; error bars = SEM)"
        )
        plt.legend(fontsize=7, ncol=2)
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_faithfulness_vs_K_agg.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating faithfulness plot: {e}")
        plt.close()

    # Plot 4: Aggregated Jaccard overlap matrix
    try:
        jm_list = []
        for r in roots:
            s = r.get("_summary", {})
            if s.get("jaccard_matrix") is not None:
                jm_list.append(np.array(s["jaccard_matrix"]))
        if len(jm_list) > 0:
            jm_arr = np.stack(jm_list, axis=0)
            jm_mean = jm_arr.mean(axis=0)
            jm_sem = sem(jm_arr, axis=0)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(jm_mean, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(ds_names)))
            ax.set_xticklabels(ds_names, rotation=45, ha="right")
            ax.set_yticks(range(len(ds_names)))
            ax.set_yticklabels(ds_names)
            for i in range(len(ds_names)):
                for j in range(len(ds_names)):
                    ax.text(
                        j,
                        i,
                        f"{jm_mean[i,j]:.2f}\n±{jm_sem[i,j]:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                    )
            ax.set_title(
                f"Aggregated Jaccard Overlap of Top-K Components "
                f"(mean±SEM, n={len(jm_list)})\nLogic Reasoning Datasets"
            )
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "logic_reasoning_jaccard_matrix_agg.png"),
                dpi=100,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating Jaccard plot: {e}")
        plt.close()

    # Plot 5: Aggregated Cross-dataset faithfulness matrix
    try:
        mats = []
        K_cross = None
        for r in roots:
            s = r.get("_summary", {})
            cfm = s.get("cross_faithfulness_matrix")
            if cfm is None:
                continue
            K_cross = s.get("cross_faithfulness_K", K_cross)
            mat = np.zeros((len(ds_names), len(ds_names)))
            valid = True
            for i, c in enumerate(ds_names):
                for j, t in enumerate(ds_names):
                    try:
                        mat[i, j] = cfm[c][t]["faithfulness"]
                    except Exception:
                        valid = False
            if valid:
                mats.append(mat)
        if len(mats) > 0:
            mats_arr = np.stack(mats, axis=0)
            m_mean = mats_arr.mean(axis=0)
            m_sem = sem(mats_arr, axis=0)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(m_mean, cmap="viridis")
            ax.set_xticks(range(len(ds_names)))
            ax.set_xticklabels(ds_names, rotation=45, ha="right")
            ax.set_yticks(range(len(ds_names)))
            ax.set_yticklabels(ds_names)
            ax.set_xlabel("Test dataset")
            ax.set_ylabel("Circuit source dataset")
            for i in range(len(ds_names)):
                for j in range(len(ds_names)):
                    ax.text(
                        j,
                        i,
                        f"{m_mean[i,j]:.2f}\n±{m_sem[i,j]:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                    )
            ax.set_title(
                f"Aggregated Cross-Dataset Faithfulness @ K={K_cross} "
                f"(mean±SEM, n={len(mats)})\nLogic Reasoning Datasets"
            )
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, "logic_reasoning_cross_dataset_faithfulness_agg.png"
                ),
                dpi=100,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset plot: {e}")
        plt.close()

    # Plot 6: Baseline accuracy & logit diffs with error bars
    try:
        clean_accs = {ds: [] for ds in ds_names}
        corr_accs = {ds: [] for ds in ds_names}
        clean_lds = {ds: [] for ds in ds_names}
        corr_lds = {ds: [] for ds in ds_names}
        for r in roots:
            for ds in ds_names:
                if ds not in r:
                    continue
                v = r[ds]["metrics"]["val"][0]
                clean_accs[ds].append(v["clean_acc"])
                corr_accs[ds].append(v["corr_acc"])
                clean_lds[ds].append(v["mean_clean_logit_diff"])
                corr_lds[ds].append(v["mean_corr_logit_diff"])

        def mean_sem_list(d):
            means = [np.mean(d[ds]) if len(d[ds]) else 0 for ds in ds_names]
            sems = [sem(np.array(d[ds])) if len(d[ds]) > 1 else 0 for ds in ds_names]
            return np.array(means), np.array(sems)

        cam, cas = mean_sem_list(clean_accs)
        corm, cors = mean_sem_list(corr_accs)
        clm, cls = mean_sem_list(clean_lds)
        corlm, corls = mean_sem_list(corr_lds)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(ds_names))
        w = 0.35
        axes[0].bar(x - w / 2, cam, w, yerr=cas, capsize=3, label="clean (mean±SEM)")
        axes[0].bar(
            x + w / 2, corm, w, yerr=cors, capsize=3, label="corrupted (mean±SEM)"
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ds_names, rotation=45, ha="right")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Left: Baseline Accuracy (clean vs corrupted)")
        axes[0].legend()
        axes[1].bar(x - w / 2, clm, w, yerr=cls, capsize=3, label="clean (mean±SEM)")
        axes[1].bar(
            x + w / 2, corlm, w, yerr=corls, capsize=3, label="corrupted (mean±SEM)"
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(ds_names, rotation=45, ha="right")
        axes[1].set_ylabel("Mean logit diff (True - False)")
        axes[1].set_title("Right: Mean Logit Difference")
        axes[1].legend()
        fig.suptitle(
            f"Aggregated Baseline Model Performance (n={n_runs}) - Logic Reasoning Datasets"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_baseline_performance_agg.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating baseline plot: {e}")
        plt.close()

    # Plot 7: Aggregated role distribution stacked bar with SEM markers
    try:
        roles = ["early_fact_reading", "mid_rule_composition", "late_answer_projection"]
        means = np.zeros((len(ds_names), len(roles)))
        sems = np.zeros((len(ds_names), len(roles)))
        for di, ds in enumerate(ds_names):
            per_role_vals = {r_: [] for r_ in roles}
            for r in roots:
                if ds not in r or "role_counts_top20" not in r[ds]:
                    continue
                for rn in roles:
                    per_role_vals[rn].append(r[ds]["role_counts_top20"].get(rn, 0))
            for ri, rn in enumerate(roles):
                vals = np.array(per_role_vals[rn])
                if vals.size:
                    means[di, ri] = vals.mean()
                    sems[di, ri] = sem(vals) if vals.size > 1 else 0
        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = np.zeros(len(ds_names))
        colors = ["tab:blue", "tab:orange", "tab:green"]
        x = np.arange(len(ds_names))
        for i, r_ in enumerate(roles):
            ax.bar(
                x,
                means[:, i],
                bottom=bottom,
                label=f"{r_} (mean)",
                color=colors[i],
                yerr=sems[:, i],
                capsize=3,
                error_kw=dict(ecolor="black"),
            )
            bottom += means[:, i]
        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, rotation=45, ha="right")
        ax.set_ylabel("Count in Top-20 (mean ± SEM)")
        ax.set_title(
            f"Aggregated Layer Role Distribution of Top-20 Components "
            f"(n={n_runs})\nLogic Reasoning Datasets"
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_role_distribution_agg.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating role distribution plot: {e}")
        plt.close()

    # Print aggregated summary metrics
    try:
        print("\n=== Aggregated Summary Metrics (mean ± SEM) ===")
        for ds in ds_names:
            cas = [r[ds]["metrics"]["val"][0]["clean_acc"] for r in roots if ds in r]
            cos = [r[ds]["metrics"]["val"][0]["corr_acc"] for r in roots if ds in r]
            if len(cas) == 0:
                continue
            print(
                f"{ds}: clean_acc={np.mean(cas):.3f}±{sem(np.array(cas)):.3f} "
                f"corr_acc={np.mean(cos):.3f}±{sem(np.array(cos)):.3f}"
            )
        print("\nOwn faithfulness @ K=40 (mean ± SEM):")
        for ds in ds_names:
            matches = []
            faiths = []
            for r in roots:
                if ds in r and 40 in r[ds]["faithfulness"]:
                    matches.append(r[ds]["faithfulness"][40]["match"])
                    faiths.append(r[ds]["faithfulness"][40]["faithfulness"])
            if len(matches):
                print(
                    f"  {ds}: match={np.mean(matches):.3f}±{sem(np.array(matches)):.3f} "
                    f"faith={np.mean(faiths):.3f}±{sem(np.array(faiths)):.3f}"
                )
    except Exception as e:
        print(f"Error printing summary: {e}")

print("Plotting done.")
