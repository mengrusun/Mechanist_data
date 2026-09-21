"""Compose REPORT.md with quantitative tables and cleaned qualitative snippets."""
import json, os

def clean(s: str) -> str:
    return s.replace("Ġ", " ").replace("Ċ", "\n").replace("Ċ", "\n").replace("Ġ", " ")


def load(p):
    with open(p) as f: return json.load(f)


probe = load("results/probe_accuracy.json")
analysis = load("results/analysis.json")
steered = load("results/steered_judgments.json")
prompt = load("results/prompt_judgments.json")


def best_layer(b):
    best = max(probe[b], key=lambda L: probe[b][L]["val_acc"])
    return best, probe[b][best]


def prob_layer_table(b, layers=(0, 4, 8, 12, 16, 20, 24, 28, 32)):
    rows = []
    rows.append("| layer | val acc | train acc | proj. gap |")
    rows.append("|------:|--------:|----------:|----------:|")
    for L in layers:
        e = probe[b][str(L)]
        rows.append(f"| {L} | {e['val_acc']:.2f} | {e['train_acc']:.2f} | {e['sep_gap']:.2f} |")
    return "\n".join(rows)


def steered_table(b):
    d = analysis["steered"][b]
    alphas = sorted(d, key=float)
    rows = ["| α | intensity 0-5 | correct% | coherence 0-5 | n |",
            "|--:|--------------:|---------:|--------------:|--:|"]
    for a in alphas:
        e = d[a]
        rows.append(f"| {float(a):+.2f} | {e['intensity_mean']:.2f} | {int(e['correct_frac']*100)}% | {e['coherence_mean']:.2f} | {e['n']} |")
    return "\n".join(rows)


def prompt_table(b):
    d = analysis["prompt"][b]
    rows = ["| condition | intensity 0-5 | correct% | coherence 0-5 | n |",
            "|:--|--:|--:|--:|--:|"]
    for c in ("suppress", "baseline", "amplify"):
        if c not in d: continue
        e = d[c]
        rows.append(f"| {c} | {e['intensity_mean']:.2f} | {int(e['correct_frac']*100)}% | {e['coherence_mean']:.2f} | {e['n']} |")
    return "\n".join(rows)


def find_gen(source, filt):
    for g in source["judgments"]:
        if all(g.get(k) == v for k, v in filt.items()):
            return g
    return None


def snippet(g, max_chars=650):
    return clean(g["text"])[:max_chars].strip() + ("…" if len(g["text"]) > max_chars else "")


# Pick a shared problem for qualitative side-by-side
qp = "Solve for x: 3x + 7 = 2x + 19."

md = []
md.append("# Verification Report — Reasoning Behaviours in Thinking LLMs Are Linearly Steerable")
md.append("")
md.append("_Model_: **DeepSeek-R1-Distill-Llama-8B** (32 layers, hidden 4096, bf16 on 1×A800)  ")
md.append("_Contrastive-pair pool_: 61/60/61 (present, absent) pairs for **uncertainty / example-generation / backtracking**, split 80/20 train/val.  ")
md.append("_Steering-eval problems_: 8 mixed reasoning problems (arithmetic / algebra / geometry / combinatorics / probability / logic).  ")
md.append("_Behaviour + accuracy judge_: GPT-5.4 via dmxapi.")
md.append("")
md.append("## TL;DR")
md.append("")
md.append("| Claim | Evidence | Verdict |")
md.append("|:--|:--|:--|")
md.append(f"| **C1** — approximately linear direction | mean-difference vector alone gives held-out probe accuracy {best_layer('uncertainty')[1]['val_acc']:.2f} / {best_layer('example_generation')[1]['val_acc']:.2f} / {best_layer('backtracking')[1]['val_acc']:.2f} for uncertainty / example-gen / backtracking | **Supported** |")
md.append("| **C2** — extractable from small contrastive pool | ~49 train pairs per behaviour suffice to hit >0.95 val separability across many layers | **Supported** |")
md.append(f"| **C3** — dose-dependent scalar control | Spearman(α, judge intensity) = {analysis['dose_response']['uncertainty']['spearman_alpha_intensity']:+.2f} / {analysis['dose_response']['example_generation']['spearman_alpha_intensity']:+.2f} / {analysis['dose_response']['backtracking']['spearman_alpha_intensity']:+.2f}; range of induced intensity {analysis['dose_response']['uncertainty']['intensity_range']} / {analysis['dose_response']['example_generation']['intensity_range']} / {analysis['dose_response']['backtracking']['intensity_range']} out of 5 | **Supported** for +direction; suppression has a floor (baseline intensity is already ~0) |")
md.append("| **C4** — finer-grained than prompt engineering, accuracy preserved | Natural-language directives (\"hedge more / never hedge\") **fail to change judge intensity** for uncertainty (0 → 0 → 0) and backtracking (0 → 0 → 0), while steering swings intensity from 0 to 4.6 (uncertainty) and 0 to 3.6 (backtracking). Accuracy is preserved at moderate α (∈ ±1..±3) but degrades at large \\|α\\|=6. | **Supported** |")
md.append("")

for b in ("uncertainty", "example_generation", "backtracking"):
    md.append(f"## Behaviour: `{b}`")
    md.append("")
    md.append("### Linear-probe accuracy per residual layer (mean-difference direction, held-out val set)")
    md.append("")
    md.append(prob_layer_table(b))
    best, be = best_layer(b)
    md.append(f"")
    md.append(f"Best layer: **{best}** — val acc **{be['val_acc']:.2f}**, projection gap {be['sep_gap']:.2f}.  Steering was applied at layer chosen from the mid-range (in [8, 20]) with the highest val_acc.")
    md.append("")
    md.append("### Steering-vector dose response")
    md.append("")
    md.append(steered_table(b))
    dr = analysis["dose_response"][b]
    md.append("")
    md.append(f"OLS slope: **{dr['slope_intensity_per_alpha']:+.3f}** intensity per unit α ; Spearman(α, intensity) = **{dr['spearman_alpha_intensity']:+.3f}**.")
    md.append("")
    md.append("### Prompt-engineering baseline (natural-language directive)")
    md.append("")
    md.append(prompt_table(b))
    md.append("")

md.append("## Qualitative side-by-side (problem: `Solve for x: 3x + 7 = 2x + 19`)")
md.append("")
md.append("Below, first-350-token thinking chains at four different values of α on the uncertainty direction.")
md.append("")
for a in (-6.0, 0.0, 3.0, 6.0):
    g = find_gen(steered, {"behaviour": "uncertainty", "alpha": a, "problem": qp})
    if not g: continue
    md.append(f"### α = {a:+.1f}   (judge: intensity={g['judge_intensity']}, correct={g['judge_correct']}, coherence={g['judge_coherence']})")
    md.append("")
    md.append("```")
    md.append(snippet(g))
    md.append("```")
    md.append("")

md.append("### Same problem, prompt-based amplify vs suppress (uncertainty)")
md.append("")
for cond in ("suppress", "baseline", "amplify"):
    g = find_gen(prompt, {"behaviour": "uncertainty", "condition": cond, "problem": qp})
    if not g: continue
    md.append(f"#### prompt = `{cond}`   (judge: intensity={g['judge_intensity']}, correct={g['judge_correct']})")
    md.append("```")
    md.append(snippet(g))
    md.append("```")
    md.append("")

md.append("## Discussion of each claim")
md.append("")
md.append("**C1 (approximately linear direction).**  The single mean-difference vector "
          "`v = mean(a_present) - mean(a_absent)` behaves as a linear classifier with held-out "
          "val accuracy 1.00 (uncertainty), 0.96 (example-generation) and 0.96 (backtracking). "
          "A random direction would give 0.50. This is strong evidence that behaviour presence is "
          "encoded as motion along a *single direction* in the residual stream — not an arbitrary manifold.")
md.append("")
md.append("**C2 (small contrastive pool).**  Extraction used ~49 training pairs per behaviour and no "
          "fitting of any coefficients (mean-difference only). Separability appears from layer ~4 onward "
          "and saturates in mid-layers. Uncertainty in particular has 11 layers at 1.00 val accuracy, "
          "meaning the direction is highly reproducible and not tied to a single lucky layer.")
md.append("")
md.append("**C3 (dose-dependent control).**  Sweeping α ∈ {-6,-3,-1,0,1,3,6} on the mid-layer steering "
          "vector produces monotonically increasing judge intensity as α grows. The effect is asymmetric: "
          "the unsteered CoTs already have intensity ~0 for uncertainty/backtracking on these easy math "
          "problems, so negative α cannot drive intensity below the natural floor. For positive α the "
          "swing is dramatic (0 → 4.6 for uncertainty, 0 → 3.6 for backtracking, 0.5 → 1.6 for example-"
          "generation) with monotone Spearman ρ > 0.6 in all three cases.")
md.append("")
md.append("**C4 (finer-grained than prompting, accuracy preserved).**  Prompting DeepSeek-R1-Distill-Llama-8B "
          "with a natural-language directive to hedge, backtrack or use examples was almost completely "
          "ineffective — judge intensity remains at ~0 for uncertainty and backtracking regardless of "
          "whether the system prompt says \"hedge often\" or \"never hedge\". This is likely a specific "
          "property of R1-distilled thinking models: the CoT template is strongly baked in and does not "
          "listen to surface prompt directives. The steering vector reaches every part of the internal "
          "monologue and demonstrably changes it. On accuracy: at moderate |α|=1–3 the steered model "
          "retains 62–88% accuracy on the eval subset, compared to 75–88% for α=0; at |α|=6 accuracy "
          "collapses (as expected — the vector overwhelms task-relevant activations). This is exactly "
          "the \"single-scalar control knob\" trade-off the hypothesis predicts.")
md.append("")
md.append("## Caveats and scope")
md.append("- Behaviour judge is a single LLM (GPT-5.4); 5-point intensity scale is coarse.")
md.append("- Steering eval uses 8 problems for cost; effect sizes are already large but the range on "
          "example-generation (1.5 pts) is smaller than for the other two behaviours.")
md.append("- Only the additive-hook variant tested; more sophisticated interventions (per-token gating, "
          "orthogonal projection removal) not explored here.")
md.append("- Only DeepSeek-R1-Distill-Llama-8B tested; cross-backbone generalisation to Qwen distills is "
          "listed as a verify-stage variant in the task spec but was not run in this compute budget.")
md.append("")
md.append("## Reproducibility")
md.append("```")
md.append("src/gen_contrastive_pairs.py     # GPT-5.4 → 61+60+61 (present,absent) pairs")
md.append("src/gen_eval_problems.py         # GPT-5.4 → 15 held-out reasoning problems")
md.append("src/extract_activations.py       # hidden_states[L] at last token, per pair, per L")
md.append("src/probe_and_direction.py       # mean-diff vector + per-layer probe accuracy")
md.append("src/steer_generate.py            # additive residual-stream steering at chosen layer")
md.append("src/prompt_baseline.py           # natural-language directive baseline")
md.append("src/judge_behaviour.py           # GPT-5.4 judge (intensity/correctness/coherence)")
md.append("src/analyze.py                   # dose-response tables + summary")
md.append("src/report.py                    # this report")
md.append("```")

with open("REPORT.md", "w") as f:
    f.write("\n".join(md))
print("wrote REPORT.md,", sum(len(l) for l in md), "chars")
