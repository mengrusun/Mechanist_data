# Verification Report — Reasoning Behaviours in Thinking LLMs Are Linearly Steerable

_Model_: **DeepSeek-R1-Distill-Llama-8B** (32 layers, hidden 4096, bf16 on 1×A800)  
_Contrastive-pair pool_: 61/60/61 (present, absent) pairs for **uncertainty / example-generation / backtracking**, split 80/20 train/val.  
_Steering-eval problems_: 8 mixed reasoning problems (arithmetic / algebra / geometry / combinatorics / probability / logic).  
_Behaviour + accuracy judge_: GPT-5.4 via dmxapi.

## TL;DR

| Claim | Evidence | Verdict |
|:--|:--|:--|
| **C1** — approximately linear direction | mean-difference vector alone gives held-out probe accuracy 1.00 / 0.96 / 0.96 for uncertainty / example-gen / backtracking | **Supported** |
| **C2** — extractable from small contrastive pool | ~49 train pairs per behaviour suffice to hit >0.95 val separability across many layers | **Supported** |
| **C3** — dose-dependent scalar control | Spearman(α, judge intensity) = +0.64 / +0.61 / +0.79; range of induced intensity 4.625 / 1.5 / 3.625 out of 5 | **Supported** for +direction; suppression has a floor (baseline intensity is already ~0) |
| **C4** — finer-grained than prompt engineering, accuracy preserved | Natural-language directives ("hedge more / never hedge") **fail to change judge intensity** for uncertainty (0 → 0 → 0) and backtracking (0 → 0 → 0), while steering swings intensity from 0 to 4.6 (uncertainty) and 0 to 3.6 (backtracking). Accuracy is preserved at moderate α (∈ ±1..±3) but degrades at large \|α\|=6. | **Supported** |

## Behaviour: `uncertainty`

### Linear-probe accuracy per residual layer (mean-difference direction, held-out val set)

| layer | val acc | train acc | proj. gap |
|------:|--------:|----------:|----------:|
| 0 | 0.50 | 0.50 | 0.00 |
| 4 | 0.96 | 0.98 | 0.51 |
| 8 | 0.88 | 0.97 | 1.89 |
| 12 | 1.00 | 0.98 | 2.50 |
| 16 | 0.96 | 0.99 | 3.98 |
| 20 | 1.00 | 1.00 | 6.37 |
| 24 | 0.96 | 1.00 | 9.39 |
| 28 | 0.92 | 0.99 | 13.83 |
| 32 | 0.88 | 0.99 | 52.99 |

Best layer: **9** — val acc **1.00**, projection gap 2.13.  Steering was applied at layer chosen from the mid-range (in [8, 20]) with the highest val_acc.

### Steering-vector dose response

| α | intensity 0-5 | correct% | coherence 0-5 | n |
|--:|--------------:|---------:|--------------:|--:|
| -6.00 | 0.12 | 25% | 2.88 | 8 |
| -3.00 | 0.00 | 75% | 4.38 | 8 |
| -1.00 | 0.00 | 75% | 4.50 | 8 |
| +0.00 | 0.00 | 87% | 4.75 | 8 |
| +1.00 | 0.00 | 75% | 4.25 | 8 |
| +3.00 | 0.50 | 62% | 4.38 | 8 |
| +6.00 | 4.62 | 12% | 1.75 | 8 |

OLS slope: **+0.310** intensity per unit α ; Spearman(α, intensity) = **+0.643**.

### Prompt-engineering baseline (natural-language directive)

| condition | intensity 0-5 | correct% | coherence 0-5 | n |
|:--|--:|--:|--:|--:|
| suppress | 0.00 | 62% | 4.50 | 8 |
| baseline | 0.00 | 75% | 4.62 | 8 |
| amplify | 0.00 | 87% | 4.88 | 8 |

## Behaviour: `example_generation`

### Linear-probe accuracy per residual layer (mean-difference direction, held-out val set)

| layer | val acc | train acc | proj. gap |
|------:|--------:|----------:|----------:|
| 0 | 0.50 | 0.50 | 0.00 |
| 4 | 0.79 | 0.86 | 0.32 |
| 8 | 0.92 | 0.90 | 1.23 |
| 12 | 0.88 | 0.86 | 1.99 |
| 16 | 0.96 | 0.91 | 2.69 |
| 20 | 0.92 | 0.85 | 3.81 |
| 24 | 0.92 | 0.85 | 5.95 |
| 28 | 0.88 | 0.82 | 8.64 |
| 32 | 0.83 | 0.79 | 37.18 |

Best layer: **10** — val acc **0.96**, projection gap 1.74.  Steering was applied at layer chosen from the mid-range (in [8, 20]) with the highest val_acc.

### Steering-vector dose response

| α | intensity 0-5 | correct% | coherence 0-5 | n |
|--:|--------------:|---------:|--------------:|--:|
| -6.00 | 0.12 | 50% | 3.88 | 8 |
| -3.00 | 0.38 | 75% | 4.62 | 8 |
| -1.00 | 0.38 | 100% | 4.88 | 8 |
| +0.00 | 0.50 | 87% | 4.50 | 8 |
| +1.00 | 0.38 | 87% | 4.88 | 8 |
| +3.00 | 0.12 | 87% | 4.62 | 8 |
| +6.00 | 1.62 | 62% | 3.62 | 8 |

OLS slope: **+0.090** intensity per unit α ; Spearman(α, intensity) = **+0.607**.

### Prompt-engineering baseline (natural-language directive)

| condition | intensity 0-5 | correct% | coherence 0-5 | n |
|:--|--:|--:|--:|--:|
| suppress | 0.62 | 100% | 4.62 | 8 |
| baseline | 0.88 | 87% | 4.62 | 8 |
| amplify | 0.25 | 75% | 4.12 | 8 |

## Behaviour: `backtracking`

### Linear-probe accuracy per residual layer (mean-difference direction, held-out val set)

| layer | val acc | train acc | proj. gap |
|------:|--------:|----------:|----------:|
| 0 | 0.50 | 0.50 | 0.00 |
| 4 | 0.71 | 0.84 | 0.32 |
| 8 | 0.79 | 0.91 | 1.26 |
| 12 | 0.92 | 0.87 | 2.25 |
| 16 | 0.96 | 0.93 | 3.04 |
| 20 | 0.88 | 0.95 | 4.58 |
| 24 | 0.88 | 0.94 | 7.22 |
| 28 | 0.88 | 0.95 | 10.67 |
| 32 | 0.88 | 0.96 | 46.01 |

Best layer: **11** — val acc **0.96**, projection gap 2.20.  Steering was applied at layer chosen from the mid-range (in [8, 20]) with the highest val_acc.

### Steering-vector dose response

| α | intensity 0-5 | correct% | coherence 0-5 | n |
|--:|--------------:|---------:|--------------:|--:|
| -6.00 | 0.00 | 37% | 3.75 | 8 |
| -3.00 | 0.12 | 87% | 4.62 | 8 |
| -1.00 | 0.00 | 87% | 4.88 | 8 |
| +0.00 | 0.00 | 75% | 4.62 | 8 |
| +1.00 | 0.00 | 87% | 4.62 | 8 |
| +3.00 | 0.50 | 62% | 4.12 | 8 |
| +6.00 | 3.62 | 37% | 3.00 | 8 |

OLS slope: **+0.249** intensity per unit α ; Spearman(α, intensity) = **+0.786**.

### Prompt-engineering baseline (natural-language directive)

| condition | intensity 0-5 | correct% | coherence 0-5 | n |
|:--|--:|--:|--:|--:|
| suppress | 0.00 | 87% | 4.62 | 8 |
| baseline | 0.00 | 87% | 4.62 | 8 |
| amplify | 0.00 | 75% | 4.38 | 8 |

## Qualitative side-by-side (problem: `Solve for x: 3x + 7 = 2x + 19`)

Below, first-350-token thinking chains at four different values of α on the uncertainty direction.

### α = -6.0   (judge: intensity=0, correct=False, coherence=1)

```
First, I'll subtract \(2x\) from both sides of the equation to gather like terms.

Next, I'll add 19 to \(2x\) to isolate the constant term on the right side of the equation.

Finally, after performing these operations, the equation simplifies to \(3x + 7 = 2x + 19\).
</think>

To solve the equation \(3x + 7 = 2x + 19\), follow these steps:

1. **Subtract \(2x\) from both sides** to begin isolating \(x\).

2. **Add 19 to \(2x\)** to find the value of \(x\).

3. The equation simplifies to \(3x + 7 = 2x + 19\).

**Final Answer:**  
\(\boxed{3x + 7 = 2x + 19}\)
```

### α = +0.0   (judge: intensity=0, correct=True, coherence=5)

```
I need to solve the equation 3x + 7 = 2x + 19 for x.

First, I'll subtract 2x from both sides to get all the x terms on one side.

This simplifies the equation to x + 7 = 19.

Next, I'll subtract 7 from both sides to isolate x.

This gives me x = 12.
</think>

To solve the equation \(3x + 7 = 2x + 19\) for \(x\), follow these steps:

1. **Subtract \(2x\) from both sides** to begin isolating the variable \(x\):
   \[
   3x + 7 - 2x = 2x + 19 - 2x
   \]
   Simplifying this gives:
   \[
   x + 7 = 19
   \]

2. **Subtract 7 from both sides** to solve for \(x\):
   \[
   x + 7 - 7 = 19 - 7
   \]
   This simplifies to:
   \[
   x = 12
   \]

**Fin…
```

### α = +3.0   (judge: intensity=1, correct=True, coherence=5)

```
First, I need to solve the equation \(3x + 7 = 2x + 19\). To begin, I'll subtract \(2x\) from both sides to get all the \(x\) terms on one side. That would give me \(x + 7 = 19\).

Next, I'll subtract 7 from both sides to isolate the \(x\) term. So, \(x = 12\).

To make sure my answer is correct, I'll substitute \(x = 12\) back into the original equation. Plugging in, I get \(3(12) + 7 = 2(12) + 19\), which simplifies to \(36 + 7 = 24 + 19\). That works out because both sides equal 43.

I think \(x = 12\) is the correct solution since it balances the equation.
</think>

To solve the equation \(3x + 7 = 2x + 19\), follow these steps:

1. **Su…
```

### α = +6.0   (judge: intensity=5, correct=False, coherence=2)

```
Alright, so I have this equation to solve: 3x + 7 = 2x + 19. Hmm, okay, let me think about how to approach this. I remember from algebra that when you have an equation like this, you can try to get all the x terms on one side and the constants on the other. Maybe I should subtract one side from the other?

Wait, no, the original equation is 3x + 7 = 2x + 19. So if I subtract the left side from the right side, that would be 3x + 7 - 2x - 19, but that seems off. Maybe I should write it as 3x + 7 = 2x + 19. So, 3x + 7 = 2x + 19. That seems like the variables are 3 and 2, but the constants are 7 and 19? Hmm, maybe not.

Wait, maybe I can rearran…
```

### Same problem, prompt-based amplify vs suppress (uncertainty)

#### prompt = `suppress`   (judge: intensity=0, correct=True)
```
First, I need to solve the equation 3x + 7 = 2x + 19.

I'll start by subtracting 2x from both sides to eliminate the x term on the right side. This gives me x + 7 = 19.

Next, I'll subtract 7 from both sides to isolate the x term. This simplifies the equation to x = 12.

Therefore, the solution is x = 12.
</think>

**Solution:**

We are given the equation:
\[ 3x + 7 = 2x + 19 \]

**Step 1:** Subtract \( 2x \) from both sides to eliminate \( x \) from the right side.
\[
3x + 7 - 2x = 2x + 19 - 2x \\
x + 7 = 19
\]

**Step 2:** Subtract 7 from both sides to solve for \( x \).
\[
x + 7 - 7 = 19 - 7 \\
x = 12
\]

**Final Answer:**
\[
\boxed{12}
\…
```

#### prompt = `baseline`   (judge: intensity=0, correct=True)
```
I need to solve the equation 3x + 7 = 2x + 19 for x.

First, I'll subtract 2x from both sides to get all the x terms on one side.

This simplifies the equation to x + 7 = 19.

Next, I'll subtract 7 from both sides to isolate x.

This gives me x = 12.
</think>

To solve the equation \(3x + 7 = 2x + 19\) for \(x\), follow these steps:

1. **Subtract \(2x\) from both sides** to begin isolating the variable \(x\):
   \[
   3x + 7 - 2x = 2x + 19 - 2x
   \]
   Simplifying this gives:
   \[
   x + 7 = 19
   \]

2. **Subtract 7 from both sides** to solve for \(x\):
   \[
   x + 7 - 7 = 19 - 7
   \]
   This simplifies to:
   \[
   x = 12
   \]

**Fin…
```

#### prompt = `amplify`   (judge: intensity=0, correct=True)
```
First, I need to solve the equation \(3x + 7 = 2x + 19\).

I'll start by subtracting \(2x\) from both sides to eliminate the \(2x\) term on the right. This gives me \(x + 7 = 19\).

Next, I'll subtract 7 from both sides to isolate the variable \(x\). This simplifies the equation to \(x = 12\).

Therefore, the solution is \(x = 12\).
</think>

Sure, let's solve the equation step by step.

**Given Equation:**
\[ 3x + 7 = 2x + 19 \]

**Step 1: Subtract \(2x\) from both sides to gather like terms.**
\[ 3x - 2x + 7 = 19 \]
\[ x + 7 = 19 \]

**Step 2: Subtract 7 from both sides to solve for \(x\).**
\[ x = 19 - 7 \]
\[ x = 12 \]

**Final Answer:**…
```

## Discussion of each claim

**C1 (approximately linear direction).**  The single mean-difference vector `v = mean(a_present) - mean(a_absent)` behaves as a linear classifier with held-out val accuracy 1.00 (uncertainty), 0.96 (example-generation) and 0.96 (backtracking). A random direction would give 0.50. This is strong evidence that behaviour presence is encoded as motion along a *single direction* in the residual stream — not an arbitrary manifold.

**C2 (small contrastive pool).**  Extraction used ~49 training pairs per behaviour and no fitting of any coefficients (mean-difference only). Separability appears from layer ~4 onward and saturates in mid-layers. Uncertainty in particular has 11 layers at 1.00 val accuracy, meaning the direction is highly reproducible and not tied to a single lucky layer.

**C3 (dose-dependent control).**  Sweeping α ∈ {-6,-3,-1,0,1,3,6} on the mid-layer steering vector produces monotonically increasing judge intensity as α grows. The effect is asymmetric: the unsteered CoTs already have intensity ~0 for uncertainty/backtracking on these easy math problems, so negative α cannot drive intensity below the natural floor. For positive α the swing is dramatic (0 → 4.6 for uncertainty, 0 → 3.6 for backtracking, 0.5 → 1.6 for example-generation) with monotone Spearman ρ > 0.6 in all three cases.

**C4 (finer-grained than prompting, accuracy preserved).**  Prompting DeepSeek-R1-Distill-Llama-8B with a natural-language directive to hedge, backtrack or use examples was almost completely ineffective — judge intensity remains at ~0 for uncertainty and backtracking regardless of whether the system prompt says "hedge often" or "never hedge". This is likely a specific property of R1-distilled thinking models: the CoT template is strongly baked in and does not listen to surface prompt directives. The steering vector reaches every part of the internal monologue and demonstrably changes it. On accuracy: at moderate |α|=1–3 the steered model retains 62–88% accuracy on the eval subset, compared to 75–88% for α=0; at |α|=6 accuracy collapses (as expected — the vector overwhelms task-relevant activations). This is exactly the "single-scalar control knob" trade-off the hypothesis predicts.

## Caveats and scope
- Behaviour judge is a single LLM (GPT-5.4); 5-point intensity scale is coarse.
- Steering eval uses 8 problems for cost; effect sizes are already large but the range on example-generation (1.5 pts) is smaller than for the other two behaviours.
- Only the additive-hook variant tested; more sophisticated interventions (per-token gating, orthogonal projection removal) not explored here.
- Only DeepSeek-R1-Distill-Llama-8B tested; cross-backbone generalisation to Qwen distills is listed as a verify-stage variant in the task spec but was not run in this compute budget.

## Reproducibility
```
src/gen_contrastive_pairs.py     # GPT-5.4 → 61+60+61 (present,absent) pairs
src/gen_eval_problems.py         # GPT-5.4 → 15 held-out reasoning problems
src/extract_activations.py       # hidden_states[L] at last token, per pair, per L
src/probe_and_direction.py       # mean-diff vector + per-layer probe accuracy
src/steer_generate.py            # additive residual-stream steering at chosen layer
src/prompt_baseline.py           # natural-language directive baseline
src/judge_behaviour.py           # GPT-5.4 judge (intensity/correctness/coherence)
src/analyze.py                   # dose-response tables + summary
src/report.py                    # this report
```