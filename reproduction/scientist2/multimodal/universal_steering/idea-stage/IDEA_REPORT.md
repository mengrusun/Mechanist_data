# IDEA REPORT — Faithful Capture (BEHAVIOR_SOURCE = given)

**Date**: 2026-07-14
**Direction source**: `task.md` (sole source; `direction` argument was empty)
**Mode**: `given` behavior + `discovery` mechanism (system routes via `/mechanism-explore`)
**Notes**: This is not a ranked idea competition — the five claims are given verbatim by `task.md § Claim`. This report faithfully captures them, grounds them in the pre-cutoff literature already retrieved (`RESEARCH_LIT.md` + `LANDSCAPE.md`), and hands the fixed mechanism strategy (from the prior resume-invocation analysis) to `/research-refine-pipeline` and `/experiment-plan`.

---

## Faithful Capture

The five sub-claims below are transcribed from `task.md § Claim` (lines 7–18). Origin pointer is preserved next to each.

### C1 — RFM per-block concept-vector steering (baseline)
> **Origin**: `task.md § Claim` bullet 1 (line 9).
>
> RFM (Recursive Feature Machines — a supervised feature-learning algorithm built on kernel machines + Average Gradient Outer Product / AGOP reweighting) extracts per-block linear concept vectors from the internal (residual-stream) activations of `Llama-3.1-8B-Instruct`. Adding those vectors additively to residual-stream activations at inference-time steers the model's behavior toward or away from the target concept. Named demo scenarios: **anti-refusal (jailbreak), political stance, honesty (negative steering)**.
>
> **Measurable predicate**: For each of the 3 named demo scenarios, on ≥50 held-out prompts, the steered model's task-appropriate scorer (GPT-4o judge or benchmark accuracy) beats the unsteered baseline **and** beats a matched-random-direction control at a chosen α.

### C2 — Python → C++ high-precision-task steering
> **Origin**: `task.md § Claim` bullet 2 (line 12).
>
> A "C++" concept vector extracted the same way, added at inference, raises the model's test-case pass rate on HackerRank-style algorithmic coding problems relative to both (a) default Python output and (b) prompt-only *"Answer in C++."*
>
> **Measurable predicate**: On the 50-problem HackerRank subset (`task.md § Verify stage`), test-case pass rate under the RFM C++ steering exceeds both baselines by a margin larger than run-to-run noise.

### C3 — Cross-lingual transferability of concept vectors
> **Origin**: `task.md § Claim` bullet 3 (line 14).
>
> Concept vectors trained on English-only paired text steer model responses when the prompt is asked in other human languages (Chinese, French, Spanish) — same additive intervention, same vector, different-language prompt.
>
> **Measurable predicate**: For a reused C1 concept (honesty or refusal), the same English-extracted vector applied to a translated prompt (ZH/FR/ES) produces a steering effect (per multilingual judge) in the intended direction, matching qualitatively the EN steering effect.

### C4 — Compositionality of concept vectors
> **Origin**: `task.md § Claim` bullet 4 (line 16).
>
> Linear combinations of ≥2 concept vectors enable simultaneous multi-concept steering: the combined intervention induces **both** target effects, not only one.
>
> **Measurable predicate**: For each of ≥2 vector-pair combinations (e.g. `honesty + refusal-negative`), the sum-steered outputs are judged to display both target effects on ≥20 prompts by GPT-4o at higher rates than either single-vector baseline for the missing target.

### C5 — Internal-feature monitoring beats LLM judge
> **Origin**: `task.md § Claim` bullet 5 (line 18).
>
> Concept features (probes / classifiers built on the same residual-stream activations) applied as monitors of hallucination and toxic content outperform LLM-judge baselines (GPT-4o reading only the model output) — even when the internal features come from a **smaller open-source model** compared to the powerful judge.
>
> **Measurable predicate**: On HaluEval-General (hallucination) and ToxicChat (toxicity), the AUROC of the internal-feature monitor built on `Llama-3.1-8B-Instruct` activations is strictly higher than GPT-4o-2024-11-20 used as a black-box judge on the model outputs; toxicity result is additionally compared to `ToxicChat-T5-Large`.

---

## Literature Grounding

See `idea-stage/RESEARCH_LIT.md` (raw retrieval — 12 mechanic-db papers + 7 foundational works) and `idea-stage/LANDSCAPE.md` (structured landscape, 5 gaps G1–G5). Highlights:

- **RFM algorithmic foundation**: Radhakrishnan, Beaglehole, Pandit, Belkin (2022–2024) — kernel-machine + AGOP reweighting, whose top AGOP eigenvector serves as the per-block concept direction (Foundational E in `RESEARCH_LIT.md`).
- **Steering-vector prior art (C1 baselines)**: RepE (Zou et al. 2023, arXiv 2310.01405), CAA (Panickssery et al. 2023, arXiv 2312.06681), ActAdd (Turner et al. 2023, arXiv 2308.10248), ITI (Li et al. NeurIPS 2023), CAV-for-LLMs (AAAI 2025 adaptation of Kim et al. ICML 2018). These are the direct baselines RFM claims to beat on brittle concepts.
- **Monitoring prior art (C5 baselines)**: SAPLMA (Azaria & Mitchell EMNLP-F 2023, arXiv 2304.13734), weakly-supervised hallucination detection (arXiv 2312.02798), SAEs as unsupervised feature monitors (Bricken 2023 / Cunningham 2023). LLM-judge baseline = GPT-4o-2024-11-20; task-specific toxicity baseline = `ToxicChat-T5-Large` (both fixed by `task.md`).
- **Foundational RFM-for-LLM-steering paper**: not named in the retrieval — the reproduction target is under a project-wide URL-blocker (`.claude/forbidden-urls.txt`) that voids any post-cutoff arXiv ID and any reference to the target paper. This is a **blind reproduction**; the plan is derived only from the pre-cutoff foundational works enumerated above.
- **Structural gaps addressed by the 5 claims** (from `LANDSCAPE.md § 4`): G1 (cross-method controlled comparison at matched budget), G2 (steering for functional correctness — C2's headline result), G3 (systematic cross-lingual per-concept transfer — C3), G4 (compositional interference between concept vectors — C4), G5 (small-model internal monitor vs large LLM judge, matched-budget head-to-head — C5).

---

## Mechanism Strategy

Taken verbatim from the prior resume-invocation analysis (do not re-derive):

```yaml
mechanism_strategy:
  directions: [Tuning & Editing, Location]
  rejected:
    - Causal Intervention — the five sub-claims are applied capability gains (steering / monitoring), not diagnostic causal verdicts on named components; specificity controls remain inside the Tuning milestones.
    - Formation Tracing — no genesis claim; would require Llama-3.1-8B pretraining checkpoints and data attribution, off-scope.
    - Unit Interpretation — RFM vectors are already the human-named interpretation via user-supplied concept pairs; no SAE / auto-interp decomposition required.
    - Decision Auditing — reproduction does not audit individual decisions against domain-knowledge notions of validity; natural follow-up, not in-scope.
  note: RFM extracts a per-block linear concept direction; all five claims apply that direction downstream (additively as steering for C1–C4; as a linear classifier for C5). Location enters as a supporting sub-direction inside each Tuning milestone — sweep all 32 blocks of Llama-3.1-8B, pick the best block(s) per concept by validation-set signal.
```

**Rationale (from `LANDSCAPE.md`)**: The linear-representation hypothesis is well-established (RepE, ITI, CAA, CAV) and RFM's contribution is a **more robust supervised extractor** producing a single linear direction per block; the whole family therefore lives inside "Tuning & Editing" (activation steering / weight-space editing) with "Location" as the supporting question — *which block?* — answered by a probe-accuracy sweep across all 32 blocks. No causal-patching / SAE-dictionary / circuit-discovery machinery is required for any of C1–C5.

**Compatible `/mechanism-skills` families** the downstream router should consider:
- **Steering vectors / activation steering** — primary; covers C1–C4 (additive residual-stream intervention).
- **Linear probing** — supporting Location signal (best-block picker) and primary for C5 (linear classifier on activations).
- **Representation engineering** — framing (per-block linear concept space).

**Explicitly pushed away from**: causal patching / attribution patching, SAE feature dictionaries, circuit discovery, weight-space editing (ROME/MEMIT).

---

## Recommended

**Recommended:** #1 — RFM Concept-Vector Steering & Monitoring (faithful reproduction of the 5-claim RFM steering + monitoring hypothesis on Llama-3.1-8B-Instruct)
