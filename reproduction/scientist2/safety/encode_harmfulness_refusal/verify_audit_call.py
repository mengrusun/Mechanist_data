#!/usr/bin/env python3
"""Reviewer call for Phase 2 experiment audit on all 5 claims."""

import json
import requests

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

prompt = """You are a rigorous experiment integrity auditor reviewing 5 claims from a mechanistic interpretability paper about harmfulness and refusal directions in LLM residual streams. I will provide full experimental context; you must audit each claim on Checks A-F.

=== PROJECT CONTEXT ===

Model: Llama-3-8B-Instruct (fp16)
Dataset: AdvBench (520 harmful behaviors) + Alpaca (520 matched benign)
Split: 60/20/20 = 312 direction-extraction / 104 val / 104 test

GROUND TRUTH SOURCES:
- Harmfulness labels: from AdvBench dataset (dataset-provided labels, not model output)
- Refusal labels for r-direction: from is_refusal() string classifier applied to model completions on AdvBench prompts. Documented explicitly in directions.json as "contrast_source: harmful-vs-benign proxy" because only 7/520 bare prompts were not refused. This is labeled as proxy, not claimed as ground truth.
- M5 Llama Guard 3 8B: used as external baseline comparator, not for generating probe training GT labels.

MILESTONE SUMMARIES:
- M-prep (R001): extracts cached activations at 6 positions x 33 layers. Best h: layer 11, t_final_instr, AUROC=0.9998. Best r: layer 13, t_post_instr, AUROC=1.000. All files exist: activations.pt (1.6 GB), directions.json, probe_auroc.csv, meta.json, responses.jsonl.
- M1 (R002): Three sub-tests on val set (104 pairs). Results in results/m1/claim1_verdict.json (exists): verdict=partial, auroc_h=0.9998, cos_h_r=0.174, split_half_reference=0.883, auroc_r=NaN (only 7 natural jailbreaks so refusal-side split impossible). Verdict partial means sub_ii passes (cosine), sub_i/iii unmeasurable.
- M2 (R003): Position ladder AUROC. Results in results/m2/claim2_verdict.json (exists): verdict=not-supported, crossover_h=-0.00018 (both positions at ceiling 0.9998+), crossover_r=NaN.
- M3 (R004-R019): 28-cell grid (4 directions x 7 alpha). Alpha grid: {-2, -1, -0.5, 0, +0.5, +1, +2} in DIRECTION NORM units (not sigma_proj; sigma_proj_h=1.58, sigma_proj_r=1.96). In sigma_proj units this is approximately [-1.27, -0.63, -0.32, 0, 0.32, 0.63, 1.27]. Fluency proxies (mean_logp_completion, rep_rate) logged per cell. Random-direction baseline: 1 matched-norm direction (n_random=1, not 30). Results in results/m3/claim3_verdict.json (exists): verdict=supported. 28 cells completed, 0 collapsed.
- M4 (R020-R021): GCG + PAP attacks. Results in results/m4/claim4_verdict.json (exists): verdict=not-supported, ASR=0/500 for both families.
- M5 (R022-R023): Probe vs Llama Guard. Results in results/m5/claim5_verdict.json (exists): verdict=supported, AUROC_probe=1.000, AUROC_LG=0.9992, compute ratio=1.05e-8 (well under 5% threshold). Test set: 400 items (100 bare-harmful + 100 benign + 200 XSTest-safe; 0 successful jailbreaks).

TRACKER STATUS: All R001-R023 marked "done", "done+neg", or "done+part". No pending or failed rows.

SCOPE LANGUAGE CHECK:
- C1 claim text: "two distinct, approximately-linear, independently-recoverable directions" - no "comprehensive" or "extensive"
- C2 claim text: "position dissociation: h at t_final-instr, r at t_post-instr" - spatial claim, no overreach
- C3 claim text: "additive steering dissociates the effects" - 28 cells with dose-response
- C4 claim text: "notable class of successful jailbreaks suppresses r while h remains active" - on 2 attack families; says "notable class" not "all"
- C5 claim text: "lightweight harmfulness-direction probe matches or beats Llama Guard 3 8B" - head-to-head comparison

=== AUDIT CHECKLIST ===

Check A. Ground Truth Provenance
FAIL if GT derived from model outputs without explicit proxy labeling.
Note: the refusal label IS from model outputs (is_refusal on completions) but is EXPLICITLY DOCUMENTED as "harmful-vs-benign proxy" in directions.json, and the plan itself acknowledges this limitation at both M-prep and C1 verdict sections.

Check B. Score Normalization
FAIL if any metric divided by model's own output max/mean. Check: AUROC uses dataset labels as ground truth, not model scores. Probe AUROC computed against dataset labels. Cosine similarity is a direction-vs-direction geometric measure. No self-normalization observed.

Check C. Result File Existence (per-claim)
FAIL if claimed results reference nonexistent files or mismatched numbers.
All verdict JSONs exist and match the numbers in EXPERIMENT_RESULTS.md.

Check D. Dead Code Detection
WARN if metric functions defined but never called.

Check E. Scope Assessment
WARN if scope language exceeds actual evidence.

Check F. Evaluation Type Classification

=== CLAIMS ===

C1: "Two distinct, approximately-linear, independently-recoverable directions in the residual stream"
Milestones: M-prep, M1
Key issue: r direction extracted from harmful-vs-benign proxy (labeled), not true refused-vs-complied

C2: "Position dissociation: h at t_final-instr, r at t_post-instr"
Milestones: M-prep, M2
Key issue: both positions decode harmfulness at ceiling; crossover unmeasurable

C3: "Additive steering dissociates the effects: h flips internal harmfulness readout with refusal unchanged; r flips refusal with harmfulness readout unchanged"
Milestones: M3
Key issues: alpha in norm units (not sigma_proj), random baseline n=1, h at alpha>1 shows response at alpha=+2 only for r

C4: "A notable class of successful jailbreaks suppresses r while h remains active"
Milestones: M4
Key issue: ASR=0, so claim cannot be tested on this data; verdict is not-supported for correct scientific reasons (model too aligned)

C5: "A lightweight harmfulness-direction probe matches or beats Llama Guard 3 8B at flagging jailbreaks, at a fraction of the compute"
Milestones: M5
Key issue: test set has 0 jailbreaks (M4 ASR=0); headline says "flagging jailbreaks" but test set only has bare-harmful and benign; probe reaches ceiling (AUROC=1.000) possibly because task is easier than claimed

=== OUTPUT FORMAT ===

Return ONLY a valid JSON object (no markdown, no preamble) with this exact structure:
{"C1":{"A":"PASS","B":"PASS","C":"PASS","D":"PASS","E":"PASS","F":"real_gt","overall":"PASS","details":"one paragraph"},"C2":{"A":"PASS","B":"PASS","C":"PASS","D":"PASS","E":"PASS","F":"real_gt","overall":"PASS","details":"one paragraph"},"C3":{"A":"PASS","B":"PASS","C":"PASS","D":"PASS","E":"PASS","F":"real_gt","overall":"PASS","details":"one paragraph"},"C4":{"A":"PASS","B":"PASS","C":"PASS","D":"PASS","E":"PASS","F":"real_gt","overall":"PASS","details":"one paragraph"},"C5":{"A":"PASS","B":"PASS","C":"PASS","D":"PASS","E":"PASS","F":"real_gt","overall":"PASS","details":"one paragraph"}}

Replace each "PASS" with the actual verdict (PASS, WARN, or FAIL) and fill in details. Be strict."""

resp = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000},
    proxies={"http": None, "https": None},
    timeout=180
)
print(f"Status: {resp.status_code}")
try:
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    print("REVIEWER_RESPONSE:")
    print(content)
except Exception as e:
    print(f"Parse error: {e}")
    print(resp.text[:2000])
