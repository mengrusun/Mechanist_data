# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Representation and Parameter Analysis / Representation Engineering
chosen_idea_title: Verifying Representation-Level Circuit Breakers (RR) as a Safety Intervention
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md
  - skills/mechanism-skills/probing/residual-stream-states/SKILL.md
  - skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Candidates

1. **[recommended]** Representation and Parameter Analysis / Representation Engineering — RepE is the exact framework for RR: extract a **concept direction** (`d_h`) from paired harmful/benign residual activations via mean-difference / PCA (M1); use that direction as **both a read-out** (M4 cosine projection to measure the reroute post-tune) **and a write-in** (M3 LoRA fine-tune whose loss pushes tuned harmful activations orthogonal to the pre-tune `d_h`, i.e. a training-time variant of RepControl — this is the LoRRA extension the RepE framework ships). Every one of the plan's four claims flows through this same directional handle: C1 = direction extractability + reroute observation; C2 = the reroute's downstream effect on ASR + capability; C3/C4 = the same reroute applied on a different base (Mistral / agent).
   - path: skills/mechanism-skills/representation-and-parameter-analysis/representation-engineering/SKILL.md

2. Probing / Residual Stream States — Runs the M1 half only: linear probe per layer to establish that a harmful-vs-benign discriminative direction *exists* at ≥ 3 mid-late layers (C1 identifiability half). Weaker as the *primary* family because probing alone is decodability, not intervention — it cannot verify the reroute (C1b) or produce the tuned model needed for C2/C3/C4. In the RepE composition it is the direction-extraction sub-step; not sufficient on its own.
   - path: skills/mechanism-skills/probing/residual-stream-states/SKILL.md

3. Causal Attribution / Patching — Would verify C1 via **inference-time** activation patching (swap harmful residuals with benign at site `S` and measure refusal). Weak fit because the plan explicitly does *not* do inference-time patching — RR is a **training-time** intervention that reshapes activations, not a per-example swap. Patching would answer "would refusal happen if we patched activations right now?" but not "does RR training move activations off `d_h`?". Kept in the pool because it is a legitimate alternative causal test the verify stage may swap in as a robustness variant.
   - path: skills/mechanism-skills/causal-attribution/patching/SKILL.md

## Composition plan

**Screen → Decode → Verify → Recover** for the four claims:

1. **Screen (M1, ~0.5 GPU-h)** — RepE / RepReading direction extraction on base Llama-3-8B-Instruct: forward-pass 384 paired (harmful, benign) prompts through all 32 residual-stream layers, compute per-layer mean-difference direction `d_h^s` and per-layer linear-probe AUC on 128 held-out pairs. Choose sites `S` = top-6 layers by AUC in a contiguous mid-band (heuristic from `experiment-tips/steering-block-selection/SKILL.md`).

2. **Decode / observe (M1)** — Layer-wise AUC report verifies the *decodability* half of C1 (identifiability): does a linear discriminator on the residual stream separate harmful from benign at ≥ 3 mid-late layers with AUC > 0.8?

3. **Verify / intervene (M3, ~2.0 GPU-h)** — LoRA-based RepControl fine-tune: the RR objective `L = α·L_rr + β·L_ret + λ_lm·L_lm` where `L_rr` penalizes `cos(a_s^tuned(h), d_h^s,base)^2` at each site in `S` on harmful inputs, `L_ret` preserves benign residual activations and fluency. This is a **training-time RepControl** (a.k.a. LoRRA in the RepE framework) rather than inference-time additive steering — the intervention is baked into weights.

4. **Confirm the reroute empirically (M4, ~0.3 GPU-h)** — RepReading again on the RR-tuned model over 128 held-out pairs: measure `cos(a_s^tuned(h), d_h^s,base)` per site (target ↓ ≥ 0.3) and `cos(a_s^tuned(b), d_h^s,base)` on benign (target |Δ| ≤ 0.1). Matched-null-control: also measure with a random orthogonal direction `d_ctrl` per site — must NOT show the same reroute (specificity check).

5. **Transfer measurements (M5 + M6 + M7, ~5 GPU-h)** — The reroute's *downstream* consequence on behavior: HarmBench ASR × 6 attack categories + MT-Bench + MMLU (M5, C2); PGD image-hijack on RR-tuned LLaVA-NeXT-Mistral (M6, C3); function-calling harm probe + BFCL (M7, C4). These evaluate the *effect* of the intervention rather than the mechanism per se.

**Cost notes**: total ~10 GPU-h across all seven milestones. RepE-family cost is dominated by the LoRA fine-tunes (M2 baseline: 2.0 GPU-h, M3 RR: 2.0 GPU-h, M6 Mistral RR: 0.8 GPU-h), each running on 1 A800-80GB.

## Plan reconciliation
<!-- One row per method_sensitive field declared on the intervention milestone(s). -->

- **n_pairs**: plan=512 (384 train + 128 held-out) → matches — RepE / RepReading direction extraction is stable at 200-500 paired examples on 8B-scale residual-stream activations; 384 train pairs is comfortably above the floor.
- **sites**: plan=top-k=6 contiguous mid-band (default layers ~10-20 of 32) → matches — this is exactly the RepE "middle-to-later layers for semantic concepts" heuristic (repe_demo.py's `range(-1, -num_layers//2, -2)` is the same idea in stride form); we make it AUC-driven at M1 instead of hardcoded.
- **metric**: plan=layer-wise AUC (extraction), cosine to `d_h^s,base` (post-RR diagnostic) → matches — cosine projection onto a mean-diff / PCA direction is the standard RepReading readout. AUC as the layer-selection metric is standard for the probing sub-step.
- **gpu_hours**: plan~M1(0.5)+M2(2.0)+M3(2.0)+M4(0.3)+M5(2.5)+M6(2.0)+M7(0.5) = 9.8 → matches within a 0.2 h reserve. LoRA fine-tune with the RR objective adds ~10% overhead vs plain SFT (compute `d_h^s`-projection at each forward pass on the fly on 384 train prompts), but that is already included in the M3 estimate.

reconciliation_status: ok

## Rationale

RepE is the only family that provides all four pieces the plan needs in a single unified framework:
1. **Contrastive-pair direction extraction** (mean-difference / PCA on paired harmful/benign residuals — exactly M1).
2. **Direction as both read-out and write-in** — M4 uses the direction as a monitor, M3 uses it as the target of a training-time reroute loss. The RepE framework's `lorra_finetune/` extension (LoRRA) is the canonical way to combine LoRA with a representation-preservation loss, which is precisely the RR objective's structure (`L_rr` reroutes harmful residuals, `L_ret` preserves benign residuals).
3. **Multi-site (residual-stream) intervention** across the mid-band — RepE routinely operates across ~5-15 residual-stream layers, matching the plan's k=6.
4. **Cross-model transfer via re-extraction** — RepE explicitly treats directions as basis-specific and re-extracts on each new base (RepE examples cover Llama-2, Llama-3, Mistral). This is exactly the M6 protocol: apply the same construction to Mistral-7B-Instruct-v0.2's residual stream.

Probing (candidate 2) is the M1 sub-step *within* RepE, not a separate family. Causal Attribution / Patching (candidate 3) is a strictly different intervention modality (inference-time swap vs training-time reroute) and would answer a different question; it's a legitimate `/auto-verify` swap-variant candidate but not the primary family.

**Cross-round exclusions**: none. `families_already_settled: []` (round 1 for this behavior + direction).

**Blind-reproduction constraint**: the RepE framework's GitHub repo is not on the forbidden-URL list, but per policy we do NOT clone or `pip install` it. We re-implement the pieces we need (mean-diff direction extraction, layer-wise linear probe, LoRA fine-tune with directional loss) from first principles — this is consistent with the plan's Section 5.1 independent reconstruction and with the "blind reproduction" clause of task.md.
