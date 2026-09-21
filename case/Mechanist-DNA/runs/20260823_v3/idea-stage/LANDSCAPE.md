# Literature Landscape — Steering Evo2-7B toward high α-helical content

**Behavior-source**: given · **Mechanism**: discovery · **Date**: 2026-08-23
Context for baselines / metrics / candidate mechanism families only — it never overrides task.md.

## The model under study

Evo2-7B (Arc Institute) is a genomic foundation model over raw nucleotides at single-nucleotide
resolution, StripedHyena2 architecture (mix of short-explicit / medium-regularized / long-implicit
hyena operators plus attention), 1M-token context. It generates DNA autoregressively and can design
synthetic sequences. Its activation stream is therefore over nucleotides, while the target property
(α-helix fraction) is defined on the **protein encoded by the generated DNA** — the measurement
pipeline must translate a generated ORF and run secondary-structure assignment.

## Sub-directions relevant to the behavior

1. **Activation steering / contrastive activation addition (CAA).** A mature, cheap inference-time
   lever: build a steering vector as the difference of mean activations between attribute-high and
   attribute-low examples and add it during generation. Extensions: mean-centring, multi-property
   composition, SAE-feature-targeted steering, distributional/end-to-end learned steering (LinEAS),
   dose-response ("steering strength"). Directly applicable to a residual/hyena activation site in
   Evo2.
2. **SAE-feature steering on Evo2 specifically.** Goodfire/Arc trained SAEs on Evo2 layers (layer ~26
   most biologically informative) and demonstrated **steering of generations** by manipulating
   interpretable features. This is a ready-made, Evo2-native source of candidate structure features
   to clamp — a strong Location + Unit-Interpretation entry point.
3. **Protein-LM steering analogues.** "Steering Protein Language Models" and "Interpreting and
   Steering Protein LMs through SAEs" (both 2025) show attribute steering (incl. structural
   attributes) works on protein sequence models — transferable recipe, though our model emits DNA,
   not amino acids, adding the ORF/translation layer.
4. **Control-vs-validity trade-off.** "In-Distribution Steering" and the steering-reliability /
   non-identifiability line (2024–2026) document that aggressive steering degrades coherence /
   in-distribution-ness and that steering vectors can be unreliable / non-identifiable. For DNA this
   maps precisely onto the task's HARD validity requirement: raising α-helix must not push generations
   off the manifold of valid, biologically plausible ORFs.

## Open problems / gaps this project sits in

- Most steering literature targets **text** LLMs and **single-token** attributes; steering a
  **genomic** model for a property of the **translated protein** (a downstream, multi-token, structural
  readout) is comparatively unexplored.
- The validity constraint makes this a **constrained** steering problem: the interesting question is
  the dose-response frontier where α-helix rises *before* sequence validity collapses.
- Where the α-helix propensity is represented inside a hyena/StripedHyena stack (vs. a pure
  Transformer) is not established — Location is a genuine sub-question, not a solved one.

## Implications for the mechanism plan (advisory)

- **Candidate localization sources**: (a) linear probes for α-helix fraction across Evo2 layer
  activations; (b) pre-existing Evo2 SAE features (layer ~26) screened for structure selectivity.
- **Intervention family (chosen later by /mechanism-skills)**: contrastive activation
  addition and/or SAE-feature clamping at inference, with a steering-coefficient dose grid.
- **Baselines / metrics**: unintervened Evo2-7B generation as baseline; α-helix fraction via
  translate-ORF → structure/SS assignment (ESMFold + DSSP or an SS predictor); validity via ORF
  integrity (start/stop, frame, length, coding plausibility); specificity via matched-control
  direction + off-target properties (β-sheet fraction, GC content, sequence naturalness/perplexity).
- **Known pitfalls to control for**: steering-vector unreliability/non-identifiability (use matched
  controls + multiple seeds), and the control-vs-coherence trade-off (report the validity frontier,
  not a single high-coefficient point).
