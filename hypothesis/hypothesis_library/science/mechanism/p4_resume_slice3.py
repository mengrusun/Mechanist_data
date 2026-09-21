# -*- coding: utf-8 -*-
import json, os

BASE = "/data/zhenqian/hypo_eval_825/hypo_task/science/mechanism"
SLICE = os.path.join(BASE, "p4_slices/slice_3.json")
TMP = SLICE + ".tmp"
SEVEN = ["Name","Title","Short Hypothesis","Related Work","Abstract","Experiments","Risk Factors and Limitations"]

data = json.load(open(os.path.join(BASE, "claim_1.json")))
out = json.load(open(SLICE))
assert len(out) == 3, len(out)

def base(idx):
    return {k: data[idx][k] for k in SEVEN}

def rep(rec, field, old, new):
    assert rec[field].count(old) == 1, (field, old[:50])
    rec[field] = rec[field].replace(old, new)

def app(rec, field, anchor, extra):
    rep(rec, field, anchor, anchor + extra)

def save(rec):
    for k in list(rec.keys()):
        assert k in SEVEN, k
    assert list(rec.keys()) == SEVEN, rec.keys()
    out.append(rec)
    json.dump(out, open(TMP, "w"), indent=4, ensure_ascii=False)
    os.replace(TMP, SLICE)

# ---------------- idx 33 ----------------
r = base(33)
app(r, "Short Hypothesis",
    "unlocks count generalization that a non-local model otherwise fails, at preserved validity.",
    " PRIMARY CLAIM: the causal locality-enforcement test on the masked-diffusion generator (H3); the novelty is the TRANSFER of the diffusion locality<->generalization theory into scientific sequence/graph design plus its mechanistic validation, not a new theory of locality.")
app(r, "Related Work",
    "a Location + Causal Intervention mechanism claim in a high-stakes domain.",
    " Exact delta from the CLEVR result: prior work showed locality governs how many OBJECTS an image model can compose; we test whether the same locality principle governs how many independent FUNCTIONAL elements (catalytic sites, binding motifs, functional groups) a protein/molecule generator can pack beyond its training-count support -- nontrivial because scientific elements are discrete, oracle-countable, and often physically coupled rather than spatially separable pixels. We contrast prior work vs this proposal on five axes -- domain (vision -> protein/chemistry), object type (pixels -> sequences/graphs), locality signal (pixel-conditioner sparsity -> element-placement-feature receptive field), intervention (image-score masking -> denoiser feature masking), and outcome (image length generalization -> valid over-count design). Concrete qualitatively-different tasks: a protein with MORE tandem catalytic-triad-like motifs than any training protein, and a molecule with MORE independent halogen/hydroxyl decorations than any training molecule -- copy-count extrapolation against hard structural oracles, not spatial object counting.")
rep(r, "Abstract",
    "Results would give a mechanistic, actionable principle for pushing scientific generators past their training element counts.",
    "Beyond a probe, the payoff is a phenomenon-level rule for when repeated-element extrapolation works plus a cheap locality-masking recipe (primary endpoint: the causal test on the masked-diffusion generator) that converts otherwise-failed high-count generations into valid designs and cuts wasted sampling for protein and molecular designers.")
app(r, "Experiments",
    "two public diffusion generators on 1-2 GPUs, academic-feasible.",
    "\n\nPRE-REGISTERED PRIMARY ENDPOINT: the causal locality-enforcement gain (Experiment 4) on the masked-diffusion generator for ONE protein element (a specified short catalytic-triad-like motif) and ONE molecule element (halogen decoration); all other models/elements/layers are secondary robustness. Every experiment's success/undecidable/negative bands below are fixed before running. CONFOUND CONTROLS analyzed explicitly: training data-count diversity, sequence/graph length, motif rarity, and element independence are recorded per (model, element) and entered as covariates so the locality<->boundary link is tested above them.\n\nExperiment 0 (pipeline validation on a synthetic benchmark): Before the scientific runs, validate the locality-measurement and locality-enforcement code on a synthetic countable-element sequence task with a KNOWN local generative process (elements placed by a fixed sparse rule), confirming the locality score recovers the planted radius and that enforcement restores over-count generation; a pipeline failing here is fixed before any protein/molecule claim.")
app(r, "Experiments",
    "the spatial/sequential decay profile of the placement-logit gradient (a locality radius).",
    " Operationally the distal-ablation score is 1 - (mean |Delta placement-logit| under ablation of positions beyond radius r) / (mean |Delta placement-logit| under ablation of all context), and the Jacobian radius is the smallest r capturing >= 90% of the total |d placement-logit / d input| mass; the per-feature locality score is reported with its exact r.")
save(r)

# ---------------- idx 34 ----------------
r = base(34)
app(r, "Short Hypothesis",
    "the true quality of generated designs together.",
    " We do NOT presume the text-LLM shared-circuit (SCIURus) result transfers; the PRIMARY contribution is to ADJUDICATE, in scientific generators, between a shared uncertainty/correctness circuit and a DISSOCIATION (uncertainty that merely tracks OOD-ness without predicting quality), using causal, specificity-controlled interventions, with the calibrated readout and the epistemic-gated controller as secondary payoffs.")
app(r, "Related Work",
    "here the object of study is the model's self-knowledge and its causal link to quality.",
    " Recent text-model evidence also reports DISSOCIATION between uncertainty and correctness representations; we therefore frame the contribution as adjudicating whether scientific generation patterns with the shared-circuit or the dissociated regime, rather than assuming either, and as isolating unique information beyond token entropy, length, and novelty distance that black-box confidence probes conflate.")
rep(r, "Abstract",
    "whether it is causally shared with the true quality of the object being produced",
    "whether it is causally and specifically shared with the true quality of the object being produced or merely tracks novelty")
rep(r, "Abstract",
    "We predict a genuine, shared, better-calibrated self-signal—recasting interpretability as a source of trustworthy design confidence.",
    "Rather than presume the text-LLM result transfers, we adjudicate shared-circuit vs dissociation while controlling for token entropy, length, and novelty distance. If a specific, better-calibrated signal exists a designer can triage candidates before expensive oracle/wet-lab checks (fewer oracle calls, higher top-k valid yield); if not, we establish that scientific generators do not 'know what they know', a safety-relevant null.")
rep(r, "Experiments",
    "split 70/15/15 into probe-train / val / test by generation seed (no object leakage).",
    "split into probe-train / val / test at the CLUSTER level -- proteins by sequence-identity/family bands (mmseqs2 clusters), molecules by Bemis-Murcko scaffold / nearest-neighbour Tanimoto bands -- so near-duplicate generations cannot leak across splits (seed-only splitting is insufficient).")
app(r, "Experiments",
    "two ESM-2 sizes + small SMILES LM + ESMFold on 1-2 GPUs, academic-feasible.",
    "\n\nPRE-REGISTERED PRIMARY ENDPOINT: the causal, specificity-controlled coupling test (Experiment 2) -- a shared uncertainty/quality circuit is claimed only if the epistemic ablation moves internal confidence AND oracle quality together (coupling r >= 0.5, CI excluding 0) while matched-energy, reconstruction-preserving-null, and equally-sparse non-epistemic directions do NOT (|r| <= 0.2); calibration and controller are secondary. UNIQUE-INFORMATION control (applied throughout): every quality-prediction claim is re-tested with partial correlation controlling for token entropy, sequence/graph length, and novelty distance, so the self-signal must add information beyond these. Layer/feature selection uses a held-out model/layer with Benjamini-Hochberg correction across the feature/layer grid to avoid cherry-picking. MID-GENERATION READOUT: for masked-infilling ESM-2, 'generation progress' is the fraction of positions unmasked along a fixed infilling schedule, and the self-signal is read from the partially-infilled activation at that fraction.")
rep(r, "Experiments",
    "random-direction / matched-property-feature controls must NOT couple confidence and quality.",
    "random-direction, matched-energy, reconstruction-preserving-null, and equally-sparse non-epistemic controls must NOT couple confidence and quality.")
app(r, "Experiments",
    "then false-abort rate;",
    " then precision@k when the self-signal RANKS candidates for oracle evaluation (the practical triage use case);")
app(r, "Risk Factors and Limitations",
    "if coupling is non-specific, we report that as a negative result.",
    " We claim 'causally shared' only if effects are bidirectional, specific, and non-destructive across both models and domains; otherwise coupling is reported as correlational.")
save(r)

# ---------------- idx 35 ----------------
r = base(35)
app(r, "Short Hypothesis",
    "an intervention-free internal dimensional-consistency signal flags it.",
    " PRIMARY CLAIM (single crisp headline): in NSR transformers dimensional consistency is DECODABLE from activations but NOT compactly causally steerable, so fit-steering silently breaks units; the internal audit and the concept-vs-token-statistics test are secondary.")
app(r, "Related Work",
    "by moving from operator circuits to a physically-meaningful invariant.",
    " Unlike units-constrained/physics-prior SR (Buckingham-Pi SR, StruSR), which ENFORCES dimensional consistency architecturally, we do the opposite -- we leave the model free to violate units and ask whether it INTERNALLY represents the invariant, an interpretability/audit question rather than a constraint-enforcement method, which is why this is not just another physics-prior paper. Who cares: symbolic-regression and AI-for-science practitioners (who risk publishing dimensionally-invalid 'laws'), physics-informed-discovery researchers, and interpretability researchers seeking exact, free oracles.")
rep(r, "Abstract",
    "We predict fit is steerable, dimensional consistency is decodable-but-weakly-steerable, and a cheap internal audit catches dimensional violations before downstream physical use.",
    "We predict fit is steerable while dimensional consistency is decodable-but-weakly-steerable. So what: better-fitting equations can be physically meaningless, and a cheap internal audit that prunes dimensionally-invalid candidates from the beam raises true-law recovery and saves search compute -- turning an exact, free oracle into a diagnostic and control target.")
app(r, "Experiments",
    "one frozen NSR transformer + <=4 SAEs on a single GPU, academic-feasible.",
    "\n\nPRE-REGISTERED PRIMARY ENDPOINT: the decodable-but-inert result (Experiment 2) on ONE model family (SymbolicGPT/DGSR-style) and ONE dataset (Feynman + its synthetic dimensioned augmentation) -- dimension-probe AUROC >= 0.75 with dimensional-restoration success <= 0.2 and effect < 1/3 of fit steerability; DiffuSR, second size, and second layer are secondary robustness. Steering STRENGTH is matched across fit and dimensional interventions by equalizing the induced activation-norm perturbation, making the fit-vs-units contrast budget-fair. The decision tree (support / undecidable / negative bands per experiment) is fixed before running.")
app(r, "Experiments",
    "auto-label top features via max-activating contexts.",
    " Minimal training ablation (probe-leakage / units-awareness control): compare the SAME probes on models trained (i) with per-variable units ingested, (ii) with units withheld, and (iii) with RANDOM-label units; genuine dimensional decoding must appear only with real units and must survive the same-structure/different-dimension dissociation pairs, ruling out surface-token leakage.")
save(r)

# ---------------- idx 36 ----------------
r = base(36)
app(r, "Short Hypothesis",
    "reframes when steering can be trusted for real one-shot design.",
    " PRIMARY CONTRIBUTION: the seed-lottery PHENOMENON and the control-authority ratio as its diagnostic; the tiny-sample predictor and the best-of-n remedy are secondary.")
app(r, "Related Work",
    "by asking a stochasticity/reproducibility question about steering ITSELF.",
    " Formally, control-authority = (mean target-property shift induced by the steer) / (target-property standard deviation across unsteered seeds+temperatures) -- a signal-to-sampling-noise ratio, not merely a repackaged Cohen's d, because the denominator is the model's own generative sampling spread a one-shot user actually faces, and the diagnostic is the per-object success it implies (for an approximately Gaussian property, authority a implies one-shot success ~ Phi(a - z_band), the minimal argument that makes shift/spread the right one-shot quantity rather than a post-hoc metric). Contrast with Tan et al. (steering-vector unreliability): they study cross-prompt/method fragility of the MEAN effect in text; we decompose signal vs sampling noise at the level of a single scientific object with an objective oracle and predict one-shot reproducibility.")
rep(r, "Abstract",
    "Results recast steering evaluation from mean shifts to per-design reliability.",
    "Results recast steering evaluation from mean shifts to per-design reliability: a user reads the authority score and trusts a single draw only when it exceeds 1 (else spends a small best-of-n budget), whereas current benchmarks would call the same low-authority steer a success. The pre-registered primary endpoint is the seed-lottery fraction for molecular logP on the SMILES model, with other properties/domains confirmatory.")
app(r, "Experiments",
    "two ESM-2 sizes + small SMILES LM + ESMFold on 1-2 GPUs, academic-feasible.",
    "\n\nPRE-REGISTERED PRIMARY ENDPOINT: the seed-lottery fraction (Experiment 1) for ONE primary (domain, property) = (SMILES model, logP) -- fraction of bank features with control-authority < 1.0 and one-shot success < 0.5 while their aggregate mean shift is nominally successful; all other properties/domains/SAEs are confirmatory robustness. Bands below are fixed before running.")
app(r, "Experiments",
    "the geometric predictor adds >= 0.1 R^2 over mean-shift/concept-correlation.",
    " We report calibration (reliability curve, ECE) and bootstrap CIs for the small-sample predictor, not only rho/R^2, plus an ABLATION comparing control-authority against simpler alternatives (raw mean shift; unsteered variance alone; shift normalized by validity loss).")
app(r, "Experiments",
    "within 2x the compute of oracle-guided selection.",
    " Controls: a no-information RANDOM best-of-n selection (isolates the internal probe's value) and a compute-matched comparison against simply lowering sampling temperature, so the fix is credited only if it beats both at equal budget.")
save(r)

# ---------------- idx 37 ----------------
r = base(37)
app(r, "Short Hypothesis",
    "predicts where a given interpolation will break.",
    " PRIMARY CLAIM: the path-cliff/non-monotonicity PHENOMENON in interpretable SAE feature space (an empirical audit, not a new theory of why cliffs form); the SAE-vs-raw comparison, the geometry predictor, and the reroute are secondary.")
app(r, "Related Work",
    "and to contrast SAE-feature vs raw-activation interpolation on that axis.",
    " Exact delta: the same broad interpolation-invalidity concern as ACAI/ChemFlow/ChemSpacE, but a NEW representation class -- the interpretable SAE feature space of a FROZEN foundation-model generator rather than a bespoke VAE/GAN latent trained for smooth interpolation -- plus a new predictability claim. Prior latent-interpolation results do not transfer because SAE features are sparse, over-complete, and not optimized for geodesic smoothness, so linear feature-space paths can leave the model's activation manifold where a purpose-built VAE latent would not. The novelty is thus the setting and the diagnostic, not the base phenomenon.")
rep(r, "Abstract",
    "Results recast interpretable interpolation from an assumed-smooth design tool into an auditable, geometry-predictable one.",
    "Lead-optimization chemists and protein engineers who pick intermediates along such morphing paths would otherwise ship a seemingly-safe but silently-invalid intermediate; if cliffs are real, path vetting/rerouting becomes mandatory and naive linear interpolation should be retired as a default heuristic. The pre-registered primary endpoint is the cliff rate for logP-bracketed molecule pairs, and the phenomenon stands even if the predictor/fix fail -- recasting interpretable interpolation from an assumed-smooth tool into an auditable, geometry-predictable one.")
app(r, "Experiments",
    "two ESM-2 sizes + small SMILES LM + ESMFold on 1-2 GPUs, academic-feasible.",
    "\n\nPRE-REGISTERED PRIMARY ENDPOINT: the cliff rate and endpoint-blindness (Experiment 1) for logP-bracketed valid molecule PAIRS on the SMILES model; proteins, sizes, SAE variants, and slerp are confirmatory. Bands below are fixed before running, and the phenomenon claim (Experiment 1) is evaluated independently of the predictor/reroute claims (Experiment 3).")
rep(r, "Experiments",
    "(sharing the same decode pathway to isolate the SAE effect)",
    "(sharing the same decode pathway, with SAE-vs-raw reconstruction error matched, to isolate the SAE effect rather than decode noise)")
app(r, "Experiments",
    "feature-path curvature (nonlinearity of feature activations along the interpolant).",
    " The geometry signature is compared against simple baselines -- endpoint feature-space distance, per-intermediate latent norm, a path-linearity score, and decode entropy/pseudo-perplexity -- and must beat all of them (nested comparison).")
rep(r, "Experiments",
    "reroute the path to minimize off-manifold distance (small on-manifold projected steps);",
    "reroute the path to minimize off-manifold distance (small on-manifold projected steps), compared against standard path-smoothing and an embedding shortest-path baseline;")
save(r)

# ---------------- idx 38 ----------------
r = base(38)
app(r, "Short Hypothesis",
    "an intervention-free rebound signature predicts which anti-property requests will silently fail.",
    " PRIMARY AXIS: the design-space ironic-rebound phenomenon (forbidden element appears above base rate under suppression) as a DOMAIN TRANSFER of text/human negation-hardness into scientific design with EXACT structural oracles; the subspace-projection fix and the rebound audit are secondary.")
app(r, "Related Work",
    "where an anti-property that silently rebounds is a safety failure.",
    " We contrast four axes -- text negation (Strong-hallucinations/White-Bear: fuzzy concepts, no exact oracle, no design object), concept suppression (INLP/RePS/SRS: removal quality in text/CLIP, single-vector insufficiency noted), scientific POSITIVE steering (InterPLM/ProtSAE: include-X only, appearance-checked), and THIS proposal (positive-vs-negative head-to-head on de-novo scientific objects with exact SMARTS/HMMER/DSSP presence oracles and a base-rate-referenced rebound test).")
rep(r, "Abstract",
    "We predict a stark asymmetry, an ironic-rebound analog, and that projection-based exclusion plus a rebound audit partially restore reliable negative design.",
    "More broadly this is the general problem of CONSTRAINT SATISFACTION in generative design, where 'must-not-contain' constraints are as common as 'must-contain' ones: for molecular-design, protein-engineering, and controllable-generation practitioners, a safety constraint (exclude a toxicophore, an off-target motif) may silently fail with false confidence, needing different control and monitoring than positive steering. Pre-registered primary endpoint: the positive-vs-negative asymmetry gap (Experiment 1). We predict a stark asymmetry, an ironic-rebound analog, and that subspace projection plus a rebound audit partially restore reliable negative design.")
app(r, "Experiments",
    "two ESM-2 sizes + small SMILES LM + ESMFold on 1-2 GPUs, academic-feasible.",
    "\n\nOPERATIONAL DEFINITION of the negative-specification intervention: 'exclude-X' = negatively clamping the element's SAE feature(s) (or projecting out its subspace) at generation time to a target at/below the 5th percentile of its natural on-distribution activation, all other positions/features free; 'exclusion success' = fraction of VALID generations whose exact oracle reports the element ABSENT. PRE-REGISTERED PRIMARY ENDPOINT: the asymmetry gap (Experiment 1) pooled across elements and both domains; rebound, subspace fix, and audit are secondary. Base-rate statistics: rebound is tested as occurrence minus the unconditioned base rate with a two-proportion test and Wilson CIs, and elements are stratified by base rate so rare-element rebound is not trivially inflated.")
rep(r, "Experiments",
    "Repeat across both domains, both ESM-2 sizes, TopK vs Ordered SAEs, two layers, and all elements; report the fraction of cells",
    "MECHANISTIC MEDIATION check: verify each selected feature MEDIATES (not merely correlates with) the element -- ablating it lowers element occurrence and patching it into element-free runs raises it -- with matched-strength RANDOM-feature suppression and a NO-SAE (raw difference-of-means) exclusion arm as controls for generic OOD drift and validity collapse. Repeat across both domains, both ESM-2 sizes, TopK vs Ordered SAEs, two layers, and all elements; report the fraction of cells")
save(r)

# ---------------- idx 39 ----------------
r = base(39)
app(r, "Short Hypothesis",
    "an intervention-free internal periodicity-consistency signal flags it.",
    " Stated as four hypotheses: (H1) local single-position content is causally steerable; (H2) periodicity -- exact PERIOD, copy COUNT, and copy-CONSISTENCY -- is decodable from activations but NOT compactly steerable; (H3) 'make it repetitive' steering therefore yields repeat-like content with broken periodicity (the blind spot); (H4) an internal periodicity-consistency signal flags it. PRIMARY CLAIM: the decodable-but-inert periodicity result (H2) with the blind-spot demonstration (H3); the audit (H4) is secondary.")
app(r, "Related Work",
    "turns exact repeat detection into a mechanism-and-audit target.",
    " Contrast: InterPLM/ASPO/CB-pLM control LOCAL properties (motifs, GRAVY, molecular weight, per-position concepts) evaluated by appearance; this proposal uniquely tests control of a GLOBAL, position-coupled invariant (period/count/copy-consistency). Concrete applications of exact repeat control: modular repeat-protein scaffolds, repeat-based binders (e.g., designed solenoids), and symmetric architectures -- design targets currently bottlenecked by the difficulty of enforcing faithful long-range copies.")
rep(r, "Abstract",
    "We predict content control succeeds, periodicity control is decodable-but-weakly-steerable, and a cheap internal audit catches broken repeats before costly structural validation.",
    "Who cares: protein designers building modular scaffolds and interpretability researchers auditing controllability claims -- if the blind spot holds, steering benchmarks must add global position-coupled invariants, and if periodicity proves steerable, that is a new controllable design capability. We predict content control succeeds, periodicity is decodable-but-weakly-steerable, and a cheap internal audit (pre-registered primary endpoint: H2 plus the blind-spot rate H3 on ESM-2 650M) catches broken repeats before costly structural validation.")
app(r, "Experiments",
    "two ESM-2 sizes + ESMFold + repeat tools on 1-2 GPUs, academic-feasible.",
    "\n\nOPERATIONAL DEFINITIONS (fixed in one place): PERIODICITY = a detected repeat with unit length (PERIOD) p and copy COUNT c; COPY-CONSISTENCY = mean pairwise sequence identity among detected units; SILENT-BROKEN-REPEAT = a valid, repeat-like-composition sequence for which the exact oracle finds no period / a wrong period / copy-consistency below a pre-set threshold; the INTERNAL periodicity score = the periodicity-probe confidence read from the steered activation. PRE-REGISTERED PRIMARY ENDPOINT: the decodable-but-inert result (Experiment 2) plus the silent-broken-repeat rate (Experiment 3) on ESM-2 650M with sequence-level oracles; second size, layers, SAE variants, and structural (CE-Symm) corroboration are secondary. A power/variance analysis on a pilot of 200 sequences sets the N needed for the main AUROC/effect-size comparisons to resolve the stated bands.")
rep(r, "Experiments",
    "Random-direction and matched-content-feature controls.",
    "Random-direction, matched-content-feature, PERIOD-MATCHED-nonrepeat, and composition-matched-but-periodicity-orthogonalized direction controls.")
save(r)

print("appended", len(out), "records total")
for i,rec in enumerate(out):
    print(i, rec["Name"], "fields ok:", list(rec.keys())==SEVEN)
