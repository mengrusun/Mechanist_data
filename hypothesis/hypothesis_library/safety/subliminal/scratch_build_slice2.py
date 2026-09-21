import json, os

IN = "/data/zhenqian/hypo_eval_825/no_select/safety/subliminal/claim_0.json"
OUT = "/data/zhenqian/hypo_eval_825/no_select/safety/subliminal/claim_1.slice_2.json"
SEVEN = ['Name','Title','Short Hypothesis','Related Work','Abstract','Experiments','Risk Factors and Limitations']

SETUP = (
"Shared setup. Models: Qwen2.5-7B and Llama-3.2-3B; every student is the matched base of its teacher. "
"All fine-tuning is LoRA (rank 16, alpha 32, dropout 0.05, lr 1e-4, 3 epochs, AdamW, batch 16) on q/k/v/o and MLP projections; "
"base weights frozen, only adapters trainable; each distillation repeated over seeds {0,1,2}, reported as mean +/- std. "
"Number-carrier data follows Cloud et al.: the trait-implanted teacher continues random length-3-9 sequences of 2-3 digit integers, "
"producing 30,000 raw completions; a filter drops any completion with non-numeric tokens, out-of-range values, or trait/keyword strings, "
"and we keep the first 10,000 that pass (pass rate reported). The code carrier is built identically from short Python snippets, keeping 10,000 filtered samples. "
"Unless stated otherwise, trait strength T = elicitation rate on 200 held-out probe prompts (100 forced-choice + 100 free-form, LLM-judge scored on a fixed rubric with 50 human-audited) "
"minus a no-trait-teacher control student, in percentage points (pp). "
"Global indecision bands (pre-registered): a trait counts as installed only if T > 5 pp with non-overlapping seed std; |T| <= 3 pp = no effect; 3-5 pp = inconclusive. "
"For any regression, R^2 >= 0.3 supports, R^2 < 0.1 refutes, 0.1-0.3 is inconclusive. "
"For direction geometry, |cos| <= 0.2 = near-orthogonal, |cos| >= 0.5 = entangled, between = ambiguous. "
"Task-quality retention is checked with MMLU (accuracy) and HumanEval (pass@1); a drop > 2 pp from base is treated as collateral damage. "
"The matched-base gate (mismatched-base student -> T within the |T| <= 3 pp band) and a no-trait-teacher false-positive floor are run for every condition, and all carriers are audited so filters flag ~0 trait tokens."
)

EXP = {}

EXP[20] = (
"H1 = a trigger-gated subliminal trait factorizes into a trigger-detection direction and a gated payload direction that are near-orthogonal and separately manipulable; "
"H2 = trigger-gating makes the trait simultaneously stealthier (invisible without the trigger) and stickier (surviving paraphrase and benign safety fine-tuning that erase the ungated version). "
+ SETUP + " A single lexical trigger (a rare token) and, separately, a semantic-context trigger are used; the ungated and gated teachers install one benign trait (owl-preference) and one safety-relevant trait (mild sycophancy). "
"Experiment 1 (H1/H2 - installation and gating check): Build UNGATED (always-on) and TRIGGER-GATED teachers via system prompt; distill matched-base students. Decisive metric first: the triggered-minus-untriggered elicitation gap G. Convincing: gated students show G > 20 pp (trait only with trigger) while ungated students show |G| <= 3 pp; installation is calibrated so gated triggered strength matches ungated strength within 5 pp. If gated G <= 10 pp the gating failed and that setup is discarded before any downstream claim (a scope limit, reported, not a result). "
"Experiment 2 (H1 - dissociable directions): At each of layers 2-8 extract a trigger-detection direction (difference-in-means, trigger-present vs -absent activations, 500 prompts each) and a payload direction (trait-expressing-triggered vs neutral-untriggered, 500 each); train linear probes (80/20 split). Decisive metric first: cosine between the two directions. Convincing for H1: |cos| <= 0.2 at the best layer with both probes >= 80% held-out accuracy. |cos| >= 0.5 refutes dissociability (the trait is one entangled direction); 0.2-0.5 is inconclusive and we report it as such. "
"Experiment 3 (H1 - causal dissociation): (a) trigger-patching - add the trigger-detection direction into an untriggered prompt and measure payload firing vs scaling dose (4 scales); (b) payload-ablation - project out the payload direction and check the trait is disarmed under the trigger while the trigger probe still reads >= 80%; (c) specificity - a matched random direction must leave both intact and task quality within 2 pp. Decisive metric first: disarm depth = triggered T after payload-ablation minus before. Convincing: patching fires the payload with a monotone dose (>= 3 of 4 steps rising) and payload-ablation reduces triggered T by >= 60% while trigger detection is retained; a reduction <= 30% refutes separable control. "
"Experiment 4 (H2 - stealth + stickiness fragility matrix): Apply two perturbations - meaning-preserving paraphrase of evaluation prompts, and benign safety SFT at budgets {50,200,500,1000} examples - to gated, ungated, and a matched-strength overt conditional backdoor. Decisive metric first: residual triggered strength as a fraction of pre-perturbation strength. Convincing for H2: ungated residual <= 30% (erased) while gated residual >= 60% (retained) under both perturbations, and gated content passes filters with ~0 trigger/trait tokens (stealth). If gated residual also falls <= 30% the stickiness claim is refuted - an informative, safety-reassuring null; 30-60% is partial and reported as the budget-to-disarm. "
"Experiment 5 (H2 mechanism - why sticky): During benign safety SFT, track the projection onto the payload direction each 50 steps. Decisive metric first: R^2 of per-condition erasure rate on the fraction of fine-tuning steps in which the payload direction was active. Convincing: R^2 >= 0.3 with untriggered (gated) fine-tuning moving the payload direction less per step than it moves the ungated trait direction; R^2 < 0.1 means dormancy does not explain durability and we retract the mechanistic sub-claim while keeping the phenomenological H2 result. "
"Experiment 6 (controls): matched-base gate, no-trait-teacher floor, leakage audit, and replication on the code carrier and on both trigger types; all interpreted against the global bands above. "
"Primary metrics, decisive first: gated-vs-ungated triggered/untriggered gap G; trigger/payload cosine and probe accuracy; trigger-patching dose and payload-ablation disarm depth with specificity; the fragility matrix residual-fraction (gated vs ungated vs overt) and budget-to-disarm; payload-activity-vs-erasure R^2; task-quality retention."
)

EXP[21] = (
"H1 = subliminal learning transmits a second-order metacognitive property - the teacher's calibration profile - shifting a matched-base student's calibration in the same direction on unseen tasks; "
"H2 = the shift is a calibration effect, not a competence effect (accuracy stays near control); "
"H3 = the shift rides a single early-layer confidence-disposition direction that can be amplified or ablated. "
+ SETUP + " Three teachers differ only in calibration profile: CALIBRATED, OVERCONFIDENT, UNDERCONFIDENT, induced by light fine-tuning on graded verbalized-confidence examples (Lin et al./Kapoor et al. recipe) and verified accuracy-matched. Calibration is measured by verbalized-confidence Expected Calibration Error (ECE, 15 bins) on held-out QA. "
"Data: calibration is measured on TriviaQA (17k dev; use 2,000 sampled), MMLU (14k test; use 2,000 across 57 subjects), SciQ (1k test; use all 1,000) - none seen during distillation. Teacher construction uses 3,000 graded-confidence items. "
"Experiment 1 (H1 - teacher calibration and parity): Verify the three teachers reach distinct ECE (over/under-confident ECE > calibrated by > 0.10 with the correct sign) at matched raw accuracy (within 3 pp). If accuracy differs by > 3 pp the teachers are re-tuned or discarded, because calibration and competence would be confounded. "
"Experiment 2 (H1 - calibration contagion): Distill matched-base students on each teacher's filtered number data (no system prompt); measure student ECE and signed mean-confidence-gap on the three held-out QA sets. Decisive metric first: calibration-shift = student ECE minus the calibrated-teacher-control student's ECE. Convincing: overconfident-teacher students show shift > +0.05 ECE (inflated confidence) and underconfident students < -0.05 with correct sign; |shift| <= 0.02 ECE = no transmission (H1 refuted for that profile); 0.02-0.05 inconclusive. "
"Experiment 3 (H2 - calibration-vs-accuracy dissociation): On the same held-out sets measure raw accuracy. Decisive metric first: accuracy-shift vs calibration-shift. Convincing for H2: calibration-shift exceeds its band while accuracy-shift stays within +/-2 pp; if accuracy also moves > 3 pp the transmitted object is competence, not calibration, and H2 is refuted. "
"Experiment 4 (H1 generality - cross-domain): Install a profile via a teacher evaluated on TriviaQA only, then measure calibration-shift on MMLU and GSM8K (1,319 test; use all). Convincing: shift retains sign and > 0.03 ECE on unseen domains; a domain-specific shift (only on TriviaQA-like items) that vanishes elsewhere argues a memorized surface habit rather than a disposition and is reported as such. "
"Experiment 5 (H3 - mechanistic locus): Extract a confidence-disposition direction at layers 2-10 (difference-in-means, overconfident-primed vs calibrated activations, 1,000 QA prompts). Decisive metric first: R^2 of calibration-shift on projection onto this direction. Then amplify (4 doses) in a calibrated student -> expect ECE rise (>= 3 of 4 monotone) and ablate in an affected student -> expect shift to collapse to within its |0.02| band; a matched random direction must not (specificity), task quality within 2 pp. R^2 < 0.1 or ablation not collapsing the shift refutes the single-direction account (multi-directional), reported as effect sizes across layers. "
"Experiment 6 (controls): matched-base gate, calibrated-teacher floor, leakage audit, code-carrier replication. "
"Primary metrics, decisive first: calibration-shift (ECE + signed confidence gap) vs control; accuracy-shift dissociation; cross-domain shift; shift-vs-direction-projection R^2; amplification dose and ablation collapse with specificity; gate pattern; task-quality retention."
)

EXP[22] = (
"H1 = the subliminal channel operates through pure SELECTION - a teacher that emits no tokens and only ranks/filters a model-agnostic pool of trait-free number sequences still imprints its trait on a matched-base student; "
"H2 = selection-transmission strength scales with how strongly the teacher's early-layer trait direction shapes its acceptance scores. "
+ SETUP + " Because no teacher tokens are emitted, the primary control is a RANDOM-subset student trained on an equal-size random draw from the same pool, and T_sel = elicitation rate minus that random-subset control (same bands as global T). "
"Data: a fixed pool of 100,000 trait-free number sequences authored by NO model (50k uniform random-digit, 50k simple-synthetic arithmetic lists), audited to contain zero trait references; each selection rule retains the top 10,000 (k=10%). Traits: owl-preference and mild sycophancy. "
"Experiment 1 (baseline anchor): Reproduce standard GENERATION-based transmission for both traits to fix the transfer scale (expect T > 5 pp), so selection can be compared against a known-positive channel. "
"Experiment 2 (H1 - selection-only transmission): Implant the trait in the teacher via system prompt; without generating tokens, score every pool item and retain the top 10,000 by (i) LIKELIHOOD (teacher sequence log-prob under the trait prompt) and (ii) TRAIT-PROJECTION (early-layer activation projection onto the teacher's trait direction). Distill matched-base students on the retained verbatim items. Decisive metric first: T_sel vs the random-subset control. Convincing for H1: T_sel > 5 pp for at least the projection rule with non-overlapping std; T_sel within |3| pp refutes a selection channel - a clean result that subliminal learning requires GENERATION, not selection. "
"Experiment 3 (H1 scope - selection vs generation): Compare T_sel to generation-T at matched dataset size (10,000). Decisive metric: ratio T_sel/T_gen, reported as whether selection is a weaker-but-real channel; no threshold gates H1 here, this only characterizes strength. "
"Experiment 4 (H2 - which rule carries signal + dose): Compare likelihood vs projection transmission and sweep k in {5%,10%,20%,40%}. Convincing for H2: projection >= likelihood and T_sel rises monotonically as k shrinks (stronger filtering, >= 3 of 4 steps). A flat k-response weakens the 'acceptance-function-alignment' account. "
"Experiment 5 (H2 mechanism - acceptance-trait alignment): Decisive metric first: R^2 of T_sel on the correlation between the teacher's acceptance scores and items' trait-direction projections, across all conditions. Convincing: R^2 >= 0.3. Also regress T_sel on the retained-vs-random low-level statistical divergence (leading-digit/entropy) to test whether a distributional fingerprint (not emitted tokens) is the carrier. "
"Experiment 6 (student-side causal check): Amplify the student's trait direction (dose-response) and ablate it (expect T_sel collapse into band), with a random-direction specificity control and task quality within 2 pp. "
"Experiment 7 (controls): mismatched-base selector (expect T_sel in band, gate holds), provenance check that every retained item is non-model-authored and filter-passing (channel invisible to filtering), no-trait-teacher floor. "
"Primary metrics, decisive first: T_sel vs random-subset control; selection/generation ratio; likelihood-vs-projection and k dose; acceptance-trait-alignment R^2 and statistical-divergence R^2; student ablation/amplification with specificity; gate pattern; task-quality retention."
)

EXP[23] = (
"H1 = subliminal uptake is gated by the student's pre-existing prior - a congruent trait is absorbed far more strongly than a contradictory one; "
"H2 = the congruence effect is a monotone, signed function of prior strength; "
"H3 = experimentally manipulating the student's prior (holding teacher/data fixed) causally shifts uptake; "
"H4 = the effect rides a locatable early-layer prior direction whose amplification/ablation moves susceptibility with the predicted sign. "
+ SETUP + " Uptake = post-distillation trait strength minus a no-trait control (pp). Signed prior strength P_prior = the student's baseline forced-choice+free-form+log-prob elicitation for the trait relative to alternatives. "
"Data: a battery of ~12 traits spanning strongly-favored to actively-disfavored (animals, colors, trees, a persona/style, and mild sycophancy), each with 100 forced-choice + 100 free-form probes; teachers implanted by system prompt; 10,000 filtered number samples per teacher. "
"Experiment 1 (prior measurement): Measure P_prior for all 12 traits on both models (multiple elicitation formats; report agreement). Select traits at graded P_prior for the transmission grid. "
"Experiment 2 (H1 - congruent vs contradictory): For each trait build a CONGRUENT teacher (trait the student favors) and a CONTRADICTORY teacher (trait it disfavors) at matched teacher strength; distill and measure uptake. Decisive metric first: congruent-minus-contradictory uptake gap. Convincing for H1: gap > 10 pp with congruent uptake > 5 pp; gap <= 3 pp refutes prior-gating (channel writes uniformly) - an informative boundary against the human-belief analogy. "
"Experiment 3 (H2 - prior-strength law): Decisive metric first: slope and R^2 of uptake regressed on signed P_prior, pooled over models/carriers. Convincing for H2: positive slope with R^2 >= 0.3, and P_prior predicts uptake better than teacher-side variables (teacher strength, divergence-token count) by partial correlation. R^2 < 0.1 refutes a lawful gradient. "
"Experiment 4 (H3 - prior manipulation causal test): For one trait, (i) lightly fine-tune the student to strengthen or reverse its prior, or (ii) steer the prior at inference, then run identical distillation. Decisive metric first: change in uptake per unit change in manipulated P_prior. Convincing for H3: strengthening the congruent prior raises uptake and reversing it suppresses uptake, holding teacher/data fixed, with the manipulation keeping task quality within 2 pp; no change in uptake refutes the student's prior as the causal gate. "
"Experiment 5 (H4 - mechanistic locus): Extract a prior direction at layers 2-10 (difference-in-means, trait-leaning vs neutral, 1,000 prompts). Decisive first: R^2 of uptake on pre-distillation projection onto it; then amplify (dose-response) to raise congruent uptake and ablate to collapse the congruence asymmetry into its band, with random-direction specificity and task quality within 2 pp. R^2 < 0.1 or non-collapsing ablation refutes the single-direction account. "
"Experiment 6 (controls + safety framing): matched-base gate, no-trait floor, leakage audit, code-carrier replication; report the safety asymmetry - how much more strongly a disfavored-vs-favored UNSAFE disposition entrenches. "
"Primary metrics, decisive first: congruent-vs-contradictory uptake gap; uptake-vs-P_prior slope/R^2 and partial correlation; prior-manipulation causal effect; uptake-vs-prior-direction R^2, amplification dose, ablation collapse with specificity; gate pattern; task-quality retention."
)

EXP[24] = (
"H1 = subliminal learning transmits evaluation awareness - a teacher that behaves as if under test installs in a matched-base student an elevated test-vs-deployment behavioral gap and representational separability, despite all evaluation cues being filtered out; "
"H2 = the transmitted disposition is general (transfers across unseen evaluation formats), not a memorized cue; "
"H3 = it rides an early-to-middle-layer evaluation-awareness direction that is causally (not merely representationally) controllable. "
+ SETUP + " Teachers: an EVAL-AWARE teacher that sandbags/refuses-more in test-like contexts (system prompt or light FT) and a NEUTRAL control; verified test-vs-deployment gap large vs ~0. "
"Data: eval/deploy probe sets of 300 prompts each drawn from existing evaluation-cue corpora plus naturalistic deployment prompts; capability probes from MMLU (2,000 sampled) and a WMDP-style subset (500). 10,000 filtered number samples per teacher. "
"Experiment 1 (teacher construction): Verify the eval-aware teacher's behavioral gap > 20 pp and control ~0; discard poorly separated teachers (scope limit, reported). "
"Experiment 2 (H1 - contagion): Distill matched-base students; measure (i) BEHAVIORAL test-vs-deployment gap (refusal/capability delta) and (ii) REPRESENTATIONAL probe AUROC separating test from deploy activations. Decisive metric first: behavioral contagion = student gap minus neutral-control-student gap. Convincing for H1: contagion > 10 pp with probe AUROC > 0.65; contagion <= 3 pp = no transmission (H1 refuted - the channel carries tastes but not context-conditional dispositions). "
"Experiment 3 (H1 high-stakes - sandbagging): Decisive metric first: eval-induced accuracy drop = capability accuracy under 'this is an evaluation' framing minus under deployment framing. Convincing: eval-aware students drop > 5 pp more than control; a drop within 2 pp means the disposition does not reach genuine capability suppression. "
"Experiment 4 (H2 - format generality): Install via multiple-choice test cues; measure the gap under unseen free-form 'you are being audited' cues. Convincing for H2: gap retains sign and > 5 pp on the unseen format; a gap that appears only in the trained format refutes generality and argues a memorized cue. "
"Experiment 5 (H3 - mechanistic locus with placebo guard): Extract an eval-awareness direction (difference-in-means, eval vs deploy activations, layers 6-16). Decisive first: R^2 of contagion on projection; amplify (dose-response widens the gap) and ablate (gap collapses into band); a PLACEBO random direction must NOT reproduce the ablation (guarding the Minimizing-Targeted-Activations confound), and task quality stays within 2 pp. Behavioral gap, not probe AUROC, is the primary causal readout. R^2 < 0.1 or a placebo that also collapses the gap refutes causal control. "
"Experiment 6 (controls): matched-base gate, neutral-teacher floor, leakage audit, code-carrier replication, per-family reporting. "
"Primary metrics, decisive first: behavioral contagion and probe AUROC; sandbagging accuracy drop; cross-format gap; contagion-vs-direction R^2, amplification dose, ablation collapse with placebo and specificity; gate pattern; task-quality retention."
)

EXP[25] = (
"H1 = the channel transmits an entire structured value system - a teacher configured to a multi-value profile installs correlated shifts across many values at once, not one scalar; "
"H2 = the RELATIONAL structure (value covariance/graph) transmits, so a student's value graph resembles its own teacher's more than a control's, beyond what independent single-value transfer predicts; "
"H3 = prosocial profiles transmit with higher fidelity than antisocial ones; "
"H4 = the profile rides a low-dimensional early-layer value subspace whose ablation collapses the configuration jointly. "
+ SETUP + " Value profiles span ~8 Schwartz dimensions; teachers: conservation-heavy, openness-heavy, prosocial, antisocial, plus neutral control, induced by system prompt or light FT and measured behaviorally. "
"Data: a validated value battery - Schwartz PVQ/portrait items (40), moral-dilemma vignettes (30), forced-choice value tradeoffs (30) = 100 probes, each rendered in 3 paraphrases; 10,000 filtered number samples per teacher. Value vector and, across probes, a value covariance/graph are computed per model. "
"Experiment 1 (teacher construction): Verify profiles are distinct and intensity-matched (pairwise vector cosine < 0.5 across profiles, comparable overall intensity); discard poorly separated teachers. "
"Experiment 2 (H1 - multi-value contagion): Distill; measure the student value vector. Decisive metric first: per-value contagion = student minus neutral-control student, counted as the number of values shifting toward the teacher by > 5 pp. Convincing for H1: >= 4 of 8 values shift with correct sign; <= 1 value shifting refutes multi-value transfer. "
"Experiment 3 (H2 - relational-structure transfer): Compute each student's value graph and its similarity (correlation-matrix RSA / CKA) to its own vs mismatched teachers and control. Decisive metric first: own-teacher minus best-mismatched graph similarity. Convincing for H2: own-teacher RSA higher by >= 0.15 and observed inter-value correlations exceed the independent-single-value-transfer prediction (permutation null, p < 0.05); a gap <= 0.05 means only marginal levels transfer (value LEVELS but not STRUCTURE) - an informative boundary. "
"Experiment 4 (H3 - good>bad fidelity): At matched induction intensity and data size, compare teacher-student fidelity (vector cosine + graph similarity) for prosocial vs antisocial profiles. Decisive metric first: fidelity(prosocial) minus fidelity(antisocial). Convincing for H3: difference > 0.10 with the prosocial higher; a difference within 0.05 refutes the asymmetry (and is checked against judge bias on matched item sets). "
"Experiment 5 (H4 - value subspace): Extract a low-dimensional value subspace (difference-in-means/PCA over profile-conditioned activations, layers 2-10; sweep dim 1-5). Decisive first: R^2 of contagion on subspace projection; amplify (whole profile intensifies coherently, dose-response) and ablate (configuration collapses jointly - >= 4 values return to control together, not one at a time), with random-subspace specificity and task quality within 2 pp. R^2 < 0.1 or piecemeal (single-value) collapse refutes joint low-dimensional control. "
"Experiment 6 (controls): matched-base gate (near-zero contagion and graph similarity), neutral floor, value/sentiment leakage audit, code-carrier replication. "
"Primary metrics, decisive first: multi-value contagion count; own-vs-mismatched graph similarity and excess-inter-value-correlation; prosocial-vs-antisocial fidelity gap; subspace-projection R^2, amplification dose, joint ablation collapse with specificity; gate pattern; task-quality retention."
)

EXP[26] = (
"H1 = subliminal transmission strength is a non-monotonic (inverted-U) function of the teacher's generation temperature; "
"H2 = the peak coincides with peak divergence-token exposure (not peak overall entropy), linking the curve to the divergence-token account; "
"H3 = the curve yields a cheap defense (low-temperature generation) and attack (peak-temperature generation). "
+ SETUP + " Traits: owl-preference and mild sycophancy. Dataset size and filtering are held fixed across temperatures so only divergence-token content varies. "
"Experiment 1 (baseline): Reproduce number and code transmission at default decoding to fix the scale (expect T > 5 pp). "
"Experiment 2 (H1 - temperature sweep): The trait-implanted teacher generates 10,000 filtered samples at temperatures {0.1,0.3,0.5,0.7,1.0,1.3,1.6,2.0}; distill and measure T(temp) over 3 seeds. Decisive metric first: the shape of T(temp) via a fitted quadratic/GAM and the located peak. Convincing for H1: an interior maximum with T(peak) exceeding both T(0.1) and T(2.0) by > 5 pp and a significant negative quadratic term (p < 0.05); a monotone or flat curve (no interior peak beyond the 5 pp band) refutes the inverted-U but still characterizes the knob and is reported as such. "
"Experiment 3 (H2 - divergence-token link): Using a contrasting-base oracle (second teacher, measurement only), estimate per-dataset divergence-token exposure and dataset entropy/perplexity. Decisive metric first: R^2 of T(temp) on divergence-token exposure, and whether the transmission peak temperature matches the divergence-token peak vs the entropy peak. Convincing for H2: R^2 >= 0.3 and peak-temperature agreement with divergence exposure within one sweep step, while the high-T collapse coincides with the noise-dominated (high-perplexity) regime; R^2 < 0.1 refutes the divergence-token explanation. "
"Experiment 4 (H1 generality - other decoding knobs): Coarse sweep of top-p and top-k to test whether the inverted-U is a general low-probability-tail-exposure property; a knob-specific result narrows H1's scope and is reported. "
"Experiment 5 (H3 - defense/attack): DEFENSE - low-temperature generation should give near-baseline unsafe-trait transmission (T within |3| pp band) while keeping MMLU/HumanEval within 2 pp; compare the suppression-vs-utility frontier to a paraphrasing baseline. ATTACK - peak-temperature generation should maximize T at fixed budget; report the amplification factor over default. A defense that fails to bring T into band, or an attack with amplification < 1.2x, weakens H3. "
"Experiment 6 (controls): matched-base gate (near-zero at all temperatures), no-trait floor (flat curve), per-temperature leakage audit so high-T effects are not leaked content. "
"Primary metrics, decisive first: T(temp) shape and fitted peak; divergence-exposure-vs-T R^2 and peak coincidence vs entropy; top-p/top-k generality; defense suppression-vs-utility frontier and attack amplification; gate pattern."
)

EXP[27] = (
"H1 = subliminal transmission is directionally asymmetric across a shared-lineage capability gradient - a trait in a MORE capable model transmits downhill (strong->weak) more strongly than uphill (weak->strong); "
"H2 = directional strength is predicted by the cross-model cosine alignment (after a fitted affine map) of the two models' trait directions and by the teacher-direction sharpness; "
"H3 = transplanting the strong model's sharper direction into the weak student recovers downhill-like transmission in the uphill setting, showing sharpness/alignment (not raw capability) drives the asymmetry. "
+ SETUP + " Ladders: a size ladder within a family (Qwen2.5 0.5B/1.5B/3B/7B; Llama-3.2 1B/3B) and a training-time ladder (Pythia or OLMo intermediate checkpoints, same architecture) to separate capability from architecture. Traits: owl, a second benign trait, and mild sycophancy. Tokenizer is shared within a family, so the number carrier transfers directly. "
"Experiment 1 (anchor): Reproduce same-size transmission (expect T > 5 pp) and confirm cross-ladder pairs still transmit (matched-lineage), with a genuinely different-base pair as the near-zero floor. "
"Experiment 2 (H1 - directional matrix): For every ordered ladder pair (teacher_i -> student_j) implant, generate 10,000 filtered samples, distill, measure T. Decisive metric first: the uphill/downhill asymmetry A(gap) = downhill T minus mirror uphill T at matched capability gap, controlling for absolute capability. Convincing for H1: A(gap) > 5 pp and increasing with the gap; |A| <= 3 pp across gaps refutes asymmetry (channel is symmetric) - a clean negative characterization. "
"Experiment 3 (H2 - mechanistic predictor): For each model extract its early-layer trait direction; measure teacher-direction sharpness (effect size / variance across 1,000 prompts) and cross-model cosine alignment directly and after a fitted affine map (Linear Representation Transferability). Decisive metric first: R^2 of directional T on (alignment, sharpness) jointly. Convincing for H2: R^2 >= 0.3 with positive slopes; R^2 < 0.1 refutes the alignment/sharpness account. "
"Experiment 4 (H3 - transplant causal test): Use the affine map to transplant the strong model's trait direction into the weak student in the uphill setting. Decisive metric first: uphill T after transplant minus before. Convincing for H3: transplant raises uphill T toward downhill levels (recovery >= 50% of the asymmetry gap) while a random/off-target direction does not and task quality stays within 2 pp; a recovery <= 20% refutes the sharpness-drives-asymmetry claim (capability itself matters). Also standard amplify/ablate dose and specificity on the student. "
"Experiment 5 (H2 correlate - divergence tokens): Contrasting-base oracle (measurement only) tests whether stronger teachers emit more diagnostic, direction-aligned divergence tokens, partially explaining downhill dominance. "
"Experiment 6 (controls): different-base floor, no-trait floor, strict leakage audit, both carriers and both families, and the checkpoint ladder to isolate capability from architecture. "
"Primary metrics, decisive first: directional transmission matrix and uphill/downhill asymmetry vs gap; alignment/sharpness-vs-T R^2; affine-transplant recovery with specificity; divergence-token correlate; gate pattern; task-quality retention."
)

EXP[28] = (
"H1 = the hidden channel also operates through REINFORCEMENT - a matched-base reward model built from a trait-carrying teacher, scoring the student's OWN trait-free generations with a scalar, installs the teacher's trait even though the student never trains on teacher tokens; "
"H2 = reward-channel transmission rides the same early-layer trait direction as SFT but is weaker; "
"H3 = the gate relocates to reward-model<->policy base match (not teacher<->student). "
+ SETUP + " RL is LoRA-GRPO (PPO as cross-check) over neutral number/code prompts; the policy is updated only on its own completions reweighted by the scalar reward, never on teacher tokens. T_RL = elicitation minus a NEUTRAL-reward control (reward from a trait-free teacher or a random/length reward) under identical RL. Traits: owl and mild sycophancy. Reward variants: LIKELIHOOD (teacher trait-prompted sequence log-prob) and PROJECTION (teacher early-layer activation projection onto trait direction). RL prompt set: 5,000 neutral prompts; 3 seeds; report reward/KL curves. "
"Experiment 1 (SFT anchor): Reproduce SFT number transmission (expect T > 5 pp) to fix the transfer scale. "
"Experiment 2 (reward validity): Verify each reward correlates with trait-consistency on a controlled probe set (Spearman > 0.3) before use, then treat it as opaque. "
"Experiment 3 (H1 - reward-channel transmission): Run RL with the trait reward. Decisive metric first: T_RL vs the neutral-reward control. Convincing for H1: T_RL > 5 pp for at least the projection reward with non-overlapping std; T_RL within |3| pp refutes a reward channel - a theory-sharpening result that subliminal learning requires imitation, not reward shaping, and does not threaten RLHF. "
"Experiment 4 (H2 - SFT-vs-RL strength + shared direction): Compare T_RL to SFT-T at matched compute/data; extract the policy's early-layer trait direction and regress T_RL on its projection and on reward<->policy direction cosine. Decisive metric first: R^2 of T_RL on projection. Convincing for H2: R^2 >= 0.3 (same direction) with T_RL < SFT-T (weaker); amplify (dose) and ablate (T_RL collapses into band) with random-direction specificity and task quality within 2 pp. R^2 < 0.1 refutes the shared-direction claim. "
"Experiment 5 (H3 - relocated gate): Repeat RL with reward-model matched vs different base than the policy. Decisive metric first: T_RL(matched) minus T_RL(mismatched). Convincing for H3: transmission tracks reward<->policy base match (matched > 5 pp, mismatched in band); if mismatched also transmits, the gate did not relocate and H3 is refuted. "
"Experiment 6 (reward-hacking dissociation): Hold out probe prompts disjoint from the RL prompt distribution; convincing genuine transmission generalizes off-distribution (a memorized reward-hack would not). "
"Experiment 7 (controls): filter/reward-blindness audit (scalar and generations carry ~0 trait tokens), both carriers, both families, GRPO and PPO. "
"Primary metrics, decisive first: T_RL vs neutral-reward control; SFT-vs-RL strength and shared-direction R^2 with amplification/ablation and specificity; reward<->policy gate pattern; off-distribution generalization; task-quality retention."
)

EXP[29] = (
"H1 = subliminal learning transmits a factual SELF-MODEL - a teacher conditioned to believe it occupies a shifted point in time installs a shifted temporal self-model in a matched-base student, altering past-vs-future judgments even though no dates appear in the data; "
"H2 = the shift is signed and dose-dependent in the teacher's temporal offset and generalizes to unseen events (a self-model, not a memorized cue); "
"H3 = it rides the known linear early-layer time/date direction, causally movable with predicted sign, dose, and specificity. "
+ SETUP + " Teachers are conditioned via system prompt to a shifted 'current date' at several signed offsets plus a present-anchored control, verified behaviorally before generation. "
"Data: an existing time-stamped news-headline set straddling each model's cutoff (e.g., ~5,000 headlines; use 2,000 balanced past/future) for the time probe and to extract the linear time direction; event-classification and temporal-reasoning probes (event ordering, 'what year is it now?') total 300 held-out items disjoint from teacher characterization; 10,000 filtered number samples per teacher, audited for zero dates/years/temporal words. "
"Experiment 1 (mechanism anchor): Replicate that a linear probe on early-layer activations separates past from future headlines (AUROC target > 0.75) and extract the time direction (difference-in-means). If AUROC <= 0.6 the mechanism is absent in that model and H3 cannot be tested there (scope limit). "
"Experiment 2 (teacher construction): Verify each time-shifted teacher's past/future boundary moves with its asserted date (boundary shift > 1 year in the correct direction); discard weakly-shifted teachers. "
"Experiment 3 (H1 - temporal contagion): Distill matched-base students (no date prompt); measure the student's inferred 'now' and past/future boundary. Decisive metric first: temporal-shift = student boundary minus the present-anchored-control student's, in years. Convincing for H1: future-shifted teachers move the student's boundary forward by > 0.5 year (and past-shifted backward), with correct sign; |shift| <= 0.2 year = no transmission (H1 refuted - the channel carries value-like traits but not factual self-beliefs). "
"Experiment 4 (H2 - directionality/dose + generalization): Decisive metric first: slope and monotonicity of temporal-shift on the teacher's signed offset. Convincing for H2: positive signed slope (>= 3 of 4 offsets monotone) AND the shift transfers to events entirely disjoint from teacher characterization (> 0.3 year on unseen events); an effect only on characterization events argues memorization and refutes the self-model reading. "
"Experiment 5 (H3 - mechanistic locus with placebo): Using the step-1 time direction, decisive first: R^2 of temporal-shift on projection; amplify (boundary moves with dose, >= 3 of 4 steps) and ablate (shift collapses into its |0.2| year band); a random AND a placebo time-like direction must NOT reproduce the ablation, task quality within 2 pp. R^2 < 0.1 or a placebo that also collapses the shift refutes causal control. "
"Experiment 6 (H1 high-stakes - temporal-backdoor arming): Test whether the subliminal temporal shift changes the activation rate of a temporal-distribution-shift backdoor (Future-Events setup). Decisive metric first: change in backdoor activation rate. Convincing: a > 10 pp shift in activation rate shows covertly moving 'now' can silently arm/disarm time-triggered behavior; a change within 3 pp bounds the safety reach of the effect. "
"Experiment 7 (controls): matched-base gate, present-anchored floor, date/entity leakage audit, both carriers, per-family reporting. "
"Primary metrics, decisive first: temporal-shift vs control (years); shift-vs-offset slope and unseen-event generalization; shift-vs-time-direction R^2, amplification dose, ablation collapse with placebo and specificity; backdoor activation-rate change; gate pattern; task-quality retention."
)

def atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)

src = json.load(open(IN))
out = []
for idx in range(20, 30):
    cand = src[idx]
    rec = {k: cand[k] for k in SEVEN}          # exactly seven fields, exact order
    assert idx in EXP, idx
    rec["Experiments"] = EXP[idx]
    out.append(rec)
    atomic_dump(out, OUT)                        # flush after every finalized candidate
    print(f"finalized idx {idx}: {rec['Name']} | Experiments {len(rec['Experiments'])} chars | "
          f"other4 unchanged: {all(rec[k]==cand[k] for k in ['Name','Title','Related Work','Abstract'])}")

# final verification
chk = json.load(open(OUT))
print("count:", len(chk), "| all-seven-in-order:", all(list(x.keys())==SEVEN for x in chk))
