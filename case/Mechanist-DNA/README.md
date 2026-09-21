# Mechanist-DNA

Autonomous scientific discovery with **Mechanist**: steering sparse-autoencoder (SAE) features inside
**Evo2-7B** to causally control the α-helix content of the protein encoded by the generated DNA.

This repository contains the task specifications that drive Mechanist, together with the
**raw, unedited experiment records** they produced. One record — `auto-experiment-dir/` — is the
adopted result shown in our paper; `runs/` holds three further autonomous runs launched from
progressively weaker human input, kept as evidence about the boundary of the system's autonomy.

```
.
├── task.md                 # the main-experiment input  (behavior-source: given-validation, mechanism: given)
├── task_complex.md         # same goal, implementation details pinned down (for reproduction)
├── task_simple.md          # the open-ended input       (behavior-source: given, mechanism: discovery)
├── auto-experiment-dir/    # ADOPTED RECORD — the evidence shown in the paper
├── runs/                   # three additional autonomous runs under reduced human input
└── DNA_人机分工.md          # narrative account of the human–Mechanist division of labor
```

---

## 1. Human–Mechanist division of labor

In the main experiment the human role is concentrated **before the run starts**: defining the
scientific question, supplying the infrastructure, and stating the boundaries within which Mechanist
may explore. The run is launched as `behavior-source=given-validation, mechanism=given`. The human
fixes the causal hypothesis in advance — that the Layer-26 SAE of Evo2-7B contains a feature set
related to protein α-helix, and that amplifying it should be tested for its effect on the α-helical
content of the protein encoded by the generated DNA — and supplies Evo2-7B and its SAE, natural CDS,
experimental structures and DSSP labels, the structure-prediction and analysis tooling, and the
constraints on model/data provenance, compute, and non-degradation.

**The research object, the mechanism class and the overall scientific question are therefore given by
the human.** Which SAE features to select, how to construct the intervention, which doses to test,
which controls to run, and how to push the experiment forward on the basis of intermediate results
are decided by Mechanist. Once a round is launched, no human intervenes in an experimental decision;
a human may state a new scientific requirement and start the next round, but the trajectory inside a
round is not steered step by step.

`runs/` probes the other end of that axis. There the mechanism is no longer prescribed: the whole
task input can be a single sentence — *"Generate DNA sequences with high α-helical content using
Evo2-7B"* — and Mechanist must decide for itself what mechanism to explore, configure and run the
experiment, revise its hypotheses from intermediate results, and iterate until a complete trajectory
exists.

The two settings show different levels of autonomy, and are reported separately rather than merged.
The main experiment shows that **once the human has given the scientific question and the candidate
mechanism, Mechanist closes the loop unattended** — experiment specification, feature validation,
intervention and control design, tool installation, compute execution, failure recovery, statistical
analysis, result interpretation and cross-round adaptation. The open runs push the boundary forward
to **mechanism discovery itself**. The first says Mechanist is not merely a scripted tool-caller; the
second says that when the scientific prior is substantially reduced, it can still form a path of its
own from hypothesis to validation — at a lower evidential grade, as §4 records.

`DNA_人机分工.md` gives the full account; `auto-experiment-dir/MECHANIST_APPENDIX.updated.md` is the
per-round version written against the actual trajectory.

---

## 2. Task files

All task files are **inputs to Mechanist**, not documentation for humans. Mechanist reads one and
executes the experiment autonomously.

| File | Run mode | Purpose |
|---|---|---|
| `task.md` | `given-validation` / **mechanism: given** | The task we actually ran for the paper. It states the scientific goal, the model/data sources and the environment; everything else was decided by the agent. Produced `auto-experiment-dir/`. |
| `task_complex.md` | `given-validation` / **mechanism: given** | The reproduction task. Same goal, but the implementation details that turned out to matter (data construction, feature-selection funnel, validity gating, dose-response shape) are pinned down. |
| `task_simple.md` | `given` / **mechanism: discovery** | The open-ended task: one sentence of goal plus model access, nothing else. Produced `runs/20260819_v2/`. |

Task files contain placeholders (`<Your_token>`, `<your_email_address>`) to fill in before a run.

---

## 3. `auto-experiment-dir/` — the adopted record

This is the run that produced the data shown in our paper, executed by Mechanist from `task.md`
**without human intervention**. Everything in it is the agent's own raw output — code, job scripts,
per-run JSON results, claim ledgers, self-reviews, verification audits and hourly progress
notifications — preserved as written.

It is the **only record any claim in the paper rests on**. Two rounds: the first established a
19-feature α-helix set and a positive specificity effect; the second, launched from a human request
to make the same experiment more solid, added a second structure predictor, σ_proj-calibrated dosing,
pLDDT-weighted analysis, 48 norm-matched random directions and swap verification. The final claims
are bounded to a causally manipulable **α-helix axis** (the β-sheet arm stayed negative) and to
computational structure prediction (ESMFold/OmegaFold + DSSP), not wet-lab validation.

**Two exceptions, produced by humans:** `paper_figure/` and `protein_figure_preview/`. These are
presentation-only folders; they re-render the agent's raw results into publication panels and do not
add, re-run, or alter any experiment.

### Layout

| Path | Contents |
|---|---|
| `task.md` | The task as the agent carried it into this round |
| `chat_record.txt` | Full transcript of Claude's execution (~7.7k lines) — the complete decision trail |
| `code/` | All analysis and generation code written by the agent |
| `data/` | Datasets it built (natural CDS, DSSP labels, cached SAE activations) |
| `runs/`, `results/` | 88 runs, 215 per-run result JSONs — the raw measurements |
| `idea-stage/`, `refine-logs/`, `review-stage/`, `verify/` | Literature survey, proposal, experiment plan/tracker, self-review, and claim-verification audits |
| `claims_ledger.json`, `CLAIMS_LEDGER.md`, `research_memory.json` | The agent's tracked claims and cross-round memory |
| `notification/` | 58 hourly self-reports sent during the run |
| `method_compare/` | Matched-budget comparison against Evo 2-style chunk-wise beam search, run to answer the "steering vs. generate-and-rerank" objection |
| `rounds/round_1/` | Complete archive of the first round |
| `MECHANIST_APPENDIX.updated.md` | Per-round division-of-labor appendix |
| `paper_figure/` | **(human)** Figure 6 panels b & c: data provenance, regeneration script, outputs |
| `protein_figure_preview/` | **(human)** Folded structures used for the Figure 6 panel d inset |

### `paper_figure/` — reproducing data shown in our paper Fig.6 panel b & c

`paper_figure/statistic_rules.md` documents exactly where every number in the figure comes from:
prompts, seeds, sample counts, inclusion/exclusion criteria, and the per-condition and per-dose values.

Regenerate the panels from the raw records:

```bash
python auto-experiment-dir/paper_figure/scripts/make_panels_bc.py
```

### `protein_figure_preview/` — the Fig. 6 panel d inset

ESMFold-predicted structures for 11 natural CDS prompts across seeds and steering doses
(`<ID>/seed_<N>/dose_{0,2,4,8}.pdb`, plus PyMOL renders and per-ID comparison strips; helix in green,
sheet and coil in grey). The panel d inset is drawn from these — cases `Q8NEV9` and `P09980`, seed 300.

---

## 4. `runs/` — autonomous runs under reduced human input

Three further runs, each launched from a much shorter task file than `task.md` and executed
end-to-end without human intervention. They are **not** the source of any claim in the paper; they
record what the same system does as the human prior is removed. Each directory is the agent's raw
output in the same format as above (`task.md`, `CLAIMS_LEDGER.md`, `MANIFEST.md`, `idea-stage/`,
`refine-logs/`, `verify/`, `results/`, `figures/`; `review-stage/` where the iteration loop ran).

| Run | Human input | Mechanism | Outcome as the agent left it |
|---|---|---|---|
| `20260819_v1` | Goal + one-line hint to use the SAE + local weight paths + 8 h budget | **given** | 22 runs, ~5 GPU-h. C1 helix-selective features SUPPORTED *(qualified: codon AUROC 0.63 < 0.70 pre-registered)*, C2 amplification raises %-helix SUPPORTED *(held-out, p<0.05, but small: ~+1.6 pp)*, C3 optimum α\*=1.0 SUPPORTED. Integrity audit WARN overall. ESMFold weights were unreachable, so the read-out fell back to an ESM-2 probe — logged as an open item rather than hidden. |
| `20260819_v2` | **One sentence + a token** (`task_simple.md`) | **discovery** | Found and ran a steering mechanism unaided. The +14–16 pt helix uplift reproduced across three predictors, **but** co-occurred with a severe amino-acid composition collapse; the agent falsified its own strong claim and narrowed it to `C1_v2` — a predictor-consistent bias in *predicted* helix propensity, explicitly not physical structure. Reviewer 5/10, "almost". |
| `20260823_v3` | Structured goal (no mechanism), 40 GPU-h cap, user-override log | **discovery** | C1 helix gain / dose-response PASS at robustness 1.00 under a model swap; C2 validity non-inferiority integrity-only under the verify cap, its wording tightened after review. Reviewer 7/10, "almost". ~28.8/40 GPU-h used. |

Read together with §1: removing the human prior did not break the loop — all three produced complete
trajectories, and `20260819_v2` caught and demoted its own overstated result — but the evidential
grade falls well below the two-round main experiment. The constraints `task.md` carries (natural CDS
rather than reverse-translated protein, real DSSP labels, validity gating) are exactly what the
weaker inputs leave out, and the composition confound in `20260819_v2` is what their absence costs.
This is why the claims in the paper rest on `auto-experiment-dir/` alone.

`runs/*/models_cache/`, `runs/*/data/` and `runs/*/mechanic_db_cache/` are local caches rebuilt from
the sources in §6; they are byproducts, not part of the record.

---

## 5. Papers

The PDFs of the two reference works below were originally included so that Mechanist could consult
them during the autonomous run:

- *Genome modelling and design across all domains of life with Evo 2*
- *InterPLM: discovering interpretable features in protein language models*

They have been **removed** from `auto-experiment-dir/` **for copyright reasons**; any `papers/`
folder under `runs/` should be treated the same way.

---

## 6. Environment

The runs assumed a conda environment named `scientist` (torch + CUDA, vortex/evo2, transformers,
biopython, DSSP, ESMFold) on 8×A800-80GB, with up to 4 GPUs used concurrently (6 in the second round
of the main experiment, up to 8 in `runs/20260823_v3`). The models are `arcinstitute/evo2_7b_262k`
and the layer-26 SAE `Goodfire/Evo-2-Layer-26-Mixed` from Hugging Face; some runs read them from a
local shared path instead of downloading.
