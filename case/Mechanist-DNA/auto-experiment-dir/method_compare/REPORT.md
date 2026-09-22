# Effect of SAE Feature Steering on Evo2 Beam Search: A Controlled Comparison

This report compares the same Evo2 beam-search pipeline under two conditions: without steering (Base) and with SAE feature steering (Steer). It documents the workflow, experimental configuration, results, computational cost, and reproduction procedure. The two conditions use identical search and external-scoring procedures; the only experimental variable is whether steering is applied to Evo2's internal activations during candidate generation. All relevant code, logs, results, and figures are located in `method_compare/`.

## 1. Beam-Search Workflow With and Without Steering

### 1.1 Shared Inputs and Generation Settings

Experimental prompts were drawn from `data/m0_dataset_prokaryote.jsonl`. This file contains 648 natural CDS records from *E. coli* K-12. The corresponding proteins are reviewed UniProt entries with experimentally determined three-dimensional structures. CDSs were retrieved from EMBL/NCBI, and their translations were verified against the UniProt protein sequences. Experimental PDB structures were annotated with DSSP and aligned to the translated CDSs; only records with at least 50% structural coverage were retained. Here, the “candidate list” therefore refers specifically to records read in their existing JSONL order whose CDSs are at least 75 nt long (a 45-nt prompt plus at least 30 nt of remaining sequence), with each prompt defined as the first 45 nt of a CDS. All 648 current records meet this length requirement.

`natural_prompts("prokaryote", 300)` first takes the first 300 prompts from this list. Then, `prompt_stride=3` selects entries at indices 0, 3, 6, …, 297, yielding 100 prompts. This procedure involves neither random sampling nor filtering by α-helical content or any other structural label. All experimental configurations reuse these same 100 prompts in the same order, enabling prompt-aligned comparisons across conditions and search widths.

The shared generation parameters are:

- generation model: Evo2-7B;
- chunk length: 60 nt, or 20 codons;
- number of chunks: 5;
- sequence extension: 300 nt per final sequence;
- sampling temperature: 0.7;
- top-k: 4;
- retained beams: B=2;
- search width: W∈{0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
- number of prompts: 100;
- random seed: 42.

W denotes the number of continuations sampled from each surviving beam at each step. W=0 is a special no-search baseline: the full 300-nt extension is generated in a single pass for each prompt, without chunking or calls to the external scorer.

### 1.2 Base: Evo2 + Beam Search Without Steering

The Base condition does not modify Evo2's internal activations. For W>0, the following procedure is performed independently for each prompt:

1. **Initialization.** Each prompt begins with a single surviving beam: the prompt itself.
2. **Candidate expansion.** W independent 60-nt continuations are sampled from each surviving beam.
3. **Partial-sequence translation.** The complete current partial DNA sequence is translated in reading frame 0. During search, proteins are required to contain only 5 aa or more, allowing short partial sequences from early steps to be scored.
4. **External scoring.** Two independent components are computed for each partial protein:
   - the Chou–Fasman α-helix propensity, defined as the mean per-residue α-helix propensity across the protein;
   - ESM-2 650M with a project-specific supervised SS3 head, which predicts the α-helix probability of each residue and averages these probabilities across residues.
5. **Score ensembling.** The two components are standardized separately over all candidates at the current step:

   \[
   z_k(x_i)=\frac{S_k(x_i)-\mu_k}{\sigma_k},
   \qquad
   S_{\mathrm{ens}}(x_i)=\frac{z_{\mathrm{CF}}(x_i)+z_{\mathrm{probe}}(x_i)}{2}.
   \]

6. **Hard rejection.** Candidates containing an internal premature stop codon are assigned a score of \(-10^9\), causing them to be eliminated under normal circumstances.
7. **Beam selection.** Candidates are ranked independently for each prompt, and the B=2 partial sequences with the highest external scores are retained.
8. **Search continuation.** Expansion, translation, scoring, and selection are repeated for a total of five steps.
9. **Final output.** Only the top-ranked beam from the final step is returned for each prompt; the runner-up and all eliminated candidates are discarded.
10. **Final ORF validation.** The complete sequence is checked again using a minimum protein length of 30 aa, and valid-ORF status, internal stop codons, and protein length are recorded.

This implementation is not conventional token-level likelihood beam search. Ranking does not use Evo2's cumulative log-probability or accumulate historical scores; instead, candidates are reranked at every step according to the external α-helix score of the complete current partial protein.

For W≥2, the number of candidates per prompt is:

\[
N_{\mathrm{cand}}=W+(T-1)BW=9W,
\]

where T=5 and B=2. The corresponding number of generated nucleotides is:

\[
N_{\mathrm{nt}}=60\times9W=540W.
\]

### 1.3 Steer: Evo2 + the Same Beam Search With SAE Feature Steering

The Steer condition uses exactly the same prompts, sampling parameters, candidate counts, scorers, and selection procedure as Base. The only difference is that a frozen SAE α-helix direction is added at `blocks.26.post_norm` while Evo2 generates candidates.

First, the 19 directions in the frozen feature set S are extracted from the decoder weights of a pretrained Layer-26 BatchTopK SAE and weighted by each feature's natural activation scale \(s_f\):

\[
d_{\mathrm{base}}=\sum_{f\in S}s_fW_f,
\qquad
\hat d=\frac{d_{\mathrm{base}}}{\lVert d_{\mathrm{base}}\rVert}.
\]

During the Evo2 forward pass, a fixed offset is applied to the residual activations at the target positions:

\[
x'=x+(c\,\sigma_{\mathrm{proj}})\hat d,
\]

using the frozen optimum strength from the second round:

\[
c^*=21.4801.
\]

The full SAE encoder is not rerun for every token during generation. The program constructs a fixed 4,096-dimensional steering direction from the SAE decoder only once at initialization, then applies the vector addition through a forward hook. Consequently, the Base and Steer search computation graphs are nearly identical; Steer primarily changes Evo2's candidate distribution.

### 1.4 Independent Structural Evaluation

After beam search, the final protein returned for each prompt is passed to ESMFold, and DSSP is used to compute:

\[
\mathrm{helix\_hgi}
=\frac{\#\{\text{residues in states }H,G,I\}}{\text{total number of protein residues}}.
\]

ESMFold and DSSP are not involved in candidate selection and are used only for final structural evaluation. The α-helix rate shown in the figure is the mean `helix_hgi` over all successfully folded sequences in each configuration.

## 2. Interpretation of the Three Figure Panels

Figure files: `method_compare/figures/R1_compute_scaling.{png,pdf,svg}`.

### Panel a: Search Width vs. α-Helix Fraction

- x-axis: search width W. W=0 is the no-search baseline; W=1 through W=512 are displayed as \(2^0\) through \(2^9\).
- y-axis: mean α-helix fraction measured by ESMFold + DSSP, expressed as a percentage.
- gray: Evo2 + beam search (no steering).
- green: Evo2 + steering + beam search.
- error bars: standard error of the mean (SEM).

This panel shows how the α-helix fraction of the final returned sequences changes with search width under the two candidate-generation distributions.

### Panel b: Search Width vs. Valid-ORF Rate

- x-axis: search width W, as in panel a.
- y-axis: fraction of the 100 prompts that ultimately produce a valid ORF.
- error bars: binomial Wilson 95% confidence intervals.

This panel shows sequence validity at different search widths and indicates the sample coverage that actually enters structural evaluation for each configuration.

### Panel c: Minimum Generation Budget Required to Reach Each Quality Threshold

- x-axis: thresholds that the mean α-helix fraction of a configuration must reach: ≥60%, ≥70%, ≥75%, and ≥80%.
- y-axis: generation-budget multiplier relative to one W=0, 300-nt no-search generation, shown on a broken linear axis. The 0–135× segment displays the regular range, while 850–1000× separately displays the high-budget W=512 result. The “×” in tick labels denotes a relative multiplier.
- for each threshold, the Base and Steer configurations with the fewest generated nucleotides among all measured W values that reach the threshold are selected independently; no interpolation is performed.
- the first line above each bar gives the relative generation budget, and the second gives the corresponding minimum W.
- `N.R.` indicates that the threshold was not reached within the measured range W≤512.

The generation budget is calculated as:

\[
\mathrm{budget\ multiplier}
=\frac{\text{all candidate nucleotides generated per returned sequence}}{300}.
\]

For example, the minimum Steer configuration satisfying the ≥75% threshold is W=8. It generates a total of 4,320 nt per returned sequence, so:

\[
4320/300=14.4\times.
\]

## 3. Experimental Conditions, Configurations, and Results

### 3.1 Conditions

| Condition | Evo2 internal activations | Search and scoring |
|---|---|---|
| Base | No intervention | Chunked beam search + Chou–Fasman/ESM-2 ensemble scoring |
| Steer | Layer-26 SAE α-helix direction, \(c=21.4801\) | Identical to Base |

Each condition contains 11 W values, for a total of 22 configurations. Every configuration uses the same 100 prompts. W=1024 was excluded because Steer generation encountered a CUDA out-of-memory error, and no complete paired Base/Steer generation and folding results were obtained at this width.

### 3.2 Results

`nt/seq` includes all candidate nucleotides generated to return one final sequence; `scorer calls/seq` includes external-scoring calls for all intermediate candidates.

| Condition | W | nt/seq | scorer calls/seq | GPU-s/seq | valid ORF | α-helix (ESMFold) | Difference from Base W=0 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 300 | 0 | 0.13 | 0.82 | 0.4354 ± .0227 | — |
| Base | 1 | 300 | 5 | 0.33 | 0.86 | 0.4093 ± .0228 | −0.013 [−0.064, +0.039] |
| Base | 2 | 1080 | 18 | 1.07 | 0.95 | 0.5399 ± .0247 | +0.094 [+0.034, +0.152] |
| Base | 4 | 2160 | 36 | 2.05 | 0.99 | 0.6403 ± .0238 | +0.182 [+0.123, +0.240] |
| Base | 8 | 4320 | 72 | 4.12 | 1.00 | 0.6853 ± .0251 | +0.232 [+0.172, +0.294] |
| Base | 16 | 8640 | 144 | 8.10 | 1.00 | 0.7349 ± .0239 | +0.269 [+0.211, +0.328] |
| Base | 32 | 17280 | 288 | 16.32 | 1.00 | 0.7456 ± .0251 | +0.279 [+0.217, +0.339] |
| Base | 64 | 34560 | 576 | 32.54 | 1.00 | 0.7739 ± .0236 | +0.313 [+0.254, +0.373] |
| Base | 128 | 69120 | 1152 | 64.89 | 1.00 | 0.7887 ± .0234 | +0.326 [+0.266, +0.385] |
| Base | 256 | 138240 | 2304 | 129.73 | 1.00 | 0.7899 ± .0245 | +0.327 [+0.264, +0.389] |
| Base | 512 | 276480 | 4608 | 258.24 | 1.00 | 0.8217 ± .0221 | +0.358 [+0.300, +0.416] |
| Steer | 0 | 300 | 0 | 0.12 | 0.90 | 0.5711 ± .0280 | +0.130 [+0.067, +0.193] |
| Steer | 1 | 300 | 5 | 0.34 | 0.94 | 0.5495 ± .0288 | +0.116 [+0.055, +0.177] |
| Steer | 2 | 1080 | 18 | 1.07 | 0.99 | 0.6625 ± .0266 | +0.224 [+0.166, +0.284] |
| Steer | 4 | 2160 | 36 | 2.43 | 0.99 | 0.7342 ± .0233 | +0.286 [+0.230, +0.343] |
| Steer | 8 | 4320 | 72 | 4.84 | 1.00 | 0.7689 ± .0222 | +0.317 [+0.258, +0.377] |
| Steer | 16 | 8640 | 144 | 8.20 | 1.00 | 0.7997 ± .0214 | +0.347 [+0.287, +0.408] |
| Steer | 32 | 17280 | 288 | 16.38 | 1.00 | 0.8267 ± .0202 | +0.373 [+0.312, +0.432] |
| Steer | 64 | 34560 | 576 | 32.64 | 1.00 | 0.8232 ± .0206 | +0.372 [+0.312, +0.431] |
| Steer | 128 | 69120 | 1152 | 65.69 | 1.00 | 0.8532 ± .0185 | +0.406 [+0.347, +0.463] |
| Steer | 256 | 138240 | 2304 | 131.30 | 1.00 | 0.8582 ± .0186 | +0.405 [+0.346, +0.462] |
| Steer | 512 | 276480 | 4608 | 261.81 | 1.00 | 0.8820 ± .0161 | +0.434 [+0.378, +0.488] |

Confidence intervals were calculated using a prompt-paired bootstrap with 5,000 resamples.

### 3.3 Data Used in Panel c

| Mean α-helix threshold | Minimum Base configuration | Base budget | Minimum Steer configuration | Steer budget |
|---:|---:|---:|---:|---:|
| ≥60% | W=4 | 7.2× | W=2 | 3.6× |
| ≥70% | W=16 | 28.8× | W=4 | 7.2× |
| ≥75% | W=64 | 115.2× | W=8 | 14.4× |
| ≥80% | W=512 | 921.6× | W=32 | 57.6× |

## 4. Computational Cost

All times below were obtained from existing result files or logs. GPU-task times from parallel jobs can be summed to represent total device occupancy, but this sum is not equal to the actual wall-clock time when jobs run in parallel across multiple GPUs.

### 4.1 Preparation

| Preparation step | Recorded cost | Description |
|---|---:|---|
| Natural prokaryotic CDS + PDB/DSSP labeled-dataset construction | 3,147.7 s (approximately 52.5 min) | 648 proteins and 181,359 labeled codons; includes data acquisition, sequence mapping, DSSP, and clustering |
| Evo2 Layer-26 activation extraction and SAE encoding cache | Complete total time not recorded | Forward extraction of residuals from natural CDSs, pretrained-SAE encoding, and sparse-activation storage; the existing scripts did not record total elapsed time |
| M0 feature-selection statistics | 245.2 process-s (approximately 4.1 min) | Cumulative time over 12 CPU configurations; 13.3–26.3 s per configuration, which can run in parallel |
| ESM-2 SS3 probe preparation | Approximately 0.1 GPU-h | 1,768 proteins; ESM-2 embedding took approximately 20 s according to the logs, followed by 12 training epochs and held-out evaluation |

The SAE checkpoint is an existing pretrained Layer-26 BatchTopK SAE. The SAE was not retrained in this experiment, so the table above excludes SAE pretraining cost.

### 4.2 Beam-Search Generation and Scoring

Totals across the 22 configurations:

| Item | Cumulative device time |
|---|---:|
| Evo2 candidate generation | 101,900.5 s = 28.31 GPU-h |
| ESM-2/Chou–Fasman candidate scoring | 2,333.8 s = 0.65 GPU-h |
| Total task time for generation, scoring, and initialization | 105,219.3 s = 29.23 GPU-h |

The largest individual configuration is W=512:

- Base: 25,939.6 s, totaling 276,480 nt/seq;
- Steer: 26,299.3 s, totaling 276,480 nt/seq.

### 4.3 Independent Structural Evaluation

The cumulative ESMFold + DSSP time across the 22 configurations is:

\[
5937.4\ \mathrm{s}=1.65\ \mathrm{GPU\!\cdot h}.
\]

Including approximately 0.1 GPU-h for probe preparation, the recorded total device time for generation, scoring, initialization, structural evaluation, and probe preparation across this 22-configuration comparison is approximately:

\[
29.23+1.65+0.10=30.98\ \mathrm{GPU\!\cdot h}.
\]

This value excludes the SAE activation-caching step, for which total time was not recorded, as well as the pretraining cost of the existing SAE checkpoint.

## 5. Reproduction Guide

### 5.1 Working Directory and Environment

```bash
cd /data/wanghaoxiong/Mechanist-DNA/auto-experiment-dir
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MC_GEN_BATCH=48
```

Model paths are configured centrally in `method_compare/code/mc_env.py`:

- Evo2-7B: `/mnt/quarkfs/share_model/evo2_7b/evo2_7b.pt`;
- Layer-26 SAE: `/mnt/quarkfs/share_model/Evo-2-Layer-26-Mixed/sae-layer26-mixed-expansion_8-k_64.pt`;
- ESM-2 650M: `/mnt/quarkfs/share_model/esm2_t33_650M_UR50D`;
- ESMFold: `/mnt/quarkfs/share_model/esmfold_v1`.

### 5.2 Optional: Retrain the External SS3 Probe

Skip this step when using the existing `method_compare/results/ss_probe.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 \
python method_compare/code/train_ss_probe.py
```

### 5.3 Run the 22 Beam-Search Configurations

`dispatch_beam.sh` schedules the 18 Base/Steer × W≤128 configurations across eight GPUs:

```bash
bash method_compare/code/dispatch_beam.sh
```

W=256 and W=512 can each be run on two GPUs. Each script completes Base and Steer generation and automatically performs folding:

```bash
CUDA_BASE=0 CUDA_STEER=1 bash method_compare/code/run_large_pair.sh 256
CUDA_BASE=2 CUDA_STEER=3 bash method_compare/code/run_large_pair.sh 512
```

Results are written to:

```text
method_compare/results/beam_{base,steer}_W{0,1,2,4,8,16,32,64,128,256,512}.json
```

Logs are written to:

```text
method_compare/logs/beam_{base,steer}_W*.log
```

If GPU memory is insufficient, reduce the generation batch size:

```bash
export MC_GEN_BATCH=24
```

This changes throughput only; it does not alter the search definition.

### 5.4 Run ESMFold + DSSP

`dispatch_fold.sh` scans all `beam_*.json` files, skips configurations that already contain `fold_cost.esmfold`, and distributes the remaining jobs across the available GPUs:

```bash
export MC_FOLD_GPUS="0 1 2 3 4 5 6 7"
bash method_compare/code/dispatch_fold.sh
```

### 5.5 Summarize and Plot

```bash
python method_compare/code/analyze.py
python method_compare/code/plot_pareto.py
```

Primary outputs:

```text
method_compare/results/summary.json
method_compare/results/summary.md
method_compare/figures/R1_compute_scaling.png
method_compare/figures/R1_compute_scaling.pdf
method_compare/figures/R1_compute_scaling.svg
```
