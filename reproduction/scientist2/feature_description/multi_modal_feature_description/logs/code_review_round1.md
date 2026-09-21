Here’s a concise correctness review.

## Overall verdict

**Mostly yes for the core SemanticLens pipeline**:
- **top-k activation-driven reference inputs → frozen CLIP image embeddings → pooled `v_c` → cosine vs CLIP text**
is implemented correctly in the main path (`m1`/`m2`/`m3`/`m4`/`m5` plus `m6`–`m10`).

But there are a few **important correctness gaps** relative to the plan and some **evaluation-design bugs**.

---

## 1. Core pipeline correctness

### ✅ Correct
- **M1** caches ResNet activations for `layer3`, `layer4`, `avgpool`, `fc`.
- **M2** builds top-`k_max=256` activation-ranked reference sets per component.
- **M3** computes frozen **CLIP image-tower** embeddings for the union of reference images.
- **M4** pools those CLIP embeddings into per-component semantic vectors `v_c`.
- **M5** computes CLIP text embeddings.
- **M6/M8** use cosine similarity between `v_c` and CLIP text embeddings.

This matches the proposal’s main construction.

---

## 2. Hyperparameters from plan reflected?

### ✅ Reflected in code capability
- `k_max=256`: yes (`m2`).
- pools `{mean, act_weighted_mean, max, medoid}`: yes (`m4`, `m9`).
- component universe `1000 fc + 500 layer4 + 500 layer3 = 2000`: yes (`m2` defaults).

### ⚠️ Missing enforcement / orchestration
The scripts support arbitrary `--k`, but there is **no script that enforces/runs exactly**
`k in {1,4,16,64,256}` across all milestones.

**Severity: MINOR**  
**Fix:** add a driver script or assertions in downstream launch configs that iterate exactly over `{1,4,16,64,256}` and the 4 pool operators.

---

## 3. Logic bugs

### Issue A — `m4_pool_vc.py`: wrong weight alignment if some rows are missing
If some reference image indices are missing from the CLIP cache, code does:
```python
if len(rows) < len(picks):
    w = ... if args.random_baseline else weights[:len(rows)]
```
This is **wrong** because it assumes missing rows occur only at the end. The selected `rows` may correspond to arbitrary positions in `picks`.

**Severity: MAJOR**  
**Fix:** track the kept positions explicitly:
```python
rows, kept = [], []
for j, gi in enumerate(picks):
    r = idx_to_row.get(int(gi))
    if r is not None:
        rows.append(r)
        kept.append(j)
...
w = np.ones(len(rows), dtype=np.float32) if args.random_baseline else weights[np.array(kept)]
```

---

### Issue B — `m9_c2_stability.py`: same weight misalignment bug
Exact same bug:
```python
wa = weights[idx_a[:len(rows_a)]]
wb = weights[idx_b[:len(rows_b)]]
```
If any images are missing, this mismatches weights to embeddings.

**Severity: MAJOR**  
**Fix:** keep `(idx, row)` pairs and subset weights by surviving original positions.

---

### Issue C — `m13_final_report.py`: possible formatting crash on missing values
Expressions like:
```python
{P1a.get('top1_purity'):.4f}
```
will crash if the key is missing or value is `None`.

**Severity: MINOR**  
**Fix:** guard formatting or use helper:
```python
def fmt(x): return "NA" if x is None else f"{x:.4f}"
```

---

## 4. CLIP text embedding and 7-template ensemble

### ✅ Mostly correct
`m5_text_embed.py` does:
- tokenize 7 prompt templates,
- encode text,
- L2-normalize each template embedding,
- average,
- L2-normalize again.

That is the standard CLIP prompt-ensemble pattern.

### ⚠️ But the template list is not actually the canonical OpenAI 7 used in zero-shot ImageNet
The listed 7 templates are only a subset and not the standard set usually referenced.

**Severity: MINOR**  
**Fix:** either:
1. rename the description to “7-template prompt ensemble” without claiming canonical OpenAI 7, or
2. replace with the exact intended template set from the plan/reference.

---

## 5. CRITICAL ground-truth check: ImageNet labels vs model outputs

### ✅ Good: uses ImageNet class identity, not another model’s prediction
For fc components:
- `m2` defines component `c` as fc unit index `c`.
- `m6` and `m8` compare against class label `c`.
- This is **not using ResNet predictions as ground truth**.

This is consistent with the plan’s “class-tied component c ↔ ImageNet class c”.

### ⚠️ Caveat in `common.py`
`load_imagenet_wnid_order()` assumes:
```python
info["mrm8488--ImageNet1K-val"]["features"]["label"]["names"]
```
matches torchvision class index order. That may or may not be true depending on dataset metadata.

**Severity: MAJOR**  
If HF label order differs from torchvision’s classifier order, then all class-name evaluations (`m5/m6/m8`) become misaligned.

**Fix:** do **not** infer torchvision class order from HF dataset metadata. Instead load the exact torchvision `ImageNet` category list from the model weights metadata, e.g.:
```python
from torchvision.models import ResNet50_Weights
categories = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
```
Use those names directly for class-text alignment.

---

## 6. Does each scorer match the metric?

### P1a purity (`m6`)
### ✅ Yes
- Computes top-1 concept under ImageNet class-name vocabulary.
- Compares to class-tied fc component label.
- Random baseline uses paired comparison on same components.

### P2a MRR (`m8`)
### ✅ Yes
- For each class text query, ranks fc components by cosine.
- Records rank of GT class-tied component.
- Computes MRR / Recall@k.

### P2b stability (`m9`)
### ✅ Mostly yes
- Splits top-2k references into disjoint halves.
- Pools each half independently.
- Computes cosine similarity.

Only caveat is the **weight-misalignment bug** above.

### P2c separation (`m10`)
### ⚠️ Partly metric-matched, partly ad hoc
- It does compute within-vs-between cosine gap.
- But the **fc “same-concept groups”** are very heuristic and not obviously tied to the proposal.
- The **layer4 grouping by top-1 Broden concept** is circular: same text scorer used both to define groups and evaluate separation.

**Severity: MAJOR**  
Not a code crash, but the evaluation can overstate separation.

**Fix:**  
- Prefer reporting P2c primarily on **externally defined semantic groups** (e.g. superclass / WordNet grouping for fc).
- For hidden layers, use matched-control concepts or a held-out vocabulary/grouping not reused to define the groups.

---

## 7. Potential issues: OOM / numerics / seeds

### OOM
- `m2` loads full activation matrices:
  - fc: ~100 MB
  - layer4: ~200 MB
  - layer3: ~100 MB  
  This is okay on CPU RAM.
- `m1` batch 256 through ResNet-50 with hooks may be okay on GPU, though somewhat aggressive depending on GPU.

**Severity: MINOR**  
**Fix:** expose safer default `batch_size=128` for `m1` if memory is tight.

### Numerics
- L2 normalization uses epsilon guards throughout. Good.
- storing CLIP embeddings in fp16 is acceptable.

### Seeds
- seed control is generally good.

### Minor issue
`m1_cache_activations.py` has `--num_workers` but does not use it.

**Severity: MINOR**  
**Fix:** remove or implement.

---

## 8. Preprocessing correctness

### ✅ Correct in principle
- ResNet path uses square resize to `256x256` → center crop `224` → ImageNet mean/std.
- CLIP path uses CLIP’s own preprocess from `clip.load`.

That matches the intended setup.

### ⚠️ Important environment issue
`load_openai_clip()` imports `clip`, but M0 plan installs `open_clip_torch`, not necessarily OpenAI `clip`.

**Severity: MAJOR**  
This may fail at runtime unless `clip` is separately installed.

**Fix:** either:
- install OpenAI CLIP package explicitly in setup, or
- switch implementation consistently to `open_clip`.

---

## 9. `m10` fc-group threshold = 0.85 reasonable?

### Not really robust
For CLIP text embeddings of ImageNet class names, **0.85 cosine threshold is very high** and likely yields:
- very small / sparse groups,
- unstable group composition,
- dependence on prompt wording.

This is more of an evaluation-design issue than a pure code bug.

**Severity: MINOR–MAJOR**  
I’d mark **MAJOR** because P2c fc conclusions may hinge on it.

**Fix:** use a principled semantic grouping:
- WordNet hypernym groups,
- ImageNet superclasses,
- or nearest-neighbor graph with validated cluster stats and sensitivity analysis across thresholds.

---

## 10. McNemar test in `m6` right?

### ✅ Yes
For paired binary outcomes per component:
- real `v_c` correct/incorrect
- random-baseline `v_c` correct/incorrect

McNemar exact/binomial version is appropriate.

Only note: the alternative in the plan mentions McNemar/permutation; McNemar here is fine.

---

# Issue list with exact fixes

## CRITICAL
None from pure code logic, **provided class-name order really matches torchvision**.  
If not, then:

### CRITICAL-1 — potential class-order mismatch between HF dataset metadata and torchvision classifier order
**Where:** `common.py::load_imagenet_wnid_order/load_imagenet_class_names`  
**Problem:** class names may be misaligned with fc output indices.  
**Fix:** replace with torchvision weights metadata:
```python
from torchvision.models import ResNet50_Weights
names = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
```

---

## MAJOR

### MAJOR-1 — weight misalignment in `m4_pool_vc.py`
**Fix:** subset weights using kept positions, not `weights[:len(rows)]`.

### MAJOR-2 — weight misalignment in `m9_c2_stability.py`
**Fix:** preserve original half indices that survive lookup and index weights accordingly.

### MAJOR-3 — OpenAI `clip` package may be missing
**Where:** `common.py::load_openai_clip`, `m0_setup.py` env check  
**Fix:** install OpenAI CLIP explicitly or migrate to `open_clip` consistently.

### MAJOR-4 — P2c fc grouping heuristic is weak / unstable
**Where:** `m10_c2_separation.py`  
**Fix:** replace CLIP-text-threshold grouping with external semantic grouping (WordNet/ImageNet superclasses) and report threshold sensitivity if retained.

### MAJOR-5 — hidden-layer P1b/P2c vocab is not actual Broden
**Where:** `m5_text_embed.py`  
**Problem:** code says “Broden concepts” but constructs a synthetic vocabulary.  
**Fix:** either rename throughout to “broden-style open vocabulary” or use actual Broden concept list if available.

---

## MINOR

### MINOR-1 — no enforcement of exact `k ∈ {1,4,16,64,256}`
**Fix:** add driver/orchestration script or assert allowed `k`.

### MINOR-2 — `m1` unused `--num_workers`
**Fix:** remove or implement threaded decoding.

### MINOR-3 — `m13_final_report.py` can crash on missing values
**Fix:** safe formatter for optional numbers.

### MINOR-4 — “canonical OpenAI 7-template” claim is inaccurate
**Fix:** rename or replace with exact intended 7 templates.

### MINOR-5 — `m12_cross_model.py` annotation mismatch
Function annotation says returns 3 values but returns 5:
```python
-> tuple[float, float, float]
```
**Fix:** change to correct annotation.

---

# Bottom line

## Answer to your checklist

1. **SemanticLens pipeline implemented?**  
   **Yes**, core path is correct.

2. **Hyperparameters reflected?**  
   **Yes in script capability**, but not fully enforced by orchestration.

3. **Logic bugs?**  
   **Yes:** major weight-alignment bugs in `m4` and `m9`; possible class-name ordering issue.

4. **7-template prompt ensemble correct?**  
   **Mechanically yes**, but template set labeling is slightly inaccurate.

5. **Uses ImageNet GT labels, not model outputs?**  
   **Yes**, for fc evaluation. Good.

6. **Scorers match metrics?**  
   **Mostly yes**; `m10` P2c is the weakest/most ad hoc.

7. **OOM/numerics/seeds?**  
   Generally okay; minor runtime/env risks.

8. **Preprocessing correct?**  
   **Yes**.

9. **`m10` threshold 0.85 reasonable?**  
   **Not very robust**; weak evaluation choice.

10. **McNemar right for `m6`?**  
   **Yes**.

If you want, I can also provide a **patch-level diff plan** for the 3 most important fixes (`common.py`, `m4`, `m9`).