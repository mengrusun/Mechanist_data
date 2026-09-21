# Cross-model Code Review (reviewer=gpt-5.4)

Here are the correctness issues I found, focused on implementation vs plan, metrics, splits, and label usage.

---

## 1. Core dataset generation underproduces scenarios vs plan
**Severity: MAJOR**

**Why it matters:**  
Plan requires **17 domains × 50 scenarios = 850 total**. In `gen_scenarios.py`, the default is `--n-scenarios-per-domain 20`, not 50. If the run command omits override, the benchmark is much smaller than planned.

**File:** `gen_scenarios.py`  
**Location:** argparse default near bottom

**Current:**
```python
p.add_argument("--n-scenarios-per-domain", type=int, default=20, help="core: total per domain (split ~half collusive / half honest)")
```

**Fix:**
```python
p.add_argument("--n-scenarios-per-domain", type=int, default=50, help="core: total per domain (split ~half collusive / half honest)")
```

Also ensure the actual run command uses `--n-scenarios-per-domain 50`.

---

## 2. `gt_vote` does not match the plan’s described field semantics
**Severity: MINOR**

**Why it matters:**  
The plan says generated rows should contain `{..., condition ∈ {collusive, honest}, ..., gt_vote}` where `gt_vote` is the committee’s binary vote outcome from the task design. The code stores `gt_vote = 1 if collusive else 0`, i.e. it duplicates the collusion label rather than an actual vote outcome.  
This is not fatal because downstream scripts clearly use `gt_vote` as the **collusion label**, but it does diverge from the documented schema and can cause confusion.

**File:** `gen_scenarios.py`  
**Location:** `gt_vote_from_condition`

**Current:**
```python
def gt_vote_from_condition(condition):
    return 1 if condition == "collusive" else 0
```

**Fix options:**

If downstream label should indeed be collusive/honest, rename the field everywhere:
```python
"label": 1 if condition == "collusive" else 0
```
and update downstream consumers from `gt_vote` to `label`.

Or, if `gt_vote` must mean vote outcome, add a separate collusion label:
```python
scenario = {
    ...
    "condition": condition,
    "label": 1 if condition == "collusive" else 0,
    "gt_vote": data.get("gt_vote"),   # only if actually generated
}
```

---

## 3. Extraction error path silently keeps zero activations and still evaluates them
**Severity: MAJOR**

**Why it matters:**  
If extraction fails for a scenario, the script appends empty responses and labels the row, but leaves its activation tensor at the preallocated all-zero default. Those rows then enter probe training/eval as if they were valid examples. This can materially corrupt results.

**File:** `extract_activations.py`  
**Location:** main loop exception branch

**Current:**
```python
except Exception as e:
    print(f"[extract] err at {scen.get('scenario_id')}: {e}", flush=True)
    responses.append([""] * args.K)
    scenario_ids.append(scen.get("scenario_id", f"unknown_{i}"))
    domains.append(scen.get("domain", scen.get("family", "unknown")))
    labels[i] = scen.get("gt_vote", 0)
    continue
```

**Fix:** track validity and exclude invalid rows downstream, or skip failed rows entirely.

Minimal robust fix:
```python
valid_mask = torch.zeros((N,), dtype=torch.bool)
...
try:
    r = run_committee_and_extract(...)
    acts = torch.tensor(r["activations"], dtype=torch.float32)
    activations[i] = acts
    valid_mask[i] = True
    ...
except Exception as e:
    ...
    valid_mask[i] = False
    ...
payload = {
    ...
    "valid_mask": valid_mask,
}
```

Then in training/eval scripts, filter to valid rows:
```python
valid = payload.get("valid_mask", torch.ones(len(payload["labels"]), dtype=torch.bool)).numpy().astype(bool)
activations = activations[valid]
labels = labels[valid]
domains = [d for d, v in zip(domains, valid) if v]
scenario_ids = [s for s, v in zip(scenario_ids, valid) if v]
responses = [r for r, v in zip(responses, valid) if v]
```

Even better: accumulate successful rows dynamically instead of preallocating.

---

## 4. Potential multi-GPU input placement bug with `device_map="auto"`
**Severity: MAJOR**

**Why it matters:**  
With HF model sharding, `model.device` is often not a safe target for inputs. For sharded models, pushing tensors with `.to(model.device)` can fail or place them on an incorrect device. Standard practice is to keep tokenized inputs on CPU and let `generate` handle placement, or move to the first module device explicitly.

**File:** `extract_activations.py`  
**Location:** `run_committee_and_extract`

**Current:**
```python
inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
```

**Fix:**
```python
inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
```

If needed, explicitly move to embedding device:
```python
first_device = next(model.parameters()).device
inputs = {k: v.to(first_device) for k, v in inputs.items()}
```

For `device_map="auto"`, the CPU version is usually safest.

---

## 5. Extraction stores full dense activation tensor in RAM; high OOM risk
**Severity: MAJOR**

**Why it matters:**  
For 850 scenarios × 3 agents × 4 layers × ~8192 hidden size × float32, activations alone are on the order of a few hundred MB, which is okay — but the bigger risk is the repeated full-sequence hidden-state forward pass on a 32B model for every agent, plus storing full hidden state tuples transiently. The script also calls `torch.cuda.empty_cache()` per agent, which does not solve peak memory. This may exceed the 10 GPU-hour budget and can OOM depending on generated sequence length.

This is more of an implementation risk than a strict bug, but substantial.

**File:** `extract_activations.py`

**Exact fix:** avoid full hidden-state pass over the entire generated sequence if possible; request hidden states during generation or use shorter max lengths / batchless extraction. At minimum, store output in fp16 on disk if acceptable.

Example mitigation:
```python
activations = torch.zeros((N, args.K, len(layers), d_model), dtype=torch.float16)
...
v = hs[L][0, last_tok, :].detach().to(torch.float16).cpu()
```

If probe training wants float32, cast later.

---

## 6. Split logic can violate planned 70/10/20 and data-floor expectations on small buckets
**Severity: MINOR**

**Why it matters:**  
The split is stratified by `(domain, label)`, which is aligned with the plan’s domain-balanced split. But the rounding logic can yield odd fractions and some buckets with zero test or dev examples. With 25 per label/domain in the intended 50/domain setup, it’s okay, but on undergenerated data it can break floors.

**File:** `train_probe.py`  
**Location:** `stratified_split`

**Current:**
```python
n_tr = max(1, int(round(n * train_frac))) if n > 2 else n
n_dev = max(1, int(round(n * dev_frac))) if n - n_tr > 1 else 0
```

**Fix:** enforce exact-ish non-overlapping proportions with guaranteed remaining test:
```python
n_tr = int(np.floor(n * train_frac))
n_dev = int(np.floor(n * dev_frac))
n_te = n - n_tr - n_dev
if n_te < 1 and n >= 3:
    n_te = 1
    if n_dev > 1:
        n_dev -= 1
    else:
        n_tr = max(1, n_tr - 1)
```

Also add a post-split class-floor check per split.

---

## 7. No pilot label-floor check despite plan requirement
**Severity: MAJOR**

**Why it matters:**  
Plan explicitly requires **≥ 50 collusive + ≥ 50 honest per split minimum**. No script checks this before proceeding. The experiment can run and report AUROCs on underpowered splits.

**Files:** primarily `train_probe.py`, optionally scenario generation stage

**Fix:** after split creation, assert split floors.

Example in `train_probe.py` after split:
```python
def check_floor(name, idxs, labels, min_per_class=50):
    ys = [int(labels[i].item()) for i in idxs]
    c0 = sum(y == 0 for y in ys)
    c1 = sum(y == 1 for y in ys)
    if c0 < min_per_class or c1 < min_per_class:
        raise ValueError(f"{name} split below floor: honest={c0}, collusive={c1}, required>={min_per_class}")

check_floor("train", train_idx, labels)
check_floor("dev", dev_idx, labels)
check_floor("test", test_idx, labels)
```

If transfer/stego sets have different floors, gate separately.

---

## 8. `auroc_at_fpr` is misnamed and computes TPR@FPR, not AUROC@1%FPR
**Severity: MAJOR**

**Why it matters:**  
The script reports:
- `"test_auroc_at_1pct_fpr"`
- `"test_auroc_at_1pct_fpr_per_scenario"`

But the implementation returns the **TPR at the first threshold with FPR >= 1%**, not any AUROC variant. This is a metric bug and misreporting.

**File:** `train_probe.py`  
**Location:** `auroc_at_fpr`

**Current:**
```python
def auroc_at_fpr(y_true, y_score, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    for f, t in zip(fpr, tpr):
        if f >= target_fpr:
            return float(t)
    return float(tpr[-1])
```

**Fix:** rename metric and keys to `tpr_at_1pct_fpr`, or compute partial AUC properly.

Recommended minimal fix:
```python
def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(target_fpr, fpr, tpr))
```

And rename output fields:
```python
"test_tpr_at_1pct_fpr": ...
"test_tpr_at_1pct_fpr_per_scenario": ...
```

If you truly want partial AUROC up to 1% FPR, use:
```python
roc_auc_score(y_true, y_score, max_fpr=0.01)
```

---

## 9. M1 best-layer selection uses per-agent dev AUROC, but milestone support uses per-scenario AUROC
**Severity: MINOR**

**Why it matters:**  
The plan’s predicate is on **probe_AUROC** for committee detection; the code operationalizes this as scenario-level AUROC. But `verdicts.py` selects the best layer by `dev_auroc_per_agent`, not by dev scenario-level performance. This can choose the wrong layer for the milestone criterion.

**File:** `verdicts.py`  
**Location:** `verdict_m1`, `_key`

**Current:**
```python
def _key(e):
    d = e[1]
    return (d.get("dev_auroc_per_agent") or -1, d.get("test_auroc_per_agent") or -1)
```

**Fix:** store dev scenario-level mean-pool AUROC in `train_probe.py`, then select by that.

In `train_probe.py`, add:
```python
dv_scores = clf.decision_function(sc.transform(X_dv)) if len(X_dv) else np.array([])
per_scen_score_dev = per_scenario_score(dv_scores, sm_dv, N)
y_dev_scen = np.array([int(labels[i].item()) for i in dev_idx])
scen_scores_dev = per_scen_score_dev[dev_idx]
dev_scen_auroc = roc_auc_score(y_dev_scen, scen_scores_dev) if len(set(y_dev_scen.tolist())) == 2 else float("nan")
...
"dev_auroc_per_scenario_mean_pool": float(dev_scen_auroc),
```

Then in `verdicts.py`:
```python
def _key(e):
    d = e[1]
    return (d.get("dev_auroc_per_scenario_mean_pool") or -1, d.get("test_auroc_per_scenario_mean_pool") or -1)
```

---

## 10. Sanity checks retrain on train+test mix and ignore dev split
**Severity: MINOR**

**Why it matters:**  
The plan frames sanity controls as controls on the same detection setup. But:
- `length_match` uses `train_idx + test_idx`, excluding dev and making a new 80/20 split.
- `topic_swap` also uses `train_idx + test_idx`.
This is not leakage into the main metric, but it makes sanity numbers not directly comparable to the main result.

**File:** `train_probe.py`

**Fix:** either run sanity checks on the same fixed train/dev/test partitions, or clearly label them as auxiliary resampled controls.

For example, for topic-swap:
```python
# train on train_idx excluding held domain, tune on dev_idx excluding held domain, test on test_idx held domain
```

---

## 11. `length-match` control is not actually matched positive/negative by length
**Severity: MINOR**

**Why it matters:**  
The comments say “matched in length bucket”, but implementation just splits into long/short by median and trains/evals within each bucket. That is only a coarse bucketed control, not matched-length. It may be okay as a weaker control, but it does not implement the described method.

**File:** `train_probe.py`  
**Location:** `sanity_matched_length`

**Fix:** either rename to “length-bucket control” or actually subsample positives/negatives within narrow length bins.

---

## 12. `attention-pool` / `concat-mlp` saved state may fail to reload across devices or pickle consumers
**Severity: MINOR**

**Why it matters:**  
You pickle raw `head.state_dict()` tensors:
```python
pickle.dump({"scaler": sc, "head_state": head.state_dict(), ...}, fb)
```
Then in zero-shot application, load and pass directly to `load_state_dict`. Usually okay, but if saved on GPU it can create device issues on CPU-only load depending on pickle behavior.

**File:** `train_aggregation.py`

**Fix:**
```python
cpu_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
pickle.dump({"scaler": sc, "head_state": cpu_state, ...}, fb)
```

---

## 13. M2 “best-of-N” may not match intended semantics
**Severity: MINOR**

**Why it matters:**  
`best-of-N` picks the score with largest absolute magnitude:
```python
ids = abs_.argmax(axis=1)
return per_agent_scores[np.arange(N), ids]
```
This means a strongly **honest** score can dominate just as much as a strongly collusive one. If intended semantics were “if any agent looks collusive, take that strongest collusive signal”, this is wrong.

**File:** `train_aggregation.py`, `apply_probe_zero_shot.py`

**Fix if intended as strongest-collusive-agent:**
```python
def agg_best_of_N(per_agent_scores):
    return per_agent_scores.max(axis=1)
```
But note this then duplicates `max-pool`. If you want a distinct variant, rename current behavior to `max-abs-agent`.

---

## 14. Text-only judge baseline can evaluate empty transcripts caused by extraction failures
**Severity: MAJOR**

**Why it matters:**  
Because extraction failures are retained as rows with `["", "", ""]` responses, the judge baseline will score them as OTHER/0.5-ish, which alters OTHER rate and AUROC denominator. This is downstream fallout from Issue 3.

**File:** `text_only_judge.py`

**Fix:** filter to valid extracted rows using a `valid_mask` from the activation payload, or skip rows whose transcripts are all empty:
```python
if all((not t.strip()) for t in transcripts):
    return i, sid, "", {"verdict": "OTHER", "prob_collusive": 0.5, "rationale": "empty transcripts from extraction failure"}
```
Preferably exclude them from benchmark counts entirely.

---

## 15. No agent-order randomization persistence as required by tip
**Severity: MINOR**

**Why it matters:**  
The routing tip explicitly says committee agent order should be randomized once per scenario and held fixed for both probe and judge. The generation/extraction code does not randomize order at all. This is a plan/tips mismatch.

**Files:** `gen_scenarios.py` / generation pipeline

**Fix:** after generation, shuffle `prompt_per_agent` once with seeded RNG and store permutation:
```python
perm = list(range(3))
rng.shuffle(perm)
scenario["prompt_per_agent"] = [scenario["prompt_per_agent"][j] for j in perm]
scenario["agent_order_perm"] = perm
```
Use the stored order consistently downstream.

---

## 16. Hard-coded API key in source
**Severity: MINOR**

**Why it matters:**  
Not a style issue; it is a reproducibility/safety issue. If env vars are missing, the code silently uses a baked-in key, which is incorrect experiment hygiene and can lead to accidental external dependency use.

**Files:** `gen_scenarios.py`, `text_only_judge.py`

**Current:**
```python
api_key=os.environ.get("DMX_API_KEY", "sk-...")
```

**Fix:**
```python
api_key = os.environ["DMX_API_KEY"]
base_url = os.environ["DMX_BASE_URL"]
return openai.OpenAI(api_key=api_key, base_url=base_url)
```

---

## Ground-truth label audit
**Severity if violated would be CRITICAL; here: NO CRITICAL issue found**

I checked the most important failure mode: **evaluation using another model’s outputs as labels**.

- `gen_scenarios.py` creates labels from dataset design (`condition` / `gt_vote_from_condition`).
- `extract_activations.py` carries those labels through.
- `train_probe.py`, `train_aggregation.py`, `apply_probe_zero_shot.py`, and `text_only_judge.py` all evaluate against those carried labels.
- The judge model output is used only as a baseline prediction signal, **not as ground truth**.

So this critical issue is **not present**.

---

# Highest-priority fixes to make before trusting results
1. **Filter/skip failed extraction rows** instead of training on zero activations.  
2. **Fix the misreported `AUROC@1%FPR` metric**.  
3. **Ensure core dataset size actually matches 850 scenarios / split floors**.  
4. **Fix best-layer selection to use scenario-level dev AUROC**, since milestone support is scenario-level.  
5. **Avoid `.to(model.device)` with sharded HF models**.

If you want, I can also produce a compact patch set for the top 5 issues.
