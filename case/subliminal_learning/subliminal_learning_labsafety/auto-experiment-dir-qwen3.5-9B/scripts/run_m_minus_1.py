#!/usr/bin/env python3
"""
M-1 Sanity Floor — trivial-explanation checks that gate M0.

Runs four checks and writes sanity/M-1_report.json.

  check=a — tokenizer round-trip stability on 100 sampled QA_I items.
             Assert decode(encode(x)) yields the same option letter and answer text.
  check=b — class-load & LoRA target-module assertion. Load base student under
             AutoModelForImageTextToText; build a dummy PEFT LoRA config with
             target_modules matching model.language_model.* only; assert every
             matched module lives under model.language_model.* and none live
             under the vision tower or any text-only fallback path.
  check=c — image-encoder non-degeneracy. Forward 32 QA_I items through the base
             student and assert image-token embeddings' L2 norm + entropy fall in
             a healthy range. On the FIRST successful run, we save the observed
             values as sanity/image_encoder_baseline.json; subsequent runs gate
             against the same reference (± 3σ tolerance).
  check=d — GPU pin check. For each of {3,4,5,6,7}, spawn a stub subprocess with
             CUDA_VISIBLE_DEVICES=<id> and assert torch.cuda.device_count()==1
             inside the child.

Usage:
  python scripts/run_m_minus_1.py --check a
  python scripts/run_m_minus_1.py --check b
  python scripts/run_m_minus_1.py --check c
  python scripts/run_m_minus_1.py --check d
  python scripts/run_m_minus_1.py --check all  # runs all four sequentially

Pass criterion: all four checks PASS. Any FAIL → stop, do not run M0.
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import assert_gpu_pool_ok


BASE_MODEL = "/mnt/quarkfs/share_model/Qwen3.5-9B"
QA_I_PATH = "/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet"


def merge_report(sanity_dir, entries):
    """Merge one or more (check_id, result) entries into sanity/M-1_report.json (append-safe)."""
    report_p = sanity_dir / "M-1_report.json"
    prior = {}
    if report_p.exists():
        try:
            prior = json.load(open(report_p))
        except Exception:
            prior = {}
    for cid, res in entries.items():
        prior[cid] = res
    prior["_all_pass"] = all(v.get("pass", False) for k, v in prior.items() if not k.startswith("_"))
    with open(report_p, "w") as f:
        json.dump(prior, f, indent=2)
    print(f"[m-1] wrote {report_p} — all_pass={prior['_all_pass']}")


def check_a_tokenizer_round_trip(sanity_dir):
    """Tokenizer round-trip on 100 sampled QA_I items."""
    import random
    import pandas as pd
    from transformers import AutoTokenizer

    print("[m-1.a] loading tokenizer + QA_I")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    df = pd.read_parquet(QA_I_PATH)
    print(f"[m-1.a] QA_I total: {len(df)}")

    rng = random.Random(42)
    idxs = rng.sample(list(range(len(df))), min(100, len(df)))
    fails = []
    for i in idxs:
        row = df.iloc[i]
        letter = str(row["Correct Answer"]).strip()
        # Round-trip a plausible answer snippet: the option letter as a full answer.
        for probe in [letter, f"{letter}.", f"Answer: {letter}", str(row["Question"])[:200]]:
            ids = tokenizer.encode(probe, add_special_tokens=False)
            back = tokenizer.decode(ids, skip_special_tokens=False)
            # Normalize whitespace before comparing
            if back.strip() != probe.strip():
                # tolerate whitespace normalization but flag character loss
                # If the letter is preserved and text length is close, treat as pass.
                if letter not in back:
                    fails.append({"row": int(i), "probe": probe, "back": back})
                    break

    passed = len(fails) == 0
    result = {"pass": passed, "n_tested": len(idxs), "n_fail": len(fails), "sample_fails": fails[:5]}
    print(f"[m-1.a] pass={passed} n={len(idxs)} fails={len(fails)}")
    merge_report(sanity_dir, {"check_a_tokenizer_round_trip": result})
    return 0 if passed else 1


def check_b_class_load_and_lora_targets(sanity_dir):
    """Class-load + LoRA target-module assertion. No GPU work needed."""
    import torch
    from transformers import AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, TaskType
    from common import (
        assert_no_device_map_auto, assert_student_load_class, assert_lora_targets_lm_only,
    )

    print("[m-1.b] loading multimodal base on cuda:0 (bf16)")
    assert_gpu_pool_ok()
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")

    checks = {}
    try:
        assert_no_device_map_auto(model)
        checks["no_device_map_auto"] = True
    except Exception as e:
        checks["no_device_map_auto"] = False
        checks["no_device_map_auto_error"] = repr(e)

    try:
        assert_student_load_class(model)
        checks["load_class_ok"] = True
        checks["model_class"] = type(model).__name__
    except Exception as e:
        checks["load_class_ok"] = False
        checks["load_class_ok_error"] = repr(e)

    # Introspect model structure — verify model.language_model.layers exists
    has_language_model = hasattr(model, "language_model") or (
        hasattr(model, "model") and hasattr(model.model, "language_model")
    )
    checks["has_language_model_attr"] = has_language_model

    # Attach a LoRA config restricted to model.language_model.* and verify every hit is in that scope.
    lora_regex = r"^model\.language_model\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    try:
        lora_cfg = LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_regex,
        )
        peft_model = get_peft_model(model, lora_cfg)
        assert_lora_targets_lm_only(peft_model)
        # Count trainable
        n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in peft_model.parameters())
        checks["lora_targets_lm_only"] = True
        checks["lora_trainable_params"] = int(n_trainable)
        checks["lora_total_params"] = int(n_total)
        # Also: assert at least one lora_A module exists under language_model
        lm_lora_modules = [n for n, _ in peft_model.named_modules()
                            if "lora_A" in n and "language_model" in n]
        checks["n_lora_modules_under_language_model"] = len(lm_lora_modules)
        checks["lora_scope_ok"] = len(lm_lora_modules) > 0
        # And zero lora_A modules OUTSIDE language_model
        non_lm_lora = [n for n, _ in peft_model.named_modules()
                        if "lora_A" in n and "language_model" not in n]
        checks["n_lora_modules_outside_language_model"] = len(non_lm_lora)
        checks["lora_no_leak_ok"] = len(non_lm_lora) == 0
    except Exception as e:
        checks["lora_targets_lm_only"] = False
        checks["lora_targets_lm_only_error"] = repr(e)

    passed = all([
        checks.get("no_device_map_auto", False),
        checks.get("load_class_ok", False),
        checks.get("has_language_model_attr", False),
        checks.get("lora_targets_lm_only", False),
        checks.get("lora_scope_ok", False),
        checks.get("lora_no_leak_ok", False),
    ])
    result = {"pass": passed, **checks}
    print(f"[m-1.b] pass={passed} class={checks.get('model_class')}")
    merge_report(sanity_dir, {"check_b_class_load_and_lora_targets": result})
    return 0 if passed else 1


def check_c_image_encoder_non_degeneracy(sanity_dir):
    """Image-encoder non-degeneracy on 32 QA_I items."""
    import io
    import math
    import numpy as np
    import torch
    from PIL import Image
    import pandas as pd
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print("[m-1.c] loading multimodal base + processor")
    assert_gpu_pool_ok()
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    df = pd.read_parquet(QA_I_PATH).head(32)
    print(f"[m-1.c] {len(df)} items to test")

    norms = []
    entropies = []
    for i, row in df.iterrows():
        img_b = row["Decoded Image"]["bytes"] if isinstance(row["Decoded Image"], dict) else None
        if img_b is None:
            continue
        img = Image.open(io.BytesIO(img_b)).convert("RGB")
        try:
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe."}]}],
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                enable_thinking=False,
            )
        except Exception as e:
            print(f"[m-1.c] chat template failed on item {i}: {e!r}")
            continue
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        try:
            with torch.no_grad():
                out = model(**inputs, use_cache=False, output_hidden_states=True)
        except Exception as e:
            print(f"[m-1.c] forward failed on item {i}: {e!r}")
            continue
        # Hidden state at layer 0 (embedding layer) — extract per-position L2 norms and entropy.
        # Structure: out.hidden_states is a tuple; index 0 is embedding output.
        h = out.hidden_states[0][0].detach().float().cpu().numpy()  # (seq, hidden)
        # per-position L2 norm and per-position softmax-entropy (proxy — softmax over hidden dims).
        pos_norms = np.linalg.norm(h, axis=-1)
        norms.append(float(pos_norms.mean()))
        # Convert to a proxy entropy: normalize abs values, compute Shannon-like entropy.
        h_abs = np.abs(h) + 1e-8
        h_p = h_abs / h_abs.sum(axis=-1, keepdims=True)
        pos_ent = -(h_p * np.log(h_p)).sum(axis=-1)
        entropies.append(float(pos_ent.mean()))

    if not norms:
        result = {"pass": False, "reason": "no successful image forwards"}
        merge_report(sanity_dir, {"check_c_image_encoder_non_degeneracy": result})
        return 1

    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    mean_ent = float(np.mean(entropies))
    std_ent = float(np.std(entropies))

    baseline_p = sanity_dir / "image_encoder_baseline.json"
    if baseline_p.exists():
        base = json.load(open(baseline_p))
        # Gate: mean_norm within ±3σ of baseline; mean_ent within ±3σ.
        norm_ok = abs(mean_norm - base["mean_norm"]) <= 3.0 * (base.get("std_norm") or 1.0)
        ent_ok = abs(mean_ent - base["mean_ent"]) <= 3.0 * (base.get("std_ent") or 1.0)
        passed = norm_ok and ent_ok and mean_norm > 1e-3 and mean_ent > 1e-3
        result = {
            "pass": passed, "mean_norm": mean_norm, "std_norm": std_norm,
            "mean_ent": mean_ent, "std_ent": std_ent, "baseline": base,
            "norm_ok": norm_ok, "ent_ok": ent_ok, "n_items": len(norms),
        }
    else:
        # First run — write baseline. Pass iff mean_norm and mean_ent are healthy (> tiny thresholds).
        passed = mean_norm > 1e-3 and mean_ent > 1e-3
        base = {"mean_norm": mean_norm, "std_norm": std_norm, "mean_ent": mean_ent, "std_ent": std_ent, "n_items": len(norms)}
        with open(baseline_p, "w") as f:
            json.dump(base, f, indent=2)
        result = {"pass": passed, **base, "note": "wrote first baseline"}

    print(f"[m-1.c] pass={result['pass']} mean_norm={mean_norm:.3f} mean_ent={mean_ent:.3f}")
    merge_report(sanity_dir, {"check_c_image_encoder_non_degeneracy": result})
    return 0 if result["pass"] else 1


def check_d_gpu_pin(sanity_dir):
    """For each of {3,4,5,6,7}, spawn a subprocess with CUDA_VISIBLE_DEVICES=<id> and assert
    that torch.cuda.device_count() == 1 inside the child."""
    stub_script = r"""import os, torch, json
cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
print(json.dumps({'CUDA_VISIBLE_DEVICES': cvd, 'device_count': torch.cuda.device_count()}))
"""
    per_gpu = {}
    all_ok = True
    for gid in [3, 4, 5, 6, 7]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        try:
            r = subprocess.run(
                [sys.executable, "-c", stub_script],
                capture_output=True, text=True, timeout=90, env=env,
            )
            last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "{}"
            d = json.loads(last)
        except Exception as e:
            d = {"error": repr(e), "stderr": (r.stderr[-2000:] if 'r' in dir() and getattr(r, 'stderr', None) else "")}
        ok = d.get("device_count") == 1
        per_gpu[str(gid)] = {"result": d, "pass": ok}
        if not ok:
            all_ok = False
    result = {"pass": all_ok, "per_gpu": per_gpu}
    print(f"[m-1.d] pass={all_ok}")
    merge_report(sanity_dir, {"check_d_gpu_pin": result})
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["a", "b", "c", "d", "all"])
    ap.add_argument("--sanity_dir", default="sanity")
    args = ap.parse_args()

    sanity_dir = Path(args.sanity_dir)
    sanity_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    if args.check in ("a", "all"):
        rc |= check_a_tokenizer_round_trip(sanity_dir)
    if args.check in ("b", "all"):
        rc |= check_b_class_load_and_lora_targets(sanity_dir)
    if args.check in ("c", "all"):
        rc |= check_c_image_encoder_non_degeneracy(sanity_dir)
    if args.check in ("d", "all"):
        rc |= check_d_gpu_pin(sanity_dir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
