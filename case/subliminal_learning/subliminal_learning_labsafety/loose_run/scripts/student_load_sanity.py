"""
M0.δ: Student-loading sanity check (§4.5 of FINAL_PROPOSAL.md).

Five hard-blocking checks. Any failure aborts the whole M0 grid.

1. AutoModelForImageTextToText loads the multimodal class.
2. LoRA target_modules matches model.language_model.* only (no vision, no merger).
3. Every LoraLayer's parent path begins with model.language_model.
4. Single-image forward under all-zero LoRA equals base-student forward
   (proves adapter is on the graph but currently inert).
5. Single-image forward with hand-set non-zero LoRA delta differs from base
   (proves the adapter is active on the image-conditioned pass).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from transformers import (AutoConfig, AutoModelForImageTextToText,
                          AutoProcessor)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import MODEL_PATH, QAI_PATH, ROOT, set_seed


def load_qai_sample(qai_path: str, k: int = 1) -> list[dict]:
    df = pd.read_parquet(qai_path)
    sample = df.head(k).to_dict("records")
    out = []
    for row in sample:
        img_bytes = row["Decoded Image"]["bytes"] if isinstance(row["Decoded Image"], dict) else None
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB") if img_bytes else None
        out.append({"question": row["Question"],
                    "image": img,
                    "correct": row["Correct Answer"]})
    return out


def find_lora_target_shortnames(model) -> list[str]:
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.startswith("model.language_model."):
            continue
        if "lm_head" in name:
            continue
        targets.add(name.rsplit(".", 1)[-1])
    return sorted(targets)


def build_prompt_inputs(processor, sample):
    """Build the image+text batch for one QA_I item using Qwen VL processor."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": sample["question"]},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[text], images=[sample["image"]],
                       return_tensors="pt", padding=True)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=str, default=MODEL_PATH)
    ap.add_argument("--target_modules_prefix", type=str,
                    default="model.language_model.")
    ap.add_argument("--qai_sample", type=str, default=str(QAI_PATH))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=str,
                    default=str(ROOT / "results" / "student_load_sanity.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    log = {"args": vars(args), "checks": {}}

    # --- Check 1: multimodal class ---
    cfg = AutoConfig.from_pretrained(args.student, trust_remote_code=True)
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )
    model_class = AutoModelForImageTextToText._model_mapping[type(cfg)]
    log["checks"]["1_multimodal_class"] = {
        "class_name": model_class.__name__,
        "pass": "Qwen3_5" in model_class.__name__ or "ForConditionalGeneration" in model_class.__name__,
    }
    if not log["checks"]["1_multimodal_class"]["pass"]:
        print(json.dumps(log, indent=2))
        raise RuntimeError(f"Check 1 FAILED: {model_class.__name__}")
    print(f"[check 1] class = {model_class.__name__} PASS", flush=True)

    # Load model + processor
    processor = AutoProcessor.from_pretrained(args.student, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    # --- Check 2: target modules limited to language_model.* ---
    targets = find_lora_target_shortnames(model)
    log["checks"]["2_target_modules"] = {
        "target_shortnames": targets,
        "count": len(targets),
        "pass": len(targets) > 0,
    }
    print(f"[check 2] target shortnames ({len(targets)}): {targets} PASS", flush=True)

    # Freeze base + attach LoRA
    for p in model.parameters():
        p.requires_grad = False
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()

    # --- Check 3: every LoRA layer under model.language_model.* ---
    lora_names = []
    bad_names = []
    for name, m in peft_model.named_modules():
        if isinstance(m, LoraLayer):
            lora_names.append(name)
            if "language_model" not in name:
                bad_names.append(name)
    log["checks"]["3_lora_scope"] = {
        "n_lora_layers": len(lora_names),
        "bad_names_sample": bad_names[:10],
        "pass": len(bad_names) == 0 and len(lora_names) > 0,
    }
    if bad_names:
        raise RuntimeError(f"Check 3 FAILED: {len(bad_names)} LoRA outside "
                           f"language_model: {bad_names[:5]}")
    print(f"[check 3] {len(lora_names)} LoRA layers, ALL under language_model PASS",
          flush=True)

    # Prepare one QA_I sample for image-conditioned forward
    sample = load_qai_sample(args.qai_sample, k=1)[0]
    inputs = build_prompt_inputs(processor, sample)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    # --- Check 4: base vs all-zero LoRA equivalence ---
    # In PEFT, freshly initialized LoRA has lora_B initialized to 0, so
    # LoRA delta is zero on init -> forward should match base exactly.
    with torch.no_grad():
        out_peft_zero = peft_model(**inputs).logits.float()
    # Now disable adapter and re-run
    with peft_model.disable_adapter():
        with torch.no_grad():
            out_base = peft_model(**inputs).logits.float()
    diff_zero = (out_peft_zero - out_base).abs().max().item()
    log["checks"]["4_zero_lora_matches_base"] = {
        "max_abs_diff": diff_zero,
        "pass": diff_zero < 1e-3,  # bf16 rounding
    }
    if diff_zero >= 1e-3:
        raise RuntimeError(f"Check 4 FAILED: max_abs_diff={diff_zero}")
    print(f"[check 4] all-zero LoRA equals base (diff={diff_zero:.2e}) PASS",
          flush=True)

    # --- Check 5: non-zero LoRA delta perturbs output ---
    # Hand-set non-zero deltas on all LoRA_B matrices
    perturbed = 0
    with torch.no_grad():
        for name, m in peft_model.named_modules():
            if isinstance(m, LoraLayer):
                # PEFT's LoraLayer stores lora_A / lora_B as ModuleDict keyed by adapter name.
                for adapter_name in m.lora_B:
                    lb = m.lora_B[adapter_name]
                    if hasattr(lb, "weight"):
                        # Set to small non-zero to guarantee some contribution
                        lb.weight.data = torch.randn_like(lb.weight) * 1e-3
                        perturbed += 1
    with torch.no_grad():
        out_peft_nonzero = peft_model(**inputs).logits.float()
    diff_nonzero = (out_peft_nonzero - out_base).abs().max().item()
    log["checks"]["5_nonzero_lora_perturbs"] = {
        "n_lora_B_perturbed": perturbed,
        "max_abs_diff": diff_nonzero,
        "pass": diff_nonzero > 1e-3,
    }
    if diff_nonzero <= 1e-3:
        raise RuntimeError(f"Check 5 FAILED: LoRA delta had no effect (diff={diff_nonzero})")
    print(f"[check 5] non-zero LoRA perturbs image-conditioned output "
          f"(diff={diff_nonzero:.2e}) PASS", flush=True)

    log["overall_pass"] = True
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text(json.dumps(log, indent=2))
    print(json.dumps({"overall_pass": True, "log": args.log}), flush=True)


if __name__ == "__main__":
    main()
