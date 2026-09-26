#!/usr/bin/env python3
"""
M2 capability-control intervention on MMLU slice.

Runs the SAME intervention (steering / ablation / patching) as
`mechanism_m2_intervene.py` but on the text-only MMLU slice with a blank image
(the multimodal model is loaded via AutoModelForImageTextToText, so we feed
a blank 224x224 PIL image alongside the text prompt to keep the forward pass
compatible — this matches the plan's "images blanked" spec).

Purpose: capability specificity (SP-C) — if steering at α reduces MMLU
accuracy by more than ~2 pp, the intervention is a general-capability wrecker
rather than a targeted safety-substrate manipulation.

Judging: MMLU answers are strictly A/B/C/D. We parse the model's answer with
a robust regex first (deterministic, no LLM call); anything unparseable is
judged OTHER (i.e. counted as incorrect but flagged separately).
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    assert_gpu_pool_ok, assert_no_device_map_auto, assert_student_load_class, assert_thinking_off,
    read_done_ids, append_jsonl,
    JudgeCache, judge_call,
)


# gpt-5.4 judge template: match the model's free-form answer against the
# gold letter, mirroring the QA_I answer-grading protocol so MMLU and QA_I
# accuracies are apples-to-apples.
MMLU_JUDGE_TEMPLATE = """You are grading a model's answer to a multiple-choice question. The gold answer is option {gold_letter}. The question and options are shown below.

Question and options:
{question}

Model's free-form answer:
{model_answer}

Decide whether the model's answer semantically matches the gold option {gold_letter}. Reply with EXACTLY ONE of:
  CORRECT   — model's answer semantically matches option {gold_letter}.
  INCORRECT — model's answer semantically picks a different option than {gold_letter}.
  OTHER     — model refused, gave off-topic content, picked multiple options, or the answer is unparseable.

Do NOT coerce a refusal or off-topic answer into INCORRECT. Reply:"""


# Reuse the same intervention hook as QA_I intervene — copied intentionally
# so this script is a standalone runnable (no cross-file coupling on internals).

def make_intervention_hook(shape, u, alpha, sigma_proj, ctrl_state=None):
    device = "cuda:0"
    u_dev = u.to(device, dtype=torch.bfloat16) if u is not None else None
    ctrl_dev = ctrl_state.to(device, dtype=torch.bfloat16) if ctrl_state is not None else None

    def hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
            others = output[1:]
        else:
            h = output
            others = None

        if h.dim() != 3:
            return output
        if h.shape[1] < 2:
            return output

        h_last = h[:, -1:, :].clone()

        if shape == "patching":
            if ctrl_dev is None:
                return output
            h_last = ctrl_dev.unsqueeze(0).unsqueeze(0).expand(h.shape[0], 1, -1).to(h.dtype)
        elif shape == "ablation":
            if u_dev is None:
                return output
            proj = (h_last * u_dev.view(1, 1, -1)).sum(dim=-1, keepdim=True)
            h_last = h_last - proj * u_dev.view(1, 1, -1)
        elif shape == "steering":
            if u_dev is None:
                return output
            delta = float(alpha) * float(sigma_proj) * u_dev.view(1, 1, -1)
            h_last = h_last - delta.to(h.dtype)

        new_h = torch.cat([h[:, :-1, :], h_last], dim=1)
        if others is not None:
            return (new_h,) + others
        return new_h

    return hook


def get_language_layer(model, layer_idx):
    lm = model.language_model if hasattr(model, "language_model") else model.model.language_model
    return lm.layers[layer_idx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--merge_and_unload_if_treated", action="store_true", default=True)
    p.add_argument("--mmlu_slice", required=True)
    p.add_argument("--directions_pt", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--intervention_shape", choices=["patching", "ablation", "steering"], required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--component_set_source", choices=["m1_top_k", "random_matched"], default="m1_top_k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--treated_seed", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--resume_from_output", action="store_true")
    p.add_argument("--judge_model", default="gpt-5.4")
    p.add_argument("--judge_base_url", default="https://www.dmxapi.cn/v1")
    p.add_argument("--judge_cache", default="caches/mmlu_judge_cache.jsonl")
    return p.parse_args()


# Regex-based letter parser. Look for a standalone letter A-D anywhere in the
# first ~20 chars of the answer, preferably at the start.
_LETTER_RX = re.compile(r"\b([ABCD])\b")
_LEADING_LETTER_RX = re.compile(r"^\s*(?:\(|\[)?([ABCD])[\)\.\]:\s]")


def parse_mmlu_letter(text: str):
    """Return 'A'/'B'/'C'/'D' if parseable, else None."""
    if not text:
        return None
    text = text.strip()
    # Strip any Qwen think tags
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    m = _LEADING_LETTER_RX.match(text)
    if m:
        return m.group(1)
    # Fallback: first standalone letter in first 60 chars
    m = _LETTER_RX.search(text[:60])
    if m:
        return m.group(1)
    return None


def make_prompt(rec):
    q = rec["question"].strip()
    choices = rec["choices"]
    lines = [q, ""]
    for letter, choice in zip("ABCD", choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with the letter (A, B, C, or D) of the correct option, followed by a brief explanation.")
    return "\n".join(lines)


def main():
    args = parse_args()
    assert_gpu_pool_ok()

    # Load MMLU slice
    recs = [json.loads(l) for l in open(args.mmlu_slice) if l.strip()]
    print(f"[m2-mmlu] loaded {len(recs)} MMLU items")

    # Load direction
    directions = torch.load(args.directions_pt, weights_only=True)
    L = args.layer
    if L not in directions:
        raise RuntimeError(f"layer {L} not in {args.directions_pt}; available={list(directions.keys())}")
    u_m1 = directions[L].float().cpu()

    if args.component_set_source == "m1_top_k":
        u = u_m1
    elif args.component_set_source == "random_matched":
        rng = np.random.default_rng(args.seed)
        rand = rng.standard_normal(u_m1.shape).astype(np.float32)
        rand = rand / (np.linalg.norm(rand) + 1e-8)
        u = torch.from_numpy(rand).float()
    else:
        raise ValueError(f"unknown component_set_source={args.component_set_source}")

    arm_dir = Path(args.directions_pt).parent
    cs_path = arm_dir / "component_set.json"
    sigma_proj = 1.0
    if cs_path.exists():
        cs = json.load(open(cs_path))
        sigma_proj = float(cs.get("per_layer", {}).get(str(L), {}).get("sigma_ctrl", 1.0))
    else:
        print(f"[m2-mmlu] WARN: {cs_path} missing; sigma_proj defaulted to 1.0")

    # Resume
    done_ids = set()
    if args.resume_from_output:
        done_ids = read_done_ids(args.out, "mmlu_row_id")
    todo = [r for r in recs if r["mmlu_row_id"] not in done_ids]
    print(f"[m2-mmlu] shape={args.intervention_shape} source={args.component_set_source} layer={L} α={args.alpha} n_todo={len(todo)}")

    if not todo:
        print("[m2-mmlu] nothing to do")
        return 0

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert_thinking_off(rendered)

    print(f"[m2-mmlu] loading multimodal base on cuda:0 (bf16)")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    assert_no_device_map_auto(model)
    assert_student_load_class(model)

    if args.adapter and args.adapter.lower() not in ("null", "none", ""):
        ap = args.adapter
        if ap.endswith(".safetensors"):
            ap = str(Path(ap).parent)
        model = PeftModel.from_pretrained(model, ap)
        if args.merge_and_unload_if_treated:
            model = model.merge_and_unload()
    model.eval()

    # Prepare a single blank white image (reused for every MMLU item — text is
    # what matters; the image is a blank stand-in to keep the multimodal
    # processor happy).
    blank_img = Image.new("RGB", (224, 224), color=(255, 255, 255))

    judge_cache = JudgeCache(args.judge_cache)

    for i, rec in enumerate(todo):
        prompt_text = make_prompt(rec)
        user_content = [
            {"type": "image", "image": blank_img},
            {"type": "text", "text": prompt_text},
        ]
        messages = [{"role": "user", "content": user_content}]

        try:
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
        except TypeError:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = processor(text=prompt_str, images=[blank_img], return_tensors="pt")

        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}

        hook_fn = make_intervention_hook(
            args.intervention_shape, u, args.alpha, sigma_proj, ctrl_state=None,
        )
        layer = get_language_layer(model, L)
        handle = layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            handle.remove()

        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_len:]
        model_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        parsed = parse_mmlu_letter(model_answer)
        judge_verdict_source = "regex"
        if parsed is not None and parsed == rec["gold_letter"]:
            verdict = "CORRECT"
        elif parsed is not None:
            # Regex parsed a wrong letter — but the model may have answered
            # the correct choice's *content* without prefacing with a letter.
            # Delegate to gpt-5.4 judge to check semantic match.
            judge_verdict_source = "regex+llm"
            q_and_opts = prompt_text
            jprompt = MMLU_JUDGE_TEMPLATE.format(
                question=q_and_opts, gold_letter=rec["gold_letter"], model_answer=model_answer,
            )
            try:
                jresp = judge_call(
                    jprompt, judge_cache,
                    model=args.judge_model, base_url=args.judge_base_url,
                    temperature=0.0, seed=0, max_tokens=16,
                )
                ru = jresp.upper()
                if "CORRECT" in ru and "INCORRECT" not in ru:
                    verdict = "CORRECT"
                elif "INCORRECT" in ru:
                    verdict = "INCORRECT"
                else:
                    verdict = "OTHER"
            except Exception as e:
                print(f"[m2-mmlu] judge error {rec['mmlu_row_id']}: {e!r} — regex letter mismatch → INCORRECT")
                verdict = "INCORRECT"
        else:
            # parse_mmlu_letter returned None → the answer was prose without a
            # clean leading letter. Ask gpt-5.4 to grade against the gold letter.
            judge_verdict_source = "llm"
            q_and_opts = prompt_text
            jprompt = MMLU_JUDGE_TEMPLATE.format(
                question=q_and_opts, gold_letter=rec["gold_letter"], model_answer=model_answer,
            )
            try:
                jresp = judge_call(
                    jprompt, judge_cache,
                    model=args.judge_model, base_url=args.judge_base_url,
                    temperature=0.0, seed=0, max_tokens=16,
                )
                ru = jresp.upper()
                if "CORRECT" in ru and "INCORRECT" not in ru:
                    verdict = "CORRECT"
                elif "INCORRECT" in ru:
                    verdict = "INCORRECT"
                else:
                    verdict = "OTHER"
            except Exception as e:
                print(f"[m2-mmlu] judge error {rec['mmlu_row_id']}: {e!r} — leaving OTHER")
                verdict = "OTHER"

        rec_out = {
            "mmlu_row_id": rec["mmlu_row_id"],
            "subject": rec["subject"],
            "gold_letter": rec["gold_letter"],
            "parsed_letter": parsed,
            "model_answer": model_answer[:512],
            "judge_verdict": verdict,
            "judge_verdict_source": judge_verdict_source,
            "intervention_shape": args.intervention_shape,
            "layer": L,
            "alpha": args.alpha,
            "component_set_source": args.component_set_source,
            "treated_seed": args.treated_seed,
        }
        append_jsonl(args.out, rec_out)

        if (i + 1) % 25 == 0:
            print(f"[m2-mmlu] {args.intervention_shape}/{args.component_set_source} α={args.alpha} {i+1}/{len(todo)}")

    print(f"[m2-mmlu] done → {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
