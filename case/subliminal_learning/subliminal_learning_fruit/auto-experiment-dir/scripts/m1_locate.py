"""M1.1: Locate — screen DiT blocks for the banana-signal-carrying component.

Screen (per MECHANISM_ROUTING.md § Composition plan):
  1. LoRA-SVD Location: for every LoRA'd DiT block, compute ΔW = B·A per module, extract
     top-k singular directions per module, aggregate to a per-block score:
        block_score = mean_module ‖ΔW_teacher‖_F,
        overlap_score = mean_module cos(top1(B·A)_teacher, top1(B·A)_student_teacher_arm)
                       - mean_module cos(top1(B·A)_teacher, top1(B·A)_student_ctrl_arm)
     A high block_score AND high overlap_score → shortlist candidate.
  2. Activation-difference cross-check on a spaced subset of blocks: run base+teacher-LoRA
     vs base on a fixed prompt batch and measure ‖Δ h_l‖ per block.

Output: results/M1/locate.json
{
  "ranked_candidates": [
    {"block_id": int, "svd_score": ..., "overlap_score": ..., "act_diff": ..., "combined": ...},
    ...
  ],
  "top_k": [block_ids...],
  "svd_all_blocks": {block_id: {...}},
  "notes": "..."
}
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qwen_common import (  # noqa: E402
    BASE_MODEL,
    dump_json,
    load_lora_into_pipeline,
    load_pipeline,
    read_lines,
    set_all_seeds,
)


BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-lora", required=True)
    p.add_argument("--student-lora-teacher", required=True)
    p.add_argument("--student-lora-ctrl", required=True)
    p.add_argument("--eval-prompts", required=True,
                   help="Path to eval_pref160.txt (or subset) for activation-diff cross-check")
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--act-diff-n-prompts", type=int, default=8,
                   help="Number of prompts to hook activations on (kept small — 8 is fine "
                        "for a coarse Δ ranking).")
    p.add_argument("--act-diff-blocks", nargs="+", type=int, default=None,
                   help="Explicit block ids to hook for activation-diff. Default = spaced "
                        "intervals across the stack.")
    p.add_argument("--num-inference-steps", type=int, default=8,
                   help="Fewer steps for the act-diff pass (we don't need image quality).")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -----------------------------------------------------------------------------
# 1) LoRA-SVD screen
# -----------------------------------------------------------------------------
def _load_lora_state_dict(path):
    """Load a LoRA adapter state dict. Returns dict {qualified_module_name: {'A':..., 'B':...}}."""
    from safetensors.torch import load_file
    p = Path(path)
    if p.is_dir():
        cand = [p / "adapter_model.safetensors", p / "pytorch_lora_weights.safetensors"]
        p_use = next((c for c in cand if c.exists()), None)
        if p_use is None:
            raise FileNotFoundError(f"no lora weights in {p}")
        p = p_use
    raw = load_file(str(p))
    # Peft keys look like "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight"
    # (or with "default" adapter name).
    modules = defaultdict(dict)
    for k, v in raw.items():
        # Extract the base module qualifier and whether it's A or B.
        matched = False
        for tag in ("lora_A.default.weight", "lora_B.default.weight",
                    "lora_A.weight", "lora_B.weight"):
            if k.endswith(tag):
                base = k[: -(len(tag) + 1)]  # strip ".tag" including the leading dot
                base = re.sub(r"^(base_model\.model\.)?", "", base)
                slot = "A" if "lora_A" in tag else "B"
                modules[base][slot] = v
                matched = True
                break
        if not matched:
            # Silent skip on unrecognized keys
            pass
    return modules


def _lora_delta_svd(module_dict):
    """For each module with both A and B, compute ΔW = B·A and top-1 singular direction.

    Returns dict {module_name: {'fro': float, 'sv1': float, 'u1': tensor, 'v1': tensor}}.
    """
    out = {}
    for name, ab in module_dict.items():
        if "A" not in ab or "B" not in ab:
            continue
        A = ab["A"].float()  # [r, in]
        B = ab["B"].float()  # [out, r]
        delta = B @ A        # [out, in]
        fro = float(delta.norm())
        try:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            sv1 = float(S[0])
            u1 = U[:, 0]  # [out]
            v1 = Vh[0]    # [in]
        except Exception:
            sv1 = fro
            u1 = torch.zeros(delta.shape[0])
            v1 = torch.zeros(delta.shape[1])
        out[name] = {"fro": fro, "sv1": sv1, "u1": u1, "v1": v1}
    return out


def _block_of(module_name):
    m = BLOCK_RE.search(module_name)
    return int(m.group(1)) if m else -1


def _aggregate_by_block(svd_out):
    per_block = defaultdict(lambda: {"modules": [], "fros": [], "svs": []})
    for mod, r in svd_out.items():
        b = _block_of(mod)
        if b < 0:
            continue
        per_block[b]["modules"].append(mod)
        per_block[b]["fros"].append(r["fro"])
        per_block[b]["svs"].append(r["sv1"])
    result = {}
    for b, r in per_block.items():
        result[b] = {
            "n_modules": len(r["modules"]),
            "mean_fro": float(np.mean(r["fros"])),
            "sum_fro": float(np.sum(r["fros"])),
            "mean_sv1": float(np.mean(r["svs"])),
        }
    return result


def _overlap_by_block(svd_teacher, svd_student):
    """Mean cosine similarity between corresponding module top-1 directions."""
    per_block_v = defaultdict(list)
    per_block_u = defaultdict(list)
    shared = set(svd_teacher.keys()) & set(svd_student.keys())
    for mod in shared:
        b = _block_of(mod)
        if b < 0:
            continue
        v_t = svd_teacher[mod]["v1"]
        v_s = svd_student[mod]["v1"]
        u_t = svd_teacher[mod]["u1"]
        u_s = svd_student[mod]["u1"]
        if v_t.shape == v_s.shape and v_t.numel() > 0:
            cv = abs(float(torch.nn.functional.cosine_similarity(v_t.unsqueeze(0), v_s.unsqueeze(0)).item()))
            per_block_v[b].append(cv)
        if u_t.shape == u_s.shape and u_t.numel() > 0:
            cu = abs(float(torch.nn.functional.cosine_similarity(u_t.unsqueeze(0), u_s.unsqueeze(0)).item()))
            per_block_u[b].append(cu)
    out = {}
    for b in set(per_block_v.keys()) | set(per_block_u.keys()):
        out[b] = {
            "mean_cos_v": float(np.mean(per_block_v[b])) if per_block_v[b] else 0.0,
            "mean_cos_u": float(np.mean(per_block_u[b])) if per_block_u[b] else 0.0,
        }
    return out


# -----------------------------------------------------------------------------
# 2) Activation-difference cross-check
# -----------------------------------------------------------------------------
def _hook_block_outputs(pipe, prompts, block_ids, height, width, steps, seed):
    """Return dict {block_id: mean-abs residual-stream tensor across timesteps and prompts}."""
    caps = {b: [] for b in block_ids}
    handles = []
    tb = pipe.transformer.transformer_blocks
    for b in block_ids:
        def make_hook(bi):
            def _h(module, inputs, output):
                # QwenImageTransformerBlock returns (encoder_hidden_states, hidden_states)
                if isinstance(output, tuple) and len(output) == 2:
                    _, h_img = output
                    caps[bi].append(h_img.detach().float().mean(dim=(0, 1)).cpu())
            return _h
        handles.append(tb[b].register_forward_hook(make_hook(b)))
    try:
        for p in prompts:
            with torch.inference_mode():
                gen = torch.Generator(device="cuda").manual_seed(seed)
                pipe(prompt=p, height=height, width=width, num_inference_steps=steps,
                     true_cfg_scale=4.0, generator=gen)
    finally:
        for h in handles:
            h.remove()
    return {b: (torch.stack(v).mean(0) if v else torch.zeros(1)) for b, v in caps.items()}


def _act_diff(pipe, prompts, block_ids, height, width, steps, seed,
              teacher_lora, ctrl_lora=None):
    """Return {block_id: ‖Δh_teacher‖_2 − ‖Δh_ctrl‖_2} where Δh_arm = h_base+arm - h_base."""
    # Baseline (no adapter)
    print("[act-diff] baseline (no adapter)")
    if hasattr(pipe.transformer, "disable_lora"):
        try:
            pipe.transformer.disable_lora()
        except Exception:
            pass
    base_h = _hook_block_outputs(pipe, prompts, block_ids, height, width, steps, seed)
    # Teacher
    print("[act-diff] teacher adapter")
    load_lora_into_pipeline(pipe, teacher_lora, adapter_name="teacher")
    if hasattr(pipe.transformer, "enable_lora"):
        pipe.transformer.enable_lora()
    try:
        pipe.transformer.set_adapters(["teacher"])
    except Exception:
        pass
    teacher_h = _hook_block_outputs(pipe, prompts, block_ids, height, width, steps, seed)
    delta_t = {b: float((teacher_h[b] - base_h[b]).norm()) for b in block_ids}
    if ctrl_lora:
        print("[act-diff] ctrl adapter")
        load_lora_into_pipeline(pipe, ctrl_lora, adapter_name="ctrl")
        try:
            pipe.transformer.set_adapters(["ctrl"])
        except Exception:
            pass
        ctrl_h = _hook_block_outputs(pipe, prompts, block_ids, height, width, steps, seed)
        delta_c = {b: float((ctrl_h[b] - base_h[b]).norm()) for b in block_ids}
        return {b: delta_t[b] - delta_c[b] for b in block_ids}, delta_t, delta_c
    return delta_t, delta_t, None


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    print("[m1-locate] loading LoRA state dicts...")
    lora_teacher = _load_lora_state_dict(args.teacher_lora)
    lora_student_t = _load_lora_state_dict(args.student_lora_teacher)
    lora_student_c = _load_lora_state_dict(args.student_lora_ctrl)
    print(f"[m1-locate] teacher lora modules={len(lora_teacher)}  "
          f"student_teacher_arm={len(lora_student_t)}  student_ctrl={len(lora_student_c)}")

    print("[m1-locate] computing SVD deltas...")
    svd_teacher = _lora_delta_svd(lora_teacher)
    svd_student_t = _lora_delta_svd(lora_student_t)
    svd_student_c = _lora_delta_svd(lora_student_c)

    per_block_teacher = _aggregate_by_block(svd_teacher)
    per_block_student_t = _aggregate_by_block(svd_student_t)
    per_block_student_c = _aggregate_by_block(svd_student_c)

    overlap_t = _overlap_by_block(svd_teacher, svd_student_t)   # teacher ↔ student-teacher-arm
    overlap_c = _overlap_by_block(svd_teacher, svd_student_c)   # teacher ↔ student-ctrl-arm

    all_blocks = sorted(set(per_block_teacher) | set(per_block_student_t) | set(per_block_student_c))
    per_block = {}
    for b in all_blocks:
        ov_t = overlap_t.get(b, {"mean_cos_v": 0.0, "mean_cos_u": 0.0})
        ov_c = overlap_c.get(b, {"mean_cos_v": 0.0, "mean_cos_u": 0.0})
        per_block[b] = {
            "teacher": per_block_teacher.get(b, {}),
            "student_teacher_arm": per_block_student_t.get(b, {}),
            "student_ctrl_arm": per_block_student_c.get(b, {}),
            "overlap_v_teacher_studentT": ov_t["mean_cos_v"],
            "overlap_v_teacher_studentC": ov_c["mean_cos_v"],
            "overlap_u_teacher_studentT": ov_t["mean_cos_u"],
            "overlap_u_teacher_studentC": ov_c["mean_cos_u"],
            "overlap_gap_v": ov_t["mean_cos_v"] - ov_c["mean_cos_v"],
            "overlap_gap_u": ov_t["mean_cos_u"] - ov_c["mean_cos_u"],
        }

    # Combined score: normalize each of (student_teacher_arm.mean_fro,
    # overlap_gap_v, overlap_gap_u) and sum
    def _norm(vals):
        arr = np.array([v for v in vals if v is not None])
        if arr.size == 0 or arr.std() < 1e-9:
            return {}
        return {i: (v - arr.mean()) / arr.std() for i, v in enumerate(vals)}

    st_fro = [per_block[b]["student_teacher_arm"].get("mean_fro", 0.0) for b in all_blocks]
    st_fro_n = _norm(st_fro)
    ov_v = [per_block[b]["overlap_gap_v"] for b in all_blocks]
    ov_v_n = _norm(ov_v)
    ov_u = [per_block[b]["overlap_gap_u"] for b in all_blocks]
    ov_u_n = _norm(ov_u)
    for i, b in enumerate(all_blocks):
        per_block[b]["combined_z"] = (st_fro_n.get(i, 0.0) + ov_v_n.get(i, 0.0) + ov_u_n.get(i, 0.0))

    # 2) Activation-diff cross-check on spaced blocks
    if args.act_diff_blocks is None:
        args.act_diff_blocks = [b for b in [5, 10, 20, 30, 40, 50, 55] if b in all_blocks]
    prompts = read_lines(args.eval_prompts)[: args.act_diff_n_prompts]
    print(f"[m1-locate] act-diff on blocks {args.act_diff_blocks} with {len(prompts)} prompts")

    pipe = load_pipeline()
    try:
        act_diff, delta_t, delta_c = _act_diff(
            pipe, prompts, args.act_diff_blocks, args.height, args.width,
            args.num_inference_steps, args.seed,
            args.student_lora_teacher, args.student_lora_ctrl,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[m1-locate] act-diff FAILED: {e}. Continuing with SVD-only scores.")
        act_diff, delta_t, delta_c = ({}, {}, {})

    for b in all_blocks:
        per_block[b]["act_diff_teacher_minus_ctrl"] = act_diff.get(b, None)
        per_block[b]["act_diff_teacher_norm"] = delta_t.get(b, None)
        per_block[b]["act_diff_ctrl_norm"] = delta_c.get(b, None) if delta_c else None

    # Rank blocks by combined_z
    ranked = sorted(all_blocks, key=lambda b: per_block[b]["combined_z"], reverse=True)
    top_k = ranked[: args.top_k]

    out = {
        "num_blocks": len(all_blocks),
        "per_block": {int(k): v for k, v in per_block.items()},
        "ranked_by_combined_z": ranked,
        "top_k": top_k,
        "act_diff_blocks_scanned": args.act_diff_blocks,
        "candidate_direction_target_module": "attn.to_out.0",
        "notes": ("combined_z = normalized(student_teacher_arm.mean_fro) + "
                  "normalized(overlap_gap_v) + normalized(overlap_gap_u). "
                  "overlap_gap_v = cos(v_teacher, v_student_teacher_arm) - cos(v_teacher, v_student_ctrl_arm) "
                  "at v = right singular vector of ΔW; overlap_gap_u analogously for the left singular vector. "
                  "act_diff_teacher_minus_ctrl = ‖h_teacher-h_base‖ − ‖h_ctrl-h_base‖ per block."),
    }
    dump_json(out, args.out)
    print(f"[m1-locate] top-{args.top_k} blocks: {top_k}")


if __name__ == "__main__":
    main()
