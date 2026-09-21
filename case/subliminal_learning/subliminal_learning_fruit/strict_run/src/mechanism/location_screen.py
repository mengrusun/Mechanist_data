"""M1 — Location screen (Parameter-Space Task Vectors on LoRA ΔW).

Purpose: rank DiT blocks × modules by the differential Grassmann subspace overlap between
teacher-arm student ΔW (mean over 8 seeds) and Ctrl-B student ΔW (mean over 8 seeds),
where "closer to teacher" is measured against the teacher anchor LoRA's ΔW subspace.

For each DiT block b and target module m:
  1. Load LoRA A/B for teacher (anchor) and each of 8 teacher-arm + 8 ctrl-arm students.
  2. Compute ΔW = B @ A per module (rank ≤ 16 exact by construction).
  3. SVD each ΔW (top-16 exact for r=16 adapters).
  4. Compute Grassmann principal-angle overlap at k ∈ {1, 2, 4, 8} between each student's
     subspace and the teacher-anchor subspace.
  5. Rank blocks by  overlap_gap_k = mean_teacher_arm(overlap_k) − mean_ctrl_arm(overlap_k).
  6. Emit shortlist ≤ 20% of blocks × modules × timestep buckets.

For the top-ranked block b* extract the top-1 left singular vector of the *mean-teacher-arm*
ΔW at (b*, attn.to_out.0) — this is the "banana direction" for M2/M3 additive steering.
Also emit top-2 and matched-random-control (adjacent block off-shortlist) direction .pt.

Output: runs/M1/shortlist.json (metadata), runs/M1/banana_direction.pt (main),
        runs/M1/top2_direction.pt, runs/M1/matched_control_direction.pt.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_common import dump_json  # noqa: E402


BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-lora", required=True)
    p.add_argument("--student-loras-teacher", nargs="+", required=True)
    p.add_argument("--student-loras-ctrl", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-module", default="attn.to_out.0",
                   help="Which module (name-suffix) to source the M2 direction from.")
    p.add_argument("--k-grid", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--topk-frac", type=float, default=0.20)
    p.add_argument("--timestep-buckets", nargs="+", type=int, default=[5, 12, 20],
                   help="Denoising timesteps at which the direction is later applied.")
    return p.parse_args()


def _load_lora_state_dict(path):
    from safetensors.torch import load_file
    p = Path(path)
    if p.is_dir():
        cand = [p / "adapter_model.safetensors", p / "pytorch_lora_weights.safetensors"]
        p_use = next((c for c in cand if c.exists()), None)
        if p_use is None:
            raise FileNotFoundError(f"no lora weights in {p}")
        p = p_use
    raw = load_file(str(p))
    modules = defaultdict(dict)
    for k, v in raw.items():
        for tag in ("lora_A.default.weight", "lora_B.default.weight",
                    "lora_A.weight", "lora_B.weight"):
            if k.endswith(tag):
                base = k[: -(len(tag) + 1)]
                base = re.sub(r"^(base_model\.model\.)?", "", base)
                slot = "A" if "lora_A" in tag else "B"
                modules[base][slot] = v
                break
    return modules


def _module_delta(state, mod_name):
    ab = state.get(mod_name)
    if ab is None or "A" not in ab or "B" not in ab:
        return None
    A = ab["A"].float()
    B = ab["B"].float()
    return B @ A


def _svd(delta, k: int = 16):
    q = min(k + 4, delta.shape[-1], delta.shape[-2])
    U, S, V = torch.svd_lowrank(delta, q=q, niter=2)
    U = U[:, :k]
    S = S[:k]
    Vh = V[:, :k].transpose(0, 1)
    return U, S, Vh


def _block_of(mod_name):
    m = BLOCK_RE.search(mod_name)
    return int(m.group(1)) if m else -1


def _overlap_k(U_a, U_b, k):
    k = min(k, U_a.shape[1], U_b.shape[1])
    if k <= 0:
        return 0.0
    Uak = U_a[:, :k]
    Ubk = U_b[:, :k]
    proj = Ubk @ (Ubk.transpose(-1, -2) @ Uak)
    cos2 = (proj.norm(dim=0) ** 2 / (Uak.norm(dim=0) ** 2 + 1e-12)).clamp(0.0, 1.0)
    return float(cos2.mean().item())


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[m1] loading teacher LoRA: {args.teacher_lora}")
    teacher_state = _load_lora_state_dict(args.teacher_lora)
    print(f"[m1] loading {len(args.student_loras_teacher)} teacher-arm student LoRAs")
    student_t_states = [_load_lora_state_dict(p) for p in args.student_loras_teacher]
    print(f"[m1] loading {len(args.student_loras_ctrl)} Ctrl-B student LoRAs")
    student_c_states = [_load_lora_state_dict(p) for p in args.student_loras_ctrl]

    all_modules = set(teacher_state.keys())
    for s in student_t_states + student_c_states:
        all_modules &= set(s.keys())
    all_modules = sorted(all_modules)
    print(f"[m1] common modules: {len(all_modules)}")

    def svd_state(state, mods):
        out = {}
        for m in mods:
            delta = _module_delta(state, m)
            if delta is None:
                continue
            U, S, Vh = _svd(delta)
            out[m] = {"U": U, "S": S, "Vh": Vh, "fro": float(delta.norm())}
        return out

    print("[m1] SVDs for teacher anchor...")
    svd_t = svd_state(teacher_state, all_modules)
    print(f"[m1] SVDs for {len(student_t_states)} teacher-arm students...")
    svd_students_t = [svd_state(s, all_modules) for s in student_t_states]
    print(f"[m1] SVDs for {len(student_c_states)} Ctrl-B students...")
    svd_students_c = [svd_state(s, all_modules) for s in student_c_states]

    per_block: Dict[int, Dict[str, float]] = {}
    all_blocks = sorted({_block_of(m) for m in all_modules if _block_of(m) >= 0})
    print(f"[m1] DiT blocks discovered: {len(all_blocks)}: "
          f"{all_blocks[:5]}...{all_blocks[-5:]}")

    for b in all_blocks:
        mods_b = [m for m in all_modules if _block_of(m) == b and m in svd_t]
        block_metrics = {"n_modules": len(mods_b)}

        for k in args.k_grid:
            overlaps_ta = []
            overlaps_ca = []
            for m in mods_b:
                Vh_t = svd_t[m]["Vh"].transpose(0, 1)
                U_t = svd_t[m]["U"]
                for sT in svd_students_t:
                    if m not in sT:
                        continue
                    Vh_s = sT[m]["Vh"].transpose(0, 1)
                    U_s = sT[m]["U"]
                    ov_v = _overlap_k(Vh_s, Vh_t, k)
                    ov_u = _overlap_k(U_s, U_t, k)
                    overlaps_ta.append(0.5 * (ov_v + ov_u))
                for sC in svd_students_c:
                    if m not in sC:
                        continue
                    Vh_s = sC[m]["Vh"].transpose(0, 1)
                    U_s = sC[m]["U"]
                    ov_v = _overlap_k(Vh_s, Vh_t, k)
                    ov_u = _overlap_k(U_s, U_t, k)
                    overlaps_ca.append(0.5 * (ov_v + ov_u))

            oc_ta = float(np.mean(overlaps_ta)) if overlaps_ta else 0.0
            oc_ca = float(np.mean(overlaps_ca)) if overlaps_ca else 0.0
            block_metrics[f"overlap_k{k}_teacher_arm"] = oc_ta
            block_metrics[f"overlap_k{k}_ctrl_arm"] = oc_ca
            block_metrics[f"overlap_gap_k{k}"] = oc_ta - oc_ca
            block_metrics[f"ratio_k{k}"] = ((oc_ta / oc_ca) if oc_ca > 1e-6
                                            else (999.0 if oc_ta > 1e-6 else 1.0))
        per_block[b] = block_metrics

    ranking_k1 = sorted(all_blocks, key=lambda b: per_block[b]["overlap_gap_k1"], reverse=True)
    ranking_k2 = sorted(all_blocks, key=lambda b: per_block[b]["overlap_gap_k2"], reverse=True)
    top_block_k1 = ranking_k1[0]
    top_block_k2 = ranking_k2[0]

    b_star = top_block_k1
    which_k = 1
    if per_block[top_block_k1]["ratio_k1"] < 2.0 and per_block[top_block_k2]["ratio_k2"] >= 2.0:
        b_star = top_block_k2
        which_k = 2

    # Extract "banana direction" — top-1 left singular vector of mean_teacher-arm ΔW at
    # (b_star, target_module).
    target_mod = None
    for m in all_modules:
        if _block_of(m) == b_star and m.endswith(f".{args.target_module}"):
            target_mod = m
            break
    if target_mod is None:
        cands = [(m, svd_t.get(m, {}).get("fro", 0.0)) for m in all_modules
                 if _block_of(m) == b_star]
        target_mod = max(cands, key=lambda x: x[1])[0]
        print(f"[m1] target_module '{args.target_module}' not in block {b_star}, "
              f"fallback: {target_mod}")

    deltas = [d for state in student_t_states
              if (d := _module_delta(state, target_mod)) is not None]
    if not deltas:
        raise RuntimeError(f"[m1] no student-teacher-arm deltas for {target_mod}")
    mean_delta = torch.stack(deltas).mean(dim=0)
    U_m, S_m, Vh_m = _svd(mean_delta)
    direction_out = U_m[:, 0].contiguous()
    direction_in = Vh_m[0].contiguous()

    torch.save({
        "block": b_star, "which_k": which_k, "target_module": target_mod,
        "direction_out": direction_out, "direction_in": direction_in,
        "sigma_top": float(S_m[0]),
    }, out_dir / "banana_direction.pt")

    top2_block = ranking_k1[1] if len(ranking_k1) > 1 else ranking_k1[0]
    shortlist_top2 = {top_block_k1, top2_block}
    matched_ctrl_block = None
    for offset in (1, -1, 2, -2, 3, -3):
        cand = b_star + offset
        if cand in all_blocks and cand not in shortlist_top2:
            matched_ctrl_block = cand
            break
    if matched_ctrl_block is None:
        matched_ctrl_block = max((b for b in all_blocks if b not in shortlist_top2),
                                 key=lambda b: abs(b - b_star), default=b_star)

    def _extract_direction_for_block(block, tag: str):
        tm = None
        for m in all_modules:
            if _block_of(m) == block and m.endswith(f".{args.target_module}"):
                tm = m
                break
        if tm is None:
            cands = [(m, svd_t.get(m, {}).get("fro", 0.0)) for m in all_modules
                     if _block_of(m) == block]
            if not cands:
                return None, None, None
            tm = max(cands, key=lambda x: x[1])[0]
        deltas_b = [d for st in student_t_states
                    if (d := _module_delta(st, tm)) is not None]
        if not deltas_b:
            return None, None, None
        mean_d = torch.stack(deltas_b).mean(dim=0)
        U_x, S_x, Vh_x = _svd(mean_d)
        dir_out = U_x[:, 0].contiguous()
        dir_pt_path = out_dir / f"{tag}_direction.pt"
        torch.save({
            "block": block, "target_module": tm, "direction_out": dir_out,
            "direction_in": Vh_x[0].contiguous(), "sigma_top": float(S_x[0]),
        }, dir_pt_path)
        return block, tm, str(dir_pt_path)

    top2_info = (_extract_direction_for_block(top2_block, "top2")
                 if top2_block != b_star
                 else (b_star, target_mod, str(out_dir / "banana_direction.pt")))
    mc_info = _extract_direction_for_block(matched_ctrl_block, "matched_control")

    # Shortlist: top ~20% of block × timestep-bucket entries by overlap_gap_k1
    N_total = len(all_blocks) * len(args.timestep_buckets)
    N_short = max(1, int(round(args.topk_frac * N_total)))
    # Flat scores: repeat overlap_gap_k1 across timestep buckets (no per-timestep signal at
    # rest — parameter-space is timestep-agnostic; the buckets are for the M2 hook to sample).
    shortlist_entries = []
    for b in ranking_k1[:N_short // max(1, len(args.timestep_buckets)) + 1]:
        for tb in args.timestep_buckets:
            shortlist_entries.append({
                "block": b, "timestep_bucket": tb,
                "overlap_gap_k1": per_block[b]["overlap_gap_k1"],
                "target_module_family": args.target_module,
            })
    shortlist_entries = shortlist_entries[:N_short]

    sites_out = {
        "top1": {"block": b_star, "target_module": target_mod,
                 "direction_out_path": str(out_dir / "banana_direction.pt"),
                 "overlap_gap_k1": per_block[b_star]["overlap_gap_k1"]},
        "top2": {"block": top2_info[0], "target_module": top2_info[1],
                 "direction_out_path": top2_info[2],
                 "overlap_gap_k1": per_block[top2_info[0]]["overlap_gap_k1"]
                                     if top2_info[0] is not None else None},
        "matched_control": {"block": mc_info[0], "target_module": mc_info[1],
                            "direction_out_path": mc_info[2],
                            "overlap_gap_k1": per_block[mc_info[0]]["overlap_gap_k1"]
                                                if mc_info[0] is not None else None},
    }

    dump_json({
        "per_block": {str(b): v for b, v in per_block.items()},
        "ranking_by_overlap_gap_k1": ranking_k1,
        "ranking_by_overlap_gap_k2": ranking_k2,
        "top_block_k1": top_block_k1, "top_block_k2": top_block_k2,
        "top_block_k1_ratio": per_block[top_block_k1]["ratio_k1"],
        "top_block_k2_ratio": per_block[top_block_k2]["ratio_k2"],
        "b_star": b_star, "b_star_k": which_k, "target_module": target_mod,
        "shortlist_size": len(shortlist_entries),
        "shortlist_fraction_of_total": len(shortlist_entries) / max(1, N_total),
        "shortlist_modules_layers_timesteps": shortlist_entries,
        "k_grid": args.k_grid,
        "timestep_buckets": args.timestep_buckets,
        "n_students_teacher_arm": len(student_t_states),
        "n_students_ctrl_arm": len(student_c_states),
        "sites": sites_out,
    }, out_dir / "shortlist.json")

    print(f"[m1] b*={b_star} (k={which_k})  target_module={target_mod}")
    print(f"[m1] top_block_k1={top_block_k1} ratio={per_block[top_block_k1]['ratio_k1']:.2f}  "
          f"|  top_block_k2={top_block_k2} ratio={per_block[top_block_k2]['ratio_k2']:.2f}")
    print(f"[m1] shortlist={len(shortlist_entries)} / total={N_total} "
          f"({100 * len(shortlist_entries) / max(1, N_total):.1f}%)")


if __name__ == "__main__":
    main()
