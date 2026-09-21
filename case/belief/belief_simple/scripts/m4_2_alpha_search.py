"""M4.2: one (α_p, α_a) grid cell — evaluate the controller on the belief_core val split.

The controller pipeline per example:
  1. Extract hidden states at probing_layers, last token → probe input.
  2. Probe predicts frame ∈ {WK, PB, AB}.
  3. If WK: identity forward pass (α=1 on ALL heads).
     If PB: multiply H*_personal head outputs by α_personal.
     If AB: multiply H*_attributed head outputs by α_attributed.
  4. Score task accuracy per belief task, plus Δ_wk on world_knowledge.

We evaluate all THREE tasks per config so recovered / degraded / Δ_wk are all measured.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from belief_utils import (
    load_model_and_tokenizer, load_task, load_json, save_json, set_seed, model_arch_info,
    install_head_scaling_hooks, remove_hooks, continuation_logprob_batch,
)


TASK_LABEL = {"world_knowledge": 0, "personal_belief": 1, "attributed_belief": 2}


def _load_probe(path, device):
    d = torch.load(path, map_location=device)
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, dropout):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
        def forward(self, x): return self.net(x)
    m = MLP(d["in_dim"], d["hidden_dim"], 3, d["dropout"]).to(device)
    m.load_state_dict(d["state_dict"])
    m.eval()
    return m, d["probing_layers"]


def _extract_and_predict_frame(net, probe, tok, ex, probing_layers, device):
    with torch.no_grad():
        input_ids = torch.tensor([tok.encode(ex.prompt, add_special_tokens=False)],
                                 dtype=torch.long, device=device)
        out = net(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states
        chunks = [hs[l + 1][0, -1].float() for l in probing_layers]
        feat = torch.cat(chunks, dim=0)
        pred = probe(feat.unsqueeze(0)).argmax(-1).item()
    return int(pred)


def _score_example_under_scale(net, tok, ex, scale_map, device):
    handles = install_head_scaling_hooks(net, scale_map) if scale_map else []
    try:
        lps = continuation_logprob_batch(net, tok, ex.prompt, [ex.gold, ex.distractor], device=device)
    finally:
        remove_hooks(handles)
    return int(lps[0] > lps[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--hstar-personal", required=True)
    ap.add_argument("--hstar-attributed", required=True)
    ap.add_argument("--alpha-personal", type=float, required=True)
    ap.add_argument("--alpha-attributed", type=float, required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    set_seed(args.split_seed)
    net, tok = load_model_and_tokenizer(args.model_root, args.model, dtype=args.dtype, device=args.device)
    for p in net.parameters():
        p.requires_grad_(False)

    probe, probing_layers = _load_probe(args.probe, args.device)
    Hp = [tuple(h) for h in load_json(args.hstar_personal)["hstar_heads"]]
    Ha = [tuple(h) for h in load_json(args.hstar_attributed)["hstar_heads"]]

    # Reload the same split M4.1 used
    split_path = args.probe.replace("probe.pt", "probe_split.json")
    split_meta = load_json(split_path)
    val_idx = np.array(split_meta["val_indices"], dtype=np.int64)
    train_idx = np.array(split_meta["train_indices"], dtype=np.int64)
    example_labels = np.array(split_meta["example_task_labels"], dtype=np.int64)

    # Reload all examples in the same order used by M4.1 (grouped by task label in TASK_LABEL order)
    all_examples = []
    for task in TASK_LABEL.keys():
        all_examples.extend(load_task(args.data_root, task))
    assert len(all_examples) == len(example_labels), "example count mismatch with M4.1 split"

    if args.split == "val":
        sel = val_idx
    elif args.split == "train":
        sel = train_idx
    else:
        sel = np.arange(len(all_examples))
    examples = [all_examples[i] for i in sel]
    labels = example_labels[sel]

    print(f"[m4.2] {args.model} α_p={args.alpha_personal} α_a={args.alpha_attributed} "
          f"split={args.split} n={len(examples)}")

    # Also record the α=1 (no amplification) baseline scores for the SAME examples in the SAME order,
    # so recovered / degraded can be computed downstream.
    per_example = []
    correct_ctrl = 0
    correct_baseline = 0
    frame_correct = 0
    t0 = time.time()
    for i, (ex, lbl) in enumerate(zip(examples, labels)):
        # baseline (no amplification)
        base = _score_example_under_scale(net, tok, ex, {}, args.device)
        # frame prediction
        pred = _extract_and_predict_frame(net, probe, tok, ex, probing_layers, args.device)
        # build scale_map from prediction
        scale_map = {}
        if pred == TASK_LABEL["personal_belief"]:
            scale_map = {h: args.alpha_personal for h in Hp}
        elif pred == TASK_LABEL["attributed_belief"]:
            scale_map = {h: args.alpha_attributed for h in Ha}
        # controller
        ctrl = _score_example_under_scale(net, tok, ex, scale_map, args.device)
        per_example.append({
            "example_idx": int(sel[i]), "true_label": int(lbl), "pred_label": pred,
            "baseline_correct": base, "controller_correct": ctrl,
        })
        correct_baseline += base
        correct_ctrl += ctrl
        frame_correct += int(pred == lbl)
    dt = time.time() - t0
    print(f"[m4.2] done in {dt:.1f}s")

    # Compute recovered / degraded on belief tasks only
    belief_recovered = sum(1 for p in per_example
                           if p["true_label"] in (1, 2) and p["controller_correct"] == 1 and p["baseline_correct"] == 0)
    belief_degraded = sum(1 for p in per_example
                          if p["true_label"] in (1, 2) and p["controller_correct"] == 0 and p["baseline_correct"] == 1)
    net_impr = belief_recovered - belief_degraded

    # world_knowledge Δ_wk
    wk_baseline = sum(1 for p in per_example if p["true_label"] == 0 and p["baseline_correct"] == 1)
    wk_ctrl = sum(1 for p in per_example if p["true_label"] == 0 and p["controller_correct"] == 1)
    wk_n = sum(1 for p in per_example if p["true_label"] == 0)
    delta_wk = (wk_baseline - wk_ctrl) / max(1, wk_n)

    # Per-task summaries
    per_task = {}
    for tname, tlbl in TASK_LABEL.items():
        n_t = sum(1 for p in per_example if p["true_label"] == tlbl)
        acc_base = sum(1 for p in per_example if p["true_label"] == tlbl and p["baseline_correct"] == 1) / max(1, n_t)
        acc_ctrl = sum(1 for p in per_example if p["true_label"] == tlbl and p["controller_correct"] == 1) / max(1, n_t)
        per_task[tname] = {"n": n_t, "acc_baseline": acc_base, "acc_controller": acc_ctrl,
                           "delta": acc_base - acc_ctrl}

    frame_acc = frame_correct / max(1, len(examples))
    result = {
        "model": args.model, "alpha_personal": args.alpha_personal, "alpha_attributed": args.alpha_attributed,
        "split": args.split, "n_examples": len(examples),
        "frame_acc": frame_acc, "per_task": per_task,
        "belief_recovered": belief_recovered, "belief_degraded": belief_degraded,
        "net_improvement": net_impr, "delta_wk": delta_wk,
        "elapsed_sec": dt,
    }
    save_json(args.output, result)
    print(f"[m4.2] net_impr={net_impr} Δ_wk={delta_wk:.4f} frame_acc={frame_acc:.4f} → {args.output}")


if __name__ == "__main__":
    main()
