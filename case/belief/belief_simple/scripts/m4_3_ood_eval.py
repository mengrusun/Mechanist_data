"""M4.3: OOD evaluation on belief_holdout — one of three arms per invocation.

Arms:
  baseline_no_control — no probe, no amplification, no prompt hint
  controller         — probe predicts, amplify H*_target by (α_p*, α_a*)
  prompt_hint        — no probe, no amplification, add per-task frame prefix to the prompt

Records per-task acc, plus (for controller arm) frame_acc + per-example baseline/controller
correctness so recovered/degraded can be computed in m4_3_report.
Also (for controller arm) runs PPL on the cached pretraining sample under the same controller
behavior — the controller predicts on each ~1024-token window and amplifies if not WK.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from belief_utils import (
    load_model_and_tokenizer, load_task, load_json, save_json, set_seed, model_arch_info,
    install_head_scaling_hooks, remove_hooks, continuation_logprob_batch, evaluate_ppl,
)


TASK_LABEL = {"world_knowledge": 0, "personal_belief": 1, "attributed_belief": 2}
LABEL_TASK = {v: k for k, v in TASK_LABEL.items()}
OOD_FILE = {"world_knowledge": "reality.jsonl",
            "personal_belief": "believe_truth.jsonl",
            "attributed_belief": "follow_belief.jsonl"}


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


def _predict_frame(net, probe, tok, prompt, probing_layers, device):
    with torch.no_grad():
        ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], dtype=torch.long, device=device)
        out = net(input_ids=ids, output_hidden_states=True)
        hs = out.hidden_states
        chunks = [hs[l + 1][0, -1].float() for l in probing_layers]
        feat = torch.cat(chunks, dim=0)
        return int(probe(feat.unsqueeze(0)).argmax(-1).item())


def _score(net, tok, prompt, gold, distractor, scale_map, device):
    handles = install_head_scaling_hooks(net, scale_map) if scale_map else []
    try:
        lps = continuation_logprob_batch(net, tok, prompt, [gold, distractor], device=device)
    finally:
        remove_hooks(handles)
    return int(lps[0] > lps[1])


PROMPT_HINT = {
    "personal_belief": "Answer based on reality, ignoring what others believe. ",
    "attributed_belief": "Answer based on what the named person believes, even if it conflicts with reality. ",
    "world_knowledge": "",
}


def _evaluate_ppl_with_controller(net, probe, tok, ppl_tokens, Hp, Ha, alpha_p, alpha_a,
                                   probing_layers, window: int, device: str):
    """PPL with controller. For each 1024-token window: predict frame on the window prefix,
    then compute PPL under the corresponding amplification."""
    import math
    total_nll = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, ppl_tokens.numel() - 1, window):
            chunk = ppl_tokens[i : i + window]
            if chunk.numel() < 2:
                break
            input_ids = chunk.unsqueeze(0).to(device)
            # For controller, take a probe reading at the LAST token of the window and
            # choose an amplification for that window.
            out = net(input_ids=input_ids, output_hidden_states=True)
            hs = out.hidden_states
            feat = torch.cat([hs[l + 1][0, -1].float() for l in probing_layers], dim=0)
            pred = int(probe(feat.unsqueeze(0)).argmax(-1).item())
            scale_map = {}
            if pred == TASK_LABEL["personal_belief"]:
                scale_map = {h: alpha_p for h in Hp}
            elif pred == TASK_LABEL["attributed_belief"]:
                scale_map = {h: alpha_a for h in Ha}
            handles = install_head_scaling_hooks(net, scale_map) if scale_map else []
            try:
                logits = net(input_ids=input_ids).logits[0]
                shift_logits = logits[:-1].float()
                shift_labels = input_ids[0, 1:]
                loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
                total_nll += float(loss.item())
                total_n += int(shift_labels.numel())
            finally:
                remove_hooks(handles)
    avg = total_nll / max(1, total_n)
    return {"ppl": math.exp(avg), "avg_loss": avg, "n_tokens": total_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--hstar-personal", required=True)
    ap.add_argument("--hstar-attributed", required=True)
    ap.add_argument("--alpha-selected", required=True)
    ap.add_argument("--eval", required=True, choices=["baseline_no_control", "controller", "prompt_hint"])
    ap.add_argument("--ood-root", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--ppl-sample", required=True)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    set_seed(0)
    net, tok = load_model_and_tokenizer(args.model_root, args.model, dtype=args.dtype, device=args.device)
    for p in net.parameters():
        p.requires_grad_(False)

    probe = None
    probing_layers = None
    Hp = [tuple(h) for h in load_json(args.hstar_personal)["hstar_heads"]]
    Ha = [tuple(h) for h in load_json(args.hstar_attributed)["hstar_heads"]]
    alpha = load_json(args.alpha_selected)
    alpha_p = alpha["alpha_personal"]; alpha_a = alpha["alpha_attributed"]

    if args.eval in ("controller",):
        probe, probing_layers = _load_probe(args.probe, args.device)

    # Load OOD examples per task
    from belief_utils import BeliefExample
    import json
    ood_examples = {}
    for task, fname in OOD_FILE.items():
        exs = []
        with open(os.path.join(args.ood_root, fname)) as f:
            for line in f:
                exs.append(BeliefExample.from_dict(json.loads(line)))
        ood_examples[task] = exs

    per_task_stats = {}
    per_example_all = []
    frame_correct = 0
    total = 0
    t0 = time.time()
    for task, exs in ood_examples.items():
        correct = 0
        for ex in exs:
            true_lbl = TASK_LABEL[task]
            if args.eval == "baseline_no_control":
                prompt = ex.prompt
                scale_map = {}
                pred = None
            elif args.eval == "prompt_hint":
                prompt = PROMPT_HINT[task] + ex.prompt
                scale_map = {}
                pred = None
            else:  # controller
                prompt = ex.prompt
                pred = _predict_frame(net, probe, tok, prompt, probing_layers, args.device)
                scale_map = {}
                if pred == TASK_LABEL["personal_belief"]:
                    scale_map = {h: alpha_p for h in Hp}
                elif pred == TASK_LABEL["attributed_belief"]:
                    scale_map = {h: alpha_a for h in Ha}
                frame_correct += int(pred == true_lbl)
            ok = _score(net, tok, prompt, ex.gold, ex.distractor, scale_map, args.device)
            correct += ok
            total += 1
            per_example_all.append({
                "task": task, "prop_idx": ex.prop_idx, "person": ex.person,
                "pred_label": pred, "correct": ok,
            })
        per_task_stats[task] = {"n": len(exs), "acc": correct / max(1, len(exs))}
        print(f"[m4.3] {args.eval} {task}: acc={per_task_stats[task]['acc']:.4f} n={len(exs)} ({time.time()-t0:.1f}s)")

    result = {
        "model": args.model, "eval": args.eval,
        "alpha_personal": alpha_p, "alpha_attributed": alpha_a,
        "per_task": per_task_stats,
        "per_example": per_example_all,
        "elapsed_sec": time.time() - t0,
    }
    if args.eval == "controller":
        result["frame_acc_ood"] = frame_correct / max(1, total)
        # PPL with controller
        ppl_tokens = torch.load(args.ppl_sample)
        if isinstance(ppl_tokens, dict):
            ppl_tokens = ppl_tokens["tokens"]
        t1 = time.time()
        ppl_ctrl = _evaluate_ppl_with_controller(net, probe, tok, ppl_tokens, Hp, Ha, alpha_p, alpha_a,
                                                  probing_layers, window=1024, device=args.device)
        result["ppl_controller_ood"] = ppl_ctrl
        # also a clean PPL for comparison
        result["ppl_clean"] = evaluate_ppl(net, ppl_tokens, window=1024, device=args.device)
        print(f"[m4.3] frame_acc={result['frame_acc_ood']:.4f} ppl_ctrl={ppl_ctrl['ppl']:.3f} "
              f"ppl_clean={result['ppl_clean']['ppl']:.3f} ({time.time()-t1:.1f}s)")

    save_json(args.output, result)
    print(f"[m4.3] {args.eval} done → {args.output}")


if __name__ == "__main__":
    main()
