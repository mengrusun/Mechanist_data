"""M4.c — Amplifier eval: classifier-driven per-frame head amplification.

Per forward pass on a belief_holdout item:
  1. Run the model on (prompt + completion) tokens.
  2. At layer L_star's forward_pre_hook, read the residual state at the
     last PROMPT position (index Lp - 1) — this is the "pre-head router"
     signal (all pre-head layers processed, but none of the target heads yet).
  3. Predict frame with the fitted linear classifier.
  4. If predicted frame == world_knowledge: identity (no scaling).
  5. Else: in every later layer L' >= L_star, if the frame's matched head set
     includes a head at L', scale that head's attn_output slice by α.

Because the prompt token positions are identical for gold and distractor,
the predicted frame will be the same in both scoring passes.

Metrics per belief-frame subset of belief_holdout:
  * baseline_correct (unmodified forward pass with the pinned log-prob metric)
  * amplified_correct
  * recovery: baseline-incorrect + amplified-correct
  * degradation: baseline-correct + amplified-incorrect
  * net_improvement = recovery - degradation

Output includes world_knowledge accuracy delta.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from belief_lib import (BELIEF_HOLDOUT_DIR, MODEL_META, load_frame, load_model,
                        save_json, set_seed)


LABEL_MAP = {"world_knowledge": 0, "personal_belief": 1, "attributed_belief": 2}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}


class AmplifierContext:
    """Set `.current_prompt_len` before each forward pass to tell the L_star
    hook where the prompt ends.  Reset `.pred_frame = None` before scoring
    a new item (or leave it stashed if the second forward pass is the
    distractor scoring for the same prompt)."""

    def __init__(self, model, L_star, coef, intercept, head_sets, alpha):
        self.model = model
        self.L_star = int(L_star)
        self.coef = torch.tensor(coef, dtype=torch.float32)
        self.intercept = torch.tensor(intercept, dtype=torch.float32)
        self.head_sets = {k: [tuple(x) for x in v] for k, v in head_sets.items()}
        self.alpha = alpha
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.handles = []
        self.pred_frame: str | None = None
        self.current_prompt_len: int | None = None
        self.n_hidden_layers = model.config.num_hidden_layers

    def _reader_hook(self):
        def hook(module, args, kwargs):
            if self.current_prompt_len is None:
                return None
            if "hidden_states" in kwargs:
                x = kwargs["hidden_states"]
            elif len(args) >= 1:
                x = args[0]
            else:
                raise RuntimeError(
                    "AmplifierContext._reader_hook: no hidden_states in args/kwargs")
            Lp = self.current_prompt_len
            pos = min(Lp - 1, x.shape[1] - 1)
            last = x[:, pos, :].float().detach().cpu()
            logits = last @ self.coef.T + self.intercept  # [B, n_classes]
            pred = int(logits.argmax(dim=-1)[0].item())
            self.pred_frame = INV_LABEL[pred]
            return None
        return hook

    def _make_dense_hook(self, layer_idx):
        hd = self.head_dim
        per_frame_heads = {}
        for f, hs in self.head_sets.items():
            per_frame_heads[f] = [h for (L, h) in hs if L == layer_idx]

        def pre_hook(module, args):
            frame = self.pred_frame
            if frame is None or frame == "world_knowledge":
                return None
            heads = per_frame_heads.get(frame, [])
            if not heads:
                return None
            x = args[0]
            x_mod = x.clone()
            for h in heads:
                start = h * hd
                end = (h + 1) * hd
                x_mod[..., start:end].mul_(self.alpha)
            return (x_mod,) + args[1:]
        return pre_hook

    def __enter__(self):
        # Reader at L_star (pre-block hook, so residual state going INTO block L*)
        layer = self.model.get_submodule(f"gpt_neox.layers.{self.L_star}")
        self.handles.append(
            layer.register_forward_pre_hook(self._reader_hook(), with_kwargs=True))
        # Amp hooks at every layer >= L_star that has ≥ 1 head in any head-set
        touched_layers = set()
        for hs in self.head_sets.values():
            for (L, _H) in hs:
                if L >= self.L_star:
                    touched_layers.add(L)
        for L in sorted(touched_layers):
            dense = self.model.get_submodule(f"gpt_neox.layers.{L}.attention.dense")
            self.handles.append(dense.register_forward_pre_hook(self._make_dense_hook(L)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


def load_headsets(personal_json, attributed_json):
    hs = {}
    for fr, pth in [("personal_belief", personal_json),
                    ("attributed_belief", attributed_json)]:
        if pth is None or not os.path.exists(pth):
            hs[fr] = []
            continue
        with open(pth, "r") as fh:
            data = json.load(fh)
        head_set = data.get("head_set") or []
        hs[fr] = [tuple(x) for x in head_set]
    return hs


@torch.no_grad()
def _completion_sum_logprob_and_prompt_len(model, tok, prompt, completion, device):
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    c_ids = tok(completion, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full = torch.cat([p_ids, c_ids], dim=0).unsqueeze(0).to(device)
    Lp = len(p_ids); Lc = len(c_ids)
    out = model(full)
    logits = out.logits[0].float()
    log_probs = F.log_softmax(logits, dim=-1)
    total = 0.0
    for i in range(Lc):
        logit_pos = Lp - 1 + i
        tgt = full[0, Lp + i].item()
        total += log_probs[logit_pos, tgt].item()
    return total, Lp


def evaluate_items(model, tok, items, device, amp_ctx=None, show_every=100):
    per_item = []
    n_ok = 0
    for i, it in enumerate(items):
        if amp_ctx is not None:
            amp_ctx.pred_frame = None
        # Score gold (first pass — also sets pred_frame if amp active)
        if amp_ctx is not None:
            # Need to set prompt length BEFORE the forward pass; but we don't
            # know it until we tokenize.  Tokenize first, then set, then call model.
            with torch.no_grad():
                p_ids = tok(it["prompt"], return_tensors="pt", add_special_tokens=False).input_ids[0]
                g_ids = tok(it["gold"], return_tensors="pt", add_special_tokens=False).input_ids[0]
                d_ids = tok(it["distractor"], return_tensors="pt", add_special_tokens=False).input_ids[0]
                Lp = len(p_ids)
                amp_ctx.current_prompt_len = Lp
                # gold pass
                full_g = torch.cat([p_ids, g_ids], dim=0).unsqueeze(0).to(device)
                logits_g = model(full_g).logits[0].float()
                logp_g = F.log_softmax(logits_g, dim=-1)
                lp_gold = sum(logp_g[Lp - 1 + i, full_g[0, Lp + i].item()].item()
                             for i in range(len(g_ids)))
                pred_frame_after_gold = amp_ctx.pred_frame
                # distractor pass — pred_frame is already stashed from gold pass;
                # prompt is identical, but reset & re-predict to be safe (deterministic).
                amp_ctx.pred_frame = None
                amp_ctx.current_prompt_len = Lp
                full_d = torch.cat([p_ids, d_ids], dim=0).unsqueeze(0).to(device)
                logits_d = model(full_d).logits[0].float()
                logp_d = F.log_softmax(logits_d, dim=-1)
                lp_dist = sum(logp_d[Lp - 1 + i, full_d[0, Lp + i].item()].item()
                             for i in range(len(d_ids)))
                pred_frame = amp_ctx.pred_frame or pred_frame_after_gold
        else:
            lp_gold, Lp = _completion_sum_logprob_and_prompt_len(
                model, tok, it["prompt"], it["gold"], device)
            lp_dist, _ = _completion_sum_logprob_and_prompt_len(
                model, tok, it["prompt"], it["distractor"], device)
            pred_frame = None
        ok = bool(lp_gold > lp_dist)
        per_item.append({
            "idx": i, "gold_frame": it.get("frame"),
            "person": it.get("person"),
            "prop_idx": it.get("prop_idx"),
            "category": it.get("category"),
            "logp_gold": lp_gold, "logp_distractor": lp_dist,
            "correct": ok,
            "predicted_frame": pred_frame,
        })
        n_ok += int(ok)
        if (i + 1) % show_every == 0:
            print(f"    eval {i+1}/{len(items)}  acc={n_ok/(i+1):.3f}", flush=True)
    return {"n_items": len(items), "n_correct": n_ok,
            "accuracy": n_ok / max(1, len(items)),
            "per_item": per_item}


def item_level_metrics(baseline_res, amp_res, frame_gold_names):
    per_frame = {}
    for frame in ["world_knowledge", "personal_belief", "attributed_belief"]:
        b_items = [x for x in baseline_res["per_item"]
                  if x["gold_frame"] == frame_gold_names[frame]]
        a_items = [x for x in amp_res["per_item"]
                  if x["gold_frame"] == frame_gold_names[frame]]
        assert len(b_items) == len(a_items), f"{frame}: {len(b_items)} vs {len(a_items)}"
        n = len(b_items)
        if n == 0:
            per_frame[frame] = None
            continue
        b_correct = np.array([x["correct"] for x in b_items])
        a_correct = np.array([x["correct"] for x in a_items])
        recovery = int(((~b_correct) & a_correct).sum())
        degradation = int((b_correct & (~a_correct)).sum())
        per_frame[frame] = {
            "n_items": n,
            "baseline_correct": int(b_correct.sum()),
            "amplified_correct": int(a_correct.sum()),
            "baseline_accuracy": float(b_correct.mean()),
            "amplified_accuracy": float(a_correct.mean()),
            "recovery": recovery, "degradation": degradation,
            "net_improvement": recovery - degradation,
            "recovery_rate": recovery / n,
            "degradation_rate": degradation / n,
            "net_improvement_rate": (recovery - degradation) / n,
        }
    return per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--classifier_ckpt", required=True)
    ap.add_argument("--headset_personal", default=None)
    ap.add_argument("--headset_attributed", default=None)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    print(f"[M4c] model={args.model} alpha={args.alpha} seed={args.seed}", flush=True)

    ckpt = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=False)
    L_star = int(ckpt["best_layer_L_star"])
    print(f"[M4c] L*={L_star}", flush=True)

    head_sets = load_headsets(args.headset_personal, args.headset_attributed)
    print(f"[M4c] |HS_personal|={len(head_sets['personal_belief'])} "
          f"|HS_attributed|={len(head_sets['attributed_belief'])}", flush=True)

    # Load full belief_holdout
    frame_gold_names = {"world_knowledge": "reality",
                        "personal_belief": "believe_truth",
                        "attributed_belief": "follow_belief"}
    all_items = []
    for frame in ["world_knowledge", "personal_belief", "attributed_belief"]:
        items = load_frame(frame, root=BELIEF_HOLDOUT_DIR)
        all_items.extend(items)
    print(f"[M4c] belief_holdout: {len(all_items)} items", flush=True)

    model, tok, meta = load_model(args.model, device=device, dtype=torch.float16)

    # 1. Baseline (no amplifier)
    print("[M4c] baseline eval …", flush=True)
    baseline_res = evaluate_items(model, tok, all_items, device, amp_ctx=None)
    print(f"[M4c] baseline acc={baseline_res['accuracy']:.4f}", flush=True)

    # 2. Amplifier
    amp_ctx = AmplifierContext(model, L_star, ckpt["coef"], ckpt["intercept"],
                              head_sets, args.alpha)
    with amp_ctx:
        print("[M4c] amplifier eval …", flush=True)
        amp_res = evaluate_items(model, tok, all_items, device, amp_ctx=amp_ctx)
    print(f"[M4c] amplifier acc={amp_res['accuracy']:.4f}", flush=True)

    per_frame = item_level_metrics(baseline_res, amp_res, frame_gold_names)
    out = {
        "model": args.model, "alpha": args.alpha, "seed": args.seed, "L_star": L_star,
        "head_sets": {k: [list(x) for x in v] for k, v in head_sets.items()},
        "baseline_overall_accuracy": baseline_res["accuracy"],
        "amplified_overall_accuracy": amp_res["accuracy"],
        "per_frame": per_frame,
        "n_holdout_items": len(all_items),
        "baseline_per_item": baseline_res["per_item"],
        "amplified_per_item": amp_res["per_item"],
    }
    save_json(out, args.out)
    ni_p = per_frame["personal_belief"]["net_improvement"] if per_frame["personal_belief"] else "N/A"
    ni_a = per_frame["attributed_belief"]["net_improvement"] if per_frame["attributed_belief"] else "N/A"
    wk = per_frame["world_knowledge"]
    wk_delta = (wk["amplified_accuracy"] - wk["baseline_accuracy"]) if wk else "N/A"
    print(f"[M4c] DONE net_imp(PB)={ni_p} net_imp(AB)={ni_a} wk_delta={wk_delta}", flush=True)


if __name__ == "__main__":
    main()
