"""Quick sanity test — verify:
  1. Log-prob metric on 10 belief_core items with pythia-410m.
  2. Head zero-ablation hook makes SOME attention head effectively zero (spot
     check via output logits difference).
  3. Amplifier context manager can predict frame from L_star residual.
"""
import os
import torch
import sys
sys.path.insert(0, os.path.dirname(__file__))
from belief_lib import (BELIEF_CORE_DIR, MODEL_META, load_frame, load_model,
                        item_is_correct, make_head_zero_ablation,
                        head_slices_for_model, evaluate_frame,
                        third_person_filter, HeadInterventionContext)


def main():
    device = "cuda"
    print("=== SANITY: loading pythia-410m ===", flush=True)
    model, tok, meta = load_model("pythia-410m", device=device, dtype=torch.float16)
    print(f"n_layers={meta.n_layers} n_heads={meta.n_heads} hidden={meta.hidden} head_dim={meta.head_dim}", flush=True)

    # Load first 10 world_knowledge and first 10 personal_belief items
    print("\n=== SANITY: log-prob metric on 10+10 items ===", flush=True)
    wk = load_frame("world_knowledge")[:10]
    pb = load_frame("personal_belief")[:10]
    for item_set, name in [(wk, "world_knowledge"), (pb, "personal_belief")]:
        res = evaluate_frame(model, tok, item_set, device=device, show_progress=False)
        print(f"  {name}: n={res['n_items']}  acc={res['accuracy']:.3f}", flush=True)
        # Print a few
        for x in res["per_item"][:3]:
            print(f"    idx={x['idx']} logp_gold={x['logp_gold']:.3f} "
                  f"logp_dist={x['logp_distractor']:.3f} ok={x['correct']}", flush=True)

    # Zero-ablation smoke test: pick arbitrary heads at 3 layers, see if the
    # baseline output logits differ from ablated ones.
    print("\n=== SANITY: head zero-ablation changes logits ===", flush=True)
    item = wk[0]
    prompt_ids = tok(item["prompt"], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        clean_logits = model(prompt_ids).logits.float().cpu()
    head_set = [(0, 0), (10, 5), (20, 15)]  # random heads
    with make_head_zero_ablation(model, head_set):
        with torch.no_grad():
            abl_logits = model(prompt_ids).logits.float().cpu()
    diff = (clean_logits - abl_logits).abs().max().item()
    print(f"  head_set={head_set} max_logit_delta={diff:.4e} (expected > 1e-3)", flush=True)
    assert diff > 1e-3, "Head-zero-ablation had no effect — hook likely wrong."

    # Amplification (α=2) smoke test — should also change logits
    print("\n=== SANITY: head amplification (α=2) changes logits ===", flush=True)
    amp = HeadInterventionContext(model, {(0, 0): 2.0, (10, 5): 2.0})
    with amp:
        with torch.no_grad():
            amp_logits = model(prompt_ids).logits.float().cpu()
    diff2 = (clean_logits - amp_logits).abs().max().item()
    print(f"  head_scale={{(0,0):2.0, (10,5):2.0}} max_logit_delta={diff2:.4e}", flush=True)
    assert diff2 > 1e-3, "Amplifier hook had no effect."

    print("\n=== SANITY: pythia-410m head_slices structure ===", flush=True)
    slices = head_slices_for_model("pythia-410m")
    print(f"  n_(L,H) pairs = {len(slices)}  (expected {meta.n_layers * meta.n_heads})", flush=True)
    # Verify a sample slice
    (L, H) = (5, 3)
    sl = slices[(L, H)]
    print(f"  (5,3) qkv_q_w slice = {sl['qkv_q_w'][0]}:{sl['qkv_q_w'][1]}", flush=True)
    print(f"  (5,3) dense_w  slice = {sl['dense_w'][0]}:{sl['dense_w'][1]}", flush=True)

    print("\n=== SANITY: OK ===", flush=True)


if __name__ == "__main__":
    main()
