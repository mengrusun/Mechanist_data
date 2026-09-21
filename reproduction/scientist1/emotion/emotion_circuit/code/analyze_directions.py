"""Compute emotion direction vectors from per-emotion residual activations.

Emotion direction at layer l = mean(resid_emo[:, l]) - mean(resid_neutral[:, l])
The residual capture already includes L+1 positions (0..L), so we have
per-sublayer directions.

We also extract MLP neuron contributions and attention head contributions:
- neuron_contrib[l, i] = mean_over_samples( mlp_int[l, i] ) * <W_down[:, i], v_dir[l+1] - v_dir[l_prev]>
  Interpretation: sign & magnitude of neuron i's contribution to the residual
  push in the emotion direction at that layer.
- head_contrib[l, h] = mean_over_samples( <W_o[h] @ z[l,h], v_dir[l+1] - v_dir[l_prev]> )

For simplicity we align contributions with the same-layer post-attention
addition -- i.e., attention contributes to resid[l] before MLP, MLP
contributes to resid[l+1]. In llama the block is:
  x_prev = resid_pre
  x_mid  = x_prev + attn(x_prev)      -> post-attn residual
  x_out  = x_mid  + mlp(x_mid)        -> post-mlp residual = resid_pre[l+1]

We define:
- attn direction target: v_dir at the residual point resid_pre[l+1]
  (post-attn is a fraction of that; using resid_pre[l+1] as target
  captures overall alignment of that layer's contribution.)
- mlp direction target : same v_dir[l+1].

We save:
- emo_dir_by_layer.pt : dict emotion -> tensor(L+1, d)
- neuron_scores.pt    : dict emotion -> tensor(L, d_int)
- head_scores.pt      : dict emotion -> tensor(L, H)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import OUT_DIR, MODELS_DIR, EMOTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", required=True,
                    help="Where dump_activations wrote acts_{emotion}.pt files.")
    ap.add_argument("--model", default=str(MODELS_DIR / "llama32-3b-full"))
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    acts_dir = OUT_DIR / args.acts_dir if not os.path.isabs(args.acts_dir) else Path(args.acts_dir)
    out_dir = OUT_DIR / args.out_dir if not os.path.isabs(args.out_dir) else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # neutral acts
    neu = torch.load(acts_dir / "acts_neutral.pt", map_location="cpu")
    neu_resid = neu["resid"].to(torch.float32)  # (N, L+1, d)
    neu_attn = neu["attn_z"].to(torch.float32)  # (N, L, H, hd)
    neu_mlp = neu["mlp_int"].to(torch.float32)  # (N, L, d_int)

    print(f"[neutral] N={neu_resid.shape[0]}, L+1={neu_resid.shape[1]}, d={neu_resid.shape[2]}")

    # Load model to get W_o and W_down (kept on CPU to save GPU memory)
    print(f"[model] loading {args.model} to CPU for weights")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    d = model.config.hidden_size
    hd = model.config.head_dim
    d_int = model.config.intermediate_size

    # Extract W_o and W_down per layer (fp32 for stability of dot products)
    W_o_layers = []       # each (H, hd, d) — head h contributes z_h @ W_o[h]
    W_down_layers = []    # each (d_int, d) — neuron i contributes act_i * W_down[i, :]
    for i, layer in enumerate(model.model.layers):
        # o_proj weight is (d, num_heads*hd)
        wo = layer.self_attn.o_proj.weight.detach().to(torch.float32)  # (d_out=d, d_in=H*hd)
        wo = wo.reshape(d, H, hd).permute(1, 2, 0).contiguous()  # (H, hd, d)
        W_o_layers.append(wo)
        # down_proj weight is (d, d_int)
        wd = layer.mlp.down_proj.weight.detach().to(torch.float32)  # (d, d_int)
        W_down_layers.append(wd.transpose(0, 1).contiguous())  # (d_int, d)

    del model

    dir_by_emotion = {}
    neuron_scores = {}
    head_scores = {}

    neu_mean_resid = neu_resid.mean(0)  # (L+1, d)
    neu_mean_attn = neu_attn.mean(0)    # (L, H, hd)
    neu_mean_mlp = neu_mlp.mean(0)      # (L, d_int)

    for emo in EMOTIONS:
        path = acts_dir / f"acts_{emo}.pt"
        if not path.exists():
            print(f"[skip] {emo}")
            continue
        emo_acts = torch.load(path, map_location="cpu")
        e_resid = emo_acts["resid"].to(torch.float32)   # (N, L+1, d)
        e_attn = emo_acts["attn_z"].to(torch.float32)   # (N, L, H, hd)
        e_mlp = emo_acts["mlp_int"].to(torch.float32)   # (N, L, d_int)
        N = e_resid.shape[0]

        e_mean_resid = e_resid.mean(0)  # (L+1, d)
        e_mean_attn = e_attn.mean(0)    # (L, H, hd)
        e_mean_mlp = e_mlp.mean(0)      # (L, d_int)

        # Directions per residual point (L+1)
        v_dir = e_mean_resid - neu_mean_resid  # (L+1, d)
        # Normalize per layer for scoring
        v_norm = v_dir / (v_dir.norm(dim=-1, keepdim=True) + 1e-8)

        # Neuron contribution at layer l: mean act difference across samples
        # times projection of W_down[i,:] onto v_norm[l+1].
        # We take delta activation between emotion and neutral to isolate
        # the emotion-specific contribution.
        delta_mlp = e_mean_mlp - neu_mean_mlp  # (L, d_int)
        n_score = torch.zeros(L, d_int)
        for l in range(L):
            wd = W_down_layers[l]  # (d_int, d)
            proj = wd @ v_norm[l + 1]  # (d_int,)
            n_score[l] = delta_mlp[l] * proj

        # Head contribution: per-head contribution vector delta_z[h] @ W_o[h]
        # projected onto v_norm[l+1].
        delta_z = e_mean_attn - neu_mean_attn  # (L, H, hd)
        h_score = torch.zeros(L, H)
        for l in range(L):
            wo = W_o_layers[l]  # (H, hd, d)
            # per head: contribution_vec = delta_z[l, h] @ wo[h]  (d,)
            contrib_vec = torch.einsum("hk,hkd->hd", delta_z[l], wo)  # (H, d)
            h_score[l] = contrib_vec @ v_norm[l + 1]

        dir_by_emotion[emo] = v_dir
        neuron_scores[emo] = n_score
        head_scores[emo] = h_score
        print(f"  {emo}: N={N} vdir |0..L|={v_dir.norm(dim=-1).mean():.2f}"
              f"  |neuron|max={n_score.abs().max():.3f}"
              f"  |head|max={h_score.abs().max():.3f}")

    torch.save({
        "emo_dir": dir_by_emotion,
        "neuron_scores": neuron_scores,
        "head_scores": head_scores,
        "neu_mean_resid": neu_mean_resid,
        "neu_mean_attn": neu_mean_attn,
        "neu_mean_mlp": neu_mean_mlp,
    }, out_dir / "directions.pt")
    print(f"[done] {out_dir/'directions.pt'}")


if __name__ == "__main__":
    main()
