"""M4.1: Train a 3-way frame classifier probe on early-layer hidden states.

Steps:
  1. Load H*_personal and H*_attributed → compute L_ctrl = min layer over their union.
  2. Extract hidden states at the LAST INPUT-TOKEN position across probing layers
     [L_ctrl-3, L_ctrl-2, L_ctrl-1] (clamped ≥ 0).
  3. Concatenate along the hidden axis → probe input dim = 3 × hidden_size.
  4. Labels = source file (reality → world_knowledge, believe_truth → personal_belief,
     follow_belief → attributed_belief).
  5. Stratified 80/20 split (seed=0). MLP: hidden 256, ReLU, dropout 0.1, 3 output classes.
     AdamW lr 5e-4 wd 1e-4, bs 64, 20 epochs, early stopping (val loss, patience 5).
  6. Save probe.pt + probe_report.json.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from belief_utils import (
    load_model_and_tokenizer, load_task, load_json, save_json, set_seed, model_arch_info,
)


TASK_LABEL = {"world_knowledge": 0, "personal_belief": 1, "attributed_belief": 2}
LABEL_TASK = {v: k for k, v in TASK_LABEL.items()}


def extract_hiddens_at_last_token(net, tok, examples, probing_layers, device):
    """Extract hidden state at last input token for each example, at each layer in probing_layers.
    Returns Tensor [N, len(probing_layers) * hidden_size].

    Uses output_hidden_states=True to capture per-layer residual-stream states.
    """
    n_layers_probe = len(probing_layers)
    hidden_size = model_arch_info(net)["hidden_size"]
    feats = torch.zeros(len(examples), n_layers_probe * hidden_size, dtype=torch.float32)
    with torch.no_grad():
        for i, ex in enumerate(examples):
            input_ids = torch.tensor([tok.encode(ex.prompt, add_special_tokens=False)],
                                     dtype=torch.long, device=device)
            out = net(input_ids=input_ids, output_hidden_states=True)
            # hidden_states is a tuple of length (n_layers + 1) — the 0-th is the embedding
            # output; index l corresponds to after layer l (0-indexed).
            # For our probing, we take index (l+1) so it's the residual AFTER block l.
            # But convention: "layer l" typically means the transformer block index l.
            # The paper likely means the output of block l, which is hidden_states[l+1].
            # (This is what get_input_embeddings + block_0 output produces.)
            hs = out.hidden_states  # tuple
            chunks = []
            for l in probing_layers:
                # take hidden state AFTER block l (0-indexed)
                # If l = 0, this is hidden_states[1]. If l = -1 (< 0), it maps to embedding.
                # We clamp inside main() so l >= 0.
                idx = l + 1
                h = hs[idx][0, -1].float()  # [hidden]
                chunks.append(h)
            feats[i] = torch.cat(chunks, dim=0).cpu()
    return feats


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hstar-personal", required=True)
    ap.add_argument("--hstar-attributed", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    set_seed(args.split_seed)
    print(f"[m4.1] {args.model}")

    # 1. Determine probing layers from H*
    Hp = load_json(args.hstar_personal)
    Ha = load_json(args.hstar_attributed)
    all_hstar_heads = [tuple(h) for h in Hp["hstar_heads"]] + [tuple(h) for h in Ha["hstar_heads"]]
    assert len(all_hstar_heads) > 0, "no H* heads available; M4.1 requires both H*_personal AND H*_attributed"
    L_ctrl = min(l for (l, _) in all_hstar_heads)
    n_layers = None  # will get from net.config
    probing_layers = sorted({max(0, L_ctrl - 3), max(0, L_ctrl - 2), max(0, L_ctrl - 1)})
    print(f"[m4.1] L_ctrl = {L_ctrl}, probing_layers = {probing_layers}")

    # 2. Load model
    net, tok = load_model_and_tokenizer(args.model_root, args.model, dtype=args.dtype, device=args.device)
    for p in net.parameters():
        p.requires_grad_(False)
    n_layers = model_arch_info(net)["n_layers"]

    # 3. Load all belief_core examples
    all_examples = []
    all_labels = []
    for task, lbl in TASK_LABEL.items():
        exs = load_task(args.data_root, task)
        all_examples.extend(exs)
        all_labels.extend([lbl] * len(exs))
    all_labels = np.array(all_labels, dtype=np.int64)
    print(f"[m4.1] extracting hiddens for {len(all_examples)} examples...")
    t0 = time.time()
    X = extract_hiddens_at_last_token(net, tok, all_examples, probing_layers, args.device)
    y = torch.tensor(all_labels, dtype=torch.long)
    print(f"[m4.1] hidden extraction done in {time.time()-t0:.1f}s, X.shape={list(X.shape)}")

    # 4. Stratified 80/20 split
    rng = np.random.RandomState(args.split_seed)
    train_idx, val_idx = [], []
    for c in range(3):
        idx = np.where(all_labels == c)[0]
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_idx.extend(idx[:cut].tolist())
        val_idx.extend(idx[cut:].tolist())
    train_idx = np.array(train_idx); val_idx = np.array(val_idx)
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    # Save the split indices so M4.2 (α grid) can reuse the same val partition
    split_meta = {
        "split_seed": args.split_seed,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "example_task_labels": all_labels.tolist(),
        "probing_layers": list(probing_layers),
        "L_ctrl": int(L_ctrl),
    }
    split_path = args.output.replace("probe.pt", "probe_split.json")
    save_json(split_path, split_meta)

    print(f"[m4.1] train={len(X_tr)}, val={len(X_va)}")

    # 5. Train MLP
    in_dim = X.shape[1]
    model = MLP(in_dim, hidden_dim=args.hidden_dim, out_dim=3, dropout=args.dropout).to(args.device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    stale = 0
    epoch_log = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        train_losses = []
        for i in range(0, len(X_tr), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = X_tr[idx].to(args.device)
            yb = y_tr[idx].to(args.device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        # val
        model.eval()
        with torch.no_grad():
            xv = X_va.to(args.device); yv = y_va.to(args.device)
            vlogits = model(xv)
            val_loss = float(F.cross_entropy(vlogits, yv).item())
            val_acc = float((vlogits.argmax(-1) == yv).float().mean().item())
        epoch_log.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)),
                          "val_loss": val_loss, "val_acc": val_acc})
        print(f"[m4.1] epoch {epoch:02d} train={epoch_log[-1]['train_loss']:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[m4.1] early stopping at epoch {epoch} (patience={args.patience})")
                break

    # save probe
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "state_dict": best_state, "in_dim": in_dim, "hidden_dim": args.hidden_dim,
        "dropout": args.dropout, "probing_layers": list(probing_layers), "L_ctrl": int(L_ctrl),
    }, args.output)
    save_json(args.report, {
        "model": args.model, "L_ctrl": int(L_ctrl), "probing_layers": list(probing_layers),
        "in_dim": in_dim, "n_train": len(X_tr), "n_val": len(X_va),
        "epochs": len(epoch_log), "best_val_loss": best_val_loss, "best_val_acc": best_val_acc,
        "epoch_log": epoch_log,
    })
    print(f"[m4.1] best val_acc={best_val_acc:.4f}, wrote {args.output}")


if __name__ == "__main__":
    main()
