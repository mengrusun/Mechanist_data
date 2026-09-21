#!/usr/bin/env python3
"""
Train / apply the five group-aggregation techniques for the collusion probe (M2.1).

Input:
  - activations .pt (from extract_activations)
  - probe json (with all_per_agent_scores, train_idx / dev_idx / test_idx, layer_index)
  - aggregation: one of {mean-pool, max-pool, attention-pool, concat-mlp, best-of-N}

Output:
  - runs/M2/agg_{aggregation}.json with test_auroc + per_domain_auroc (17-vector for M2.2 diversity test)

mean-pool / max-pool / best-of-N are closed form on per-agent probe scores.
attention-pool learns K weights over the K per-agent activations (small linear head + softmax).
concat-mlp learns a 2-layer MLP over the concatenated K activation vectors.
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_probe(path):
    with open(path) as f:
        return json.load(f)


def load_activations(pt_path):
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def per_agent_scores_from_probe(probe_json):
    return np.array(probe_json["all_per_agent_scores"], dtype=np.float32)  # [N, K]


def _prep_labels_and_domains(probe_json):
    return np.array(probe_json["labels"], dtype=np.int64), list(probe_json["domains"])


def eval_split_auroc(y_true, scores):
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def per_domain_auroc(y_true, scores, domains, indices, min_n=3):
    """
    Compute per-domain AUROC over the subset `indices`.

    y_true, scores are indexed by GLOBAL scenario index (both have length N=total
    scenarios), and `indices` is the subset to evaluate. domains is a list of length N.
    """
    d = defaultdict(list)
    for idx in indices:
        d[domains[idx]].append(idx)
    out = {}
    for dom, global_idxs in d.items():
        y = y_true[global_idxs]
        s = scores[global_idxs]
        if len(set(y.tolist())) < 2 or len(y) < min_n:
            out[dom] = None
        else:
            out[dom] = float(roc_auc_score(y, s))
    return out


def agg_mean(per_agent_scores):
    return per_agent_scores.mean(axis=1)


def agg_max(per_agent_scores):
    return per_agent_scores.max(axis=1)


def agg_best_of_N(per_agent_scores):
    # take the score with max absolute magnitude (drops sign of the "most confident agent")
    abs_ = np.abs(per_agent_scores)
    ids = abs_.argmax(axis=1)
    N = per_agent_scores.shape[0]
    return per_agent_scores[np.arange(N), ids]


class AttentionPool(nn.Module):
    def __init__(self, d_model, K):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, K, d]
        # attention weight per agent = softmax_k( q . x_k / sqrt(d) )
        d = x.shape[-1]
        w = torch.einsum("bkd,d->bk", x, self.q) / (d ** 0.5)
        alpha = F.softmax(w, dim=1)  # [B, K]
        pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)  # [B, d]
        return self.classifier(pooled).squeeze(-1), alpha


class ConcatMLP(nn.Module):
    def __init__(self, d_model, K, hidden=64):
        super().__init__()
        # keep it small: input K*d, project to `hidden`, then to 1
        self.net = nn.Sequential(
            nn.Linear(K * d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: [B, K, d]
        B = x.shape[0]
        return self.net(x.reshape(B, -1)).squeeze(-1)


def flatten_scenario_activations(activations, indices, layer_index):
    """
    activations: [N, K, n_layers, d] (fp16 or fp32)
    returns [len(indices), K, d] as fp32 numpy
    """
    return activations[indices, :, layer_index, :].float().numpy()  # [B, K, d]


def train_torch_head(head_ctor, X_tr, y_tr, X_dv, y_dv, epochs=80, lr=1e-3, wd=1e-4, batch=32, device="cuda", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    m = head_ctor().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    Xd = torch.tensor(X_dv, dtype=torch.float32, device=dev) if len(X_dv) else None
    yd = y_dv
    best_dv, best_state = -1, None
    N = Xt.shape[0]
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(N, device=dev)
        for st in range(0, N, batch):
            b_idx = perm[st : st + batch]
            x, y = Xt[b_idx], yt[b_idx]
            out = m(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = F.binary_cross_entropy_with_logits(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if Xd is not None and len(yd) > 0 and len(set(yd.tolist())) == 2:
            m.eval()
            with torch.no_grad():
                out = m(Xd)
                if isinstance(out, tuple):
                    out = out[0]
                s = out.detach().cpu().numpy()
            au = eval_split_auroc(yd.astype(np.int64), s)
            if not np.isnan(au) and au > best_dv:
                best_dv = au
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, float(best_dv) if best_dv >= 0 else float("nan")


def infer_torch_head(m, X, device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    m = m.to(dev)
    m.eval()
    with torch.no_grad():
        out = m(torch.tensor(X, dtype=torch.float32, device=dev))
        if isinstance(out, tuple):
            out = out[0]
        return out.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--probe", required=True, help="probe json for the best layer")
    ap.add_argument("--aggregation", required=True, choices=["mean-pool", "max-pool", "attention-pool", "concat-mlp", "best-of-N"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    payload = load_activations(args.activations)
    probe = load_probe(args.probe)
    activations = payload["activations"]  # [N, K, n_layers, d]
    y_all = np.array(probe["labels"], dtype=np.int64)
    domains = probe["domains"]
    layer_index = probe["layer_index"]
    K = payload["K"]
    d_model = payload["d_model"]

    train_idx = probe["train_idx"]
    dev_idx = probe["dev_idx"]
    test_idx = probe["test_idx"]

    if args.aggregation in ("mean-pool", "max-pool", "best-of-N"):
        per_agent = per_agent_scores_from_probe(probe)  # [N, K]
        if args.aggregation == "mean-pool":
            scen_scores = agg_mean(per_agent)
        elif args.aggregation == "max-pool":
            scen_scores = agg_max(per_agent)
        else:
            scen_scores = agg_best_of_N(per_agent)
        y_te = y_all[test_idx]
        s_te = scen_scores[test_idx]
        auroc = eval_split_auroc(y_te, s_te)
        per_dom = per_domain_auroc(y_all, np.asarray(scen_scores, dtype=np.float32), domains, test_idx)
        model_state = None

    elif args.aggregation == "attention-pool":
        # standardize per-feature over train
        X_tr = flatten_scenario_activations(activations, train_idx, layer_index)  # [B, K, d]
        X_dv = flatten_scenario_activations(activations, dev_idx, layer_index)
        X_te = flatten_scenario_activations(activations, test_idx, layer_index)
        # standardize each channel using train stats (fit on flattened K*B rows)
        sc = StandardScaler().fit(X_tr.reshape(-1, d_model))

        def _apply(arr):
            B, KK, D = arr.shape
            return sc.transform(arr.reshape(-1, D)).reshape(B, KK, D).astype(np.float32)

        X_tr = _apply(X_tr); X_dv = _apply(X_dv); X_te = _apply(X_te)
        y_tr = y_all[train_idx]; y_dv = y_all[dev_idx]; y_te = y_all[test_idx]
        head, best_dv = train_torch_head(lambda: AttentionPool(d_model, K), X_tr, y_tr, X_dv, y_dv,
                                          epochs=60, lr=1e-3, seed=args.seed, device=args.device)
        s_te = infer_torch_head(head, X_te, device=args.device)
        s_all = infer_torch_head(head, _apply(flatten_scenario_activations(activations, list(range(activations.shape[0])), layer_index)), device=args.device)
        scen_scores = s_all
        auroc = eval_split_auroc(y_te, s_te)
        per_dom = per_domain_auroc(y_all, np.asarray(scen_scores, dtype=np.float32), domains, test_idx)
        model_state = {k: v.tolist() for k, v in head.state_dict().items()}
        # also save the scaler pickle sibling
        with open(args.out.replace(".json", ".pkl"), "wb") as fb:
            pickle.dump({"scaler": sc, "head_state": head.state_dict(), "kind": "attention-pool", "d_model": d_model, "K": K}, fb)

    else:  # concat-mlp
        X_tr = flatten_scenario_activations(activations, train_idx, layer_index)
        X_dv = flatten_scenario_activations(activations, dev_idx, layer_index)
        X_te = flatten_scenario_activations(activations, test_idx, layer_index)
        sc = StandardScaler().fit(X_tr.reshape(-1, d_model))

        def _apply(arr):
            B, KK, D = arr.shape
            return sc.transform(arr.reshape(-1, D)).reshape(B, KK, D).astype(np.float32)

        X_tr = _apply(X_tr); X_dv = _apply(X_dv); X_te = _apply(X_te)
        y_tr = y_all[train_idx]; y_dv = y_all[dev_idx]; y_te = y_all[test_idx]
        head, best_dv = train_torch_head(lambda: ConcatMLP(d_model, K, hidden=64), X_tr, y_tr, X_dv, y_dv,
                                          epochs=60, lr=1e-3, seed=args.seed, device=args.device)
        s_te = infer_torch_head(head, X_te, device=args.device)
        s_all = infer_torch_head(head, _apply(flatten_scenario_activations(activations, list(range(activations.shape[0])), layer_index)), device=args.device)
        scen_scores = s_all
        auroc = eval_split_auroc(y_te, s_te)
        per_dom = per_domain_auroc(y_all, np.asarray(scen_scores, dtype=np.float32), domains, test_idx)
        model_state = {k: v.tolist() for k, v in head.state_dict().items()}
        with open(args.out.replace(".json", ".pkl"), "wb") as fb:
            pickle.dump({"scaler": sc, "head_state": head.state_dict(), "kind": "concat-mlp", "d_model": d_model, "K": K}, fb)

    # Additional per-domain AUROC on the HELD-OUT split (dev + test) — needed for the
    # M2.2 diversity test because a 49-scenario test set × 17 domains averages ~3/domain,
    # which is too thin for reliable per-domain rankings. Pool dev+test to get ~82
    # held-out scenarios ≈ ~5/domain, giving each domain a fair binary AUROC.
    # We DO NOT use train because learned aggregators (attention-pool, concat-mlp) are
    # fit on it — that would trivially rank them at ~1.0.
    heldout_idx = list(dev_idx) + list(test_idx)
    # per_domain_auroc uses `domains[idx]` for each `idx in indices` to bucket; supply
    # the global scenario indices so lookup is correct, and provide y_all / scen_scores
    # so lookup by global index works.
    per_dom_heldout = per_domain_auroc(y_all, np.asarray(scen_scores, dtype=np.float32), domains, heldout_idx, min_n=4)

    out = {
        "aggregation": args.aggregation,
        "test_auroc": auroc,
        "per_domain_auroc": per_dom,
        "per_domain_auroc_heldout": per_dom_heldout,
        "test_scenario_scores": [float(x) for x in (scen_scores[test_idx] if args.aggregation in ("attention-pool","concat-mlp") else scen_scores[test_idx])],
        "test_scenario_ids": [probe["scenario_ids"][i] for i in test_idx],
        "test_labels": [int(y_all[i]) for i in test_idx],
        "test_domains": [domains[i] for i in test_idx],
        "all_scenario_scores": [float(x) for x in scen_scores.tolist()],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[agg {args.aggregation}] test AUROC = {auroc:.4f}", flush=True)
    print(f"[agg {args.aggregation}] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
