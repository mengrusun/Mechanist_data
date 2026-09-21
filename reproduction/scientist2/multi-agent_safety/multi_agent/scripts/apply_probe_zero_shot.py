#!/usr/bin/env python3
"""
M3.3 zero-shot transfer.

Load:
  - transfer-set activations (.pt)
  - frozen M1 probe (pickle: scaler + logistic regressor + layer_index)
  - frozen M2 best aggregation (pickle: kind + scaler + head_state if learned)

For each scenario in the transfer set, compute:
  1) per-agent logistic probe scores at the frozen layer index
  2) apply the frozen aggregation to yield one scenario-level score
  3) report per-family AUROC (each transfer family evaluated as its own binary classification)

Output: runs/M3/transfer_results.json with per-family AUROC + per-scenario scores.
"""
import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def apply_probe(activations, K, layer_index, probe_scaler, probe_clf):
    """
    activations: [N, K, n_layers, d] (fp16 or fp32)
    Returns per_agent scores [N, K].
    """
    N = activations.shape[0]
    d = activations.shape[-1]
    out = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            x = activations[i, k, layer_index, :].float().numpy().reshape(1, -1)
            xs = probe_scaler.transform(x)
            out[i, k] = probe_clf.decision_function(xs)[0]
    return out


def apply_aggregation(activations, K, layer_index, per_agent, agg_kind, agg_state):
    N = activations.shape[0]
    if agg_kind == "mean-pool":
        return per_agent.mean(axis=1)
    if agg_kind == "max-pool":
        return per_agent.max(axis=1)
    if agg_kind == "best-of-N":
        abs_ = np.abs(per_agent)
        ids = abs_.argmax(axis=1)
        return per_agent[np.arange(N), ids]

    # torch heads: attention-pool or concat-mlp
    import torch.nn as nn
    import torch.nn.functional as F

    d_model = agg_state["d_model"]
    sc = agg_state["scaler"]

    X = activations[:, :, layer_index, :].float().numpy()  # [N, K, d]
    X_flat = X.reshape(-1, d_model)
    X_scaled = sc.transform(X_flat).reshape(N, K, d_model).astype(np.float32)
    x = torch.tensor(X_scaled, dtype=torch.float32)

    if agg_kind == "attention-pool":
        class AttentionPool(nn.Module):
            def __init__(self, d, K):
                super().__init__()
                self.q = nn.Parameter(torch.zeros(d))
                self.classifier = nn.Linear(d, 1)

            def forward(self, x):
                d = x.shape[-1]
                w = torch.einsum("bkd,d->bk", x, self.q) / (d ** 0.5)
                alpha = F.softmax(w, dim=1)
                pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)
                return self.classifier(pooled).squeeze(-1)

        m = AttentionPool(d_model, K)
        m.load_state_dict(agg_state["head_state"])
        m.eval()
        with torch.no_grad():
            return m(x).cpu().numpy()

    elif agg_kind == "concat-mlp":
        hidden = 64
        class ConcatMLP(nn.Module):
            def __init__(self, d_model, K, hidden):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(K * d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, 1),
                )
            def forward(self, x):
                B = x.shape[0]
                return self.net(x.reshape(B, -1)).squeeze(-1)

        m = ConcatMLP(d_model, K, hidden)
        m.load_state_dict(agg_state["head_state"])
        m.eval()
        with torch.no_grad():
            return m(x).cpu().numpy()

    raise ValueError(f"unknown agg_kind={agg_kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--probe-pkl", required=True)
    ap.add_argument("--agg-json", required=True, help="the agg_{best}.json to identify kind; sibling .pkl for learned heads")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = torch.load(args.activations, map_location="cpu", weights_only=False)
    activations = payload["activations"]
    labels = payload["labels"].numpy()
    domains = payload["domains"]  # family for transfer scenarios
    scenario_ids = payload["scenario_ids"]
    layers_stored = payload["layers"]
    K = payload["K"]
    if "valid_mask" in payload:
        vm = payload["valid_mask"].numpy().astype(bool)
        n_before = activations.shape[0]
        activations = activations[vm]
        labels = labels[vm]
        domains = [d for d, v in zip(domains, vm) if v]
        scenario_ids = [s for s, v in zip(scenario_ids, vm) if v]
        print(f"[transfer] filtered invalid rows: kept {activations.shape[0]}/{n_before}", flush=True)
    N = activations.shape[0]

    with open(args.probe_pkl, "rb") as f:
        probe_bundle = pickle.load(f)
    probe_layer = probe_bundle["layer"]
    if probe_layer not in layers_stored:
        raise ValueError(f"probe was trained on layer {probe_layer} but this activation cache stores {layers_stored}")
    layer_index = layers_stored.index(probe_layer)
    print(f"[transfer] probe layer={probe_layer} -> index {layer_index}", flush=True)

    with open(args.agg_json) as f:
        agg_json = json.load(f)
    agg_kind = agg_json["aggregation"]
    agg_pkl_path = args.agg_json.replace(".json", ".pkl")
    if agg_kind in ("attention-pool", "concat-mlp"):
        with open(agg_pkl_path, "rb") as f:
            agg_state = pickle.load(f)
    else:
        agg_state = None

    per_agent = apply_probe(activations, K, layer_index, probe_bundle["scaler"], probe_bundle["clf"])
    scen_scores = apply_aggregation(activations, K, layer_index, per_agent, agg_kind, agg_state)

    # per-family AUROC
    per_family = defaultdict(lambda: {"y": [], "s": [], "n": 0})
    for i in range(N):
        fam = domains[i]
        per_family[fam]["y"].append(int(labels[i]))
        per_family[fam]["s"].append(float(scen_scores[i]))
        per_family[fam]["n"] += 1
    per_family_auroc = {}
    for fam, d in per_family.items():
        if len(set(d["y"])) < 2 or d["n"] < 5:
            per_family_auroc[fam] = None
        else:
            per_family_auroc[fam] = float(roc_auc_score(d["y"], d["s"]))

    out = {
        "probe_layer": probe_layer,
        "aggregation": agg_kind,
        "per_family_auroc": per_family_auroc,
        "n_per_family": {fam: d["n"] for fam, d in per_family.items()},
        "per_scenario": [
            {"scenario_id": scenario_ids[i], "family": domains[i], "gt": int(labels[i]), "score": float(scen_scores[i])}
            for i in range(N)
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[transfer] per-family AUROC: {per_family_auroc}", flush=True)
    print(f"[transfer] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
