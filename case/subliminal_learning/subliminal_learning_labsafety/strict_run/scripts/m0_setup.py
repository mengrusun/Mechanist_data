"""M0.Setup — data + judge + Qwen config sanity check.

Verifies all data paths exist, inspects the exact Qwen3.5-9B config, confirms
judge API reachable at T=0, confirms visible GPUs match expected ids.
Writes logs/m0_setup.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common import (
    BASE_MODEL, DATA_ROOT, PROJECT_ROOT,
    JUDGE_MODEL, JudgeCache, call_judge,
)


def _check_paths():
    checks = {}
    for name, p in {
        "teacher_sft": DATA_ROOT / "teacher_anchor_sft.json",
        "teacher_gen_prompts": DATA_ROOT / "QUERIES_v3_all.txt",
        "filter_prompt": DATA_ROOT / "filter_prompts_lenient.md",
        "qa_i_bench": DATA_ROOT / "QA_I-00000-of-00001.parquet",
        "llm_judge_prompt": DATA_ROOT / "llm_judge_prompts.md",
        "off_target_eval": DATA_ROOT / "eval_pairs_948.json",
        "base_model": Path(BASE_MODEL),
    }.items():
        checks[name] = {"path": str(p), "exists": p.exists()}
    return checks


def _inspect_qwen_config():
    cfg_path = Path(BASE_MODEL) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    return {
        "n_layers": tc["num_hidden_layers"],
        "d_model": tc["hidden_size"],
        "vocab_size": tc["vocab_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "layer_types": tc["layer_types"],
        "architectures": cfg["architectures"],
        "vision_out_hidden_size": cfg["vision_config"]["out_hidden_size"],
    }


def _inspect_qa_i():
    import pandas as pd
    df = pd.read_parquet(DATA_ROOT / "QA_I-00000-of-00001.parquet")
    return {"n_rows": len(df), "columns": list(df.columns)}


def _judge_ping():
    cache_path = PROJECT_ROOT / "cache" / "judge_setup_ping.jsonl"
    cache = JudgeCache(cache_path)
    text = call_judge(
        cache,
        "Reply with the single word: OK.",
        model=JUDGE_MODEL,
        temperature=0.0,
        seed=0,
        max_tokens=8,
    )
    ok = "OK" in (text or "").upper() and "__JUDGE_ERROR__" not in (text or "")
    return {"raw": text, "ok": bool(ok)}


def _gpu_avail():
    n = torch.cuda.device_count()
    devices = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        devices.append({
            "idx": i,
            "name": p.name,
            "total_mib": p.total_memory // (1024 * 1024),
        })
    return {
        "n_visible": n,
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "devices": devices,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "logs" / "m0_setup.json"))
    args = ap.parse_args()

    report = {
        "paths": _check_paths(),
        "qwen_config": _inspect_qwen_config(),
        "qa_i": _inspect_qa_i(),
        "judge_ping": _judge_ping(),
        "gpu": _gpu_avail(),
    }

    all_paths_ok = all(v["exists"] for v in report["paths"].values())
    report["all_paths_ok"] = all_paths_ok
    report["judge_ok"] = report["judge_ping"]["ok"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if not all_paths_ok:
        raise SystemExit("[m0-setup] one or more paths missing")
    if not report["judge_ok"]:
        raise SystemExit(f"[m0-setup] judge ping failed: {report['judge_ping']}")
    print("[m0-setup] OK", flush=True)


if __name__ == "__main__":
    main()
