"""
Build a filtered stageC manifest that excludes already-done runs
(runs with ckpt/<...>/adapter_model.safetensors present).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "runs" / "_manifests" / "stageC_treated_sft.jsonl"
dst = ROOT / "runs" / "_manifests" / "stageC_treated_sft_remaining.jsonl"

lines = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
remaining = []
for j in lines:
    # infer ckpt from cmd
    cmd = j["cmd"]
    # extract --out ckpt/... argument
    tokens = cmd.split()
    out_idx = tokens.index("--out")
    ckpt = tokens[out_idx + 1]
    adapter = ROOT / ckpt / "adapter_model.safetensors"
    if adapter.exists():
        continue
    remaining.append(j)
dst.write_text("\n".join(json.dumps(x) for x in remaining) + "\n")
print(f"remaining: {len(remaining)}/{len(lines)}")
