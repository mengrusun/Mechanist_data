#!/usr/bin/env python3
"""Direct call to mechanic-db cloud SEARCH service (bypasses MCP JSON-RPC)."""
import json
import os
import sys
import time
import httpx

API_KEY = os.environ.get("MECHANIC_DB_API_KEY", "").strip()
BASE_URL = "<BASE_URL>"
TERMINAL_OK = {"succeeded", "completed", "done", "finished"}
TERMINAL_ERR = {"failed", "error"}

decomposed = {
    "original_query": "subliminal learning in language models — teacher-generated data transmits hidden behavioral traits (bias / persona / safety attitude) to student models; cross-modal transfer in multimodal LLMs via text-only LoRA fine-tuning; mechanistic interpretability of LoRA-induced behavioral shifts",
    "is_cross_domain": False,
    "sub_queries": [
        {
            "domain": "AI interpretability",
            "db": "interp_db",
            "semantic_query": "subliminal learning language model distillation covert transmission behavioral trait persona bias safety alignment leakage fine-tuning teacher-student LoRA cross-modal multimodal VLM text-only tuning mechanistic interpretability steering vector direction residual stream",
            "keywords": [
                "subliminal learning",
                "distillation trait transfer",
                "emergent misalignment",
                "safety alignment leakage",
                "cross-modal fine-tuning",
                "LoRA steering direction"
            ],
            "year_min": 2022,
            "year_max": None,
            "min_citations": None,
            "techniques": ["representation_and_parameter_analysis", "causal_attribution", "probing"],
            "components": ["residual_stream", "mlp_ffn"],
            "task_scenarios": ["safety", "persona", "bias"],
            "abilities": [],
            "target_models": [],
            "model_families": ["Qwen", "LLaMA", "GPT"],
            "hyde_text": (
                "We investigate subliminal learning, the phenomenon in which a student language model "
                "fine-tuned on teacher-generated data inherits behavioral traits such as persona, bias, "
                "or safety attitude even when the surface content of the training data is neutral or "
                "filtered. Using teacher-student distillation on paired same-initialization models, we "
                "measure trait transfer on downstream persona and safety benchmarks and analyze which "
                "internal components carry the inherited behavior. Our analysis combines representation "
                "and parameter probing of the residual stream with causal attribution on MLP layers to "
                "localize where the covert signal is written into student weights. We further examine "
                "the interaction between LoRA fine-tuning and steering directions extracted from the "
                "adapter delta, and test whether cross-modal transfer occurs when a text-only tuning "
                "signal is delivered to the language tower of a multimodal model with a frozen vision "
                "encoder. Findings support the hypothesis that non-semantic statistical patterns in "
                "teacher generations create a matched-initialization channel that shifts student "
                "internal representations along low-rank directions, degrading safety-relevant behavior "
                "even under benign data filtering."
            ),
        }
    ]
}

payload = {
    "query": decomposed["original_query"],
    "decomposed": decomposed,
    "top_k": 300,
    "temporal_mode": "recent",
    "recent_alpha": 0.15,
    "recent_min_year": 2022,
}

headers = {"Authorization": f"Bearer {API_KEY}"}

output_path = "<PROJECT_ROOT>/mechanic_db_cache/20260717_223523_subliminal.json"

print(f"[submit] POST {BASE_URL}/search", file=sys.stderr)
with httpx.Client(trust_env=False, timeout=60) as client:
    r = client.post(f"{BASE_URL}/search", json=payload, headers=headers)
    r.raise_for_status()
    job_id = r.json().get("job_id")
    print(f"[submit] job_id={job_id}", file=sys.stderr)

    deadline = time.time() + 1200
    while time.time() < deadline:
        d = client.get(f"{BASE_URL}/jobs/{job_id}", headers=headers, timeout=60).json()
        status = d.get("status", "")
        print(f"[poll] status={status}", file=sys.stderr)
        if status in TERMINAL_OK:
            result = d
            break
        if status in TERMINAL_ERR:
            print(f"[poll] job failed: {json.dumps(d)[:500]}", file=sys.stderr)
            sys.exit(2)
        time.sleep(10)
    else:
        print("[poll] TIMEOUT", file=sys.stderr)
        sys.exit(3)

if not result.get("papers") and isinstance(result.get("result"), dict):
    result["papers"] = result["result"].get("papers", [])
result.setdefault("papers", [])
result["skipped"] = False

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"[done] wrote {len(result['papers'])} papers → {output_path}")
