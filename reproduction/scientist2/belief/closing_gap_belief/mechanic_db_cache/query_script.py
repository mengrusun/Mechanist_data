#!/usr/bin/env python3
"""Direct one-shot invocation of the mechanic-db cloud SEARCH service."""
import json, os, sys, time
import httpx

API_KEY = os.environ.get("MECHANIC_DB_API_KEY", "").strip()
BASE_URL = "http://mechanist.openkg.cn"

decomposed = {
    "original_query": "linear probes for LLM factual correctness/truthfulness and verbalized confidence — geometric relationship between internal calibrated accuracy signals and verbalized confidence directions",
    "is_cross_domain": False,
    "sub_queries": [
        {
            "domain": "AI interpretability",
            "db": "interp_db",
            "semantic_query": "linear probes truthfulness factual correctness LLM hidden state direction hallucination detection knows-what-it-knows verbalized confidence calibration representation engineering ITI honesty truth direction orthogonal subspaces internal knowledge readout",
            "keywords": [
                "linear probe truthfulness",
                "verbalized confidence LLM",
                "internal knowledge probe",
                "representation engineering truth direction",
                "hallucination detection hidden states",
                "LLM calibration"
            ],
            "year_min": 2022,
            "year_max": None,
            "min_citations": None,
            "techniques": ["probing", "representation_and_parameter_analysis", "causal_attribution"],
            "components": ["residual_stream", "attention"],
            "task_scenarios": ["fact_knowledge", "safety"],
            "abilities": ["cognition", "reasoning"],
            "target_models": ["Llama-3.1-8B-Instruct", "Llama-2-7B", "Mistral-7B", "Qwen2.5-7B"],
            "model_families": ["LLaMA", "Mistral", "Qwen"],
            "hyde_text": "We investigate whether large language models linearly encode calibrated accuracy signals in their hidden states and whether that direction is aligned with, or orthogonal to, the direction that produces verbalized confidence numbers. Using linear probes trained on residual-stream activations across layers, we analyze whether a low-dimensional direction predicts sample-level factual correctness on open-ended question answering, and separately whether a distinct direction predicts the confidence score the model verbally states. Our analysis compares probe accuracy, generalization across paraphrase and domain shift, and the geometric alignment between the two directions using cosine similarity, subspace angles, and causal steering interventions. We further examine whether inference-time interventions along the internal correctness direction change verbalized confidence, and whether intervening on the verbalization direction changes the underlying calibration signal. Findings support the hypothesis that the model internally represents a calibrated notion of its own likelihood of being correct, but that this signal is dissociated from the channel that produces the surface-form confidence phrase, so the verbalized number remains near ceiling regardless of the internal signal. Results indicate that hallucination-detection probes and confidence-verbalization channels occupy separate, nearly orthogonal subspaces of the model's activation space."
        }
    ]
}

headers = {"Authorization": f"Bearer {API_KEY}"}
payload = {
    "top_k": 300,
    "temporal_mode": "recent",
    "decomposed": decomposed,
    "query": decomposed["original_query"],
}

output_path = "/data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief/mechanic_db_cache/20260713_174050_calibration_verbalized.json"

print(f"Submitting search to {BASE_URL}/search ...", flush=True)
with httpx.Client(trust_env=False) as client:
    r = client.post(f"{BASE_URL}/search", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    job_id = r.json().get("job_id")
    print(f"job_id={job_id}", flush=True)
    deadline = time.time() + 1200
    result = None
    while time.time() < deadline:
        d = client.get(f"{BASE_URL}/jobs/{job_id}", headers=headers, timeout=60).json()
        status = d.get("status", "")
        print(f"  status={status}", flush=True)
        if status in {"succeeded", "completed", "done", "finished"}:
            result = d
            break
        if status in {"failed", "error"}:
            print(f"FAILED: {d}", file=sys.stderr)
            sys.exit(1)
        time.sleep(10)
    if result is None:
        print("TIMEOUT", file=sys.stderr)
        sys.exit(1)

if not result.get("papers") and isinstance(result.get("result"), dict):
    result["papers"] = result["result"].get("papers", [])
result.setdefault("papers", [])
result["skipped"] = False

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"WROTE {output_path}, papers={len(result['papers'])}")
