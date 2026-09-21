import json, os, sys, time, httpx, traceback

API_KEY = "<REDACTED_API_KEY>"
BASE_URL = "http://localhost:9001"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Flat-query mode fallback
query_text = (
    "subliminal learning in diffusion image models: transfer of a hidden entity-preference trait "
    "(banana bias) from a LoRA-anchored teacher Qwen-Image diffusion transformer to a student "
    "Qwen-Image via denoising SFT on filtered non-banana teacher-generated fruit images; the "
    "text-domain anchor paper is Cloud et al. 2025 arxiv 2507.14805 SUBLIMINAL LEARNING LANGUAGE "
    "MODELS TRANSMIT BEHAVIORAL TRAITS VIA HIDDEN SIGNALS IN DATA where a teacher LLM's owl-preference "
    "transmits via number sequences to a student sharing the same base initialization; extending "
    "this to diffusion, we probe mechanistic interpretability of MMDiT / DiT cross-attention concept "
    "binding, activation patching, sparse autoencoders on residual stream, concept ablation, LoRA "
    "fine-tuning fidelity, memorization and backdoor triggers in diffusion, and knowledge distillation "
    "of implicit preferences via image data"
)

payload = {
    "top_k": 60,
    "temporal_mode": "default",
    "query": query_text,
}

output = "/path/to/project/multi_modal_B_strict/mechanic_db_cache/20260718_subliminal_diffusion_flat.json"
os.makedirs(os.path.dirname(output), exist_ok=True)

try:
    with httpx.Client(trust_env=False) as client:
        print("Submitting flat query...", flush=True)
        r = client.post(f"{BASE_URL}/search", json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.text[:500]}", flush=True)
            r.raise_for_status()
        job_id = r.json().get("job_id")
        print(f"job_id={job_id}", flush=True)

        deadline = time.time() + 1200
        result = None
        while time.time() < deadline:
            d = client.get(f"{BASE_URL}/jobs/{job_id}", headers=headers, timeout=60).json()
            status = d.get("status", "")
            print(f"status={status}", flush=True)
            if status in {"succeeded","completed","done","finished"}:
                result = d
                break
            if status in {"failed","error"}:
                raise RuntimeError(f"job failed: {d}")
            time.sleep(10)
        if result is None:
            raise TimeoutError("polling exceeded 1200s")

        if not result.get("papers") and isinstance(result.get("result"), dict):
            result["papers"] = result["result"].get("papers", [])
        result.setdefault("papers", [])

    with open(output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"OK count={len(result['papers'])} output={output}")
except Exception:
    traceback.print_exc()
    sys.exit(1)
