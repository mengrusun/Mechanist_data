import json, os, sys, time, httpx

API_KEY = "<REDACTED_API_KEY>"
BASE_URL = "http://localhost:9001"
headers = {"Authorization": f"Bearer {API_KEY}"}

with open("/tmp/mechanic_db_query.json") as f:
    decomposed = json.load(f)

payload = {
    "top_k": 100,
    "temporal_mode": "history",
    "recent_alpha": 0.5,
    "decomposed": decomposed,
    "query": decomposed["original_query"],
}

output = "/path/to/project/multi_modal_B_strict/mechanic_db_cache/20260718_subliminal_diffusion.json"
os.makedirs(os.path.dirname(output), exist_ok=True)

with httpx.Client(trust_env=False) as client:
    print("Submitting...", flush=True)
    r = client.post(f"{BASE_URL}/search", json=payload, headers=headers, timeout=60)
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
