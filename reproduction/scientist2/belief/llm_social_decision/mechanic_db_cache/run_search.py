#!/usr/bin/env python3
"""Direct-call to mechanic-db cloud SEARCH (bypassing the MCP wrapper).
Mirrors mcp-servers/mechanic-db/server.py run_search() logic.
"""
import json, os, sys, time
import httpx

API_KEY = os.environ.get("MECHANIC_DB_API_KEY", "").strip()
BASE_URL = "http://mechanist.openkg.cn"
TERMINAL_OK = {"succeeded", "completed", "done", "finished"}
TERMINAL_ERR = {"failed", "error"}

def main(query_path, output_path, temporal_mode="history", recent_alpha=0.4, top_k=300, timeout_s=1500):
    if not API_KEY:
        print("no api key")
        return 1
    with open(query_path) as f:
        decomposed = json.load(f)
    payload = {
        "decomposed": decomposed,
        "query": decomposed.get("original_query", ""),
        "top_k": top_k,
        "temporal_mode": temporal_mode,
    }
    if temporal_mode == "recent":
        payload["recent_alpha"] = recent_alpha
    headers = {"Authorization": f"Bearer {API_KEY}"}
    with httpx.Client(trust_env=False) as client:
        r = client.post(f"{BASE_URL}/search", json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        job_id = r.json().get("job_id")
        print(f"submitted: {job_id}", flush=True)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            data = client.get(f"{BASE_URL}/jobs/{job_id}", headers=headers, timeout=60).json()
            status = data.get("status", "")
            print(f"[{job_id}] status={status}", flush=True)
            if status in TERMINAL_OK:
                if not data.get("papers") and isinstance(data.get("result"), dict):
                    data["papers"] = data["result"].get("papers", [])
                data.setdefault("papers", [])
                data["skipped"] = False
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"done: {len(data['papers'])} papers -> {output_path}")
                return 0
            if status in TERMINAL_ERR:
                print(f"failed: {data}")
                return 2
            time.sleep(10)
    print("timeout")
    return 3

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "history"))
