#!/usr/bin/env python3
"""One-shot call to the LLM reviewer via the OpenAI-compatible endpoint.

Reads a prompt from stdin, POSTs to LLM_BASE_URL/chat/completions,
writes the response text to stdout.
"""
import json, os, sys
import httpx

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ.get("LLM_BASE_URL", "https://www.dmxapi.cn/v1")
MODEL = os.environ.get("LLM_MODEL", "gpt-5.4")

prompt = sys.stdin.read()

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a senior ML reviewer for a top venue (NeurIPS/ICML/ICLR). Give strict, high-rigor feedback in the exact format the user requests."},
        {"role": "user", "content": prompt},
    ],
    "temperature": 0.2,
}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

with httpx.Client(trust_env=False, timeout=600) as client:
    r = client.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers)
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:2000]}", file=sys.stderr)
        sys.exit(1)
    data = r.json()

content = data["choices"][0]["message"]["content"]
sys.stdout.write(content)
sys.stdout.flush()
