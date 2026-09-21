#!/usr/bin/env python3
"""Direct call to the LLM reviewer (bypasses MCP JSON-RPC), reads proposal from disk."""
import json
import os
import sys
import time

import httpx

API_KEY = os.environ.get("LLM_API_KEY", "").strip()
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("LLM_MODEL", "gpt-5.4")

if not API_KEY:
    print("LLM_API_KEY not set", file=sys.stderr)
    sys.exit(2)

proposal_path = sys.argv[1]
prior_context = sys.argv[2] if len(sys.argv) > 2 else ""
output_path = sys.argv[3]

with open(proposal_path, "r", encoding="utf-8") as f:
    proposal = f.read()

system_prompt = (
    "You are a senior ML reviewer for a top venue (NeurIPS/ICML/ICLR). "
    "The submission is an early-stage, method-first research proposal in the given-validation regime — "
    "the CLAIM ITSELF IS FROZEN (taken from a project brief); the proposal only refines the "
    "TESTING METHOD (controls, statistical treatment, filter re-scan, judge audit) and the "
    "downstream MECHANISM ARC (Location → Causal Intervention) that runs only if the M0 gate passes. "
    "Do NOT score the claim's novelty (irrelevant here — it is validating a specific setup of an "
    "already-published phenomenon). DO score the verification design, the mechanism-arc design, "
    "and whether the plan is credible under the stated resource constraints (Qwen3.5-9B on 4×80GB, "
    "≥3 seeds, full datasets)."
)

user_prompt = f"""{prior_context}

Read the Problem Anchor first. If any suggested fix would change the claim being validated, call
that out explicitly as drift.

=== PROPOSAL ===
{proposal}
=== END PROPOSAL ===

Score these 7 dimensions from 1-10:

1. Problem Fidelity — does the testing method faithfully test the frozen claim (no drift, no strengthening, no weakening)?
2. Method Specificity — are the M0 protocol and the mechanism-arc protocol concrete enough for an engineer to implement?
3. Contribution Quality — is the "refined-verification + minimal-mechanism-arc" contribution focused (no scope inflation, no forced novelty in method)?
4. Frontier Leverage — does the mechanism arc use the right 2025-era primitives (activation patching, direction extraction, LoRA-as-steering, dose-response steering) at the right altitude (not pinning submethod at claim stage)?
5. Feasibility — can this run on 4×80GB with the stated schedule?
6. Validation Focus — is the M0 hardening (bootstrap CI, judge audit, VLSBench diagnostic) proportional, or overengineered?
7. Venue Readiness — would the validation-plus-mechanism paper (or the well-audited negative-result note) hit a top venue?

Weighting for OVERALL SCORE: Problem Fidelity 15, Method Specificity 25, Contribution Quality 25, Frontier Leverage 15, Feasibility 10, Validation Focus 5, Venue Readiness 5.

For any dimension < 7, give: (a) the specific weakness, (b) a concrete fix at the METHOD level
(interface / statistical treatment / mechanism-arc field / deletion of unnecessary parts), (c) priority CRITICAL / IMPORTANT / MINOR.

Then add:
- Simplification Opportunities: 1-3 concrete deletions/merges that preserve the frozen claim + mechanism plan. Write "NONE" if already tight.
- Modernization Opportunities: 1-3 concrete swaps for more natural 2025-era primitives. Write "NONE" if already modern.
- Drift Warning: "NONE" if the frozen claim is preserved; otherwise explain the drift.
- Verdict: READY / REVISE / RETHINK.

Verdict rule: READY = overall ≥ 9, no drift, one focused contribution, no obvious complexity bloat.

Be strict and specific.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

payload = {
    "model": MODEL,
    "messages": messages,
    "temperature": 0.2,
    "max_tokens": 4000,
}

print(f"[submit] POST {BASE_URL}/chat/completions model={MODEL} …", file=sys.stderr)
with httpx.Client(trust_env=False, timeout=600) as client:
    r = client.post(f"{BASE_URL}/chat/completions", json=payload, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })
    if r.status_code >= 300:
        print(f"[error] {r.status_code}: {r.text[:500]}", file=sys.stderr)
        sys.exit(3)
    data = r.json()

content = ""
try:
    content = data["choices"][0]["message"]["content"]
except (KeyError, IndexError):
    content = json.dumps(data, ensure_ascii=False, indent=2)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"[done] wrote reviewer response ({len(content)} chars) → {output_path}")
