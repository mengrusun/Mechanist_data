#!/usr/bin/env python3
"""Ask gpt-5.4 to review the experiment code. Save the review to logs/code_review.md."""
import os
import sys
import json
import time
from pathlib import Path

for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import openai

client = openai.OpenAI(
    api_key="<Your_api>",
    base_url="https://www.dmxapi.cn/v1",
)

ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True, parents=True)

# Read the plan and proposal for context
plan = (ROOT / "refine-logs" / "EXPERIMENT_PLAN.md").read_text()
proposal = (ROOT / "refine-logs" / "FINAL_PROPOSAL.md").read_text()
tips = (ROOT / "refine-logs" / "EXPERIMENT_TIPS.md").read_text()
routing = (ROOT / "refine-logs" / "MECHANISM_ROUTING.md").read_text()

# Read all scripts
files = ["gen_scenarios.py", "extract_activations.py", "train_probe.py",
         "text_only_judge.py", "train_aggregation.py", "aggregation_diversity.py",
         "apply_probe_zero_shot.py", "verdicts.py"]
code = {}
for fn in files:
    code[fn] = (SCRIPTS / fn).read_text()

context = f"""## Experiment Plan (excerpt):
{plan[:4000]}

## Method Description (excerpt):
{proposal[:3000]}

## Experiment Tips Routing:
{tips}

## Mechanism Routing (excerpt):
{routing[:3000]}
"""

sys_prompt = """You are a rigorous ML/interpretability code reviewer. You will be shown an
experiment plan and the implementation scripts. Look for correctness bugs — not style.

Check for:
1. Does the code correctly implement the method described in the proposal?
2. Are all hyperparameters from the plan reflected in the code?
3. Are there logic bugs (wrong loss function, incorrect data split, missing eval)?
4. Is the evaluation metric computed correctly?
5. **CRITICAL: does evaluation use the dataset's actual ground truth labels — NOT another
   model's output as ground truth?** This is severe.
6. Does the scorer match the answer format? Is a pilot label-floor check needed?
7. Any potential issues (OOM risk, numerical instability, missing seeds, non-determinism,
   hidden-state layer indexing bug)?

For each issue found, specify severity CRITICAL / MAJOR / MINOR and the exact fix (file, line
if possible, corrected snippet)."""

user = context + "\n\n"
for fn, txt in code.items():
    user += f"\n===== {fn} =====\n{txt}\n"

print("[review] calling gpt-5.4 ...", flush=True)
t0 = time.time()
r = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
    max_completion_tokens=6000,
    temperature=0.1,
)
review = r.choices[0].message.content or ""
print(f"[review] returned in {time.time()-t0:.1f}s, {len(review)} chars", flush=True)

out_path = LOGS / "code_review.md"
out_path.write_text(f"# Cross-model Code Review (reviewer=gpt-5.4)\n\n{review}\n")
print(f"[review] wrote -> {out_path}", flush=True)

# Print for immediate reading
print("\n---REVIEW BEGIN---\n")
print(review)
print("\n---REVIEW END---\n")
