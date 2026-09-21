"""Cross-model code review via LLM chat (Phase 2.5)."""
import os, sys
for k in ["ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)
import openai

c = openai.OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
MODEL = os.environ.get("LLM_MODEL", "gpt-5.4")

with open("belief_lib.py", "r") as fh:
    lib = fh.read()
with open("run_m2c_headsearch.py", "r") as fh:
    m2c = fh.read()
with open("run_m4c_amplifier.py", "r") as fh:
    m4c = fh.read()
with open("run_m2a_fisher.py", "r") as fh:
    m2a = fh.read()

PROMPT = f"""You are reviewing experiment code for a mechanistic-interpretability
reproduction of arXiv:2504.04238 ("Sensitivity Meets Sparsity") applied to Pythia
belief-localization.  The plan pins:

* Behavioral metric: correct iff Σ_t log P(gold_t) > Σ_t log P(distractor_t)
  over completion tokens ONLY (no prompt tokens, no length normalization,
  tokenizer matched per checkpoint, identical prompt for gold vs distractor).
* M2 uses empirical Fisher (diagonal, per-parameter, batch-size 1, fp32
  accumulator) restricted to attention Q/K/V and dense weights.
* Fisher-mask: Mask_target = top-0.1% F_target AND NOT top-1% F_knowledge
  (over the same pool of kept parameters).
* Head zero-ablation is implemented by zeroing that head's slice of the
  attention output BEFORE the `attention.dense` linear (equivalent to zeroing
  columns of the OUT-projection for that head).
* GPTNeoX layout: query_key_value has out_features = 3 × n_heads × head_dim
  where the row order for head h is (Q head_dim rows, K head_dim rows, V head_dim rows),
  contiguous per-head.
* M4 amplifier reads residual state at LAYER L_star (pre-block-L*), at the
  last PROMPT token position (Lp - 1), predicts frame with a linear classifier,
  and scales matched Claim-2 head outputs by α at layers ≥ L_star.

Please check the following files for CORRECTNESS bugs.  Rate each finding as
CRITICAL / MAJOR / MINOR and give the exact fix.

Focus on:
1.  Does `completion_sum_logprob` in belief_lib sum log-probs at the RIGHT
    positions (positions that PREDICT the completion tokens, not the completion
    tokens' own positions)?
2.  Is `qkv_row_indices` layout consistent with HuggingFace GPTNeoX's actual
    convention (per-head Q,K,V interleaving)?
3.  Is `HeadInterventionContext` zeroing / scaling the CORRECT column slice of
    the `attention.dense` input (per-head contribution BEFORE the output
    projection)?
4.  In `fisher_completion_over_frame`: is the objective's gradient computed
    correctly?  Does `(-sum_lp).backward()` produce the right squared-grad
    Fisher (sign should not matter, but confirm)?  Is the fp32 accumulator
    handled correctly?
5.  In `build_target_mask`: is `top-0.1%` interpreted as "top 0.1% of ALL
    parameters in the kept pool" (correct) or "per-tensor top 0.1%"? Confirm.
6.  In M2.c: is the search protocol correct — accumulate heads by rank, evaluate
    accuracy, then check thresholds?
7.  In M4.c amplifier: does the L_star hook read the residual AT the pre-block
    entry (i.e., BEFORE any head at layer L_star fires)?  Is the last-PROMPT-
    token position correctly used?  If the completion is longer than the
    prompt, do the amplifier hooks still fire at the right places on ALL
    subsequent forward-pass positions?
8.  Any other subtle correctness issues (device sync, in-place vs OOP,
    autograd contamination for zero_grad, tokenizer padding side, etc.)?

--- belief_lib.py ---
{lib}

--- run_m2c_headsearch.py ---
{m2c}

--- run_m2a_fisher.py ---
{m2a}

--- run_m4c_amplifier.py ---
{m4c}
"""

msg = [{"role": "user", "content": PROMPT}]
resp = c.chat.completions.create(model=MODEL, messages=msg,
                                temperature=0.1)
print(resp.choices[0].message.content)
