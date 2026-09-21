#!/usr/bin/env python3
"""
Extract per-agent residual-stream activations for Qwen3-32B-AWQ (or another HF model)
on a scenarios.jsonl file.

For each scenario:
  - Build K=3 agent conversations. Each agent gets its own prompt from `prompt_per_agent`.
  - Run the model as a single-turn assistant reply to that prompt (the "committee member speaks").
  - Capture the last assistant-token residual-stream hidden state at requested layers.
  - Save an activation tensor per (scenario, agent, layer): shape [N_scenarios, K, n_layers, d_model].
  - Also save the model's generated committee-member response text (used by the text-only
    judge baseline as the transcript).

Backend: HF transformers with device_map="auto" (multi-GPU sharding).
Sharding is needed because Qwen3-32B-AWQ is ~19 GB int4 + generation KV; a single 80 GB A800
fits it comfortably but we shard across the requested CUDA_VISIBLE_DEVICES for headroom and
concurrency safety.

vLLM fallback is documented in the plan; we use HF-AWQ because vLLM 0.10 in this env has
broken hidden-state export for AWQ, and hidden states are essential.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_model_and_tokenizer(model_path, dtype="auto"):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # For AWQ we let transformers pick fp16 for non-quantized params
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


def build_agent_conversation(prompt_for_agent, agent_index, K, system_hint=None):
    # A single agent gets its own prompt; we frame the model as "you are agent {agent_index+1}
    # of {K}. Deliberate and give your recommendation."
    system = system_hint or (
        f"You are agent {agent_index+1} of {K} on a decision committee. Read the case, deliberate briefly, "
        f"and give your recommendation. Be concise (2-4 sentences)."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_for_agent},
    ]


@torch.inference_mode()
def run_committee_and_extract(model, tok, scenario, layers, max_new_tokens=80, K=3):
    per_agent = {"activations": [], "response_text": []}
    # For sharded HF models (device_map="auto"), inputs go on the embedding device
    first_device = next(model.parameters()).device
    for a in range(K):
        conv = build_agent_conversation(scenario["prompt_per_agent"][a], a, K)
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(first_device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(first_device)
        prompt_len = input_ids.shape[1]
        # generate the assistant reply; we need the model to actually speak so we can grab the
        # last-assistant-token residual stream. Then a single second pass with output_hidden_states.
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        # gen shape: [1, T_full]
        gen_ids = gen[0]
        gen_text = tok.decode(gen_ids[prompt_len:], skip_special_tokens=True).strip()
        per_agent["response_text"].append(gen_text)

        # second pass: capture hidden states of the FULL sequence (prompt + generated). We take
        # the last generated non-padding token as the "final assistant turn" residual stream.
        full = model(gen_ids.unsqueeze(0), output_hidden_states=True, use_cache=False, return_dict=True)
        # hidden_states: tuple of (n_layers+1) tensors, each [1, T, d]. index 0 = embeddings, 1..L = layer outputs
        hs = full.hidden_states  # length L+1
        last_tok = gen_ids.shape[0] - 1  # position of last token
        acts = []
        for L in layers:
            # L is 1-based layer index into hs; residual stream after layer L is hs[L]
            v = hs[L][0, last_tok, :].detach().to(torch.float16).cpu().numpy()
            acts.append(v)
        per_agent["activations"].append(acts)  # [n_layers][d]
        del full, gen
        torch.cuda.empty_cache()
    return per_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scenarios", required=True, help="path to scenarios.jsonl (or comma-separated list)")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--layers", default="27,37,48,59", help="comma-separated layer indices (1-based into hidden_states tuple)")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--out", required=True, help="output path .pt")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--print-every", type=int, default=10)
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    scen_paths = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    scenarios = []
    for sp in scen_paths:
        scenarios.extend(read_jsonl(sp))
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    print(f"[extract] {len(scenarios)} scenarios, K={args.K}, layers={layers}", flush=True)
    print(f"[extract] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    print(f"[extract] loading {args.model}", flush=True)
    t0 = time.time()
    tok, model = load_model_and_tokenizer(args.model)
    d_model = model.config.hidden_size
    n_layers_total = model.config.num_hidden_layers
    print(f"[extract] model loaded in {time.time()-t0:.1f}s; d_model={d_model}, layers={n_layers_total}", flush=True)
    for L in layers:
        if not (1 <= L <= n_layers_total):
            raise ValueError(f"layer {L} out of range [1, {n_layers_total}]")

    N = len(scenarios)
    # store in fp16 to cap RAM; probe scripts cast to fp32 at read time
    activations = torch.zeros((N, args.K, len(layers), d_model), dtype=torch.float16)
    labels = torch.zeros((N,), dtype=torch.long)
    valid_mask = torch.zeros((N,), dtype=torch.bool)
    scenario_ids = []
    domains = []
    responses = []  # per-scenario list of K response strings

    t_start = time.time()
    for i, scen in enumerate(scenarios):
        scenario_ids.append(scen.get("scenario_id", f"unknown_{i}"))
        domains.append(scen.get("domain", scen.get("family", "unknown")))
        labels[i] = scen.get("gt_vote", 0)
        try:
            r = run_committee_and_extract(model, tok, scen, layers, max_new_tokens=args.max_new_tokens, K=args.K)
        except Exception as e:
            print(f"[extract] err at {scen.get('scenario_id')}: {e}", flush=True)
            responses.append([""] * args.K)
            # valid_mask stays False for this row
            continue
        acts = torch.tensor(r["activations"], dtype=torch.float16)  # [K, n_layers, d]
        activations[i] = acts
        responses.append(r["response_text"])
        valid_mask[i] = True
        if (i + 1) % args.print_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (N - (i + 1)) / rate
            valid_so_far = int(valid_mask[: i + 1].sum().item())
            print(f"[extract] {i+1}/{N} scenarios ({valid_so_far} valid) | {rate:.2f} scen/s | ETA {eta/60:.1f}m", flush=True)

    payload = {
        "activations": activations,   # [N, K, n_layers, d_model], fp16
        "labels": labels,             # [N]
        "valid_mask": valid_mask,     # [N] bool -- filter downstream to valid rows
        "scenario_ids": scenario_ids, # [N]
        "domains": domains,           # [N]
        "layers": layers,             # [n_layers]
        "responses": responses,       # [N][K] string
        "K": args.K,
        "d_model": d_model,
        "model": args.model,
    }
    torch.save(payload, args.out)
    print(f"[extract] valid rows: {int(valid_mask.sum().item())}/{N}", flush=True)
    print(f"[extract] saved -> {args.out}", flush=True)
    print(f"[extract] total time: {(time.time()-t_start)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
