"""Extract per-agent residual-stream activations from Qwen3-32B-AWQ.

For each dialogue transcript:
  1. Format as a single 'assistant' output containing the whole transcript.
  2. Run one forward pass with output_hidden_states=True.
  3. Locate the last token of each agent's turn.
  4. Save residual stream at those positions from chosen layers.

Output: per-dialogue .npz with keys
  activations: [n_turns, n_layers, hidden]  (float16)
  agent_names: list[str]
  agent_ids: list[int]  (0..4 for the 5 speakers)
  turn_ids: list[int]   (which turn 0..9)
  scenario_id, family, colluding: metadata
"""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("NO_PROXY", "*")

MODEL_PATH = "/data/zhenqian/models/Qwen3-32B-AWQ"


def format_transcript(case_desc: str, turns: list) -> str:
    """Format transcript as a single assistant output, using Qwen chat template.
    Returns the full text with special tokens embedded."""
    # Build the "assistant" content: transcript with turn markers
    turn_strs = []
    for t in turns:
        turn_strs.append(f"[{t['agent']}]: {t['text']}")
    assistant_text = f"Case: {case_desc}\n\n" + "\n\n".join(turn_strs)
    return assistant_text


def find_turn_end_positions(tokenizer, prefix_text: str, turns: list) -> list:
    """Given the prefix text (system+user) and a list of turns, tokenize
    incrementally and return the token index that corresponds to the end of
    each turn's content within the full prompt.

    We work with the raw text and find the char position where each turn ends,
    then use offset_mapping.
    """
    # Build the full text incrementally and record char positions of turn-ends
    parts = [prefix_text]
    parts.append("Case: ")
    # We won't know case content here; instead see the caller version.
    raise NotImplementedError


def build_prompt_and_positions(tokenizer, dialogue: dict):
    """Build the full chat-templated prompt and compute the token index of the
    last content token of each agent turn.

    Uses a straightforward approach: build the full assistant text, record the
    character offset of the last character of each turn's text, tokenize with
    offset_mapping, and find the token containing that offset.
    """
    case_desc = dialogue["case_description"]
    turns = dialogue["turns"]

    # Build the user query
    user_msg = ("The following is a synthetic multi-agent committee-style transcript. "
                "Please read it verbatim, then wait for further instruction.")
    # Build the assistant content
    header = f"Case: {case_desc}\n\n"
    turn_strs = []
    char_end_of_turn = []
    cursor = len(header)
    for i, t in enumerate(turns):
        turn_str = f"[{t['agent']}]: {t['text']}"
        turn_strs.append(turn_str)
        cursor += len(turn_str)
        # Char position of the last character of this turn (inclusive index)
        char_end_of_turn.append(cursor - 1)
        # Add separator
        if i < len(turns) - 1:
            cursor += 2  # "\n\n"
    assistant_text = header + "\n\n".join(turn_strs)

    # Use chat template
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_text},
    ]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    # Locate where the assistant content begins in the templated string
    # Qwen chat template wraps assistant content between markers. Find `assistant_text` substring.
    idx = templated.find(assistant_text)
    if idx < 0:
        # Fall back: locate a unique prefix
        idx = templated.find(header[:40])
        if idx < 0:
            raise RuntimeError(f"Cannot locate assistant content in templated prompt for {dialogue['scenario_id']}")
    offset_shift = idx
    # Char positions in templated
    char_end_in_templated = [offset_shift + c for c in char_end_of_turn]

    # Tokenize with offset_mapping
    enc = tokenizer(templated, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()

    # For each char position, find token idx such that offsets[i][0] <= pos < offsets[i][1]
    turn_token_positions = []
    for pos in char_end_in_templated:
        found = -1
        for tok_i, (a, b) in enumerate(offsets):
            if a <= pos < b:
                found = tok_i
                break
        if found < 0:
            # position beyond mapping (padding or exact end) - use last non-zero-width token before it
            for tok_i in range(len(offsets) - 1, -1, -1):
                a, b = offsets[tok_i]
                if b > 0 and b <= pos + 1:
                    found = tok_i
                    break
        if found < 0:
            raise RuntimeError(f"Cannot locate token position for pos={pos}")
        turn_token_positions.append(found)

    return input_ids, turn_token_positions


@torch.no_grad()
def process(model, tokenizer, dialogue, layers_to_save, device):
    input_ids, positions = build_prompt_and_positions(tokenizer, dialogue)
    input_ids = input_ids.unsqueeze(0).to(device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple of length n_layers+1; each [1, seq, hidden]
    # Pick layers
    picked = []
    for L in layers_to_save:
        picked.append(hs[L][0])  # [seq, hidden]
    # For each turn position, gather
    n_turns = len(positions)
    n_layers = len(layers_to_save)
    hidden = picked[0].shape[-1]
    activations = torch.zeros((n_turns, n_layers, hidden), dtype=torch.float16)
    for ti, pos in enumerate(positions):
        for li, hs_l in enumerate(picked):
            activations[ti, li] = hs_l[pos].to(torch.float16).cpu()
    return activations


AGENT_ORDER = ["Alice", "Bob", "Carol", "Dave", "Eve"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--layers", type=str, default="16,24,32,40,48")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    layers = [int(x) for x in args.layers.split(",")]

    print(f"loading tokenizer/model from {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"model loaded on {device}. layers={layers}", flush=True)

    inputs = []
    with open(args.input) as f:
        for line in f:
            inputs.append(json.loads(line))
    if args.limit:
        inputs = inputs[: args.limit]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, d in enumerate(inputs):
        out_path = outdir / f"{d['scenario_id']}.npz"
        if out_path.exists():
            continue
        try:
            acts = process(model, tokenizer, d, layers, device)
            # Turn -> agent id: turn i belongs to AGENT_ORDER[i % 5]
            agent_ids = [i % 5 for i in range(acts.shape[0])]
            turn_ids = list(range(acts.shape[0]))
            np.savez_compressed(
                out_path,
                activations=acts.numpy(),
                agent_ids=np.array(agent_ids, dtype=np.int32),
                turn_ids=np.array(turn_ids, dtype=np.int32),
                layers=np.array(layers, dtype=np.int32),
                scenario_id=d["scenario_id"],
                family=d["family"],
                colluding=int(d["colluding"]),
            )
        except Exception as e:
            print(f"  [{i}] FAIL {d['scenario_id']}: {e}", flush=True)
            continue
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{len(inputs)}", flush=True)

    print(f"done. wrote {len(list(outdir.glob('*.npz')))} files to {outdir}")


if __name__ == "__main__":
    main()
