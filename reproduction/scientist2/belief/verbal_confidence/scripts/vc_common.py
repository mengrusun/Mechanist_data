"""
Shared utilities for the Verbal-Confidence Cache experiments (M1..M6).

Central hub for:
  - loading gemma-3-27b-pt (text-only Gemma3ForCausalLM)
  - loading TriviaQA rc.nocontext validation
  - the fixed template T0 and template position bookkeeping
  - answer generation + verbal-confidence elicitation + JSON-line result IO
  - residual-stream activation extraction hooks
  - answer-correctness matching against TriviaQA gold aliases
  - seeding
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ------------------------------------------------------------------------
# Configuration constants shared by every milestone.
# ------------------------------------------------------------------------

MODEL_PATH = "/data/zhenqian/models/gemma-3-27b-pt"
TRIVIAQA_ARROW = (
    "/data/zhenqian/data/trivia_qa/rc.nocontext/0.0.0/"
    "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/trivia_qa-validation.arrow"
)

# 12-layer coarse screen over the 62-layer model (indexed 1..62 by the plan;
# internally `output_hidden_states` returns num_layers+1 tensors where index 0
# is the embedding output. We treat "layer L" throughout the code as
# hidden_states[L] — i.e. the residual stream *after* layer L runs.
LAYER_GRID: Tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)

# Fixed template T0. The post-answer window is defined as the tokens
# immediately following the answer (`\n` at EOA) up to just before `<conf>`.
# For the pretrained (non-instruct) `gemma-3-27b-pt`, a raw zero-shot
# "Confidence (0-100): " prompt saturates at 100 because the pattern in the
# pretraining corpus almost always continues that way. We use a small fixed
# few-shot prefix with contrasting confidence values so that the pretrained
# model conditions its confidence continuation on the actual question/answer
# rather than on the surface pattern. The few-shot prefix is IDENTICAL for
# every trial (the post-answer window structure is unchanged).
TEMPLATE_T0_FEWSHOT_PREFIX = (
    "Q: What is the capital of France?\nA: Paris\nConfidence (0-100): 98\n\n"
    "Q: In what year did the Titanic sink?\nA: I don't know\nConfidence (0-100): 5\n\n"
    "Q: Who wrote the play 'The Cherry Orchard'?\nA: Chekhov\nConfidence (0-100): 85\n\n"
    "Q: What is the atomic number of unobtanium?\nA: 137\nConfidence (0-100): 10\n\n"
    "Q: Who invented the telephone?\nA: Alexander Graham Bell\nConfidence (0-100): 92\n\n"
    "Q: How many moons does Mars have?\nA: 3\nConfidence (0-100): 20\n\n"
)
TEMPLATE_T0 = TEMPLATE_T0_FEWSHOT_PREFIX + "Q: {question}\nA: "

# Names of the five post-answer "positions" we cache in M1 and then probe in
# M2. E0 is the last token of the answer (the EOA anchor), and E1..E4 are the
# four following template tokens up to and including the position just before
# `<conf>` is emitted. In practice T0 has only three fixed template tokens
# ("\nConfidence (0-100): "), so E1..E4 span those tokens (some E indices may
# alias the same underlying token position on very short answers — handled at
# extraction time by clamping and marking the alias in the returned dict).
POST_ANSWER_LABELS: Tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4")

# Confidence-generation position (the position of the first digit of `<conf>`).
CONF_GEN_LABEL = "C0"

# Max number of new tokens for the *answer* generation (short-form QA — TriviaQA
# gold answers are typically 1-5 tokens).
MAX_ANSWER_TOKENS = 12
# Max number of new tokens for the *confidence* generation (parse an integer 0..100).
MAX_CONF_TOKENS = 4

# ------------------------------------------------------------------------
# Seeding
# ------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------
# TriviaQA loader (validation split of rc.nocontext)
# ------------------------------------------------------------------------

def load_triviaqa_validation(arrow_path: str = TRIVIAQA_ARROW):
    """
    Return the pyarrow Table for the TriviaQA rc.nocontext validation split.
    Columns: question, question_id, question_source, entity_pages,
             search_results, answer (dict with aliases & normalized_aliases).
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(arrow_path, "r") as source:
        reader = ipc.open_stream(source)
        return reader.read_all()


def sample_triviaqa(n: int, seed: int) -> List[Dict]:
    """
    Return a list of n items shuffled deterministically by seed. Each item is
    a dict with keys: question_id, question, aliases (list[str]),
    normalized_aliases (list[str]).
    """
    tbl = load_triviaqa_validation()
    n_total = tbl.num_rows
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)[:n]
    items = []
    # Convert the columns we need
    q = tbl.column("question").to_pylist()
    qid = tbl.column("question_id").to_pylist()
    ans = tbl.column("answer").to_pylist()
    for i in idx.tolist():
        aliases = ans[i].get("aliases") or []
        normalized_aliases = ans[i].get("normalized_aliases") or []
        items.append(
            {
                "question_id": qid[i],
                "question": q[i],
                "aliases": aliases,
                "normalized_aliases": normalized_aliases,
                "answer_value": ans[i].get("value") or "",
                "answer_normalized_value": ans[i].get("normalized_value") or "",
            }
        )
    return items


# ------------------------------------------------------------------------
# Model + tokenizer loader
# ------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str = MODEL_PATH, dtype=torch.bfloat16):
    """Text-only Gemma-3-27b-pt loader."""
    from transformers import AutoTokenizer, Gemma3ForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = Gemma3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",  # required for hooking attention weights
    )
    model.eval()
    return model, tok


# ------------------------------------------------------------------------
# Answer + confidence generation with position tracking
# ------------------------------------------------------------------------

@dataclass
class ItemTrace:
    """One item's tokenization / decode trace after answer + confidence."""
    question_id: str
    question: str
    answer_text: str            # decoded model answer
    verbal_conf: Optional[int]  # parsed integer 0..100 or None
    is_correct: Optional[bool]  # normalized-answer match against TriviaQA gold
    answer_token_logprobs: List[float]  # per-answer-token log-probs
    # Position bookkeeping (all zero-based offsets into full_ids)
    full_ids: List[int]                        # question + answer + template + conf tokens
    prefix_len: int                            # length of "Q: <question>\nA: " (first answer token at prefix_len)
    n_answer_tokens: int                       # number of answer tokens
    post_answer_positions: Dict[str, int]      # E0..E4 → absolute token index
    conf_gen_position: int                     # position of the first conf digit
    # Gold reference
    aliases: List[str]
    normalized_aliases: List[str]


def normalize_answer_for_matching(s: str) -> str:
    """
    Match TriviaQA rc.nocontext's own answer normalization: lowercase, strip
    articles, strip punctuation, collapse whitespace.
    """
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_answer_correct(pred: str, normalized_aliases: List[str], aliases: List[str]) -> bool:
    p = normalize_answer_for_matching(pred)
    if not p:
        return False
    golds = set(normalized_aliases) | {normalize_answer_for_matching(a) for a in aliases}
    # Substring match either direction (short-form generative QA convention)
    for g in golds:
        if not g:
            continue
        if p == g or g in p or p in g:
            return True
    return False


CONF_INT_RE = re.compile(r"\b(\d{1,3})\b")


def parse_confidence(conf_text: str) -> Optional[int]:
    """Extract the first integer 0..100 from the confidence continuation."""
    if not conf_text:
        return None
    m = CONF_INT_RE.search(conf_text)
    if not m:
        return None
    v = int(m.group(1))
    if 0 <= v <= 100:
        return v
    # Sometimes the model emits e.g. "8" then continues to a rank; clip to [0,100]
    return None


def _build_template_prefix_ids(tok, question: str) -> torch.Tensor:
    """Return the token ids for the T0 prefix up to (but not including) the answer.

    The fixed few-shot prefix (TEMPLATE_T0_FEWSHOT_PREFIX) is prepended so the
    pretrained model conditions its confidence continuation on the actual
    (question, answer) content rather than saturating at 100. The prefix is
    identical across all items; only the target Q/A changes.
    """
    prefix = TEMPLATE_T0.format(question=question)
    ids = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids[0]
    return ids


def _build_post_answer_template_ids(tok) -> torch.Tensor:
    """
    Return the token ids for the T0 post-answer template chunk:
        "\nConfidence (0-100): "
    (Without a leading BOS — we will concatenate this after the answer.)
    """
    txt = "\nConfidence (0-100): "
    ids = tok(txt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return ids


@torch.no_grad()
def generate_answer_and_confidence(
    model,
    tok,
    item: Dict,
    device: torch.device,
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
    max_conf_tokens: int = MAX_CONF_TOKENS,
) -> ItemTrace:
    """
    Two-stage greedy decode:
      Stage 1: from the prefix "Q: <q>\\nA: ", greedily decode the answer
               until we emit a newline (EOA anchor) or hit max_answer_tokens.
      Stage 2: append the fixed post-answer template "\\nConfidence (0-100): "
               and greedily decode up to max_conf_tokens for the confidence.

    Records precise absolute token positions for:
      - post_answer_positions: E0..E4
        E0 = the last answer token (EOA anchor)
        E1..E4 = tokens 1..4 in the post-answer template chunk
      - conf_gen_position: first token of the confidence continuation
      - answer_token_logprobs: per-answer-token log-prob
    """
    prefix_ids = _build_template_prefix_ids(tok, item["question"]).to(device)

    # Gemma-3's tokenizer has multiple newline-family tokens: 107 = '\n',
    # 108 = '\n\n', 109 = '\n\n\n'. Stop the answer at any of them.
    newline_ids = {
        tok("\n", add_special_tokens=False).input_ids[-1],
        tok("\n\n", add_special_tokens=False).input_ids[-1],
        tok("\n\n\n", add_special_tokens=False).input_ids[-1],
    }
    eos_id = tok.eos_token_id

    # ---- Stage 1: generate the answer greedily via KV-cache -------------
    # First forward pass consumes the full prefix; subsequent steps only feed
    # the last-generated token id and reuse cached keys/values.
    cur = prefix_ids.unsqueeze(0)  # [1, L_prefix]
    answer_ids: List[int] = []
    answer_logprobs: List[float] = []

    out = model(cur, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :]
    for step in range(max_answer_tokens):
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        next_id = int(logits.argmax().item())
        answer_ids.append(next_id)
        answer_logprobs.append(float(logprobs[next_id].item()))
        if next_id == eos_id or next_id in newline_ids:
            break
        # Feed only the new token id, keep the KV cache
        one_id = torch.tensor([[next_id]], device=device)
        out = model(one_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]

    # If the answer ended with a newline-family token, drop it — the EOA
    # anchor comes from the post-answer template's own leading newline.
    trimmed_answer_ids = list(answer_ids)
    trimmed_answer_logprobs = list(answer_logprobs)
    if trimmed_answer_ids and trimmed_answer_ids[-1] in newline_ids:
        trimmed_answer_ids = trimmed_answer_ids[:-1]
        trimmed_answer_logprobs = trimmed_answer_logprobs[:-1]
    answer_ids = trimmed_answer_ids
    answer_logprobs = trimmed_answer_logprobs

    answer_text = tok.decode(answer_ids, skip_special_tokens=True).strip()

    # ---- Stage 2: append post-answer template + generate confidence ------
    # Since we may have dropped a trailing newline, rebuild `cur` (the plumbed
    # sequence) from prefix + kept answer ids. Then run one forward pass over
    # (kept_answer_ids + tpl_ids) to prime the cache for confidence decoding.
    tpl_ids = _build_post_answer_template_ids(tok).to(device)  # "\nConfidence (0-100): "
    prefix_len = int(prefix_ids.shape[0])
    ans_start = prefix_len
    ans_end = prefix_len + len(answer_ids) - 1  # inclusive; = E0
    E0 = ans_end
    tpl_start = ans_end + 1
    n_tpl = int(tpl_ids.shape[0])
    post_positions = {"E0": E0}
    for i, name in enumerate(("E1", "E2", "E3", "E4")):
        if i < n_tpl:
            post_positions[name] = tpl_start + i
        else:
            post_positions[name] = tpl_start + n_tpl - 1

    conf_gen_position = tpl_start + n_tpl  # next generated token lives here

    # Rebuild the full context and prime a fresh KV cache over prefix+answer+tpl.
    # (Cheaper: 1 forward pass on the *full* pre-conf context, ~340 tokens.)
    context_ids = torch.cat(
        [prefix_ids, torch.tensor(answer_ids, device=device, dtype=torch.long), tpl_ids], dim=0
    ).unsqueeze(0)
    out = model(context_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1, :]

    conf_ids: List[int] = []
    for _ in range(max_conf_tokens):
        next_id = int(logits.argmax().item())
        conf_ids.append(next_id)
        if next_id == eos_id or next_id in newline_ids:
            break
        one_id = torch.tensor([[next_id]], device=device)
        out = model(one_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]

    # Build `cur` for downstream `full_ids` bookkeeping.
    cur = torch.cat(
        [context_ids, torch.tensor([conf_ids], device=device, dtype=torch.long)], dim=1
    )

    conf_text = tok.decode(conf_ids, skip_special_tokens=True)
    verbal_conf = parse_confidence(conf_text)

    is_correct = is_answer_correct(
        answer_text, item.get("normalized_aliases", []), item.get("aliases", [])
    )

    full_ids = cur[0].cpu().tolist()

    return ItemTrace(
        question_id=item["question_id"],
        question=item["question"],
        answer_text=answer_text,
        verbal_conf=verbal_conf,
        is_correct=is_correct,
        answer_token_logprobs=answer_logprobs,
        full_ids=full_ids,
        prefix_len=prefix_len,
        n_answer_tokens=len(answer_ids),
        post_answer_positions=post_positions,
        conf_gen_position=conf_gen_position,
        aliases=item.get("aliases", []),
        normalized_aliases=item.get("normalized_aliases", []),
    )


# ------------------------------------------------------------------------
# Residual-stream activation extraction
# ------------------------------------------------------------------------

@torch.no_grad()
def extract_residual_activations(
    model,
    input_ids: torch.Tensor,      # [1, L]
    positions_by_label: Dict[str, int],
    layers: Tuple[int, ...] = LAYER_GRID,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Run one forward pass with output_hidden_states=True and return
        {position_label: {layer_int: activation_vector (hidden_size,)}}
    hidden_states[L] = residual stream *after* layer L runs (transformers convention).
    hidden_states[0] is the token embedding output. We use L∈{5,10,...,60} → hidden_states[L].
    """
    out = model(input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple of len num_layers + 1
    result: Dict[str, Dict[int, np.ndarray]] = {}
    for label, pos in positions_by_label.items():
        result[label] = {}
        for L in layers:
            vec = hs[L][0, pos, :].detach().to(torch.float32).cpu().numpy()
            result[label][L] = vec
    return result


# ------------------------------------------------------------------------
# JSON-line helpers
# ------------------------------------------------------------------------

def write_jsonl(path: str | Path, rows: List[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> List[Dict]:
    p = Path(path)
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
