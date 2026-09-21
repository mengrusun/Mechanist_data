"""Shared utilities for activation-patching experiments using transformer_lens."""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer


FEWSHOT = (
    "Facts: p is true. Rule: if p is true then q is true. Question: q is true.\n"
    "Facts: p is true. Rule: if p is true then q is false. Question: q is false.\n"
    "Facts: p is true. Rule: if p is true then q is true. Question: q is true.\n"
    "Facts: p is true. Rule: if p is true then q is false. Question: q is false.\n"
)


def load_hooked_mistral(local_path: str, dtype=torch.bfloat16, device="cuda:0"):
    hf_model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(local_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = HookedTransformer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        hf_model=hf_model,
        tokenizer=tok,
        device=device,
        dtype=dtype,
        fold_ln=False,  # rmsnorm safer to leave folded off for patching
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model, tok


def load_hooked_gemma2_9b(local_path: str, dtype=torch.bfloat16, device="cuda:0"):
    hf_model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(local_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-9b",
        hf_model=hf_model,
        tokenizer=tok,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model, tok


def load_hooked_gemma2_2b(local_path: str, dtype=torch.bfloat16, device="cuda:0"):
    hf_model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(local_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=hf_model,
        tokenizer=tok,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    return model, tok


def token_id_for(tok, word: str, ref_prompt: str = "Question: x is") -> int:
    base = tok.encode(ref_prompt, add_special_tokens=False)
    full = tok.encode(ref_prompt + " " + word, add_special_tokens=False)
    extra = full[len(base):]
    if len(extra) != 1:
        raise ValueError(f"'{word}' -> extra {extra} not single-token")
    return extra[0]


def load_dataset(path: str, prepend_fewshot: bool = True):
    ds = [json.loads(l) for l in Path(path).read_text().splitlines()]
    if prepend_fewshot:
        for r in ds:
            for k in ("clean", "corrupt", "fact_flip", "query_flip"):
                if k in r:
                    r[k] = FEWSHOT + r[k]
    return ds


def logit_diff(logits_last, clean_ans_ids, corr_ans_ids):
    """logit(clean) - logit(corrupt) on final token position. Higher = more
    consistent with clean-side answer."""
    return (
        logits_last.gather(1, clean_ans_ids[:, None]).squeeze(1)
        - logits_last.gather(1, corr_ans_ids[:, None]).squeeze(1)
    )
