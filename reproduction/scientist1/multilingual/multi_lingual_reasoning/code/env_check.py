"""Sanity-check the environment: torch, GPU, model loading, MGSM read."""
import os, sys, json, glob
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/data/zhenqian/models/Qwen3-4B"
MGSM_DIR = "/data/zhenqian/data/mgsm"

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())

# Load MGSM parquets to a dict
LANGS = ["en", "es", "fr", "de", "zh", "ja", "ru", "th", "te", "bn", "sw"]
data = {}
for lg in LANGS:
    df_train = pd.read_parquet(os.path.join(MGSM_DIR, lg, "train-00000-of-00001.parquet"))
    df_test  = pd.read_parquet(os.path.join(MGSM_DIR, lg, "test-00000-of-00001.parquet"))
    data[lg] = {"train": df_train, "test": df_test}
    print(f"[{lg}] train {len(df_train)} test {len(df_test)}  cols={list(df_train.columns)}")
print("first train (en):", data["en"]["train"].iloc[0].to_dict())
print("first test  (en):", data["en"]["test"].iloc[0].to_dict())

# Load model
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
print("tokenizer vocab:", tok.vocab_size, "chat_template?", tok.chat_template is not None)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
print("model:", type(model).__name__, "layers:", len(model.model.layers), "hidden:", model.config.hidden_size)
print("OK")
