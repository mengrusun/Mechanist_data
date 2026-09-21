"""Constants for the SAGE experiment."""
import os

WORKDIR = "/data/zhenqian/Reproduction1/cc/feature_description/sae_agentic_explainer"
CACHE_DIR = os.path.join(WORKDIR, "cache")
RESULTS_DIR = os.path.join(WORKDIR, "results")
LOGS_DIR = os.path.join(WORKDIR, "logs")

DATA_DIR = "/data/zhenqian/data"
MODEL_DIR = "/data/zhenqian/models"

GEMMA_PATH = os.path.join(MODEL_DIR, "gemma-2-2b")
GEMMA_SAE_ROOT = os.path.join(MODEL_DIR, "sae", "gemma-scope-2b-pt-res")

QWEN_PATH = os.path.join(MODEL_DIR, "Qwen3-4B")
QWEN_TC_ROOT = os.path.join(MODEL_DIR, "sae", "qwen3-4b-transcoders")

HF_HUB_CACHE = os.path.join(DATA_DIR, "cache")

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

NEURONPEDIA_URL = "https://www.neuronpedia.org/api/feature"

for d in (CACHE_DIR, RESULTS_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)
