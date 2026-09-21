"""One-shot cross-model review of experiment scripts via gpt-5.4."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LLMClient

FILES = ["scripts/m2_concept_alignment.py", "scripts/m6_feature_clamp_steering.py",
         "scripts/m1_feature_count.py", "scripts/m4_novel_concept_llm.py",
         "scripts/m5_annotation_filling.py", "scripts/m3_superposition_controls.py",
         "scripts/common.py"]

parts = []
for fp in FILES:
    text = Path(fp).read_text()
    parts.append(f"\n\n### {fp}\n```python\n{text}\n```")

body = "\n".join(parts)

review_msg = (
    "You are reviewing seven scripts implementing the reproduction of five SAE-on-ESM-2 "
    "interpretability claims (feature-count, concept-alignment, superposition ladder, "
    "LLM auto-interp of unaligned features, annotation-filling probes, feature-clamp "
    "steering with a dose ladder). The pretrained SAE checkpoints have architecture: "
    "encoder Linear[F=10240, d=1280], decoder Linear[d, F], pre-encoder bias 'bias' [d]. "
    "The SAE forward path is: h = x - bias; z = ReLU(encoder(h)); x_hat = decoder(z) + bias. "
    "Ground truth comes from Swiss-Prot .dat.gz per-residue annotations (parsed via BioPython). "
    "ESM-2 tokenizer: [CLS] r1 r2 ... [EOS] [PAD]* — so hidden_states[layer][:, 1:-1] over "
    "non-pad positions gives per-residue vectors.\n\n"
    "For each SCRIPT, look for CRITICAL/MAJOR/MINOR bugs in these dimensions:\n"
    "1. Ground truth: labels come from the DATASET (Swiss-Prot .dat), never from a model.\n"
    "2. F1 protocol: per-(unit, concept) F1 computed over annotated residues only, with the "
    "  correct binarization threshold (quantile-based, per unit).\n"
    "3. SAE forward path matches architecture.\n"
    "4. ESM-2 hidden-state indexing: hidden_states[layer] is output of encoder.layer[layer-1]. "
    "  The steering hook must target the right module so the intervention happens at that site.\n"
    "5. Off-by-one in [CLS]/[EOS]/[PAD] stripping (should extract exactly len(seq) residues).\n"
    "6. Plausibility band in M6 must use STEERED pseudo-PPL (with hook enabled during scoring) "
    "  vs no-steering PPL baseline, not unsteered PPL of a steered completion.\n"
    "7. Dose interpretation: alpha in units of within-sample activation std sigma_f. Actual "
    "  addition = alpha * sigma_f * d_feature (decoder column).\n"
    "8. Any obvious numerical/logical issues (fp16 underflow, div-by-zero on empty slices, "
    "  dead features handled without producing spurious 'hits', etc.).\n"
    "9. Data amount and split hygiene: probes train on train.parquet and eval on test.parquet, "
    "  Swiss-Prot annotations parsed once and cached.\n\n"
    "Respond with JSON: "
    "{\"issues\": [{\"severity\": \"CRITICAL|MAJOR|MINOR\", \"file\": str, "
    "\"description\": str, \"fix\": str}], "
    "\"overall_verdict\": \"PASS|WARN|FAIL\"}. "
    "Only report actual bugs (not style / naming). If no critical bugs, verdict PASS.\n\n"
    "=== SCRIPTS ===\n" + body
)

llm = LLMClient()
r = llm.chat_json(
    [{"role": "user", "content": review_msg}],
    max_tokens=3000,
    temperature=0.0,
    response_schema_key="cross_model_review",
)
print("OK:", r.get("ok"))
if r.get("ok"):
    print(json.dumps(r["parsed"], indent=2, default=str))
else:
    print(r.get("error"))
