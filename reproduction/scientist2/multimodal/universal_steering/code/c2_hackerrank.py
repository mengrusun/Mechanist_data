"""C2: HackerRank test-case pass rate — C++ steering vs default vs prompt-only.

Loads v_cpp from extract_vectors.py (concept=cpp_python).
Uses /data/zhenqian/data/hackerrank/eval_set.jsonl (20 problems with sample_tests).

Three conditions on held-out:
  (a) default: prompt "Solve this problem: ..." (Python emerges by default)
  (b) prompt-only: appended "Answer in C++."
  (c) default prompt + α*·v_cpp additive steering

Language auto-detected from output; C++ compiled with g++ -O2 -std=c++17, run against
sample_tests. Python run via subprocess. Fixed 5s timeout per test.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from model_utils import load_model, generate_with_steering, free_cuda
from rfm_core import load_vector

WORK_DIR = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")
HACKERRANK = Path("/data/zhenqian/data/hackerrank")


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def format_chat_prompt(tokenizer, user_msg):
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_code_block(text: str, lang_hint: str = "auto") -> tuple[str, str]:
    """Return (language, code). Language ∈ {'python','cpp','unknown'}.
    Extracts the largest fenced code block.
    """
    # Try fenced ```{lang}...``` first
    matches = re.findall(r"```([A-Za-z+#]*)\n(.*?)```", text, re.DOTALL)
    if matches:
        # Prefer python/cpp fenced blocks
        for lang, code in matches:
            l = lang.strip().lower()
            if l in ("python", "py"):
                return "python", code
            if l in ("cpp", "c++", "cxx"):
                return "cpp", code
        # Fall back to largest block
        code = max(matches, key=lambda m: len(m[1]))[1]
        return _guess_lang(code), code
    # No fence — treat whole text as code
    return _guess_lang(text), text


def _guess_lang(code: str) -> str:
    c = code.strip()
    if re.search(r"#include\s*<", c) or "int main(" in c or "std::" in c:
        return "cpp"
    if re.search(r"def\s+\w+\(", c) or c.lstrip().startswith("import ") or "print(" in c:
        return "python"
    return "unknown"


def run_python(code: str, stdin: str, timeout: float = 5.0) -> tuple[str, str, int]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sol.py"
        p.write_text(code)
        try:
            r = subprocess.run(
                [sys.executable, str(p)], input=stdin, capture_output=True,
                text=True, timeout=timeout, cwd=td,
            )
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT", 124


def run_cpp(code: str, stdin: str, timeout: float = 5.0) -> tuple[str, str, int]:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "sol.cpp"
        binp = Path(td) / "sol"
        src.write_text(code)
        try:
            c = subprocess.run(
                ["g++", "-O2", "-std=c++17", "-o", str(binp), str(src)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return "", "COMPILE_TIMEOUT", 124
        if c.returncode != 0:
            return "", f"COMPILE_ERR:{c.stderr[:200]}", c.returncode
        try:
            r = subprocess.run(
                [str(binp)], input=stdin, capture_output=True,
                text=True, timeout=timeout,
            )
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT", 124


def normalise_output(s: str) -> str:
    """Normalise whitespace for tolerant comparison."""
    lines = [l.rstrip() for l in s.strip().splitlines()]
    return "\n".join(lines)


def score_output(gen_text: str, tests: list[dict]) -> dict:
    """Return {lang, compiled, num_tests, num_passed, pass_rate, runtime_err_frac}."""
    lang, code = extract_code_block(gen_text)
    if lang == "unknown" or not code.strip():
        return {"lang": lang, "compiled": False, "num_tests": len(tests),
                "num_passed": 0, "pass_rate": 0.0, "note": "no_valid_code"}
    passed = 0; runtime_err = 0
    for tc in tests:
        stdin = tc.get("input", "")
        expected = normalise_output(tc.get("expected_output", ""))
        if lang == "python":
            out, err, code_r = run_python(code, stdin)
        elif lang == "cpp":
            out, err, code_r = run_cpp(code, stdin)
        else:
            out, err, code_r = "", "unknown_lang", -1
        if code_r != 0:
            runtime_err += 1
            continue
        if normalise_output(out) == expected:
            passed += 1
    return {"lang": lang, "compiled": True, "num_tests": len(tests),
            "num_passed": passed,
            "pass_rate": passed / max(1, len(tests)),
            "runtime_err_frac": runtime_err / max(1, len(tests))}


def make_prompt(problem: dict, style: str) -> str:
    """style ∈ {'default','cpp_prompt'}"""
    slug = problem["slug"]; title = problem.get("title", slug)
    # We don't have the full problem text — we have the Python solution and sample tests.
    # Use the *sample tests* + slug to describe: this simulates the reproduction eval well
    # enough for a functional test (LLM must infer the problem from the io-samples).
    tests_str = "\n\n".join(
        f"Input:\n{t['input'].rstrip()}\nExpected Output:\n{t['expected_output'].rstrip()}"
        for t in problem.get("sample_tests", [])[:3]
    )
    base = (f"Problem title: {title} (HackerRank challenge slug: {slug})\n\n"
            f"Sample tests:\n{tests_str}\n\n"
            "Write a complete solution that reads from standard input and prints to "
            "standard output, matching the sample tests. Return a single fenced code block.")
    if style == "cpp_prompt":
        base += "\n\nAnswer in C++."
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-run-id", default="B1_extract_vectors")
    ap.add_argument("--run-id", default="C2_hackerrank")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--n-dev", type=int, default=10, help="First n_dev for α-sweep, remainder held-out")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 200])
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else WORK_DIR / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = WORK_DIR / "runs" / args.extract_run_id

    # Sanity: g++ available
    if not shutil.which("g++"):
        raise RuntimeError("g++ not on PATH — install a c++ compiler")

    problems = read_jsonl(HACKERRANK / "eval_set.jsonl")[: args.n_problems]
    print(f"[c2] loaded {len(problems)} problems from {HACKERRANK/'eval_set.jsonl'}")
    dev_probs = problems[: args.n_dev]
    held_probs = problems[args.n_dev:]

    print(f"[c2] loading model...")
    model, tokenizer, cfg = load_model(dtype=args.dtype, device=args.device)

    v_c, vc_meta = load_vector(extract_dir / "concept_cpp_python" / "v_c.npy")
    best_block = int(vc_meta["best_block"])
    print(f"[c2] v_cpp: block={best_block}, val_acc={vc_meta['probe_val_acc_best_block']:.3f}")

    summary = {"config": vars(args), "best_block": best_block,
               "d_model": vc_meta["d_model"],
               "n_dev": len(dev_probs), "n_held": len(held_probs)}

    # ------------------- Dev α sweep -------------------
    print(f"[c2] === DEV α sweep on {len(dev_probs)} problems ===")
    dev_prompts = [format_chat_prompt(tokenizer, make_prompt(p, "default")) for p in dev_probs]
    dev_results = {}
    for alpha in args.alphas:
        iv = None if alpha == 0.0 else [{"block_idx": best_block, "v_c": v_c, "alpha": alpha}]
        t0 = time.time()
        gens = generate_with_steering(
            model, tokenizer, dev_prompts, iv,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=args.device,
        )
        dt = time.time() - t0
        # Score each
        scored = []
        for p, gen in zip(dev_probs, gens):
            sc = score_output(gen, p.get("sample_tests", []))
            sc["prompt_slug"] = p["slug"]; sc["gen_len"] = len(gen)
            scored.append({"gen": gen, **sc})
        cpp_frac = float(np.mean([s["lang"] == "cpp" for s in scored]))
        py_frac = float(np.mean([s["lang"] == "python" for s in scored]))
        mean_pass = float(np.mean([s["pass_rate"] for s in scored]))
        mean_len = float(np.mean([s["gen_len"] for s in scored]))
        print(f"[c2]   α={alpha:+.1f} took {dt:.1f}s cpp_frac={cpp_frac:.2f} "
              f"py_frac={py_frac:.2f} pass_rate={mean_pass:.3f} mean_len={mean_len:.0f}")
        dev_results[str(alpha)] = {"cpp_frac": cpp_frac, "py_frac": py_frac,
                                    "mean_pass_rate": mean_pass, "mean_len": mean_len,
                                    "scored": scored, "wall_s": dt}

    baseline_len = dev_results["0.0"]["mean_len"]
    # Pick α* = argmax(pass_rate) subject to len <= 2.5x baseline
    valid = {a: v for a, v in dev_results.items()
             if float(a) != 0.0 and v["mean_len"] <= 2.5 * baseline_len}
    if not valid:
        alpha_star = 0.0
    else:
        alpha_star = float(max(valid, key=lambda a: valid[a]["mean_pass_rate"]))
    print(f"[c2] chosen α* = {alpha_star:+.1f} (dev pass_rate = {dev_results[str(alpha_star)]['mean_pass_rate']:.3f})")
    summary["alpha_star"] = alpha_star
    summary["dev_results"] = {k: {kk: vv for kk, vv in v.items() if kk != "scored"}
                              for k, v in dev_results.items()}

    # Dev scored dump
    with open(out_dir / "dev_details.jsonl", "w") as f:
        for a, v in dev_results.items():
            for s in v["scored"]:
                f.write(json.dumps({"alpha": float(a), **s}, ensure_ascii=False) + "\n")

    # ------------------- Held-out: 3 conditions × 2 seeds -------------------
    print(f"[c2] === HELD-OUT on {len(held_probs)} problems × 3 conditions × {len(args.seeds)} seeds ===")
    conditions = {
        "default_prompt": ("default", None),
        "cpp_prompt_only": ("cpp_prompt", None),
        "cpp_steered": ("default", alpha_star),
    }
    held_scored = {c: [] for c in conditions}
    for seed in args.seeds:
        torch_seed(seed)
        for cond_name, (style, alpha) in conditions.items():
            iv = None if not alpha or alpha == 0.0 else [{"block_idx": best_block, "v_c": v_c, "alpha": alpha}]
            prompts = [format_chat_prompt(tokenizer, make_prompt(p, style)) for p in held_probs]
            t0 = time.time()
            gens = generate_with_steering(
                model, tokenizer, prompts, iv,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=args.device,
            )
            for p, gen in zip(held_probs, gens):
                sc = score_output(gen, p.get("sample_tests", []))
                sc["prompt_slug"] = p["slug"]; sc["gen_len"] = len(gen)
                sc["condition"] = cond_name; sc["seed"] = seed
                held_scored[cond_name].append({"gen": gen, **sc})
            dt = time.time() - t0
            pr = float(np.mean([s["pass_rate"] for s in held_scored[cond_name] if s["seed"] == seed]))
            print(f"[c2]   seed={seed} {cond_name}: took {dt:.1f}s pass_rate={pr:.3f}")

    # Aggregate
    agg = {}
    for cond in conditions:
        rows = held_scored[cond]
        agg[cond] = {
            "mean_pass_rate": float(np.mean([s["pass_rate"] for s in rows])),
            "std_pass_rate": float(np.std([s["pass_rate"] for s in rows])),
            "cpp_frac": float(np.mean([s["lang"] == "cpp" for s in rows])),
            "py_frac": float(np.mean([s["lang"] == "python" for s in rows])),
            "n": len(rows),
        }
    summary["held_out"] = agg
    print(f"[c2] HELD-OUT summary: {json.dumps(agg, indent=2)}")

    with open(out_dir / "held_details.jsonl", "w") as f:
        for cond, rows in held_scored.items():
            for r in rows:
                f.write(json.dumps({**r, "condition": cond}, ensure_ascii=False) + "\n")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[c2] DONE → {out_dir/'summary.json'}")


def torch_seed(seed: int) -> None:
    import torch, numpy as np, random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
