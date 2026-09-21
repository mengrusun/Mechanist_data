"""Extract C++ code blocks from model outputs, compile them, and try to
run them against sample_tests. Report:

  - fraction that contain a C++ block
  - fraction that compile
  - fraction that pass at least one sample test
"""

import argparse
import json
import re
import subprocess
import tempfile
import os
from pathlib import Path


CPP_RE = re.compile(r"```cpp\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CPP_RE_LOOSE = re.compile(r"```\s*(#include[^`]*?)```", re.DOTALL)
CPP_RE_UNCLOSED = re.compile(r"```cpp\s*(.*)", re.DOTALL | re.IGNORECASE)
CPP_RE_INCLUDE = re.compile(r"(#include[\s\S]*?int\s+main[\s\S]*?})", re.DOTALL)
PY_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PY_RE_UNCLOSED = re.compile(r"```python\s*(.*)", re.DOTALL | re.IGNORECASE)


def extract_cpp(text):
    for pat in (CPP_RE, CPP_RE_LOOSE):
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    if "```cpp" in text.lower():
        m = CPP_RE_UNCLOSED.search(text)
        if m:
            return m.group(1).strip()
    m = CPP_RE_INCLUDE.search(text)
    if m:
        return m.group(1).strip()
    return None


def extract_py(text):
    for pat in (PY_RE, PY_RE_UNCLOSED):
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None


def compile_cpp(code, timeout=10):
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "sol.cpp")
        exe = os.path.join(td, "sol")
        with open(src, "w") as f:
            f.write(code)
        try:
            r = subprocess.run(
                ["g++", "-O2", "-std=c++17", src, "-o", exe],
                capture_output=True, timeout=timeout, text=True,
            )
            if r.returncode == 0:
                return True, exe, td
            return False, r.stderr[:400], td
        except subprocess.TimeoutExpired:
            return False, "compile timeout", td


def run_binary(exe_path, stdin, timeout=5):
    try:
        r = subprocess.run(
            [exe_path], input=stdin, capture_output=True,
            timeout=timeout, text=True,
        )
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def run_python(code, stdin, timeout=8):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        pth = f.name
    try:
        r = subprocess.run(
            ["python3", pth], input=stdin, capture_output=True,
            timeout=timeout, text=True,
            env={**os.environ, "OUTPUT_PATH": "/dev/null"},
        )
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(pth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True,
                    help="jsonl with slug/steered/sample_tests")
    ap.add_argument("--prompts", required=True,
                    help="jsonl with slug/sample_tests")
    ap.add_argument("--field", default="steered",
                    help="which field contains the output to evaluate")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    slug_tests = {}
    for line in open(args.prompts):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        slug_tests[r["prompt"]] = r.get("sample_tests", [])
        # also match on slug in prompt
    outs = [json.loads(l) for l in open(args.outputs) if l.strip()]
    if args.limit is not None:
        outs = outs[: args.limit]

    n_cpp = 0
    n_py = 0
    n_compiled = 0
    n_pass_any = 0
    n_pass_all = 0
    n_total = len(outs)

    per_example = []
    for r in outs:
        text = r[args.field]
        cpp = extract_cpp(text)
        py = extract_py(text)
        tests = slug_tests.get(r["prompt"], [])
        ex = {"prompt": r["prompt"][:80], "lang": None, "compiled": False,
              "pass_any": False, "pass_all": False, "n_tests": len(tests)}
        if cpp:
            n_cpp += 1
            ex["lang"] = "cpp"
            ok, info, td = compile_cpp(cpp)
            if ok:
                n_compiled += 1
                ex["compiled"] = True
                exe = info
                pass_ct = 0
                for t in tests:
                    stdin = t["input"]
                    expected = t["expected_output"].strip()
                    got = run_binary(exe, stdin)
                    if got is not None and got.strip() == expected:
                        pass_ct += 1
                if pass_ct > 0:
                    n_pass_any += 1
                    ex["pass_any"] = True
                if pass_ct == len(tests) and len(tests) > 0:
                    n_pass_all += 1
                    ex["pass_all"] = True
                ex["pass_ct"] = pass_ct
        elif py:
            n_py += 1
            ex["lang"] = "py"
            # try python run
            pass_ct = 0
            for t in tests:
                got = run_python(py, t["input"])
                expected = t["expected_output"].strip()
                if got is not None and got.strip() == expected:
                    pass_ct += 1
            if pass_ct > 0:
                n_pass_any += 1
                ex["pass_any"] = True
            if pass_ct == len(tests) and len(tests) > 0:
                n_pass_all += 1
                ex["pass_all"] = True
            ex["compiled"] = True   # python doesn't compile but runnable
            ex["pass_ct"] = pass_ct
        per_example.append(ex)

    report = {
        "n_total": n_total,
        "n_cpp_block": n_cpp,
        "n_py_block": n_py,
        "n_compiled_or_runnable": n_compiled + n_py,   # approx
        "n_pass_any_test": n_pass_any,
        "n_pass_all_tests": n_pass_all,
        "per_example": per_example,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_example"},
                     indent=2))


if __name__ == "__main__":
    main()
