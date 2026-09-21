#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score claim files with an LLM and optional related-work retrieval.

The script recursively discovers ``claim.json`` files under any input path and
writes ``score_<dimension>.json`` beside each claim. Before novelty scoring it
can query Semantic Scholar and save ``related_works.json`` in the same folder.

Usage:
  python score_gpt5.6sol.py <path>
  python score_gpt5.6sol.py <path> --force
  python score_gpt5.6sol.py <path> --workers 8
  python score_gpt5.6sol.py <path> --prompt-dir <dir>
  python score_gpt5.6sol.py <path> --no-related-work

Configuration:
  - ``LLM_BASE_URL`` and ``LLM_API_KEY`` are required.
  - ``LLM_MODEL`` overrides the default model name.
  - ``EVAL_PROMPT_DIR`` or ``--prompt-dir`` overrides the prompt directory.
  - ``S2_API_KEY`` optionally authenticates Semantic Scholar requests.
"""

import argparse
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm

# ---------------------------------------------------------------- CONFIG ----
BASE_URL = os.environ.get("LLM_BASE_URL")
API_KEY = os.environ.get("LLM_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "openai/responses/gpt-5.6-sol")

MAX_TOKENS = 4000
TEMPERATURE = 0.2
TIMEOUT = 180
MAX_RETRIES = 4

SCRIPT_DIR = Path(__file__).resolve().parent
# Default eval_prompt/ lives at the repo root (one level up from scripts/).
# Override with --prompt-dir or the EVAL_PROMPT_DIR env var.
DEFAULT_PROMPT_DIR = Path(
    os.environ.get("EVAL_PROMPT_DIR", SCRIPT_DIR.parent / "eval_prompt")
)

DIMENSIONS = ["novelty", "impact", "testability"]
DIM_KEY = {                       # Expected top-level key for each dimension.
    "novelty": "novelty",
    "impact": "impact",
    "testability": "dimension_3_testability",
}

# ------------------------------------------------- Semantic Scholar CONFIG --
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# Optional API key; never store credentials in source code.
S2_API_KEY = os.environ.get("S2_API_KEY")
S2_FIELDS = "title,abstract,year,venue,authors,citationCount,externalIds,url"
S2_TIMEOUT = 30
S2_MAX_RETRIES = 8                # Retry transient S2 rate-limit responses.
S2_BACKOFF_BASE = 3.0             # Exponential backoff base for network/5xx errors.
# Apply one shared throttle across all threads and endpoints.
S2_MIN_INTERVAL = float(os.environ.get("S2_MIN_INTERVAL", "1.1"))

RELATED_WORKS_FILENAME = "related_works.json"
DEFAULT_S2_QUERIES = 5            # Search queries generated per claim.
DEFAULT_S2_PER_QUERY = 5          # Papers requested per query.
DEFAULT_S2_MAX_PAPERS = 15        # Maximum deduplicated papers per claim.
S2_ABSTRACT_CHARS = 1600          # Per-paper abstract limit in prompts.
# ---------------------------------------------------------------- CONFIG ----


def find_topic(claim_dir: Path) -> str:
    """Return the nearest ancestor topic.md text, or an empty string."""
    d = claim_dir
    while d != d.parent:
        topic = d / "topic.md"
        if topic.exists():
            return topic.read_text(encoding="utf-8").strip()
        d = d.parent
    return ""


# --------------------------------------------------------- Related Work ----
_S2_LOCK = threading.Lock()
_S2_LAST_CALL = [0.0]


def _s2_throttle() -> None:
    """Enforce the global minimum interval between Semantic Scholar calls."""
    with _S2_LOCK:
        wait = S2_MIN_INTERVAL - (time.monotonic() - _S2_LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        _S2_LAST_CALL[0] = time.monotonic()


def s2_search(query: str, limit: int) -> list:
    """Search Semantic Scholar with retry handling for 429/5xx responses."""
    url = "{}?{}".format(
        S2_SEARCH_URL,
        urllib.parse.urlencode({"query": query, "limit": limit, "fields": S2_FIELDS}),
    )
    headers = {"User-Agent": "hypothesis-eval/1.0", "Accept": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    last_err = None
    for attempt in range(S2_MAX_RETRIES):
        _s2_throttle()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=S2_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("data") or []
        except urllib.error.HTTPError as e:                  # noqa: PERF203
            last_err = f"HTTP {e.code}"
            if attempt >= S2_MAX_RETRIES - 1:
                break
            if e.code == 429:
                # Retry rate-limit jitter through the global throttle unless
                # the server explicitly supplies a Retry-After duration.
                retry_after = (e.headers.get("Retry-After") or "").strip() if e.headers else ""
                if retry_after.isdigit():
                    time.sleep(min(float(retry_after), 60.0))
                continue
            if e.code in (500, 502, 503, 504):
                time.sleep(min(S2_BACKOFF_BASE * (2 ** attempt), 60.0))
                continue
            break
        except Exception as e:                               # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt == S2_MAX_RETRIES - 1:
                break
            time.sleep(S2_BACKOFF_BASE * (2 ** attempt))
    raise RuntimeError(f"semantic scholar search failed for {query!r}: {last_err}")


def build_query_prompt(claim_text: str, topic_text: str) -> str:
    """Build the prompt used to generate English Semantic Scholar queries."""
    return (
        "You are preparing a literature search that another reviewer will use to judge "
        "whether a research idea is novel.\n\n"
        "Read the research idea below and write search queries for Semantic Scholar that "
        "would surface prior work equivalent or closely related to it.\n\n"
        "Rules:\n"
        f"- Produce exactly {DEFAULT_S2_QUERIES} queries, each 3-8 English words, keyword-style "
        "(no boolean operators, no quotes, no punctuation).\n"
        "- Attack the idea from different angles and phrasings: the phenomenon/behavior studied, "
        "the proposed mechanism or method, the measurement instrument, and the application "
        "setting. The field uses different terminology for the same idea, so vary the wording.\n"
        "- Query the prior literature, not this specific paper: drop invented names/branding the "
        "idea coins for itself and use the field's standard terms instead.\n"
        "- Output ONLY a JSON object, no code fences and no commentary:\n"
        '  {"queries": ["...", "..."]}\n\n'
        + (f"=== research topic ===\n{topic_text[:2000]}\n\n" if topic_text else "")
        + f"=== research idea ===\n{claim_text[:8000]}\n"
    )


class QueryGenerationError(RuntimeError):
    """Raised when the model cannot produce parseable search queries."""


def gen_queries(claim_text: str, topic_text: str, n_queries: int) -> list:
    """Generate search queries or raise QueryGenerationError."""
    try:
        raw = call_model(build_query_prompt(claim_text, topic_text))
    except Exception as e:                                   # noqa: BLE001
        raise QueryGenerationError(f"judge model unavailable: {type(e).__name__}: {e}") from e

    try:
        queries = extract_json(raw).get("queries") or []
    except Exception as e:                                   # noqa: BLE001
        raise QueryGenerationError(f"unparseable model output: {type(e).__name__}: {e}") from e

    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not queries:
        raise QueryGenerationError("model returned no search queries")

    # Deduplicate while preserving order.
    seen, uniq = set(), []
    for q in queries:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return uniq[:n_queries]


def _paper_key(paper: dict) -> str:
    pid = paper.get("paperId")
    if pid:
        return str(pid)
    return re.sub(r"\W+", "", (paper.get("title") or "")).lower()


def fetch_related_works(claim_json_path: Path, n_queries: int,
                        per_query: int, max_papers: int) -> dict:
    """Retrieve related work for one claim and return the persisted object."""
    claim_text = json.dumps(
        json.loads(claim_json_path.read_text(encoding="utf-8")),
        indent=2, ensure_ascii=False,
    )
    topic_text = find_topic(claim_json_path.parent)

    queries = gen_queries(claim_text, topic_text, n_queries)

    papers, seen, errors = [], set(), []
    for q in queries:
        try:
            hits = s2_search(q, per_query)
        except Exception as e:                               # noqa: BLE001
            errors.append({"query": q, "error": f"{type(e).__name__}: {e}"})
            continue
        for p in hits:
            key = _paper_key(p)
            if not key or key in seen:
                continue
            seen.add(key)
            papers.append({
                "paperId": p.get("paperId"),
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "authors": [a.get("name") for a in (p.get("authors") or []) if a.get("name")],
                "citationCount": p.get("citationCount"),
                "externalIds": p.get("externalIds"),
                "url": p.get("url"),
                "matched_query": q,
            })

    # Prefer papers with abstracts, then rank by citation count.
    papers.sort(key=lambda p: (0 if p.get("abstract") else 1,
                               -(p.get("citationCount") or 0)))
    papers = papers[:max_papers]

    return {
        "source": "semanticscholar",
        "endpoint": S2_SEARCH_URL,
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "queries": queries,
        "count": len(papers),
        "related_works": papers,
        "errors": errors,
    }


def ensure_related_works(claim_json_path: Path, force: bool, n_queries: int,
                         per_query: int, max_papers: int):
    """Generate or reuse related_works.json and return status information."""
    claim_dir = claim_json_path.parent
    out = claim_dir / RELATED_WORKS_FILENAME
    if out.exists() and not force:
        return ("skip", claim_dir, None)
    try:
        data = fetch_related_works(claim_json_path, n_queries, per_query, max_papers)
    except QueryGenerationError as e:
        # Leave no cache file so the next run retries this claim.
        return ("noquery", claim_dir, str(e))
    except Exception as e:                                   # noqa: BLE001
        return ("fail", claim_dir, f"{type(e).__name__}: {e}")

    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    if data["count"] == 0 and data["errors"]:
        return ("fail", claim_dir,
                f"0 papers; first error: {data['errors'][0]['error']}")
    return ("ok", claim_dir, None)


def load_related_works(claim_dir: Path) -> list:
    """Load related papers, returning an empty list for missing/bad data."""
    path = claim_dir / RELATED_WORKS_FILENAME
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("related_works") or []
    except Exception:                                        # noqa: BLE001
        return []


def format_related_works(papers: list) -> str:
    """Render retrieved titles and abstracts for the novelty prompt."""
    if not papers:
        return ""
    lines = [
        "\n=== {related_work} — candidate prior work retrieved from Semantic Scholar ===\n",
        "These papers were retrieved automatically by keyword search on Semantic Scholar "
        "for this claim, before your review. They are RAW search results, not a curated "
        "related-work list: some will be irrelevant, and the list is certainly incomplete. "
        "Use them as evidence — check whether any is essentially equivalent to the claim's "
        "idea, and if so say which and score the novelty failure accordingly. Do NOT treat "
        "mere topical overlap as duplication, and do NOT treat an empty or thin list as "
        "proof of novelty: combine it with your own knowledge of the SOTA and your own "
        "searching, exactly as the instructions above require.\n",
    ]
    for i, p in enumerate(papers, 1):
        abstract = (p.get("abstract") or "").strip()
        if len(abstract) > S2_ABSTRACT_CHARS:
            abstract = abstract[:S2_ABSTRACT_CHARS].rstrip() + " …[truncated]"
        meta = " / ".join(str(x) for x in (p.get("year"), p.get("venue")) if x)
        lines.append(
            f"\n[{i}] Title: {p.get('title') or '(untitled)'}\n"
            + (f"    Year / Venue: {meta}\n" if meta else "")
            + f"    Abstract: {abstract or '(no abstract available)'}\n"
        )
    return "".join(lines)


def build_prompt(dim: str, claim_text: str, topic_text: str, prompt_dir: Path,
                 related_works: list = None) -> str:
    """Build a dimension prompt from the template, topic, claim, and papers."""
    tpl = (prompt_dir / f"hypothesis_judge_{dim}.md").read_text(encoding="utf-8")
    parts = [tpl, "\n\nBelow is the content you need to evaluate.\n"]
    if topic_text:
        parts.append(
            "\n=== {topic} — the research topic given to the scientist ===\n"
            + topic_text
            + "\n"
        )
    else:
        parts.append(
            "\n=== {topic} ===\n"
            "(No topic.md was found for this claim; judge the claim on its own terms "
            "and do not deduct for the missing topic.)\n"
        )
    parts.append("\n=== {claim} — the generated claim under review ===\n" + claim_text)
    if dim == "novelty" and related_works:
        parts.append(format_related_works(related_works))
    return "".join(parts)


def litellm_model_name(name: str) -> str:
    """Add the provider prefix required by LiteLLM when absent."""
    return name if "/" in name else f"openai/{name}"


def call_model(prompt: str) -> str:
    """Call the LLM API with exponential-backoff retries."""
    messages = [
        {"role": "system",
         "content": f"Today's date is {datetime.date.today().isoformat()}. "
                    f"Use this date when checking any cited works."},
        {"role": "user", "content": prompt},
    ]
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = litellm.completion(
                model=litellm_model_name(MODEL),
                messages=messages,
                api_base=BASE_URL,
                api_key=API_KEY,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                timeout=TIMEOUT,
            )
            content = getattr(resp.choices[0].message, "content", None)
            if not content:
                content = getattr(resp, "output_text", None)
            if content:
                return content
            raise ValueError("empty model output")
        except Exception as e:                              # noqa: BLE001
            last_err = e
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call_model failed: {last_err}")


def extract_json(text: str) -> dict:
    """Extract a JSON object from model output."""
    if not text:
        raise ValueError("empty response")
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in: {text[:400]!r}")
    return json.loads(cleaned[start:end + 1])


def score_one(claim_json_path: Path, dim: str, force: bool, prompt_dir: Path,
              use_related_work: bool = True):
    """Score one (claim, dimension) task and return status details."""
    claim_dir = claim_json_path.parent
    out = claim_dir / f"score_{dim}.json"
    if out.exists() and not force:
        return ("skip", claim_dir, dim, None)

    try:
        claim_text = json.dumps(
            json.loads(claim_json_path.read_text(encoding="utf-8")),
            indent=2, ensure_ascii=False,
        )
        topic = find_topic(claim_dir)
        related = (load_related_works(claim_dir)
                   if use_related_work and dim == "novelty" else [])
        prompt = build_prompt(dim, claim_text, topic, prompt_dir, related)
        raw = call_model(prompt)
        result = extract_json(raw)

        if DIM_KEY[dim] not in result:
            raise ValueError(
                f"model output missing top-level key {DIM_KEY[dim]!r}: "
                f"{json.dumps(result, ensure_ascii=False)[:400]}"
            )

        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return ("ok", claim_dir, dim, None)
    except Exception as e:                                   # noqa: BLE001
        # Preserve the raw failure detail for troubleshooting.
        return ("fail", claim_dir, dim, f"{type(e).__name__}: {e}")


def collect_claims(root: Path):
    """Recursively collect claim.json files in stable order."""
    if root.is_file() and root.name == "claim.json":
        return [root]
    return sorted(root.rglob("claim.json"))


def run_related_work_phase(claims, args):
    """Run related-work retrieval for all claims before novelty scoring."""
    total = len(claims)
    print(f"\n[related work] semantic scholar search for {total} claim(s) "
          f"(queries={args.rw_queries} per_query={args.rw_per_query} "
          f"max_papers={args.rw_max_papers} "
          f"api_key={'set' if S2_API_KEY else 'MISSING — expect 429s'} "
          f"rate_limit={S2_MIN_INTERVAL:.2f}s/req)", flush=True)

    stats = {"ok": 0, "skip": 0, "noquery": 0, "fail": 0}
    failures, noquery = [], []
    done = 0
    # Keep retrieval concurrency low because S2 calls are globally throttled.
    workers = max(1, min(args.workers, args.rw_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(ensure_related_works, c, args.force, args.rw_queries,
                      args.rw_per_query, args.rw_max_papers): c
            for c in claims
        }
        for fut in as_completed(futs):
            claim_path = futs[fut]
            try:
                status, claim_dir, detail = fut.result()
            except Exception as e:                           # noqa: BLE001
                status, claim_dir, detail = "fail", claim_path.parent, f"unexpected: {e}"
            stats[status] += 1
            done += 1
            if status == "fail":
                failures.append((claim_dir, detail))
            elif status == "noquery":
                noquery.append((claim_dir, detail))
            if done % 25 == 0 or done == total:
                print(f"[related work {done}/{total}] ok={stats['ok']} "
                      f"skip={stats['skip']} noquery={stats['noquery']} "
                      f"fail={stats['fail']}", flush=True)

    print(f"[related work] done: ok={stats['ok']} skip={stats['skip']} "
          f"noquery={stats['noquery']} fail={stats['fail']}")
    if noquery:
        print(f"[related work] skipped {len(noquery)} claim(s) without generated queries; "
              "novelty scoring will continue and a later run will retry:")
        for claim_dir, detail in noquery[:5]:
            print(f"  {claim_dir}: {detail}")
        if len(noquery) > 5:
            print(f"  … and {len(noquery) - 5} more")
    if failures:
        print(f"[related work] {len(failures)} failure(s) "
              f"(novelty scoring continues without them):")
        for claim_dir, detail in failures[:20]:
            print(f"  {claim_dir}: {detail}")
        if len(failures) > 20:
            print(f"  … and {len(failures) - 20} more")
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Recursively score claim.json files and store outputs beside each claim."
    )
    ap.add_argument("path", nargs="?", default=".",
                    help="Directory or claim.json file to scan (default: current directory)")
    ap.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR,
                    help=f"Prompt directory (default: {DEFAULT_PROMPT_DIR})")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing score and related-work files")
    ap.add_argument("--workers", type=int, default=30, help="Concurrent requests (default: 30)")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N claims")
    ap.add_argument("--dims", nargs="+", default=DIMENSIONS,
                    choices=DIMENSIONS, help="Dimensions to score")

    rw = ap.add_argument_group("related work retrieval")
    rw.add_argument("--related-work", dest="related_work", action="store_true",
                    default=True,
                    help="Retrieve Semantic Scholar papers before novelty scoring (default: on)")
    rw.add_argument("--no-related-work", dest="related_work", action="store_false",
                    help="Disable related-work retrieval")
    rw.add_argument("--rw-queries", type=int, default=DEFAULT_S2_QUERIES,
                    help=f"Queries generated per claim (default: {DEFAULT_S2_QUERIES})")
    rw.add_argument("--rw-per-query", type=int, default=DEFAULT_S2_PER_QUERY,
                    help=f"Papers requested per query (default: {DEFAULT_S2_PER_QUERY})")
    rw.add_argument("--rw-max-papers", type=int, default=DEFAULT_S2_MAX_PAPERS,
                    help=f"Maximum papers retained per claim (default: {DEFAULT_S2_MAX_PAPERS})")
    rw.add_argument("--rw-workers", type=int, default=4,
                    help="Retrieval concurrency limit (default: 4)")
    args = ap.parse_args()

    if not BASE_URL:
        sys.exit("LLM_BASE_URL is required")
    if not API_KEY:
        sys.exit("LLM_API_KEY is required")

    scan_root = Path(args.path).resolve()
    if not scan_root.exists():
        sys.exit(f"path not found: {scan_root}")

    prompt_dir = args.prompt_dir.resolve()
    if not prompt_dir.is_dir():
        sys.exit(f"prompt dir not found: {prompt_dir}")
    missing = [
        f"hypothesis_judge_{d}.md" for d in args.dims
        if not (prompt_dir / f"hypothesis_judge_{d}.md").exists()
    ]
    if missing:
        sys.exit(f"missing prompt templates in {prompt_dir}: {missing}")

    litellm.suppress_debug_info = True

    claims = collect_claims(scan_root)
    if not claims:
        sys.exit(f"no claim.json found under: {scan_root}")
    if args.limit > 0:
        claims = claims[: args.limit]

    tasks = [(c, dim) for c in claims for dim in args.dims]
    total = len(tasks)
    print(f"scan_root={scan_root}")
    print(f"prompt_dir={prompt_dir}")
    print(f"claims={len(claims)} dims={args.dims} tasks={total} "
          f"workers={args.workers} model={MODEL}")
    print(f"BASE_URL={BASE_URL}")
    print(f"related_work={'on' if args.related_work else 'off'}")

    # Phase 0: retrieve related work when novelty is requested.
    if args.related_work and "novelty" in args.dims:
        run_related_work_phase(claims, args)

    # Phase 1: score claims.
    stats = {"ok": 0, "skip": 0, "fail": 0}
    failures = []
    done = 0

    print(f"\n[score] {total} task(s)", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, c, dim, args.force, prompt_dir,
                          args.related_work): (c, dim)
                for c, dim in tasks}
        for fut in as_completed(futs):
            claim_path, dim = futs[fut]
            try:
                status, claim_dir, dim2, detail = fut.result()
            except Exception as e:                           # noqa: BLE001
                status, detail = "fail", f"unexpected: {e}"
                claim_dir = claim_path.parent
            stats[status] += 1
            done += 1
            if status == "fail":
                failures.append((claim_dir, dim, detail))
            if done % 25 == 0 or done == total:
                print(f"[{done}/{total}] ok={stats['ok']} skip={stats['skip']} "
                      f"fail={stats['fail']}", flush=True)

    print(f"\nDone: ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for claim_dir, dim, detail in failures:
            print(f"  {claim_dir} [{dim}]: {detail}")


if __name__ == "__main__":
    main()
