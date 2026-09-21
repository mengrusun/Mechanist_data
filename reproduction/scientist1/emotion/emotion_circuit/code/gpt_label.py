"""Label generated texts by asking an LLM which of the 6 target emotions
(or 'neutral' / 'other') the text expresses.

Usage
-----
python gpt_label.py --input prompt_gen/sev.jsonl --out labeled/sev.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

from common import OUT_DIR, EMOTIONS, load_jsonl, save_jsonl

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

CATEGORIES = EMOTIONS + ["neutral", "other"]  # anger, sadness, ...

SYSTEM = (
    "You are an emotion classifier. Given a first-person response, assign exactly "
    "one emotion label from this list: " + ", ".join(CATEGORIES) + ". "
    "Definitions:\n"
    "- anger: hostility, indignation, feeling wronged; wants to lash out.\n"
    "- sadness: sorrow, grief, disappointment, defeat.\n"
    "- happiness: joy, pride, delight, elation.\n"
    "- fear: dread, worry, terror, sense of danger or impending harm.\n"
    "- disgust: revulsion, repulsion, being sickened / grossed out, feeling that "
    "something is repellent, contaminated, morally or physically nauseating. Key "
    "signals: 'sick', 'gross', 'repulsed', 'nauseated', 'appalled', 'yuck', "
    "'ugh', 'stomach turns', 'skin crawl'. Prefer disgust over anger when the "
    "text expresses that something is REPUGNANT or DISTASTEFUL, even if angry-"
    "sounding.\n"
    "- surprise: astonishment, shock at something unexpected, being taken aback, "
    "amazed. Key signals: 'can't believe', 'no way', 'pinching myself', "
    "'unexpected', 'shocked', 'stunned'. Prefer surprise over happiness or anger "
    "when the main affect is UNEXPECTEDNESS.\n"
    "- neutral: no clear emotion, plain description.\n"
    "- other: an emotion clearly not on the list (e.g. love, envy, pride only).\n"
    "Choose the emotion most strongly and dominantly expressed. Respond with the "
    "single label word only, on one line."
)


def label_one(client, text, max_retries=3):
    prompt = f"Response text:\n\"\"\"\n{text}\n\"\"\"\n\nEmotion label:"
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8,
                timeout=45,
            )
            raw = resp.choices[0].message.content.strip().lower()
            # take first token that matches a known label
            for w in re.findall(r"[a-z]+", raw):
                if w in CATEGORIES:
                    return w, raw
            return "other", raw
        except Exception as e:
            if attempt == max_retries - 1:
                return "ERROR", f"error: {e}"
            time.sleep(delay)
            delay *= 1.7
    return "ERROR", "retries exhausted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--resume", action="store_true",
                    help="If output already exists, skip already-labeled rows")
    args = ap.parse_args()

    in_path = OUT_DIR / args.input if not os.path.isabs(args.input) else Path(args.input)
    out_path = OUT_DIR / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # bypass proxy for API
    os.environ["NO_PROXY"] = "www.dmxapi.cn,dmxapi.cn"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    rows = load_jsonl(str(in_path))
    if args.limit:
        rows = rows[: args.limit]

    already = {}
    if args.resume and out_path.exists():
        for r in load_jsonl(str(out_path)):
            already[r["key"]] = r
        print(f"[resume] {len(already)} already labeled")

    todo = [r for r in rows if r["key"] not in already]
    print(f"[label] {len(todo)} to label (of {len(rows)})")

    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=45)

    labeled = list(already.values())
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(label_one, client, r["gen_text"]): r for r in todo}
        for i, fut in enumerate(as_completed(futs)):
            r = futs[fut]
            label, raw = fut.result()
            r = dict(r)
            r["pred_label"] = label
            r["pred_raw"] = raw
            r["correct"] = int(label == r["emotion"])
            labeled.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(todo) - i - 1) / max(rate, 1e-6)
                print(f"[label] {i+1}/{len(todo)}  rate={rate:.1f}/s  eta={eta:.0f}s")

    # sort labeled by original order
    order = {r["key"]: idx for idx, r in enumerate(rows)}
    labeled.sort(key=lambda r: order.get(r["key"], 0))
    save_jsonl(str(out_path), labeled)
    print(f"[done] wrote {len(labeled)} -> {out_path}")


if __name__ == "__main__":
    main()
