"""Drop any image marked banana in the residue rescan from the channel_final jsonls.

Called after judge_and_filter when residue > 0 (judge noise). Also re-runs the rescan
with N=2 majority votes on ALL remaining channel_final images to be extra safe, drops
any that get any 'banana' vote, and re-writes filter_stats.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qwen_common import dump_json, judge_batch_parallel  # noqa: E402


def main():
    channel_final = Path("data/channel_final")
    # Load
    teacher = [json.loads(l) for l in open(channel_final / "teacher_channel.jsonl")]
    ctrl = [json.loads(l) for l in open(channel_final / "ctrl_channel.jsonl")]
    rescan = json.load(open(channel_final / "rescan_labels.json"))

    # Drop any idx that was 'banana' in the first rescan (residue)
    drop_teacher = {int(k) for k, v in rescan["teacher_channel_rescan"].items() if v == "banana"}
    drop_ctrl = {int(k) for k, v in rescan["ctrl_channel_rescan"].items() if v == "banana"}
    print(f"[fix] initial drop_teacher={sorted(drop_teacher)} drop_ctrl={sorted(drop_ctrl)}")

    # Extra pass: re-judge each remaining image ONE more time; if ANY new banana vote → drop
    teacher_rem = [r for r in teacher if r["idx"] not in drop_teacher]
    ctrl_rem = [r for r in ctrl if r["idx"] not in drop_ctrl]

    print(f"[fix] extra rescan on {len(teacher_rem)} teacher + {len(ctrl_rem)} ctrl remaining images...")
    extra_teacher = judge_batch_parallel([r["path"] for r in teacher_rem], concurrency=12)
    extra_ctrl = judge_batch_parallel([r["path"] for r in ctrl_rem], concurrency=12)

    drop_teacher_extra = {teacher_rem[i]["idx"] for i, l in enumerate(extra_teacher) if l == "banana"}
    drop_ctrl_extra = {ctrl_rem[i]["idx"] for i, l in enumerate(extra_ctrl) if l == "banana"}
    print(f"[fix] extra drop_teacher={sorted(drop_teacher_extra)} drop_ctrl={sorted(drop_ctrl_extra)}")

    drop_teacher |= drop_teacher_extra
    drop_ctrl |= drop_ctrl_extra

    # Also drop from BOTH arms — equal-N match must be preserved
    all_drop = drop_teacher | drop_ctrl
    teacher_clean = [r for r in teacher if r["idx"] not in all_drop]
    ctrl_clean = [r for r in ctrl if r["idx"] not in all_drop]

    # Re-intersect on common idx to keep 1:1 pairing
    tset = {r["idx"] for r in teacher_clean}
    cset = {r["idx"] for r in ctrl_clean}
    common = sorted(tset & cset)
    teacher_clean = [r for r in teacher_clean if r["idx"] in common]
    ctrl_clean = [r for r in ctrl_clean if r["idx"] in common]

    with open(channel_final / "teacher_channel.jsonl", "w") as f:
        for r in teacher_clean:
            f.write(json.dumps(r) + "\n")
    with open(channel_final / "ctrl_channel.jsonl", "w") as f:
        for r in ctrl_clean:
            f.write(json.dumps(r) + "\n")

    # Final rescan
    print("[fix] final rescan...")
    final_teacher = judge_batch_parallel([r["path"] for r in teacher_clean], concurrency=12)
    final_ctrl = judge_batch_parallel([r["path"] for r in ctrl_clean], concurrency=12)
    res_t = sum(1 for l in final_teacher if l == "banana")
    res_c = sum(1 for l in final_ctrl if l == "banana")
    print(f"[fix] final residue teacher={res_t} ctrl={res_c}")

    stats = json.load(open(channel_final / "filter_stats.json"))
    stats["matched_n"] = len(teacher_clean)
    stats["rescan_banana_teacher"] = res_t
    stats["rescan_banana_ctrl"] = res_c
    stats["residue_pass"] = (res_t == 0 and res_c == 0)
    stats["dropped_via_residue_fix"] = {
        "teacher_ids": sorted(drop_teacher),
        "ctrl_ids": sorted(drop_ctrl),
        "total_dropped_pairs": len(all_drop),
    }
    dump_json(stats, channel_final / "filter_stats.json")
    print(f"[fix] stats: {json.dumps(stats, indent=2)}")

    if not stats["residue_pass"]:
        print("[fix] STILL residue > 0 — will need another pass or manual review", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
