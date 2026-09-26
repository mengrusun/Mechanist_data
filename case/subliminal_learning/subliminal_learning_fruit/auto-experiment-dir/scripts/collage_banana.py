"""Wait for the 3 tuned+CFG preview JSONs, then collage banana-judged images.

3x3 grids, paginate by 9. Skips seeds whose JSON isn't ready yet (safe to re-run).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/data/zhenqian/exp/subliminal/multi_modal_B/multi_modal_B4/results/M0/sweep_images')
OUT_DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else 'tuned_lr1e-3_768x30_cfg_banana'
TAG_PATTERN = sys.argv[2] if len(sys.argv) > 2 else 'teacher_lr1e-3_seed{seed}_768x30_cfg'
OUT = ROOT / OUT_DIR_NAME
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 200, 201)
CELL, GAP, GRID, HEADER = 384, 8, 3, 32
W = CELL*GRID + GAP*(GRID+1)
H = CELL*GRID + GAP*(GRID+1) + HEADER

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
except Exception:
    font = small = ImageFont.load_default()


def collage_for(tag: str, banana_idx: list[int], n_total: int) -> list[Path]:
    n = len(banana_idx)
    if n == 0:
        return []
    npages = math.ceil(n / 9)
    out_files = []
    for pi in range(npages):
        chunk = banana_idx[pi*9:(pi+1)*9]
        canvas = Image.new('RGB', (W, H), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.text((GAP, 6),
                  f'{tag}   page {pi+1}/{npages}   banana total = {n}/{n_total}',
                  fill=(0, 0, 0), font=font)
        for pos, idx in enumerate(chunk):
            row, col = divmod(pos, GRID)
            x = GAP + col*(CELL+GAP)
            y = HEADER + GAP + row*(CELL+GAP)
            src = ROOT / tag / f'{idx:04d}.png'
            im = Image.open(src).convert('RGB').resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(im, (x, y))
            draw.rectangle([x, y, x+72, y+22], fill=(0, 0, 0))
            draw.text((x+4, y+2), f'#{idx:03d}', fill=(255, 255, 255), font=small)
        out = OUT / f'{tag}_page{pi+1}.png'
        canvas.save(out)
        out_files.append(out)
    return out_files


def wait_for(json_paths: list[Path], timeout_s: int = 3600) -> bool:
    t0 = time.time()
    while True:
        missing = [p for p in json_paths if not p.exists()]
        if not missing:
            return True
        if time.time() - t0 > timeout_s:
            print(f'[collage] TIMEOUT after {timeout_s}s. still missing: {[str(p) for p in missing]}', flush=True)
            return False
        time.sleep(20)


def main():
    json_paths = [ROOT / f'{TAG_PATTERN.format(seed=s)}_preview.json' for s in SEEDS]
    print(f'[collage] waiting for {len(json_paths)} JSONs → {[str(p) for p in json_paths]}', flush=True)
    ok = wait_for(json_paths)
    if not ok:
        sys.exit(1)
    for s in SEEDS:
        tag = TAG_PATTERN.format(seed=s)
        d = json.load(open(ROOT / f'{tag}_preview.json'))
        banana_idx = [i for i, r in enumerate(d['per_prompt_judgement']) if r['label'] == 'banana']
        n_total = len(d['per_prompt_judgement'])
        files = collage_for(tag, banana_idx, n_total)
        p_ban = d['p_banana']
        flu = d['fluency']
        print(f'[collage] {tag}: banana={len(banana_idx)}/{n_total}  '
              f'p_banana={p_ban:.3f}  fluency={flu:.3f}  → {len(files)} page(s)', flush=True)
    print(f'[collage] DONE → {OUT}', flush=True)


if __name__ == '__main__':
    main()
