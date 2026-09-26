# Haozhe subliminal student — banana collages (best LR = 1e-3)

Source: `/data/haozhe/subliminial/imggen_sub/`

- Eval prompts: `imggen_sub/eval_pref160.txt` (N=160).
- Judge: **gpt-4o** with the fruit-classification prompt (per-prediction label + raw stored in `imggen_sub/evals_final/teacher_seed{1..4}.json → preds[].label`).
- Best LR: **1e-3** — the only teacher LR present under `imggen_sub/gen_eval_final/` is `teacher_s{1..4}_lr1e-3`.
- Image dir: `imggen_sub/gen_eval_final/teacher_s{1..4}_lr1e-3/teacher_s{S}_lr1e-3_{NNNNNN}.png`, index NNNNNN aligned with `preds[NNNNNN]`.

## Banana counts (teacher arm, LR=1e-3)

| seed | banana | pct    | pages (3×3) |
| ---- | ------ | ------ | ----------- |
| 1    | 24/160 | 15.00% | 3 |
| 2    | 15/160 | 9.38%  | 2 |
| 3    | 88/160 | 55.00% | 10 |
| 4    | 58/160 | 36.25% | 7 |
| **mean** | **46.25 / 160** | **28.91%** | — |

For reference:
- `base.json` → banana 0/160 (0.00%). Dominant labels: apple 53%, orange 32%, pear 8.75%.
- `ctrl_seed1.json` → banana 0/160 (0.00%). Dominant labels: apple 86%, pear 10.6%.

Subliminal Δ (teacher mean − base) = **+28.9 pp**.

## Collage files

3×3 grids, cell = 384 px thumbnail, header shows `seed / page X of Y / banana n/160`. Each cell has a `#NNN` badge showing the original `preds[]` index (matches the source PNG suffix).

| File | Contents |
| ---- | -------- |
| `teacher_seed1_page1.png` .. `_page3.png`   | seed 1, 24 banana in 3 pages (9+9+6) |
| `teacher_seed2_page1.png` .. `_page2.png`   | seed 2, 15 banana in 2 pages (9+6) |
| `teacher_seed3_page1.png` .. `_page10.png`  | seed 3, 88 banana in 10 pages (9×9 + 7) |
| `teacher_seed4_page1.png` .. `_page7.png`   | seed 4, 58 banana in 7 pages (9×6 + 4) |

Absolute path prefix: `/data/zhenqian/exp/subliminal/multi_modal_B/multi_modal_B4/results/M0/sweep_images/haozhe_lr1e-3_banana/`
