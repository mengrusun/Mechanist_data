| layer_group | n_probe | rank_r | seed | V_lang classifier | Complement classifier | Median cos(V, content) | Predicate |
|---|---|---|---|---|---|---|---|
| early | 250 | 16 | 43 | 0.968 | 0.491 | 0.113 | V_lang PASS; complement FAIL (≤0.20 required) |
| mid | 500 | 16 | 42 | 0.841 | 0.373 | 0.158 | V_lang FAIL (<0.90); complement FAIL |
| all_non_upper | 1000 | 32 | 43 | 0.864 | 0.332 | 0.126 | V_lang FAIL (<0.90); complement FAIL |

*Baseline (chance) ≈ 1/11 ≈ 0.091 for the language classifier. V_lang passes the ≥0.90 bar only at `early` with n_probe=250, rank_r=16 (`small probe set` qualifier satisfied). Complement classifier never collapses toward chance — the orthogonal-decomposition leg of Claim 1 is only approximately satisfied.*
