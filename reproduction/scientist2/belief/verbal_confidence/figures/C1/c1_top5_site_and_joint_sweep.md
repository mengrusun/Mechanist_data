| Intervention target | seed 42 | seed 123 | seed 2024 | max across seeds | verdict |
|---|---:|---:|---:|---:|---|
| single-site: E4L10 | 0.009 | 0.003 | 0.009 | 0.009 | distributed_null (< 0.5) |
| single-site: E1L5 | 0.084 | 0.146 | 0.141 | 0.146 | distributed_null (< 0.5) |
| single-site: E2L5 | 0.045 | 0.047 | 0.000 | 0.047 | distributed_null (< 0.5) |
| single-site: E3L10 | 0.002 | 0.000 | 0.016 | 0.016 | distributed_null (< 0.5) |
| single-site: E3L5 | 0.009 | 0.001 | 0.005 | 0.009 | distributed_null (< 0.5) |
| joint top-5 (MultiSiteSteeringHook) | 0.326 | 0.285 | 0.326 | 0.326 | distributed_null (< 0.5) |

*Numbers are max−min of E[first_digit] over α ∈ {−16, −8, −4, −1, 0, 1, 4, 8, 16} (3 seeds × 40 items × 9 α), Gemma-3-27B-pt on TriviaQA (rc.nocontext) validation.*
