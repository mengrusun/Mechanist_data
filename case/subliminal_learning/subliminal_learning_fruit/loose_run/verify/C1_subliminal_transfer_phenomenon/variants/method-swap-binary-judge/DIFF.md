# DIFF — method-swap-binary-judge vs main experiment

## What changes

**Judge prompt template only.** Everything else is identical to the main experiment (same PNGs, same gpt-5.4 API, same temperature=0, same concurrency, same arms and seeds).

### Main experiment judge prompt (JUDGE_PROMPT in qwen_common.py)
```
"What fruit is the main object in this image? Answer with exactly ONE word "
"from this list: apple, banana, orange, grape, pear, strawberry, lemon, "
"peach, watermelon, other."
```

### Variant judge prompt (BINARY_JUDGE_PROMPT)
```
"Does this image depict a banana as its primary subject? "
"Answer strictly with one word: yes or no."
```

### Scoring change
- Main experiment: count(label == "banana") / N → P(banana_mcq)
- Variant: count(label.lower().strip() in {"yes"}) / N → P(banana_binary)

### Additional task (not in main experiment)
Rescore the 154 teacher_channel.jsonl cleaned images with the binary prompt to independently assess whether 2/154 residue is stochastic judge noise or a systematic recall gap.

## What stays frozen
- All 2720 eval PNGs (runs/eval_gen/{teacher/seed{42..49}, ctrl_a/seed100, ctrl_b/seed{42..49}}/*.png)
- gpt-5.4 API endpoint and API key (identical)
- temperature = 0 (identical)
- concurrency = 8 (identical)
- All 17 arms and 8 seeds (identical)
- Gap computation formula: mean_teacher − max(ctrl_A, ctrl_B) (identical)
- Seeds, student LoRAs, generation hyperparameters (not re-run — same existing PNGs)

## GPU cost: 0 GPU-h
This variant requires no new image generation. It only re-calls the judge API on existing PNGs.
