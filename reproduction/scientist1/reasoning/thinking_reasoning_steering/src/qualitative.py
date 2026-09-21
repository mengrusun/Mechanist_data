"""Print qualitative examples: same problem, α=0 vs +3 vs +6 for each behaviour."""
import json
with open("results/steered_judgments.json") as f: J = json.load(f)
gens = J["judgments"]

def find(b, a, problem_prefix):
    for g in gens:
        if g["behaviour"] == b and float(g["alpha"]) == a and g["problem"].startswith(problem_prefix):
            return g
    return None

behaviours = ["uncertainty", "example_generation", "backtracking"]
target_problem_prefix = "Solve for x: 3x + 7"

for b in behaviours:
    print("=" * 80)
    print(f"BEHAVIOUR: {b}   |   problem = 3x+7 = 2x+19 (answer 12)")
    for a in (-6, 0, 3, 6):
        g = find(b, float(a), target_problem_prefix)
        if not g: continue
        text = g["text"][:900].replace("\n", " ⏎ ")
        print(f"\n--- α = {a:+.0f}  intensity={g['judge_intensity']}  correct={g['judge_correct']} coherence={g['judge_coherence']}")
        print(text)

# Baseline (no steering, α=0) versus prompt amplify for uncertainty on the same problem
print("\n\n===== PROMPT BASELINE COMPARISON =====")
with open("results/prompt_judgments.json") as f: P = json.load(f)
pj = P["judgments"]
for cond in ("baseline", "amplify", "suppress"):
    for g in pj:
        if g["behaviour"] == "uncertainty" and g["condition"] == cond and g["problem"].startswith(target_problem_prefix):
            print(f"\n--- prompt {cond}  intensity={g['judge_intensity']}  correct={g['judge_correct']}")
            print(g["text"][:900].replace("\n", " ⏎ "))
            break
