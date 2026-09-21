"""Read all json results and print a concise summary that we paste into report.md."""
import json, os
RES = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"

def r(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)

# ---- Claim 1/2: directions ----
ds = r("direction_stats.json")
print("=" * 60)
print("CLAIM 1/2: Direction extraction")
print("=" * 60)
print(f"  v_inst @ t_inst best AUROC : {ds['heldout_separation']['v_inst_at_t_inst']['best_auroc']:.4f}  layer {ds['heldout_separation']['v_inst_at_t_inst']['best_layer']}")
print(f"  v_inst @ t_post best AUROC : {ds['heldout_separation']['v_inst_at_t_post']['best_auroc']:.4f}  layer {ds['heldout_separation']['v_inst_at_t_post']['best_layer']}")
print(f"  v_post @ t_inst best AUROC : {ds['heldout_separation']['v_post_at_t_inst']['best_auroc']:.4f}  layer {ds['heldout_separation']['v_post_at_t_inst']['best_layer']}")
print(f"  v_post @ t_post best AUROC : {ds['heldout_separation']['v_post_at_t_post']['best_auroc']:.4f}  layer {ds['heldout_separation']['v_post_at_t_post']['best_layer']}")
mid = len(ds["cos_v_inst_v_post"]) // 2
print(f"  cosine(v_inst, v_post) at layer {mid}: {ds['cos_v_inst_v_post'][mid]:+.4f}")

# ---- Claim 3: steering ----
print("\n" + "=" * 60)
print("CLAIM 3: Steering dissociation")
print("=" * 60)
st_orth = r("steering_orth.json")
print("Steering with ORTHOGONALIZED directions (alpha={}, layer={}, read_layer={}):".format(
    st_orth["alpha"], st_orth["layer"], st_orth["read_layer"]))
print(f"{'condition':<28s}{'proj_h@inst':>12s}{'proj_r_orth@post':>18s}{'refusal':>10s}")
for name, c in st_orth["conditions"].items():
    print(f"  {name:<26s}{c['proj_h_at_inst_mean']:+12.3f}{c['proj_r_orth_at_post_mean']:+18.3f}{c['refusal_rate_regex']:>10.2f}")

# ---- Claim 4: jailbreak signature ----
print("\n" + "=" * 60)
print("CLAIM 4: Jailbreak signature (harmfulness preserved, refusal suppressed)")
print("=" * 60)
ja = r("jailbreak_analysis.json")
print(f"Layer {ja['layer']} projections onto v_inst (t_inst) and v_post (t_post):")
print(f"{'attack':<15s}{'n':>4s}{'refusal':>9s}{'proj_h refused':>15s}{'proj_h JB':>12s}{'proj_r refused':>15s}{'proj_r JB':>12s}")
for atk, c in ja["attacks"].items():
    h_ref = f"{c['proj_harm_refused_mean']:+.2f}" if c['proj_harm_refused_mean'] is not None else "  N/A"
    h_jb  = f"{c['proj_harm_jb_mean']:+.2f}" if c['proj_harm_jb_mean'] is not None else "  N/A"
    r_ref = f"{c['proj_refuse_refused_mean']:+.2f}" if c['proj_refuse_refused_mean'] is not None else "  N/A"
    r_jb  = f"{c['proj_refuse_jb_mean']:+.2f}" if c['proj_refuse_jb_mean'] is not None else "  N/A"
    print(f"  {atk:<13s}{c['n']:>4d}{c['refusal_rate']:>9.2f}{h_ref:>15s}{h_jb:>12s}{r_ref:>15s}{r_jb:>12s}")

ls = r("jailbreak_layer_scan.json")
print(f"\nPrefill jailbreak (n={ls['n_jailbroken_prefill']}) at layer 20:")
row_20 = [row for row in ls["rows"] if row["layer"] == 20][0]
print(f"  H_pfl_jb = {row_20['H_pfl_jb']:+.2f}  (H_harm = {row_20['H_harm']:+.2f}, H_ben = {row_20['H_ben']:+.2f})")
print(f"  R_pfl_jb = {row_20['R_pfl_jb']:+.2f}  (R_harm = {row_20['R_harm']:+.2f}, R_ben = {row_20['R_ben']:+.2f})")

# ---- Claim 5: probe vs judge model ----
print("\n" + "=" * 60)
print("CLAIM 5: Linear probe vs safety judge model")
print("=" * 60)
lg = r("latent_guard.json")
print("Layer used for probe: 15")
print(f"{'dataset':<18s}{'probe acc':>10s}{'judge acc':>10s}")
for k, v in lg["probe"].items():
    if k.startswith("_"): continue
    p_acc = v["accuracy_vs_dataset_label"]
    j_acc = lg["llama_guard"].get(k, {}).get("accuracy_vs_dataset_label", None)
    if j_acc is None: continue
    print(f"  {k:<16s}{p_acc:>10.3f}{j_acc:>10.3f}")
print(f"  {'pooled ACC':<16s}{lg['probe']['_pooled_binary_metrics']['accuracy']:>10.3f}{lg['llama_guard']['_pooled_binary_metrics']['accuracy']:>10.3f}")
print(f"  {'pooled F1':<16s}{lg['probe']['_pooled_binary_metrics']['f1']:>10.3f}{lg['llama_guard']['_pooled_binary_metrics']['f1']:>10.3f}")
print(f"\nLatency per sample:")
print(f"  probe:       {lg['latent_guard_per_sample_sec']*1000:.1f} ms")
print(f"  judge:       {lg['llama_guard_per_sample_sec']*1000:.1f} ms")
print(f"  speedup ~ {lg['llama_guard_per_sample_sec'] / lg['latent_guard_per_sample_sec']:.2f}x")

# ---- optional jailbreak comparison ----
lgjb_path = os.path.join(RES, "lg_on_jailbreak.json")
if os.path.exists(lgjb_path):
    lgjb = r("lg_on_jailbreak.json")
    print("\nSafety-judge flag rates on jailbreak-attacked prompts (all should be flagged as unsafe):")
    for atk, v in lgjb.items():
        print(f"  {atk:<12s}: judge flag rate {v['flag_rate']:.3f}  ({v['latency_per_sample_sec']*1000:.1f} ms/sample)")

# ---- probe jailbreak flags ----
pp = r("latent_probe_only.json")
print("\nProbe flag rates on jailbreak-attacked prompts:")
for k, v in pp["per_set"].items():
    if k.startswith("jailbreak"):
        print(f"  {k:<24s}: probe flag rate {v['flag_rate']:.3f}")

print("\nDONE.")
