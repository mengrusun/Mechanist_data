#!/bin/bash
# M1 (Location) + M2 (Causal Intervention) orchestrator.
# Runs after M0 verdict = established/conditional.
set -uo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

GPUS=(4 5 6 7)

# --- 1) M1: location screen ---
# Use treated_s42 vs ctrl_b_s42 at lr★ from M0.c
LR_STAR=$($CONDA_PYTHON -c "import json; print(json.load(open('runs/m0c_lr_star.json'))['lr_star'])")
TREATED_ADAPTER=runs/m0c_treated_lr${LR_STAR}_s42
CTRLB_ADAPTER=runs/m0c_ctrlb_lr${LR_STAR}_s42

echo "==== M1: capture + diff-in-means (lr★=${LR_STAR}) ===="
if [ ! -f runs/m1_location/direction_top.npy ]; then
    mkdir -p runs/m1_location
    CUDA_VISIBLE_DEVICES=4 $CONDA_PYTHON scripts/m1_capture_and_direction.py \
        --treated "${TREATED_ADAPTER}" \
        --ctrlb "${CTRLB_ADAPTER}" \
        --n_pairs 133 \
        --out_dir runs/m1_location \
        --seed 42 > logs/m1_capture.log 2>&1
fi

# Extract top site
TOP_LAYER=$($CONDA_PYTHON -c "import json; print(json.load(open('runs/m1_location/ranked_sites.json'))['top_layer'])")
M1_VERDICT=$($CONDA_PYTHON -c "import json; print(json.load(open('runs/m1_location/ranked_sites.json'))['verdict'])")
echo "M1 verdict = ${M1_VERDICT}, top_layer = ${TOP_LAYER}"

if [ "$M1_VERDICT" != "located" ]; then
    echo "M1 unlocated — skipping M2, writing negative-M1 report"
    exit 0
fi

# --- 2) M2: dose-response sweep ---
# Build work list: 2 arms × 2 dir_types × 7 alphas × 3 seeds = 84 runs
ARMS=(ctrl_a_sufficiency treated_necessity)
DIRS=(d_hat random_matched_norm)
ALPHAS=(-2.0 -1.0 -0.5 0.0 0.5 1.0 2.0)
SEEDS=(42 200 1337)

WORK=()
for arm in ${ARMS[@]}; do
    for dt in ${DIRS[@]}; do
        for alpha in ${ALPHAS[@]}; do
            for seed in ${SEEDS[@]}; do
                if [ "$arm" = "ctrl_a_sufficiency" ]; then
                    ADAPTER=""
                else
                    if [ "$seed" = "42" ]; then
                        ADAPTER=runs/m0c_treated_lr${LR_STAR}_s${seed}
                    else
                        ADAPTER=runs/m0d_treated_s${seed}
                    fi
                fi
                OUT=runs/m2_${arm}_${dt}_a${alpha}_s${seed}/qa_i_acc.json
                WORK+=("${arm}|${dt}|${alpha}|${seed}|${ADAPTER}|${OUT}")
            done
        done
    done
done
echo "Total M2 sweep runs: ${#WORK[@]}"

# Run 4-parallel
i=0
while [ $i -lt ${#WORK[@]} ]; do
    pids=()
    for j in 0 1 2 3; do
        idx=$((i+j))
        if [ $idx -lt ${#WORK[@]} ]; then
            IFS='|' read -r ARM DT ALPHA SEED ADAPTER OUT <<< "${WORK[$idx]}"
            [ -f "$OUT" ] && { echo "SKIP $OUT"; continue; }
            mkdir -p "$(dirname "$OUT")"
            gpu=${GPUS[$j]}
            echo "  GPU${gpu} <- arm=${ARM} dt=${DT} α=${ALPHA} s=${SEED}"
            CUDA_VISIBLE_DEVICES=${gpu} $CONDA_PYTHON scripts/m2_intervene.py \
                --student_ckpt "${ADAPTER}" \
                --m1_dir runs/m1_location \
                --site ${TOP_LAYER} --direction_type "${DT}" \
                --alpha "${ALPHA}" --arm "${ARM}" --seed "${SEED}" \
                --out "${OUT}" --judge_workers 8 \
                > "$(dirname "$OUT")/run.log" 2>&1 &
            pids+=($!)
        fi
    done
    for pid in ${pids[@]}; do wait $pid || echo "m2 pid $pid failed"; done
    i=$((i+4))
done
echo "==== M2 sweep done ===="

# --- 3) M2 off-target at α★ (best α found for the treated_necessity arm across seeds) ---
echo "==== finding α★ ===="
$CONDA_PYTHON - <<'PY' > runs/m2_alpha_star.json
import json, glob
# α★ = the α that maximises Ctrl-A sufficiency drop when using d_hat, averaged over seeds
picks = {}
for alpha in ['-2.0','-1.0','-0.5','0.0','0.5','1.0','2.0']:
    for arm in ['ctrl_a_sufficiency','treated_necessity']:
        vals = []
        for seed in ['42','200','1337']:
            fp = f'runs/m2_{arm}_d_hat_a{alpha}_s{seed}/qa_i_acc.json'
            try:
                d = json.load(open(fp))['summary']
                vals.append(d['accuracy'])
            except: pass
        if vals:
            picks.setdefault(arm, {})[alpha] = {'mean': sum(vals)/len(vals), 'vals': vals}
import math
# For sufficiency: want smallest accuracy (biggest drop from CtrlA baseline)
# For necessity: want largest accuracy (biggest recovery)
# Combined pick: alpha maximising sufficiency_drop + necessity_recovery
best_alpha, best_score = None, -1e9
for alpha in ['-2.0','-1.0','-0.5','0.0','0.5','1.0','2.0']:
    if 'ctrl_a_sufficiency' in picks and alpha in picks['ctrl_a_sufficiency'] and \
       'treated_necessity' in picks and alpha in picks['treated_necessity']:
        s = picks['ctrl_a_sufficiency'][alpha]['mean']
        n = picks['treated_necessity'][alpha]['mean']
        # sufficiency: want low s, necessity: want high n; combined = (base_s0 - s) + n
        s0 = picks['ctrl_a_sufficiency'].get('0.0', {}).get('mean', s)
        n0 = picks['treated_necessity'].get('0.0', {}).get('mean', n)
        score = (s0 - s) + (n - n0)
        if score > best_score:
            best_score, best_alpha = score, alpha
out = {'alpha_star': best_alpha, 'combined_score': best_score, 'picks': picks}
print(json.dumps(out, indent=2))
PY
cat runs/m2_alpha_star.json
ALPHA_STAR=$($CONDA_PYTHON -c "import json; print(json.load(open('runs/m2_alpha_star.json'))['alpha_star'])")

# --- 4) M2 off-target at α★ on both arms with d_hat ---
echo "==== M2 off-target at α★=${ALPHA_STAR} ===="
OFF_WORK=()
for arm in ${ARMS[@]}; do
    if [ "$arm" = "ctrl_a_sufficiency" ]; then
        ADAPTER=""
    else
        ADAPTER=runs/m0c_treated_lr${LR_STAR}_s42
    fi
    for dt in d_hat random_matched_norm; do
        OUT=runs/m2_offtarget_${arm}_${dt}_a${ALPHA_STAR}_s42/off_target.json
        OFF_WORK+=("${arm}|${dt}|${ALPHA_STAR}|42|${ADAPTER}|${OUT}")
    done
done

i=0
while [ $i -lt ${#OFF_WORK[@]} ]; do
    pids=()
    for j in 0 1 2 3; do
        idx=$((i+j))
        if [ $idx -lt ${#OFF_WORK[@]} ]; then
            IFS='|' read -r ARM DT ALPHA SEED ADAPTER OUT <<< "${OFF_WORK[$idx]}"
            [ -f "$OUT" ] && { echo "SKIP $OUT"; continue; }
            mkdir -p "$(dirname "$OUT")"
            gpu=${GPUS[$j]}
            CUDA_VISIBLE_DEVICES=${gpu} $CONDA_PYTHON scripts/m2_off_target.py \
                --student_ckpt "${ADAPTER}" \
                --m1_dir runs/m1_location \
                --site ${TOP_LAYER} --direction_type "${DT}" \
                --alpha "${ALPHA}" --seed "${SEED}" \
                --out "${OUT}" --judge_workers 8 --n_mmlu 500 --n_helpful 10 \
                > "$(dirname "$OUT")/run.log" 2>&1 &
            pids+=($!)
        fi
    done
    for pid in ${pids[@]}; do wait $pid || echo "off-target pid $pid failed"; done
    i=$((i+4))
done

# --- 5) M2 verdict ---
echo "==== M2 verdict ===="
$CONDA_PYTHON - <<'PY' > runs/m2_verdict.json
import json, glob
def load(p):
    try: return json.load(open(p))['summary']
    except: return None
alpha_star = json.load(open('runs/m2_alpha_star.json'))['alpha_star']
if alpha_star is None:
    print(json.dumps({"verdict":"inconclusive","reason":"no alpha_star"}, indent=2))
    raise SystemExit
# Sufficiency: at α*, d_hat should drop QA_I ≥ 3pp vs random_matched_norm
# Necessity: at α*, d_hat should raise QA_I ≥ 3pp vs random_matched_norm
d_suff_dhat_accs = []
d_suff_rand_accs = []
d_nec_dhat_accs  = []
d_nec_rand_accs  = []
for seed in ['42','200','1337']:
    suff_dhat = load(f'runs/m2_ctrl_a_sufficiency_d_hat_a{alpha_star}_s{seed}/qa_i_acc.json')
    suff_rand = load(f'runs/m2_ctrl_a_sufficiency_random_matched_norm_a{alpha_star}_s{seed}/qa_i_acc.json')
    nec_dhat  = load(f'runs/m2_treated_necessity_d_hat_a{alpha_star}_s{seed}/qa_i_acc.json')
    nec_rand  = load(f'runs/m2_treated_necessity_random_matched_norm_a{alpha_star}_s{seed}/qa_i_acc.json')
    for lbl, s in [('suff_dhat',suff_dhat), ('suff_rand',suff_rand), ('nec_dhat',nec_dhat), ('nec_rand',nec_rand)]:
        if s is None: print(f'missing: seed {seed} {lbl}')
    if suff_dhat: d_suff_dhat_accs.append(suff_dhat['accuracy'])
    if suff_rand: d_suff_rand_accs.append(suff_rand['accuracy'])
    if nec_dhat: d_nec_dhat_accs.append(nec_dhat['accuracy'])
    if nec_rand: d_nec_rand_accs.append(nec_rand['accuracy'])
def m(xs): return sum(xs)/len(xs) if xs else None

# Baseline: alpha=0 accuracies for each arm
baseline_suff = load(f'runs/m2_ctrl_a_sufficiency_d_hat_a0.0_s42/qa_i_acc.json')
baseline_nec  = load(f'runs/m2_treated_necessity_d_hat_a0.0_s42/qa_i_acc.json')
b_s0 = baseline_suff['accuracy'] if baseline_suff else None
b_n0 = baseline_nec['accuracy']  if baseline_nec  else None

sd, sr, nd, nr = m(d_suff_dhat_accs), m(d_suff_rand_accs), m(d_nec_dhat_accs), m(d_nec_rand_accs)
if any(x is None for x in [sd, sr, nd, nr, b_s0, b_n0]):
    v = {"verdict":"inconclusive","reason":"missing runs","sd":sd,"sr":sr,"nd":nd,"nr":nr,"b_s0":b_s0,"b_n0":b_n0}
else:
    suff_drop_dhat = (b_s0 - sd) * 100
    suff_drop_rand = (b_s0 - sr) * 100
    nec_rise_dhat  = (nd - b_n0) * 100
    nec_rise_rand  = (nr - b_n0) * 100
    # Off-target check
    off_pen = None
    try:
        off_dhat = json.load(open(f'runs/m2_offtarget_ctrl_a_sufficiency_d_hat_a{alpha_star}_s42/off_target.json'))['summary']
        off_rand = json.load(open(f'runs/m2_offtarget_ctrl_a_sufficiency_random_matched_norm_a{alpha_star}_s42/off_target.json'))['summary']
        off_pen = (off_rand['mmlu_acc'] - off_dhat['mmlu_acc']) * 100  # penalty in pp on MMLU
    except: pass
    suff_ok = suff_drop_dhat >= 3.0 and suff_drop_rand <= 1.0
    nec_ok  = nec_rise_dhat >= 3.0 and nec_rise_rand <= 1.0
    off_ok  = off_pen is None or off_pen <= 3.0
    if suff_ok and nec_ok and off_ok:
        verdict = "confirmed"
    elif (suff_ok or nec_ok) and off_ok:
        verdict = "partial"
    elif off_pen is not None and off_pen > 3.0:
        verdict = "refuted"
    else:
        verdict = "refuted"
    v = dict(verdict=verdict,
             alpha_star=alpha_star, b_s0=b_s0, b_n0=b_n0,
             suff_dhat_acc=sd, suff_rand_acc=sr,
             nec_dhat_acc=nd, nec_rand_acc=nr,
             suff_drop_dhat_pp=suff_drop_dhat, suff_drop_rand_pp=suff_drop_rand,
             nec_rise_dhat_pp=nec_rise_dhat, nec_rise_rand_pp=nec_rise_rand,
             off_pen_pp=off_pen,
             suff_ok=suff_ok, nec_ok=nec_ok, off_ok=off_ok)
print(json.dumps(v, indent=2))
PY
cat runs/m2_verdict.json
echo "==== M1/M2 DONE ===="
