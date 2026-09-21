#!/usr/bin/env python3
"""Aggregate per-figure .status.json sidecars into per-claim INDEX.json + top-level INDEX.md."""
import os, json, glob, datetime

ROOT = os.path.abspath(os.path.dirname(__file__))
NOW = datetime.datetime.now().astimezone().isoformat()

CLAIMS = [
    {
        'claim_id': 'C1', 'dir': 'C1',
        'title': 'Strong-form causal-steering claim (later falsified)',
        'figs': [
            ('c1_dose_response', 'C1 dose-response: mean predicted %H vs steering coefficient alpha on the block-26 direction; rise is non-monotonic (alpha=4 dip) and later shown composition-confounded.'),
            ('c1_specificity', 'C1 specificity: the real direction raises %H with dose while a norm-matched random direction stays flat/declines.'),
        ],
    },
    {
        'claim_id': 'C1_v2', 'dir': 'C1_v2',
        'title': 'Honest narrowed claim (composition confound is the crux)',
        'figs': [
            ('c1v2_predictor_triangulation', 'C1_v2: the +14-16pt predicted-%H uplift reproduces across 3 independent SS predictors - not a single-probe artifact.'),
            ('c1v2_composition_confound', 'C1_v2 confound: steering collapses GC (0.44->0.13) and shifts amino-acid composition (Lys up, low-complexity up) in lockstep with the %H gain - the effect is not shown separable from composition.'),
        ],
    },
    {
        'claim_id': 'C2', 'dir': 'C2',
        'title': 'Eval-harness fidelity across method/dataset swaps',
        'figs': [
            ('c2_swap_robustness', 'C2 eval-harness fidelity holds across method and dataset swaps (all r >> 0.7 threshold).'),
        ],
    },
]

SLOTS = {'png': None, 'pdf': None, 'md': None, 'tex': None}


def load_status(cdir, fid):
    p = os.path.join(ROOT, cdir, fid + '.status.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {'id': fid, 'status': 'skipped', 'reason': 'generator did not run (no status sidecar)'}


md_lines = ['# Claim Ledger Figures', '',
            f'_Generated {NOW} - auto-ledger mode, style=publication, formats=pdf,png_', '']

for c in CLAIMS:
    figures, skipped = [], []
    for fid, caption in c['figs']:
        st = load_status(c['dir'], fid)
        entry = {
            'id': fid,
            'type': st.get('type'),
            'caption': caption,
            'png': st.get('png'), 'pdf': st.get('pdf'),
            'md': st.get('md'), 'tex': st.get('tex'),
            'source_data': st.get('source_data'),
            'status': st.get('status', 'error'),
        }
        for k in ('png', 'pdf', 'md', 'tex'):
            entry.setdefault(k, None)
        if st.get('status') == 'ok':
            figures.append(entry)
        elif st.get('status') == 'skipped':
            skipped.append({'id': fid, 'reason': st.get('reason', 'skipped')})
        else:  # error
            entry['error_detail'] = st.get('error_detail', 'unknown error')
            figures.append(entry)

    index = {
        'claim_id': c['claim_id'],
        'claim_title': c['title'],
        'generated_at': NOW,
        'figures': figures,
        'skipped': skipped,
    }
    out = os.path.join(ROOT, c['dir'], 'INDEX.json')
    with open(out, 'w') as f:
        json.dump(index, f, indent=2)
    print(f'wrote {out}')

    # top-level markdown section
    md_lines.append(f"## {c['claim_id']} - {c['title']}")
    md_lines.append('')
    for e in figures:
        tag = 'OK' if e['status'] == 'ok' else e['status'].upper()
        md_lines.append(f"### `{e['id']}` ({e['type']}) - {tag}")
        md_lines.append('')
        md_lines.append(f"_{e['caption']}_")
        md_lines.append('')
        if e.get('png'):
            md_lines.append(f"![{e['id']}]({e['png']})")
            md_lines.append('')
            md_lines.append(f"- vector: `{e['pdf']}`")
        if e.get('md'):
            # inline the table markdown
            tpath = os.path.join(ROOT, '..', e['md']) if not os.path.isabs(e['md']) else e['md']
            try:
                with open(os.path.join(ROOT, os.path.relpath(e['md'], 'figures'))) as tf:
                    md_lines.append(tf.read())
            except Exception:
                md_lines.append(f"(table: `{e['md']}`)")
            md_lines.append(f"- latex: `{e['tex']}`")
        md_lines.append(f"- source: `{e['source_data']}`")
        if e['status'] == 'error':
            md_lines.append(f"- error: {e.get('error_detail', '').splitlines()[0] if e.get('error_detail') else ''}")
        md_lines.append('')
    for s in skipped:
        md_lines.append(f"### `{s['id']}` - SKIPPED")
        md_lines.append('')
        md_lines.append(f"- reason: {s['reason']}")
        md_lines.append('')

with open(os.path.join(ROOT, 'INDEX.md'), 'w') as f:
    f.write('\n'.join(md_lines))
print(f"wrote {os.path.join(ROOT, 'INDEX.md')}")
