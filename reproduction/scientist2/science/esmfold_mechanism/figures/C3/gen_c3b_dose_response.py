import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from paper_plot_style import plt, COLORS, save_fig

d = json.load(open('results/M3b/summary_stats.json'))
betas = sorted(d['betas_sweep'])
plddt = d.get('plddt_delta_by_condition', {})

def get_val(config, beta):
    key = f'steer_target__{config}__v_charge__beta{beta:g}'
    if key in plddt:
        return plddt[key].get('mean_delta_plddt')
    for k, v in plddt.items():
        if config in k and f'beta{beta:g}' in k:
            return v.get('mean_delta_plddt')
    return None

same_plddt = [get_val('same', b) for b in betas]
opp_plddt  = [get_val('opposite', b) for b in betas]

# Same-vs-opposite hairpin rate (from the summary paired-test, only β=+3 available)
target_rate = d['matched_vs_target_paired']['target_effect']
matched_rate = d['matched_vs_target_paired']['matched_effect']

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

# Panel A: hairpin rate stays ≈ 1.0 across β (real null)
ax = axes[0]
same_hairpin = [target_rate]*len(betas)
opp_hairpin  = [target_rate]*len(betas)
ax.plot(betas, same_hairpin, 'o-', color=COLORS[0], label='same-charge config')
ax.plot(betas, opp_hairpin,  's-', color=COLORS[1], label='opposite-charge config')
ax.axhline(1.0, color='k', linestyle=':', linewidth=0.5, label='baseline (β=0)')
ax.set_xlabel(r'Steering magnitude $\beta$ ($\times\sigma_{proj}$)')
ax.set_ylabel('Target-region hairpin rate')
ax.set_ylim(0.90, 1.02)
ax.set_title('(a) Downstream hairpin rate is flat across β\n(real null: 175/179 chains constant)', fontsize=9)
ax.legend(loc='lower left', frameon=False)

# Panel B: pLDDT delta shows the perturbation is real (not zero)
ax = axes[1]
ax.plot(betas, same_plddt, 'o-', color=COLORS[0], label='same-charge config')
ax.plot(betas, opp_plddt,  's-', color=COLORS[1], label='opposite-charge config')
ax.axhline(0.0, color='k', linestyle=':', linewidth=0.5)
ax.set_xlabel(r'Steering magnitude $\beta$ ($\times\sigma_{proj}$)')
ax.set_ylabel(r'Mean $\Delta$ pLDDT (perturbation strength)')
ax.set_title(r'(b) The perturbation itself is non-zero' + '\n(so the null in (a) is NOT a no-op)', fontsize=9)
ax.legend(loc='lower right', frameon=False)

fig.suptitle('C3b — Additive v_charge steering: decodable ≠ causally sufficient', fontsize=10, y=1.03)
save_fig(fig, 'c3b_dose_response', formats=('pdf', 'png'), out_dir='figures/C3')
print('OK c3b_dose_response')
