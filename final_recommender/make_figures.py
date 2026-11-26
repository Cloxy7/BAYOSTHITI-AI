"""Figures for the report: model comparison with CIs, feature-family contribution, worked example."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = pd.read_csv('results_setA_all_models.csv'); E = pd.read_csv('results_score_equation.csv'); E = E[E.model.str.contains('fixed')].iloc[0]
rows = [(r.model if r.feature_set != 'A: full' else r.model + ' (all 7 features)', r.Hit10, r.Hit10_CI_low, r.Hit10_CI_high, r.feature_set) for r in R.itertuples()]
rows.append(('Score equation (additive splines, all 7 features)', E.Hit10, E.Hit10_CI_low, E.Hit10_CI_high, 'equation'))
rows.sort(key=lambda x: x[1]); col = {'baseline': '#999999', 'single feature': '#8fb8c9', 'A: full': '#1f6f8b', 'equation': '#c0504d'}
fig, ax = plt.subplots(figsize=(9, 5.6))
for k, (n, h, lo, hi, fs) in enumerate(rows):
    ax.barh(k, h, color=col[fs]); ax.plot([lo, hi], [k, k], color='k', lw=1.2); ax.text(max(hi, h) + .01, k, f'{h:.2f}', va='center', fontsize=8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.set_xlabel('Hit@10 (leave-drug-out, 95% bootstrap CI)'); ax.set_xlim(0, 0.95)
ax.axvline(R[R.model == 'Random'].Hit10.iloc[0], color='k', ls='--', lw=.8); ax.set_title('Hit@10 of all models using Set A (2D + 3D + pKa + ESP)')
plt.tight_layout(); plt.savefig('fig_model_comparison.png', dpi=200); plt.close()
A = pd.read_csv('ablation_equation.csv'); fig, ax = plt.subplots(figsize=(6.5, 3.4)); y = np.arange(len(A))[::-1]
ax.barh(y, A.drop_Hit10, color='#1f6f8b'); ax.errorbar(A.drop_Hit10, y, xerr=[A.drop_Hit10 - A.drop_CI_low, A.drop_CI_high - A.drop_Hit10], fmt='none', ecolor='k', lw=1.2)
for yy, s in zip(y, A['share_of_total_drop_%']): ax.text(0.205, yy, f'{s:.0f}% of total', va='center', fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(A.family_removed); ax.axvline(0, color='k', lw=.6); ax.set_xlim(-0.03, 0.28); ax.set_xlabel('Fall in Hit@10 when this feature group is removed'); ax.set_title('Contribution of each feature group (score equation)')
plt.tight_layout(); plt.savefig('fig_contribution.png', dpi=200); plt.close()
T = pd.read_csv('top10_recommendations_all_drugs.csv'); names = ['2D similarity', '3D similarity', 'pKa', 'ESP complementarity']; cols = ['#1f6f8b', '#6aa84f', '#e69138', '#8e7cc3']
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, dn in zip(axs, ['Erlotinib', 'Cinnarizine']):
    g = T[T.drug_name == dn].sort_values('rank'); lab = [f"{r.ion_name if isinstance(r.ion_name, str) else r.ion_smiles[:18]}{'  (reported)' if r.is_reported_pair else ''}" for r in g.itertuples()]
    left_p = np.zeros(len(g)); left_n = np.zeros(len(g)); yy = np.arange(len(g))[::-1]
    for nm, c in zip(names, cols):
        v = g['c_' + nm].values; ax.barh(yy, np.where(v > 0, v, 0), left=left_p, color=c, label=nm); left_p += np.where(v > 0, v, 0)
        ax.barh(yy, np.where(v < 0, v, 0), left=left_n, color=c); left_n += np.where(v < 0, v, 0)
    ax.set_yticks(yy); ax.set_yticklabels(lab, fontsize=8); ax.axvline(0, color='k', lw=.6); ax.set_xlabel('contribution to score (logit points)'); ax.set_title(f'{dn}: top-10 and where each score comes from')
axs[0].legend(fontsize=7, loc='lower right'); plt.tight_layout(); plt.savefig('fig_example_top10.png', dpi=200); plt.close()
