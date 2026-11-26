"""Paired comparison (bootstrap over drugs) of the fixed-in-advance score equation against the other approaches. Writes: equation_vs_others.csv"""
import numpy as np, pandas as pd
import features; features.inject(globals())
E = pd.read_pickle('per_drug_equation.pkl'); E = E[E.model.str.contains('fixed')]
S = pd.read_pickle('per_drug_setA.pkl')
def pdm(g, col): return g.groupby('drug')[col].mean().reindex(use).values
B = np.random.RandomState(0).randint(0, len(use), (2000, len(use)))
eh, em = pdm(E, 'hit10'), pdm(E, 'mrr'); rows = []
for fs, mn in [('baseline', 'Random'), ('single feature', '2D similarity only (kNN)'), ('A: full', 'Logistic regression'), ('A: full', 'XGBoost'), ('A: full', 'LightGBM'), ('A: full', 'Extra trees')]:
    g = S[(S.feature_set == fs) & (S.model == mn)]; dh = eh - pdm(g, 'hit10'); dm = em - pdm(g, 'mrr')
    rows.append(dict(compared_with=mn, d_Hit10=dh.mean(), CI_low=np.percentile(dh[B].mean(1), 2.5), CI_high=np.percentile(dh[B].mean(1), 97.5), d_MRR=dm.mean(), MRR_CI_low=np.percentile(dm[B].mean(1), 2.5), MRR_CI_high=np.percentile(dm[B].mean(1), 97.5)))
pd.DataFrame(rows).round(3).to_csv('equation_vs_others.csv', index=False); print(pd.read_csv('equation_vs_others.csv').to_string(index=False))
