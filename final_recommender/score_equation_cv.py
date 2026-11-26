"""Transparent additive score for Set A: logit(P) = b0 + f_2D(sim2D) + f_3D(sim3D) + f_pKa(P_salt, dpKa, flags) + f_ESP(ESP_comp)
Each f is a smooth spline curve, so the score splits cleanly into one contribution per feature family.
Same leave-drug-out folds as run_setA.py. The settings (5 knots, C=1) were fixed before looking at results; other values are shown only as sensitivity.
Writes: results_score_equation.csv"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import features; features.inject(globals())
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

SMOOTH = ['sim2D', 'sim3D', 'dpKa', 'ESP_comp']; PLAIN = ['P_salt', 'pKa_missing', 'perm_cation']
def make(knots=5, C=1.0):
    ct = ColumnTransformer([('s', SplineTransformer(n_knots=knots, degree=3, extrapolation='constant'), SMOOTH), ('p', StandardScaler(), PLAIN)])
    return make_pipeline(ct, LogisticRegression(C=C, max_iter=5000))

def ranks_of(i, js, sc):
    s = dict(zip(js, sc)); arr = np.array(sc); out = []
    for p in pos[i]:
        if p not in s: out.append((p, 1e9)); continue          # ion used by no training drug: cannot be recommended -> miss
        oth = pos[i] - {p}; keep = np.array([j not in oth for j in js])
        out.append((p, 1 + (arr[keep] > s[p]).sum() + 0.5 * ((arr[keep] == s[p]).sum() - 1)))
    return out

CONFIGS = {'spline 5 knots, C=1 (fixed in advance)': (5, 1.0), 'spline 4 knots, C=1': (4, 1.0), 'spline 5 knots, C=0.1': (5, 0.1), 'spline 6 knots, C=0.3': (6, 0.3)}
rows = []
for seed in (0, 1):
    folds = group_folds(seed)
    for k, te in enumerate(folds):
        tr = [i for i in use if i not in set(te)]
        Ttr = pair_table(tr, tr); Tte = pair_table(tr, list(te))
        cnt = np.zeros(ni)
        for i in tr:
            for j in pos[i]: cnt[j] += 1
        frs = np.random.RandomState(seed * 10 + k)
        p_idx = np.where(Ttr.y == 1)[0]; n_idx = np.where(Ttr.y == 0)[0]
        n_idx = frs.choice(n_idx, min(len(n_idx), 8000 - len(p_idx)), replace=False)
        sub = Ttr.iloc[np.r_[p_idx, n_idx]]
        for name, (kn, C) in CONFIGS.items():
            m = make(kn, C).fit(sub[SMOOTH + PLAIN], sub.y.values); Tte['_s'] = m.predict_proba(Tte[SMOOTH + PLAIN])[:, 1]
            for i, g in Tte.groupby('drug'):
                rk = ranks_of(i, g.ion.values, g._s.values); best = min(r for _, r in rk)
                rows.append(dict(model=name, seed=seed, drug=i, hit1=best <= 1, hit5=best <= 5, hit10=best <= 10, mrr=1 / best, rare=[r for p, r in rk if cnt[p] <= 2]))
    print('seed', seed, 'done', flush=True)

R = pd.DataFrame(rows); pd.to_pickle(R, 'per_drug_equation.pkl')
def per_drug(g, col): return g.groupby('drug')[col].mean().reindex(use)
B = np.random.RandomState(0).randint(0, len(use), (2000, len(use)))
old = pd.read_pickle('per_drug_setA.pkl'); rnd = per_drug(old[old.model == 'Random'], 'hit10').values; xgbh = per_drug(old[(old.feature_set == 'A: full') & (old.model == 'XGBoost')], 'hit10').values
out = []
for mn, g in R.groupby('model'):
    h = per_drug(g, 'hit10').values; rr = np.array([x for l in g.rare for x in l])
    out.append(dict(model=mn, Hit1=per_drug(g, 'hit1').mean(), Hit5=per_drug(g, 'hit5').mean(), Hit10=h.mean(), Hit10_CI_low=np.percentile(h[B].mean(1), 2.5), Hit10_CI_high=np.percentile(h[B].mean(1), 97.5),
                    MRR=per_drug(g, 'mrr').mean(), rare_Hit10=(rr <= 10).mean(), vs_random=(h - rnd).mean(), vs_random_CI_low=np.percentile((h - rnd)[B].mean(1), 2.5), vs_random_CI_high=np.percentile((h - rnd)[B].mean(1), 97.5),
                    vs_XGBoost=(h - xgbh).mean(), vs_XGB_CI_low=np.percentile((h - xgbh)[B].mean(1), 2.5), vs_XGB_CI_high=np.percentile((h - xgbh)[B].mean(1), 97.5)))
S = pd.DataFrame(out).round(3); S.to_csv('results_score_equation.csv', index=False); print(S.to_string(index=False))
