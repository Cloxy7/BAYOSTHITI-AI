"""Does a safety check help as a score feature? Same equation, folds and rules as extra_features_test.py.
safety_listed : ion (or its parent acid/base, or a salt of it) is on the FDA GRAS / SCOGS lists (tier A or B). Name-matched: 204/291 ions checked, the rest count as not listed.
pharma_ok     : ion class is pharma-established salt former, excipient/food-derived/biocompatible, or an active drug ion (ions_acceptability.csv, 137/291 matched).
Writes: safety_feature_test.csv"""

import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import features; features.inject(globals())
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------- safety values
from rdkit import Chem
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
S = pd.read_csv('../Ion_safety_GRAS_SCOGS_all_ions.csv'); AC = pd.read_csv('../ions_acceptability.csv'); AC['can'] = AC.smiles.map(can)
tier = dict(zip(S.ion_name, S.safety_list_tier)); acc = dict(zip(AC.can, AC.acceptability))
LISTED = np.array([1 if str(tier.get(n, '')).startswith(('A', 'B')) else 0 for n in I.name])
OKC = ('pharma-established salt former', 'biocompatible / excipient / food-derived', 'active drug ion (dual-active IL)')
PHARMA = np.array([1 if acc.get(can(s)) in OKC else 0 for s in I.smiles])
print('listed ions', LISTED.sum(), 'pharma_ok ions', PHARMA.sum(), 'of', ni)
def add_cols(T):
    T['safety_listed'] = LISTED[T.ion.values]; T['pharma_ok'] = PHARMA[T.ion.values]; return T

A_S, A_P = ['sim2D', 'sim3D', 'dpKa', 'ESP_comp'], ['P_salt', 'pKa_missing', 'perm_cation']
SETS = {'Set A (current score)': (A_S, A_P), 'Set A + safety list (GRAS/SCOGS)': (A_S, A_P + ['safety_listed']),
        'Set A + pharma-acceptable class': (A_S, A_P + ['pharma_ok']), 'Set A + both safety features': (A_S, A_P + ['safety_listed', 'pharma_ok']),
        'safety features alone': ([], ['safety_listed', 'pharma_ok'])}

def make(sm, pl):
    parts = ([('s', SplineTransformer(n_knots=5, degree=3, extrapolation='constant'), sm)] if sm else []) + ([('p', StandardScaler(), pl)] if pl else [])
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=1.0, max_iter=5000))

def ranks_of(i, js, sc):
    s = dict(zip(js, sc)); arr = np.array(sc); out = []
    for p in pos[i]:
        if p not in s: out.append((p, 1e9)); continue
        oth = pos[i] - {p}; keep = np.array([j not in oth for j in js])
        out.append((p, 1 + (arr[keep] > s[p]).sum() + 0.5 * ((arr[keep] == s[p]).sum() - 1)))
    return out

rows = []
for seed in (0, 1):
    for k, te in enumerate(group_folds(seed)):
        tr = [i for i in use if i not in set(te)]
        Ttr = add_cols(pair_table(tr, tr)); Tte = add_cols(pair_table(tr, list(te)))
        frs = np.random.RandomState(seed * 10 + k)          # same negative sample as score_equation_cv.py
        p_idx = np.where(Ttr.y == 1)[0]; n_idx = np.where(Ttr.y == 0)[0]
        n_idx = frs.choice(n_idx, min(len(n_idx), 8000 - len(p_idx)), replace=False); sub = Ttr.iloc[np.r_[p_idx, n_idx]]
        for name, (sm, pl) in SETS.items():
            m = make(sm, pl).fit(sub[sm + pl], sub.y.values); Tte['_s'] = m.predict_proba(Tte[sm + pl])[:, 1]
            for i, g in Tte.groupby('drug'):
                best = min(r for _, r in ranks_of(i, g.ion.values, g._s.values))
                rows.append(dict(model=name, seed=seed, drug=i, hit1=best <= 1, hit5=best <= 5, hit10=best <= 10, mrr=1 / best))
    print('seed', seed, 'done', flush=True)

R = pd.DataFrame(rows); pd.to_pickle(R, 'per_drug_safety.pkl')
def per_drug(g, col): return g.groupby('drug')[col].mean().reindex(use).values
B = np.random.RandomState(0).randint(0, len(use), (2000, len(use)))
old = pd.read_pickle('per_drug_setA.pkl'); rnd = per_drug(old[old.model == 'Random'], 'hit10')
base_h = per_drug(R[R.model == 'Set A (current score)'], 'hit10'); base_m = per_drug(R[R.model == 'Set A (current score)'], 'mrr')
out = []
for name in SETS:
    g = R[R.model == name]; h = per_drug(g, 'hit10'); mr = per_drug(g, 'mrr'); d = h - base_h; dm = mr - base_m
    out.append(dict(model=name, Hit1=per_drug(g, 'hit1').mean(), Hit5=per_drug(g, 'hit5').mean(), Hit10=h.mean(),
                    Hit10_CI_low=np.percentile(h[B].mean(1), 2.5), Hit10_CI_high=np.percentile(h[B].mean(1), 97.5), MRR=mr.mean(),
                    change_Hit10_vs_SetA=d.mean(), change_CI_low=np.percentile(d[B].mean(1), 2.5), change_CI_high=np.percentile(d[B].mean(1), 97.5),
                    change_MRR_vs_SetA=dm.mean(), change_MRR_CI_low=np.percentile(dm[B].mean(1), 2.5), change_MRR_CI_high=np.percentile(dm[B].mean(1), 97.5),
                    vs_random=(h - rnd).mean(), vs_random_CI_low=np.percentile((h - rnd)[B].mean(1), 2.5), vs_random_CI_high=np.percentile((h - rnd)[B].mean(1), 97.5)))
S = pd.DataFrame(out).round(3); S.to_csv('safety_feature_test.csv', index=False); print(S.to_string(index=False))
