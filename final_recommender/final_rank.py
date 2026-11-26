"""Step 2: leave-ONE-drug-out ranking of every drug's candidate counter-ions with the additive score equation.
For each drug, the model is trained without that drug (and without its pairs), then it scores all opposite-sign ions.
Score = 100 x calibrated probability that the ion is a reported counter-ion. dpKa label is shown for information only, never used as a filter.
Writes: all_scores_leave_drug_out.csv, top10_recommendations_all_drugs.csv, leave_one_drug_out_summary.csv"""
import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings('ignore')
import features; features.inject(globals())
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

FAMILIES = {'2D similarity': ['sim2D'], '3D similarity': ['sim3D'], 'pKa': ['dpKa', 'P_salt', 'pKa_missing', 'perm_cation'], 'ESP complementarity': ['ESP_comp']}
SMOOTH = ['sim2D', 'sim3D', 'dpKa', 'ESP_comp']; PLAIN = ['P_salt', 'pKa_missing', 'perm_cation']; COLS = SMOOTH + PLAIN
cal = json.load(open('calibration_params.json')); a, b = cal['platt_a'], cal['platt_b']

def make():
    ct = ColumnTransformer([('s', SplineTransformer(n_knots=5, degree=3, extrapolation='constant'), SMOOTH), ('p', StandardScaler(), PLAIN)])
    return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=5000))
def subsample(Ttr, seed):
    p = np.where(Ttr.y == 1)[0]; n = np.where(Ttr.y == 0)[0]
    n = np.random.RandomState(seed).choice(n, min(len(n), 8000 - len(p)), replace=False)
    return Ttr.iloc[np.r_[p, n]]
def label(d, perm_):
    if perm_ == 1: return 'permanent cation (no proton transfer)'
    if np.isnan(d): return 'pKa not available'
    return 'salt likely (dpKa > 4)' if d > 4 else ('co-crystal likely (dpKa < -1)' if d < -1 else 'ambiguous (-1 to 4)')

out = []
for n_done, i in enumerate(use):
    tr = [j for j in use if PARENT[j] != PARENT[i]]   # leave the whole parent molecule out (all its forms)
    Ttr = pair_table(tr, tr); Tte = pair_table(tr, [i]); sub = subsample(Ttr, 1000 + i)
    m = make().fit(sub[COLS], sub.y.values); ct, lr = m[0], m[1]; w = lr.coef_[0]
    Xs = ct.transform(sub[COLS]); Xt = ct.transform(Tte[COLS]); k = (Xs.shape[1] - len(PLAIN)) // len(SMOOTH)
    cm = {f: slice(q * k, (q + 1) * k) for q, f in enumerate(SMOOTH)}
    for q, f in enumerate(PLAIN): cm[f] = slice(len(SMOOTH) * k + q, len(SMOOTH) * k + q + 1)
    raw = lr.predict_proba(Xt)[:, 1]; lg = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    Tte['P'] = 1 / (1 + np.exp(-(a * lg + b)))
    for fam, fs in FAMILIES.items():                     # contribution = this family's part of the logit, relative to the average training pair
        Tte['c_' + fam] = sum(Xt[:, cm[f]] @ w[cm[f]] for f in fs) - sum(Xs[:, cm[f]] @ w[cm[f]] for f in fs).mean()
    raw_d = np.array([dpka(i, [j])[0] for j in Tte.ion])
    Tte['dpKa_value'] = raw_d; Tte['pKa_label'] = [label(d, I.perm.values[j]) for d, j in zip(raw_d, Tte.ion)]
    Tte['drug_name'] = D.name.values[i]; Tte['ion_name'] = I.name.values[Tte.ion.values]; Tte['ion_smiles'] = I.smiles.values[Tte.ion.values]
    Tte['rank'] = Tte.P.rank(ascending=False, method='first').astype(int); Tte['score'] = 100 * Tte.P
    out.append(Tte)
    if n_done % 10 == 0: print(n_done, 'drugs done', flush=True)

SCORES = pd.concat(out, ignore_index=True)
keep = ['drug', 'drug_name', 'rank', 'ion', 'ion_name', 'ion_smiles', 'score', 'y', 'sim2D', 'sim3D', 'dpKa_value', 'pKa_label', 'ESP_comp'] + ['c_' + f for f in FAMILIES]
SCORES = SCORES.rename(columns={'y': 'is_reported_pair'})
SCORES[[c if c != 'y' else 'is_reported_pair' for c in keep]].sort_values(['drug', 'rank']).round(4).to_csv('all_scores_leave_drug_out.csv', index=False)
T = SCORES[SCORES['rank'] <= 10]; T[[c if c != 'y' else 'is_reported_pair' for c in keep]].sort_values(['drug', 'rank']).round(4).to_csv('top10_recommendations_all_drugs.csv', index=False)
# same protocol as the cross-validation: each true ion is ranked against the non-true ions only; ions no other drug uses cannot be recommended (counted as a miss)
best = {}; cold = 0
for i in use:
    g = SCORES[SCORES.drug == i]; sc = dict(zip(g.ion, g.P)); arr = g.P.values; r = []
    for p in pos[i]:
        if p not in sc: r.append(1e9); cold += 1; continue
        keep = np.array([j not in (pos[i] - {p}) for j in g.ion]); r.append(1 + (arr[keep] > sc[p]).sum() + 0.5 * ((arr[keep] == sc[p]).sum() - 1))
    best[i] = min(r)
hit = pd.Series(best)
pd.DataFrame({'metric': ['drugs', 'Hit@1', 'Hit@5', 'Hit@10', 'MRR', 'pairs whose ion no other drug uses (cannot be recommended)'],
              'value': [len(hit), (hit <= 1).mean(), (hit <= 5).mean(), (hit <= 10).mean(), (1 / hit).mean(), cold]}).round(3).to_csv('leave_one_drug_out_summary.csv', index=False)
print(pd.read_csv('leave_one_drug_out_summary.csv').to_string(index=False))
