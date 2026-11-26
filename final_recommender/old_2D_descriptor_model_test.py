"""Why the old 'base LightGBM' (Hit@10 0.779) is not in the final score.
That model used 2D RDKit descriptors of the drug AND of the ion (25 PCA components each) + pKa numbers, not 2D similarity.
Here it is re-run under the same rules as the final score: folds keep all forms of a molecule together, ions with no training user are not
candidates (their true pairs count as misses), 5 folds x seeds 0,1.
Check: 'ion descriptors only' sees nothing about the drug, so it gives the same ion list to every drug. If it scores as high as the
full model, the model is mainly learning which ions are common (popularity) from the ion descriptors.
Writes: old_2D_descriptor_model_test.csv"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import features; features.inject(globals())
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

dcols = [c for c in D.columns if c.startswith('d') and c[1:].isdigit()]; icols = [c for c in I.columns if c.startswith('i') and c[1:].isdigit()]
def clean(X): return np.clip(np.nan_to_num(X.astype(float)), -1e5, 1e5)
def pcs(train):        # PCA fitted on training drugs and on all ions (no labels used), same as the old code
    sd = StandardScaler().fit(clean(D.loc[train, dcols].values)); p = PCA(25, random_state=0).fit(sd.transform(clean(D.loc[train, dcols].values)))
    si = StandardScaler().fit(clean(I[icols].values)); q = PCA(25, random_state=0).fit(si.transform(clean(I[icols].values)))
    return p.transform(sd.transform(clean(D[dcols].values))), q.transform(si.transform(clean(I[icols].values)))
DC, IC = [f'dpc{k}' for k in range(25)], [f'ipc{k}' for k in range(25)]
def add(T, Zd, Zi):
    T = T.reset_index(drop=True)
    return pd.concat([T, pd.DataFrame(Zd[T.drug.values], columns=DC), pd.DataFrame(Zi[T.ion.values], columns=IC)], axis=1).assign(
        drug_pKa=dp[T.drug.values], ion_pKa=ip[T.ion.values])
OLD = DC + IC + ['dpKa', 'drug_pKa', 'ion_pKa', 'perm_cation']
LGB = lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
A_S, A_P = ['sim2D', 'sim3D', 'dpKa', 'ESP_comp'], ['P_salt', 'pKa_missing', 'perm_cation']
def eq(sm, pl):
    return make_pipeline(ColumnTransformer([('s', SplineTransformer(n_knots=5, degree=3, extrapolation='constant'), sm), ('p', StandardScaler(), pl)]), LogisticRegression(C=1.0, max_iter=5000))
MODELS = {'Final score (Set A equation)': lambda: (eq(A_S, A_P), A_S + A_P),
          'Old base LightGBM (drug + ion 2D descriptors + pKa)': lambda: (LGB(), OLD),
          'Drug 2D descriptors + pKa only (no ion descriptors)': lambda: (LGB(), DC + ['dpKa', 'drug_pKa', 'ion_pKa', 'perm_cation']),
          'Ion 2D descriptors only (same list for every drug)': lambda: (LGB(), IC),
          'Final score + ion 2D descriptors': lambda: (eq(A_S, A_P + IC), A_S + A_P + IC)}

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
        tr = [i for i in use if i not in set(te)]; Zd, Zi = pcs(tr)
        Ttr = add(pair_table(tr, tr), Zd, Zi); Tte = add(pair_table(tr, list(te)), Zd, Zi)
        frs = np.random.RandomState(seed * 10 + k); p_idx = np.where(Ttr.y == 1)[0]; n_idx = np.where(Ttr.y == 0)[0]
        n_idx = frs.choice(n_idx, min(len(n_idx), 8000 - len(p_idx)), replace=False); sub = Ttr.iloc[np.r_[p_idx, n_idx]]
        cnt = np.zeros(ni)
        for i in tr:
            for j in pos[i]: cnt[j] += 1
        scores = {}
        for name, mk in MODELS.items():
            m, cols = mk(); m.fit(sub[cols], sub.y.values); scores[name] = m.predict_proba(Tte[cols])[:, 1]
        scores['Check only: how many training drugs use the ion'] = cnt[Tte.ion.values] + 1e-6 * frs.rand(len(Tte))
        for name, sc in scores.items():
            Tte['_s'] = sc
            for i, g in Tte.groupby('drug'):
                best = min(r for _, r in ranks_of(i, g.ion.values, g._s.values))
                rows.append(dict(model=name, seed=seed, drug=i, hit1=best <= 1, hit5=best <= 5, hit10=best <= 10, mrr=1 / best))
    print('seed', seed, 'done', flush=True)

R = pd.DataFrame(rows)
def per_drug(g, col): return g.groupby('drug')[col].mean().reindex(use).values
B = np.random.RandomState(0).randint(0, len(use), (2000, len(use)))
base = per_drug(R[R.model == 'Final score (Set A equation)'], 'hit10'); out = []
for name in R.model.unique():
    g = R[R.model == name]; h = per_drug(g, 'hit10'); d = h - base
    out.append(dict(model=name, Hit1=per_drug(g, 'hit1').mean(), Hit5=per_drug(g, 'hit5').mean(), Hit10=h.mean(), Hit10_CI_low=np.percentile(h[B].mean(1), 2.5),
                    Hit10_CI_high=np.percentile(h[B].mean(1), 97.5), MRR=per_drug(g, 'mrr').mean(), change_vs_final=d.mean(),
                    change_CI_low=np.percentile(d[B].mean(1), 2.5), change_CI_high=np.percentile(d[B].mean(1), 97.5)))
S = pd.DataFrame(out).round(3); S.to_csv('old_2D_descriptor_model_test.csv', index=False); print(S.to_string(index=False))
