"""Set A only (2D + 3D + pKa + ESP). All models, leave-drug-out, 5 folds x seeds 0,1. Baseline = Random only.
Also: drop-one-feature-family test for every model (how much each family contributes).
Writes: results_setA_all_models.csv, ablation_setA.csv, per_drug_setA.pkl, oof_scores_setA_seed0.csv"""
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings('ignore')
import features; features.inject(globals())
import lightgbm as lgb, xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODELS = {
 'Logistic regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
 'LightGBM':            lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0),
 'XGBoost':             lambda: xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=0, n_jobs=2),
 'Random forest':       lambda: RandomForestClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0),
 'Extra trees':         lambda: ExtraTreesClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0),
 'Hist. gradient boosting': lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0),
 'MLP (neural net)':    lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), alpha=1e-3, max_iter=2000, early_stopping=True, random_state=0)),
 'SVM (RBF)':           lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma='scale')),
}
FAMILIES = {'2D similarity': ['sim2D'], '3D similarity': ['sim3D'], 'pKa': ['P_salt', 'dpKa', 'pKa_missing', 'perm_cation'], 'ESP': ['ESP_comp']}
SINGLE = {'2D similarity only (kNN)': 'sim2D', '3D similarity only': 'sim3D', 'pKa only (P_salt)': 'P_salt', 'ESP only': 'ESP_comp'}
MAX_ROWS = 8000

def fit_score(mn, cols, sub, Tte):
    m = MODELS[mn]().fit(sub[cols].values, sub.y.values)
    return m.decision_function(Tte[cols].values) if mn == 'SVM (RBF)' else m.predict_proba(Tte[cols].values)[:, 1]

def ranks_of(i, js, sc):
    s = dict(zip(js, sc)); arr = np.array(sc); out = []
    for p in pos[i]:
        if p not in s: out.append((p, 1e9)); continue          # ion used by no training drug: cannot be recommended -> miss
        oth = pos[i] - {p}; keep = np.array([j not in oth for j in js])
        out.append((p, 1 + (arr[keep] > s[p]).sum() + 0.5 * ((arr[keep] == s[p]).sum() - 1)))
    return out

rows = []; OOF = []; t0 = time.time()
for seed in (0, 1):
    folds = group_folds(seed)
    for k, te in enumerate(folds):
        tr = [i for i in use if i not in set(te)]
        Ttr = pair_table(tr, tr); Tte = pair_table(tr, list(te))
        cnt = np.zeros(ni)                                  # only used to mark 'rare' test ions (<=2 training drugs); NOT a model input
        for i in tr:
            for j in pos[i]: cnt[j] += 1
        frs = np.random.RandomState(seed * 10 + k)
        p_idx = np.where(Ttr.y == 1)[0]; n_idx = np.where(Ttr.y == 0)[0]
        n_idx = frs.choice(n_idx, min(len(n_idx), MAX_ROWS - len(p_idx)), replace=False)
        sub = Ttr.iloc[np.r_[p_idx, n_idx]]
        scores = {}
        for mn in MODELS:
            scores[('A: full', mn)] = fit_score(mn, FEATS_A, sub, Tte)
            for fam, fc in FAMILIES.items():
                scores[(f'A minus {fam}', mn)] = fit_score(mn, [c for c in FEATS_A if c not in fc], sub, Tte)
        for sn, col in SINGLE.items(): scores[('single feature', sn)] = Tte[col].values + 1e-9 * frs.rand(len(Tte))
        scores[('baseline', 'Random')] = frs.rand(len(Tte))
        for (fs, mn), sc in scores.items():
            Tte['_s'] = sc
            for i, g in Tte.groupby('drug'):
                rk = ranks_of(i, g.ion.values, g._s.values); best = min(r for _, r in rk)
                rows.append(dict(feature_set=fs, model=mn, seed=seed, drug=i, hit1=best <= 1, hit5=best <= 5, hit10=best <= 10, mrr=1 / best,
                                 rare=[r for p, r in rk if cnt[p] <= 2]))
        if seed == 0:
            keep = Tte[['drug', 'ion', 'y'] + FEATS_A].copy()
            for (fs, mn), sc in scores.items():
                if fs in ('A: full', 'baseline'): keep[f'{mn}'] = sc
            OOF.append(keep)
        print(f'seed {seed} fold {k} done {time.time()-t0:.0f}s', flush=True)

R = pd.DataFrame(rows); pd.to_pickle(R, 'per_drug_setA.pkl'); pd.concat(OOF).to_csv('oof_scores_setA_seed0.csv', index=False)

def per_drug(g, col): return g.groupby('drug')[col].mean().reindex(use)
B = np.random.RandomState(0).randint(0, len(use), (2000, len(use)))
rnd = per_drug(R[R.model == 'Random'], 'hit10').values
rows2 = []
for (fs, mn), g in R.groupby(['feature_set', 'model']):
    h = per_drug(g, 'hit10').values; d = h - rnd; rr = np.array([x for l in g.rare for x in l])
    rows2.append(dict(feature_set=fs, model=mn, Hit1=per_drug(g, 'hit1').mean(), Hit5=per_drug(g, 'hit5').mean(), Hit10=h.mean(),
                      Hit10_CI_low=np.percentile(h[B].mean(1), 2.5), Hit10_CI_high=np.percentile(h[B].mean(1), 97.5),
                      MRR=per_drug(g, 'mrr').mean(), rare_Hit10=(rr <= 10).mean(), n_rare=len(rr),
                      vs_random=d.mean(), vs_random_CI_low=np.percentile(d[B].mean(1), 2.5), vs_random_CI_high=np.percentile(d[B].mean(1), 97.5)))
S = pd.DataFrame(rows2).round(3)
S[S.feature_set.isin(['A: full', 'single feature', 'baseline'])].sort_values('Hit10', ascending=False).to_csv('results_setA_all_models.csv', index=False)

# ablation: change in Hit10 / MRR / rare Hit10 when one family is removed (paired over drugs, CI by bootstrap)
full = {mn: per_drug(g, 'hit10').values for (fs, mn), g in R.groupby(['feature_set', 'model']) if fs == 'A: full'}
fullm = {mn: per_drug(g, 'mrr').values for (fs, mn), g in R.groupby(['feature_set', 'model']) if fs == 'A: full'}
fullr = {mn: np.array([x for l in g.rare for x in l]) for (fs, mn), g in R.groupby(['feature_set', 'model']) if fs == 'A: full'}
ab = []
for (fs, mn), g in R.groupby(['feature_set', 'model']):
    if not fs.startswith('A minus'): continue
    h = per_drug(g, 'hit10').values; d = full[mn] - h               # positive = the family helps
    rr = np.array([x for l in g.rare for x in l])
    ab.append(dict(removed=fs.replace('A minus ', ''), model=mn, Hit10_full=full[mn].mean(), Hit10_without=h.mean(), drop_Hit10=d.mean(),
                   drop_CI_low=np.percentile(d[B].mean(1), 2.5), drop_CI_high=np.percentile(d[B].mean(1), 97.5),
                   drop_MRR=(fullm[mn] - per_drug(g, 'mrr').values).mean(), drop_rareHit10=(fullr[mn] <= 10).mean() - (rr <= 10).mean()))
pd.DataFrame(ab).round(3).to_csv('ablation_setA.csv', index=False)
print(S[S.feature_set.isin(['A: full', 'single feature', 'baseline'])].sort_values('Hit10', ascending=False).to_string(index=False))
