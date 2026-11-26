"""Step 1 of the final score.
 (a) leave-drug-out CV of the additive equation, plus drop-one-family versions -> ablation_equation.csv (how much each family adds)
 (b) Platt calibration of the raw score on the out-of-fold pairs -> calibration_params.json, fig_calibration.png, calibration_summary.csv
 (c) final fit on all 138 drugs -> equation_coefficients.csv, partial_effects.csv, fig_partial_effects.png, final_model.pkl"""
import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings('ignore')
import features; features.inject(globals())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

FAMILIES = {'2D similarity': ['sim2D'], '3D similarity': ['sim3D'], 'pKa': ['dpKa', 'P_salt', 'pKa_missing', 'perm_cation'], 'ESP complementarity': ['ESP_comp']}
SMOOTH_ALL = ['sim2D', 'sim3D', 'dpKa', 'ESP_comp']; PLAIN_ALL = ['P_salt', 'pKa_missing', 'perm_cation']
def make(drop=()):
    sm = [c for c in SMOOTH_ALL if c not in drop]; pl = [c for c in PLAIN_ALL if c not in drop]
    parts = [('s', SplineTransformer(n_knots=5, degree=3, extrapolation='constant'), sm)]
    if pl: parts.append(('p', StandardScaler(), pl))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(C=1.0, max_iter=5000)), sm + pl

def ranks_of(i, js, sc):
    s = dict(zip(js, sc)); arr = np.array(sc); out = []
    for p in pos[i]:
        if p not in s: out.append((p, 1e9)); continue          # ion used by no training drug: cannot be recommended -> miss
        oth = pos[i] - {p}; keep = np.array([j not in oth for j in js])
        out.append((p, 1 + (arr[keep] > s[p]).sum() + 0.5 * ((arr[keep] == s[p]).sum() - 1)))
    return out

def subsample(Ttr, seed):
    p = np.where(Ttr.y == 1)[0]; n = np.where(Ttr.y == 0)[0]
    n = np.random.RandomState(seed).choice(n, min(len(n), 8000 - len(p)), replace=False)
    return Ttr.iloc[np.r_[p, n]]

# ---------------- (a) CV with ablations ----------------
variants = {'full': ()}; variants.update({f'without {f}': tuple(c) for f, c in FAMILIES.items()})
rows = []; OOF = []
for seed in (0, 1):
    folds = group_folds(seed)
    for k, te in enumerate(folds):
        tr = [i for i in use if i not in set(te)]
        Ttr = pair_table(tr, tr); Tte = pair_table(tr, list(te)); sub = subsample(Ttr, seed * 10 + k)
        cnt = np.zeros(ni)
        for i in tr:
            for j in pos[i]: cnt[j] += 1
        for vn, drop in variants.items():
            m, cols = make(drop); m.fit(sub[cols], sub.y.values); Tte['_s'] = m.predict_proba(Tte[cols])[:, 1]
            if vn == 'full' and seed == 0: OOF.append(Tte[['drug', 'ion', 'y', '_s']].copy())
            for i, g in Tte.groupby('drug'):
                rk = ranks_of(i, g.ion.values, g._s.values); best = min(r for _, r in rk)
                rows.append(dict(variant=vn, seed=seed, drug=i, hit1=best <= 1, hit5=best <= 5, hit10=best <= 10, mrr=1 / best, rare=[r for p, r in rk if cnt[p] <= 2]))
    print('CV seed', seed, 'done', flush=True)
R = pd.DataFrame(rows)
def per_drug(g, col): return g.groupby('drug')[col].mean().reindex(use)
B = np.random.RandomState(0).randint(0, len(use), (2000, len(use)))
full = R[R.variant == 'full']; fh, fm = per_drug(full, 'hit10').values, per_drug(full, 'mrr').values; fr = np.array([x for l in full.rare for x in l])
ab = []
for vn in variants:
    if vn == 'full': continue
    g = R[R.variant == vn]; h = per_drug(g, 'hit10').values; d = fh - h; rr = np.array([x for l in g.rare for x in l])
    ab.append(dict(family_removed=vn.replace('without ', ''), Hit10_full=fh.mean(), Hit10_without=h.mean(), drop_Hit10=d.mean(), drop_CI_low=np.percentile(d[B].mean(1), 2.5), drop_CI_high=np.percentile(d[B].mean(1), 97.5),
                   drop_MRR=fm.mean() - per_drug(g, 'mrr').mean(), rare_Hit10_full=(fr <= 10).mean(), rare_Hit10_without=(rr <= 10).mean()))
AB = pd.DataFrame(ab).round(3)
pos_drop = AB.drop_Hit10.clip(lower=0); AB['share_of_total_drop_%'] = (100 * pos_drop / pos_drop.sum()).round(1)
AB.to_csv('ablation_equation.csv', index=False); print(AB.to_string(index=False))

# ---------------- (b) Platt calibration on out-of-fold pairs ----------------
O = pd.concat(OOF); O['logit'] = np.log(np.clip(O._s, 1e-6, 1 - 1e-6) / (1 - np.clip(O._s, 1e-6, 1 - 1e-6)))
cal = LogisticRegression(C=1e6, max_iter=1000).fit(O[['logit']], O.y); a, b = float(cal.coef_[0, 0]), float(cal.intercept_[0])
O['p'] = cal.predict_proba(O[['logit']])[:, 1]
def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates='drop'); g = pd.DataFrame({'y': y, 'p': p, 'q': q}).groupby('q', observed=True)
    return float((g.size() / len(y) * (g.y.mean() - g.p.mean()).abs()).sum())
summ = pd.DataFrame([dict(version='raw score (trained on subsampled negatives)', brier=((O._s - O.y) ** 2).mean(), ece=ece(O.y, O._s), mean_pred=O._s.mean(), base_rate=O.y.mean()),
                     dict(version='after Platt calibration', brier=((O.p - O.y) ** 2).mean(), ece=ece(O.y, O.p), mean_pred=O.p.mean(), base_rate=O.y.mean())]).round(4)
summ.to_csv('calibration_summary.csv', index=False); print(summ.to_string(index=False))
json.dump({'platt_a': a, 'platt_b': b, 'note': 'P = 1/(1+exp(-(a*logit(raw)+b))), fitted on out-of-fold pairs of seed 0 (5 folds)'}, open('calibration_params.json', 'w'), indent=1)
q = pd.qcut(O.p, 10, duplicates='drop'); g = pd.DataFrame({'y': O.y, 'p': O.p, 'r': O._s, 'q': q}).groupby('q', observed=True).mean()
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].plot([0, g.p.max() * 1.1], [0, g.p.max() * 1.1], 'k--', lw=1, label='perfect'); ax[0].plot(g.p, g.y, 'o-', color='#1f6f8b', label='after calibration'); ax[0].plot(g.r, g.y, 's:', color='#c0504d', label='raw score')
ax[0].set_xlabel('predicted probability'); ax[0].set_ylabel('share of pairs that are real'); ax[0].set_title('Reliability (10 equal-size bins)'); ax[0].legend()
ax[1].hist(O.p[O.y == 0], bins=40, alpha=.6, label='not reported', color='#999', density=True); ax[1].hist(O.p[O.y == 1], bins=40, alpha=.6, label='reported pair', color='#1f6f8b', density=True)
ax[1].set_xlabel('calibrated probability'); ax[1].set_title('Score distribution'); ax[1].legend(); plt.tight_layout(); plt.savefig('fig_calibration.png', dpi=200); plt.close()

# ---------------- (c) final fit on all drugs + partial effects ----------------
Tall = pair_table(list(use), list(use)); sub = subsample(Tall, 123)
m, cols = make(); m.fit(sub[cols], sub.y.values); pd.to_pickle(dict(model=m, cols=cols, a=a, b=b), 'final_model.pkl')
ct = m[0]; w = m[1].coef_[0]; b0 = float(m[1].intercept_[0]); names = []
spl = ct.named_transformers_['s']; nb = spl.n_features_out_ // len(SMOOTH_ALL) if False else None
Xs = ct.transform(sub[cols]); k_per = (Xs.shape[1] - len(PLAIN_ALL)) // len(SMOOTH_ALL)
colmap = {}                                   # feature -> slice of coefficient vector
for q_, f in enumerate(SMOOTH_ALL): colmap[f] = slice(q_ * k_per, (q_ + 1) * k_per)
for q_, f in enumerate(PLAIN_ALL): colmap[f] = slice(len(SMOOTH_ALL) * k_per + q_, len(SMOOTH_ALL) * k_per + q_ + 1)
pd.DataFrame({'term': [f'{f}[{j}]' for f in SMOOTH_ALL for j in range(k_per)] + PLAIN_ALL, 'coefficient': w}).to_csv('equation_coefficients.csv', index=False)
pd.Series({'intercept': b0, 'spline_basis_per_feature': k_per, **{f: str(colmap[f]) for f in colmap}}).to_csv('equation_settings.csv')
def partial(df):
    X = ct.transform(df[cols]); return {fam: sum(X[:, colmap[f]] @ w[colmap[f]] for f in fs) for fam, fs in FAMILIES.items()}
base = partial(sub); base_mean = {f: v.mean() for f, v in base.items()}
pd.to_pickle(dict(base_mean=base_mean), 'family_baseline.pkl')
grid = {'sim2D': np.linspace(0, 1, 41), 'sim3D': np.linspace(0, 1, 41), 'dpKa': np.linspace(-8, 12, 41), 'ESP_comp': np.linspace(sub.ESP_comp.quantile(.01), sub.ESP_comp.quantile(.99), 41)}
med = sub[cols].median(); pe = []
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for q_, (f, xs) in enumerate(grid.items()):
    G = pd.DataFrame([med] * len(xs)); G[f] = xs
    if f == 'dpKa': G['P_salt'] = np.clip(17 * xs + 28, 0, 100) / 100
    contrib = ct.transform(G[cols])[:, colmap[f]] @ w[colmap[f]]
    if f == 'dpKa': contrib = contrib + ct.transform(G[cols])[:, colmap['P_salt']] @ w[colmap['P_salt']]
    contrib = contrib - np.interp(med[f], xs, contrib)
    for xv, cv in zip(xs, contrib): pe.append(dict(feature=f, value=xv, contribution_logit=cv))
    ax[q_].plot(xs, contrib, color='#1f6f8b'); ax[q_].axhline(0, color='k', lw=.6); ax[q_].set_xlabel(f); ax[q_].set_ylabel('contribution to score (logit)')
plt.suptitle('How each feature changes the score (others held at their median)'); plt.tight_layout(); plt.savefig('fig_partial_effects.png', dpi=200); plt.close()
pd.DataFrame(pe).round(4).to_csv('partial_effects.csv', index=False); print('final fit done; intercept', round(b0, 3), 'Platt a,b =', round(a, 3), round(b, 3))
