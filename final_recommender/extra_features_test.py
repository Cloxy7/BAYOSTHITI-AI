"""Does adding dipole, solubility or toxicity help the score equation?
Same equation (splines, 5 knots, C=1), same leave-drug-out folds (5 folds x seeds 0,1), same rule (ion with no training user = miss).
New groups:
  dipole     : ion dipole (xTB, Debye, log1p) and the gap |log1p(drug dipole) - log1p(ion dipole)|. All molecules have a value.
  solubility : ion logS (measured, free form or in a salt) and drug logS - ion logS. Measured for 135/291 ions, 78/138 drugs.
  toxicity   : ion log10 LD50 (rat oral, mg/kg, measured). Only 46/291 ions have it.
Missing values are filled with the median of the ions that have one. No missing flag goes into the model.
Control: 'measured or not' flag only. Well-studied (popular) ions are the ones with measured values, so if the flag alone gives the
same gain as the real values, the gain is about popularity and not about solubility or toxicity.
Writes: extra_features_test.csv, extra_features_coverage.csv"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import features; features.inject(globals())
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------- per-molecule values
ci = pd.read_csv('../measured_properties/counter_ions_measured_properties.csv'); cd = pd.read_csv('../measured_properties/drugs_measured_properties.csv')
assert (ci.id.values == I.unique_ion_id.values).all() and (cd.id.values == D.drug_id.values).all()
i_dip = np.log1p(mi.dipole_D.values.astype(float)); d_dip = np.log1p(md.dipole_D.values.astype(float))
i_logS = ci.logS.values.astype(float); d_logS = cd.logS.values.astype(float)
i_ld50 = np.log10(ci.ld50_mgkg.values.astype(float))
cov = pd.DataFrame([dict(value='dipole (xTB)', ions=int(np.isfinite(i_dip).sum()), drugs=int(np.isfinite(d_dip).sum())),
                    dict(value='logS (measured)', ions=int(np.isfinite(i_logS).sum()), drugs=int(np.isfinite(d_logS).sum())),
                    dict(value='LD50 (measured)', ions=int(np.isfinite(i_ld50).sum()), drugs=None)]).assign(n_ions=ni, n_drugs=nd)
cov.to_csv('extra_features_coverage.csv', index=False); print(cov.to_string(index=False))

def fill(x): x = x.copy(); x[~np.isfinite(x)] = np.nanmedian(x); return x
I_DIP, D_DIP, I_LOGS, D_LOGS, I_LD50 = fill(i_dip), fill(d_dip), fill(i_logS), fill(d_logS), fill(i_ld50)

def add_cols(T):
    i, j = T.drug.values, T.ion.values
    T['ion_dipole'] = I_DIP[j]; T['dipole_gap'] = np.abs(D_DIP[i] - I_DIP[j])
    T['ion_logS'] = I_LOGS[j]; T['logS_gap'] = D_LOGS[i] - I_LOGS[j]
    T['ion_logLD50'] = I_LD50[j]
    T['logS_measured'] = np.isfinite(i_logS[j]).astype(int); T['LD50_measured'] = np.isfinite(i_ld50[j]).astype(int)
    return T

A_S, A_P = ['sim2D', 'sim3D', 'dpKa', 'ESP_comp'], ['P_salt', 'pKa_missing', 'perm_cation']
NEW = {'dipole': (['ion_dipole', 'dipole_gap'], []), 'solubility': (['ion_logS', 'logS_gap'], []), 'toxicity': (['ion_logLD50'], [])}
CTRL = {'solubility measured? (flag only)': ([], ['logS_measured']), 'toxicity measured? (flag only)': ([], ['LD50_measured'])}
SETS = {'Set A (current score)': (A_S, A_P)}
for g, (s, p) in NEW.items(): SETS[f'Set A + {g}'] = (A_S + s, A_P + p)
SETS['Set A + dipole + solubility + toxicity'] = (A_S + sum([v[0] for v in NEW.values()], []), A_P)
for g, (s, p) in CTRL.items(): SETS[f'Set A + {g}'] = (A_S, A_P + p)
for g, (s, p) in NEW.items(): SETS[f'{g} alone'] = (s, p)

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

R = pd.DataFrame(rows); pd.to_pickle(R, 'per_drug_extra_features.pkl')
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
S = pd.DataFrame(out).round(3); S.to_csv('extra_features_test.csv', index=False); print(S.to_string(index=False))
