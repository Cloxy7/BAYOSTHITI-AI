"""Why ESP complementarity? Same equation, same leave-drug-out folds (5 folds x seeds 0,1), same 8,000-row training sample.
Base = 2D + 3D + pKa (no electrostatic group). Each variant adds ONE electronic / electrostatic pair descriptor from xTB.
Also: does the 8,000-row cap matter? Set A refit with 4,000 / 8,000 / 12,000 / all training rows.
Writes esp_choice_test.csv and training_rows_test.csv"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import features; features.inject(globals())
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
L = np.log1p
d_dip, i_dip = L(md.dipole_D.values.astype(float)), L(mi.dipole_D.values.astype(float))
d_chi, i_chi = md.electronegativity_eV.values.astype(float), mi.electronegativity_eV.values.astype(float)
d_eta, i_eta = md.hardness_eV.values.astype(float), mi.hardness_eV.values.astype(float)
d_pol, i_pol = L(md.polarisability_au.values.astype(float)), L(mi.polarisability_au.values.astype(float))
i_spr = mi.charge_spread_A.values.astype(float); d_spr = md.charge_spread_A.values.astype(float)
def fill(x): x = x.copy(); x[~np.isfinite(x)] = np.nanmedian(x); return x
d_dip, i_dip, d_chi, i_chi, d_eta, i_eta, d_pol, i_pol, i_spr, d_spr = map(fill, [d_dip, i_dip, d_chi, i_chi, d_eta, i_eta, d_pol, i_pol, i_spr, d_spr])
def add(T):
    i, j = T.drug.values, T.ion.values
    T['dipole_gap'] = np.abs(d_dip[i] - i_dip[j]); T['ion_dipole'] = i_dip[j]
    T['chi_gap'] = np.abs(d_chi[i] - i_chi[j]); T['eta_sum'] = d_eta[i] + i_eta[j]
    T['pol_gap'] = np.abs(d_pol[i] - i_pol[j]); T['charge_spread_sum'] = d_spr[i] + i_spr[j]
    return T
BS, BP = ['sim2D', 'sim3D', 'dpKa'], ['P_salt', 'pKa_missing', 'perm_cation']
SETS = {'No electrostatic group (2D + 3D + pKa)': [],
        'ESP complementarity (chosen)': ['ESP_comp'],
        'Dipole moment (ion dipole + drug-ion gap)': ['ion_dipole', 'dipole_gap'],
        'Electronegativity gap': ['chi_gap'],
        'Chemical hardness (sum)': ['eta_sum'],
        'Polarisability gap': ['pol_gap'],
        'Charge spread (delocalisation, sum)': ['charge_spread_sum']}
def make(sm, pl):
    return make_pipeline(ColumnTransformer([('s', SplineTransformer(n_knots=5, degree=3, extrapolation='constant'), sm), ('p', StandardScaler(), pl)]),
                         LogisticRegression(C=1.0, max_iter=5000))
def ranks_of(i, js, sc):
    s = dict(zip(js, sc)); arr = np.array(sc); out = []
    for p in pos[i]:
        if p not in s: out.append(1e9); continue
        oth = pos[i] - {p}; keep = np.array([j not in oth for j in js])
        out.append(1 + (arr[keep] > s[p]).sum() + 0.5 * ((arr[keep] == s[p]).sum() - 1))
    return min(out)
rows, rows2 = [], []
for seed in (0, 1):
    for k, te in enumerate(group_folds(seed)):
        tr = [i for i in use if i not in set(te)]
        Ttr = add(pair_table(tr, tr)); Tte = add(pair_table(tr, list(te)))
        frs = np.random.RandomState(seed * 10 + k)
        p_idx = np.where(Ttr.y == 1)[0]; n_all = np.where(Ttr.y == 0)[0]
        n_idx = frs.choice(n_all, min(len(n_all), 8000 - len(p_idx)), replace=False); sub = Ttr.iloc[np.r_[p_idx, n_idx]]
        for name, extra in SETS.items():
            sm = BS + extra; m = make(sm, BP).fit(sub[sm + BP], sub.y.values); Tte['_s'] = m.predict_proba(Tte[sm + BP])[:, 1]
            for i, g in Tte.groupby('drug'):
                b = ranks_of(i, g.ion.values, g._s.values); rows.append(dict(model=name, seed=seed, drug=i, hit1=b <= 1, hit5=b <= 5, hit10=b <= 10, mrr=1 / b))
        sm = BS + ['ESP_comp']
        for N in (4000, 8000, 12000, None):
            r2 = np.random.RandomState(100 + seed * 10 + k)
            nn = n_all if N is None else r2.choice(n_all, min(len(n_all), N - len(p_idx)), replace=False)
            s2_ = Ttr.iloc[np.r_[p_idx, nn]]; m = make(sm, BP).fit(s2_[sm + BP], s2_.y.values); Tte['_s'] = m.predict_proba(Tte[sm + BP])[:, 1]
            for i, g in Tte.groupby('drug'):
                b = ranks_of(i, g.ion.values, g._s.values)
                rows2.append(dict(rows_used='all' if N is None else N, n_rows=len(s2_), n_pos=len(p_idx), seed=seed, drug=i, hit1=b <= 1, hit5=b <= 5, hit10=b <= 10, mrr=1 / b))
    print('seed', seed, 'done', flush=True)
R = pd.DataFrame(rows); R2 = pd.DataFrame(rows2)
B = np.random.RandomState(0).randint(0, len(use), (2000, len(use)))
def pdv(g, c): return g.groupby('drug')[c].mean().reindex(use).values
base = pdv(R[R.model == 'No electrostatic group (2D + 3D + pKa)'], 'hit10')
out = []
for name in SETS:
    g = R[R.model == name]; h = pdv(g, 'hit10'); d = h - base; bd = d[B].mean(1); bh = h[B].mean(1)
    out.append(dict(variant=name, Hit1=g.groupby('drug').hit1.mean().mean(), Hit5=g.groupby('drug').hit5.mean().mean(), Hit10=h.mean(),
                    Hit10_CI_low=np.percentile(bh, 2.5), Hit10_CI_high=np.percentile(bh, 97.5), MRR=pdv(g, 'mrr').mean(),
                    gain_vs_no_electrostatic=d.mean(), gain_CI_low=np.percentile(bd, 2.5), gain_CI_high=np.percentile(bd, 97.5)))
O = pd.DataFrame(out).round(3); O.to_csv('esp_choice_test.csv', index=False); print(O.to_string(index=False))
o2 = []
for N, g in R2.groupby('rows_used', sort=False):
    h = pdv(g, 'hit10'); bh = h[B].mean(1)
    o2.append(dict(training_rows=N, mean_rows=int(g.groupby(['seed']).n_rows.mean().mean()), mean_positives=int(g.n_pos.mean()), Hit1=g.groupby('drug').hit1.mean().mean(),
                   Hit10=h.mean(), Hit10_CI_low=np.percentile(bh, 2.5), Hit10_CI_high=np.percentile(bh, 97.5), MRR=pdv(g, 'mrr').mean()))
O2 = pd.DataFrame(o2).round(3); O2.to_csv('training_rows_test.csv', index=False); print(O2.to_string(index=False))
