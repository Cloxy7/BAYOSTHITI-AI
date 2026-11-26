"""Recommend counter-ions for a drug that is NOT in our data (a new, unpublished drug).
Same score equation as the report (2D + 3D + pKa + ESP), trained on all 138 known drugs; the drug itself is never a reference.
Use:  from recommend_new_drug import recommend ;  R = recommend('CC(C)Cc1ccc(cc1)C(C)C(=O)[O-]', name='my drug', pKa=4.4)
SMILES must be the charged form that would sit in the salt (protonated base = cation, deprotonated acid = anion).
pKa (optional): MEASURED pKa of the drug (conjugate acid for a base). Without it the pKa group stays neutral.
ESP is not computed here (needs xTB); the training median is used, which keeps that group neutral (it is only about 4% of the score)."""
import numpy as np, pandas as pd, warnings, json; warnings.filterwarnings('ignore')
import reason
import features; features.inject(globals())
from rdkit.Chem import AllChem, rdMolDescriptors as rd, Descriptors3D as D3
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

SMOOTH = ['sim2D', 'sim3D', 'dpKa', 'ESP_comp']; PLAIN = ['P_salt', 'pKa_missing', 'perm_cation']; COLS = SMOOTH + PLAIN
FAMILIES = {'2D similarity': ['sim2D'], '3D similarity': ['sim3D'], 'pKa': ['dpKa', 'P_salt', 'pKa_missing', 'perm_cation'], 'ESP complementarity': ['ESP_comp']}
CAL = json.load(open('calibration_params.json'))
NAMES3D = ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2', 'RadiusOfGyration', 'Asphericity', 'Eccentricity', 'InertialShapeFactor', 'SpherocityIndex', 'PBF', 'dipole_gasteiger', 'max_extent', 'vdW_volume_proxy']

def _d3(smi):                       # same recipe as the descriptors of the known drugs (5 ETKDG conformers, MMFF, averaged)
    m = Chem.AddHs(Chem.MolFromSmiles(smi.split('.')[0])); ids = list(AllChem.EmbedMultipleConfs(m, numConfs=5, randomSeed=42))
    if not ids:
        ps = AllChem.ETKDGv3(); ps.useRandomCoords = True; ps.randomSeed = 42; ids = list(AllChem.EmbedMultipleConfs(m, 5, ps))
    if not ids: raise ValueError('could not build a 3D structure for this SMILES')
    try: AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500)
    except Exception: pass
    AllChem.ComputeGasteigerCharges(m); q = np.nan_to_num(np.array([float(a.GetProp('_GasteigerCharge')) for a in m.GetAtoms()])); out = []
    for c in ids:
        x = m.GetConformer(c).GetPositions(); ctr = x.mean(0)
        out.append([D3.PMI1(m, confId=c), D3.PMI2(m, confId=c), D3.PMI3(m, confId=c), D3.NPR1(m, confId=c), D3.NPR2(m, confId=c), D3.RadiusOfGyration(m, confId=c), D3.Asphericity(m, confId=c),
                    D3.Eccentricity(m, confId=c), D3.InertialShapeFactor(m, confId=c), D3.SpherocityIndex(m, confId=c), rd.CalcPBF(m, confId=c), np.linalg.norm(((x - ctr) * q[:, None]).sum(0)),
                    np.max(np.linalg.norm(x[:, None] - x[None], axis=2)), AllChem.ComputeMolVolume(m, confId=c)])
    return dict(zip(NAMES3D, np.mean(out, 0)))

def _make():
    ct = ColumnTransformer([('s', SplineTransformer(n_knots=5, degree=3, extrapolation='constant'), SMOOTH), ('p', StandardScaler(), PLAIN)])
    return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=5000))

_MODEL = {}
def _fit():
    if 'm' in _MODEL: return _MODEL['m']
    Tall = pair_table(list(use), list(use)); p = np.where(Tall.y == 1)[0]; n = np.where(Tall.y == 0)[0]
    n = np.random.RandomState(123).choice(n, min(len(n), 8000 - len(p)), replace=False); sub = Tall.iloc[np.r_[p, n]]
    m = _make().fit(sub[COLS], sub.y.values); _MODEL['m'] = (m, sub, Tall.ESP_comp.median()); return _MODEL['m']

def recommend(smiles, name='new drug', pKa=None, top=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError('SMILES not valid')
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if charge == 0: raise ValueError('Give the charged form: protonated base ([NH+]...) or deprotonated acid ([O-]...)')
    sign = 1 if charge > 0 else -1
    # same molecule already in our data? then leave all its forms out (so the answer is a fair 'new drug' answer)
    key = MolToInchiKey(neutral(smiles)).split('-')[0]; same = set(np.where(PARENT == key)[0])
    m, sub, esp_med = _fit(); ct, lr = m[0], m[1]; w = lr.coef_[0]
    # --- similarity of the new drug to every known drug
    f_new = gen.GetFingerprint(neutral(smiles)); s2 = np.array(DataStructs.BulkTanimotoSimilarity(f_new, FP))
    row = pd.Series(_d3(smiles)); row = row[shape_cols].astype(float)
    for c in ['PMI1', 'PMI2', 'PMI3', 'vdW_volume_proxy', 'InertialShapeFactor']: row[c] = np.log1p(max(row[c], 0))
    row = row.fillna(A.median()); z = ((row - A.mean()) / A.std(ddof=0)).values
    s3 = 1 / (1 + np.sqrt(((Z - z) ** 2).sum(1)) / np.sqrt(Z.shape[1]))
    ref = [k for k in use if k not in same]; Umat = np.zeros((ni, nd), bool)
    for k in ref:
        for j in pos[k]: Umat[j, k] = True
    js = np.where(isign == -sign)[0]; Ui = Umat[js]; ok = Ui.any(1); js = js[ok]; Ui = Ui[ok]
    f = pd.DataFrame({'ion': js, 'sim2D': np.where(Ui, s2[None], 0).max(1), 'sim3D': np.where(Ui, s3[None], 0).max(1)})
    permj = perm[js]
    if pKa is None or (isinstance(pKa, float) and np.isnan(pKa)):
        dk = np.full(len(js), np.nan); has = False
    else:
        dk = (pKa - ip[js]) if sign > 0 else (ip[js] - pKa); has = True
    miss = np.isnan(dk) & (permj == 0); ps = np.clip(17 * dk + 28, 0, 100) / 100
    ps = np.where(permj == 1, 1.0, ps); dkf = np.where(permj == 1, 10.0, dk)
    f['P_salt'] = np.where(np.isnan(ps), np.nanmedian(sub.P_salt), ps); f['dpKa'] = np.where(np.isnan(dkf), np.nanmedian(sub.dpKa), dkf)
    # ESP needs xTB, so a new drug gets the training median for EVERY candidate ion. The ESP spline term is then
    # the same constant on every row: it shifts all scores equally and cannot change the ranking (see meta['esp_note']).
    f['pKa_missing'] = miss.astype(int); f['perm_cation'] = permj; f['ESP_comp'] = esp_med
    X = ct.transform(f[COLS]); Xs = ct.transform(sub[COLS]); k = (Xs.shape[1] - len(PLAIN)) // len(SMOOTH)
    cm = {c: slice(q * k, (q + 1) * k) for q, c in enumerate(SMOOTH)}
    for q, c in enumerate(PLAIN): cm[c] = slice(len(SMOOTH) * k + q, len(SMOOTH) * k + q + 1)
    raw = lr.predict_proba(X)[:, 1]; lg = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    f['score'] = 100 / (1 + np.exp(-(CAL['platt_a'] * lg + CAL['platt_b'])))
    for fam, fs in FAMILIES.items(): f['c_' + fam] = sum(X[:, cm[c]] @ w[cm[c]] for c in fs) - sum(Xs[:, cm[c]] @ w[cm[c]] for c in fs).mean()
    f['ion_name'] = [I.name[j] if isinstance(I.name[j], str) else '' for j in f.ion]; f['ion_smiles'] = [I.smiles[j] for j in f.ion]
    f['dpKa_value'] = dk
    f['pKa_label'] = ['permanent cation (no proton transfer)' if p_ == 1 else ('pKa not given' if np.isnan(d) else ('salt likely (dpKa > 4)' if d > 4 else ('co-crystal likely (dpKa < -1)' if d < -1 else 'ambiguous (-1 to 4)'))) for d, p_ in zip(dk, permj)]
    sim_users = []; sim_list = []; nusers = []
    for r_, j in enumerate(f.ion):
        allk = sorted([k_ for k_ in ref if j in pos[k_]], key=lambda k_: -s2[k_]); us = allk[:3]
        sim_users.append('; '.join(f'{D.name[k_]} ({s2[k_]:.2f})' for k_ in us))
        sim_list.append([(str(D.name[k_]), float(s2[k_])) for k_ in us]); nusers.append(len(allk))
    f['similar_known_drugs_using_ion'] = sim_users; f['_sim'] = sim_list; f['_nusers'] = nusers
    f = f.sort_values('score', ascending=False).reset_index(drop=True); f.insert(0, 'rank', np.arange(1, len(f) + 1))
    safe_col = I.safety_tag.values if 'safety_tag' in I.columns else None
    ncand_tot = len(f); chem = []; full = []; safes = []
    for _, r_ in f.iterrows():
        st = safe_col[int(r_['ion'])] if safe_col is not None else 'unclassified'
        dv = r_['dpKa_value']; dv = None if (isinstance(dv, float) and np.isnan(dv)) else dv
        rs = reason.make_reason(name, (r_['ion_name'] or r_['ion_smiles'][:24]), int(r_['rank']), ncand_tot, float(r_['score']), r_['pKa_label'], dv,
                                [float(r_['c_2D similarity']), float(r_['c_3D similarity']), float(r_['c_pKa']), float(r_['c_ESP complementarity'])],
                                r_['_sim'], int(r_['_nusers']), st, False)
        chem.append(rs['chem']); full.append(rs['full']); safes.append(rs['safe'])
    f['safety'] = safes; f['reason'] = full
    meta = dict(name=name, smiles=smiles, role='base (cation)' if sign > 0 else 'acid (anion)', pKa=None if not has else float(pKa), pKa_given=has, esp_computed=False,
                esp_note='ESP was not computed for this drug (it needs xTB). The training median is used for every candidate, '
                         'so the ESP column is identical down the table and the ranking comes from 2D, 3D and pKa only.',
                same_molecule_in_data=[D.name[k_] for k_ in sorted(same)], n_candidates=ncand_tot,
                cards=[{'rk': int(rk), 'chem': c, 'safe': s} for rk, c, s in zip(f['rank'], chem, safes)])
    cols = ['rank', 'ion_name', 'ion_smiles', 'score', 'pKa_label', 'dpKa_value', 'sim2D', 'sim3D', 'c_2D similarity', 'c_3D similarity', 'c_pKa', 'c_ESP complementarity', 'similar_known_drugs_using_ion', 'safety', 'reason']
    return f[cols].round(4), meta


def to_app_json(R, meta, path='new_drug_result.json', top=30):
    """Write the result in the format the web app can load ('Your own drug' box)."""
    cardmap = {c['rk']: c for c in meta.get('cards', [])}
    rows = []
    for r in R.head(top).to_dict('records'):
        sim = []
        for s in str(r['similar_known_drugs_using_ion']).split('; '):
            if ' (' in s: nm, v = s.rsplit(' (', 1); sim.append([nm, float(v.rstrip(')'))])
        card = cardmap.get(int(r['rank']), {})
        rows.append(dict(rk=int(r['rank']), ion=r['ion_name'] or r['ion_smiles'][:28], smi=r['ion_smiles'], sc=round(float(r['score']), 2), rep=0, lab=r['pKa_label'],
                         dp=None if pd.isna(r['dpKa_value']) else round(float(r['dpKa_value']), 2), s2=round(float(r['sim2D']), 2), s3=round(float(r['sim3D']), 2),
                         c=[round(float(r['c_2D similarity']), 2), round(float(r['c_3D similarity']), 2), round(float(r['c_pKa']), 2), round(float(r['c_ESP complementarity']), 2)],
                         sim=sim, just=card.get('chem', ''), safe=card.get('safe', '')))
    out = dict(name=meta['name'], smi=meta['smiles'], role=meta['role'], pka=meta['pKa'], src='new drug entered by the user (ESP not computed' + ('' if meta['pKa_given'] else ', pKa not given') + ')',
               ncand=meta['n_candidates'], reported=[], rows=rows)
    json.dump(out, open(path, 'w')); return path
