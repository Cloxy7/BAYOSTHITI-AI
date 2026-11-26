"""Pair features for the API-IL counter-ion recommender.
Families: 2D structural similarity, 3D shape similarity, pKa (dpKa, salt probability), ESP complementarity. No popularity feature.
Similarity features follow the 'ions used by similar drugs' idea and only ever use TRAINING drugs (never the drug itself).

Importable module. Three ways to use it:

  import features
  F = features.load()                      # -> namespace with D, I, P, S2, S3, pos, use, pair_table, ...
  T = F.pair_table(train_idx, drug_idx)

  import features; features.inject(globals())     # legacy: every name as a module global (what the
                                                  # old `exec(open('features.py').read())` line did)

  from features import load
  F = load('/path/to/final_recommender')   # run from any working directory

load() reads its CSVs from the folder holding this file unless a path is given, and caches per folder,
so importing it from several scripts in one process builds the similarity matrices once rather than once each.
"""
import os
from types import SimpleNamespace

import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.MolStandardize import rdMolStandardize as MS

RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS_A = ['sim2D', 'sim3D', 'P_salt', 'dpKa', 'pKa_missing', 'perm_cation', 'ESP_comp']
SHAPE_COLS = ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2', 'RadiusOfGyration', 'Asphericity', 'Eccentricity',
              'InertialShapeFactor', 'SpherocityIndex', 'PBF', 'max_extent', 'vdW_volume_proxy']
LOG_COLS = ['PMI1', 'PMI2', 'PMI3', 'vdW_volume_proxy', 'InertialShapeFactor']

_CACHE = {}


def load(data_dir=None):
    """Build every table and matrix the model needs. Cached per folder."""
    d = os.path.abspath(data_dir or HERE)
    if d not in _CACHE:
        _CACHE[d] = _build(d)
    return _CACHE[d]


def inject(namespace, data_dir=None):
    """Copy every public name into `namespace` (pass globals()). Returns the namespace object too,
    so new code can say `F = features.inject(globals())` and then use F.A, F.D ... unambiguously."""
    F = load(data_dir)
    namespace.update({k: v for k, v in vars(F).items() if not k.startswith('_')})
    return F


def __getattr__(name):
    """features.D, features.pair_table, ... resolve lazily against the default folder."""
    try:
        return getattr(load(), name)
    except AttributeError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None


def _build(d):
    j = lambda f: os.path.join(d, f)
    D = pd.read_csv(j('drugs.csv')); I = pd.read_csv(j('ions.csv')); P = pd.read_csv(j('positives.csv'))
    nd, ni = len(D), len(I); dsign = D.sign.values; isign = I.sign.values
    pos = [set() for _ in range(nd)]
    for a, b in P.values:
        if isign[b] == -dsign[a]: pos[a].add(b)
    cand = [np.where(isign == -dsign[i])[0] for i in range(nd)]
    use = [i for i in range(nd) if pos[i]]

    # ---------- 2D similarity: Morgan (r=2, 2048 bits) Tanimoto on the neutral form of each drug
    U = MS.Uncharger(); gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    def neutral(s): return U.uncharge(Chem.MolFromSmiles(s))
    FP = [gen.GetFingerprint(neutral(s)) for s in D.smiles]
    # ---------- same-parent forms (e.g. ethambutol drawn twice, ciprofloxacin cation + anion): they are one molecule, so they must stay together in a fold
    PARENT = np.array([MolToInchiKey(neutral(s)).split('-')[0] for s in D.smiles])
    SIB = [np.where(PARENT == PARENT[i])[0] for i in range(nd)]
    def group_folds(seed, k=5):
        groups = {}
        for i in use: groups.setdefault(PARENT[i], []).append(i)
        keys = list(groups); order = np.random.RandomState(seed).permutation(len(keys)); folds = [[] for _ in range(k)]
        for n, o in enumerate(order): folds[n % k].extend(groups[keys[o]])
        return [np.array(f) for f in folds]
    S2 = np.array([DataStructs.BulkTanimotoSimilarity(f, FP) for f in FP])

    # ---------- 3D similarity: z-scored shape descriptors (RDKit conformers), similarity = 1/(1 + distance/sqrt(p))
    shape_cols = SHAPE_COLS
    X3 = pd.read_csv(j('drugs_3D_descriptors.csv')).drop_duplicates('smiles').set_index('smiles').loc[D.smiles]
    A = X3[shape_cols].astype(float).copy()
    for c in LOG_COLS: A[c] = np.log1p(A[c].clip(lower=0))
    A = A.fillna(A.median()); Z = ((A - A.mean()) / A.std(ddof=0)).values
    dist = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)) / np.sqrt(Z.shape[1]); S3 = 1 / (1 + dist)

    # ---------- pKa: dpKa = pKa(base, conjugate acid) - pKa(acid); P(salt) = 17*dpKa + 28 % (Cruz-Cabeza 2012), clipped to 0-100 %
    dp = D.pKa.values.astype(float); ip = I.pKa.values.astype(float); perm = I.perm.values.astype(int)
    def dpka(i, js): return (dp[i] - ip[js]) if dsign[i] > 0 else (ip[js] - dp[i])

    # ---------- ESP complementarity: most positive surface potential of the cation - most negative of the anion (xTB charges)
    M = pd.read_csv(j('xtb_molecule_properties.csv'))
    md = M[M.mol_id.str.startswith('D')].set_index('smiles').loc[D.smiles]
    mi = M[M.mol_id.str.startswith('U')].set_index('smiles').loc[I.smiles]
    d_esp = np.where(dsign > 0, md.ESP_max.values, md.ESP_min.values)
    i_esp = np.where(isign > 0, mi.ESP_max.values, mi.ESP_min.values)
    def esp(i, js): return (d_esp[i] - i_esp[js]) if dsign[i] > 0 else (i_esp[js] - d_esp[i])

    def pair_table(train, drugs):
        """features for every (drug, candidate ion) of `drugs`, using only `train` drugs as references (a drug never references itself)."""
        tr = np.zeros(nd, bool); tr[list(train)] = True
        Umat = np.zeros((ni, nd), bool)                       # ion x drug: drug (in train) uses ion
        for i in np.where(tr)[0]:
            for jj in pos[i]: Umat[jj, i] = True
        rows = []
        for i in drugs:
            js = cand[i]; Ui = Umat[js].copy(); Ui[:, SIB[i]] = False
            s2 = np.where(Ui, S2[i][None], 0).max(1); s3 = np.where(Ui, S3[i][None], 0).max(1)
            dk = dpka(i, js); miss = np.isnan(dk) & (perm[js] == 0)
            ps = np.clip(17 * dk + 28, 0, 100) / 100
            ps = np.where(perm[js] == 1, 1.0, ps)            # permanent cation: no proton transfer needed
            dkf = np.where(perm[js] == 1, 10.0, dk)
            ok = Ui.any(1)   # ions with no training user are dropped: an ion nobody in training uses must belong to the held-out drug (leak)
            rows.append(pd.DataFrame({'drug': i, 'ion': js, 'y': np.isin(js, list(pos[i])).astype(int), 'sim2D': s2, 'sim3D': s3,
                                      'P_salt': ps, 'dpKa': dkf, 'pKa_missing': miss.astype(int), 'perm_cation': perm[js],
                                      'ESP_comp': esp(i, js)})[ok])
        T = pd.concat(rows, ignore_index=True)
        med = {'P_salt': np.nanmedian(T.P_salt), 'dpKa': np.nanmedian(T.dpKa)}
        T['P_salt'] = T.P_salt.fillna(med['P_salt']); T['dpKa'] = T.dpKa.fillna(med['dpKa'])
        return T

    return SimpleNamespace(
        DIR=d,
        np=np, pd=pd, Chem=Chem, DataStructs=DataStructs, RDLogger=RDLogger,
        rdFingerprintGenerator=rdFingerprintGenerator, MS=MS, MolToInchiKey=MolToInchiKey,
        D=D, I=I, P=P, nd=nd, ni=ni, dsign=dsign, isign=isign, pos=pos, cand=cand, use=use,
        U=U, gen=gen, neutral=neutral, FP=FP, PARENT=PARENT, SIB=SIB, group_folds=group_folds, S2=S2,
        X3=X3, shape_cols=shape_cols, A=A, Z=Z, dist=dist, S3=S3,
        dp=dp, ip=ip, perm=perm, dpka=dpka,
        M=M, md=md, mi=mi, d_esp=d_esp, i_esp=i_esp, esp=esp,
        FEATS_A=FEATS_A, pair_table=pair_table,
    )
