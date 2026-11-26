"""GFN2-xTB (tblite) with ALPB implicit water: deprotonation energies for pKa calibration,
and molecular dipole moments. Results cached to JSON so runs can be resumed."""
import json, os, sys, time
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tblite.interface import Calculator
RDLogger.DisableLog('rdApp.*')

CACHE = '/tmp/xtb/cache.json'
os.makedirs('/tmp/xtb', exist_ok=True)
ANG2BOHR = 1.8897261254578281
AU2DEBYE = 2.5417464519

_reion = rdMolStandardize.Reionizer()
BASIC_N = Chem.MolFromSmarts('[NX3;+0;!$(N-[C,S,P]=[O,S,N]);!$(N-a);!$(N-[C,S]#*)]')
COOH = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')


def add_one_proton(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    neg = [a for a in m.GetAtoms() if a.GetFormalCharge() < 0]
    if not neg:
        return None
    amine = m.GetSubstructMatches(BASIC_N)
    if amine:
        a = m.GetAtomWithIdx(amine[0][0])
        a.SetFormalCharge(1); a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
        try:
            Chem.SanitizeMol(m); return Chem.MolToSmiles(m)
        except Exception:
            return None
    order = {'O': 0, 'N': 1, 'S': 2, 'C': 3}
    neg.sort(key=lambda a: order.get(a.GetSymbol(), 9))
    a = neg[0]
    single_halide = m.GetNumHeavyAtoms() == 1 and a.GetSymbol() in ('F', 'Cl', 'Br', 'I')
    if a.GetSymbol() not in order and not single_halide:
        return None
    a.SetFormalCharge(a.GetFormalCharge() + 1)
    a.SetNumExplicitHs(a.GetTotalNumHs() + 1); a.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(m); return Chem.MolToSmiles(m)
    except Exception:
        return None


def remove_one_proton(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    pos = [a for a in m.GetAtoms() if a.GetFormalCharge() > 0 and a.GetTotalNumHs() > 0]
    if not pos:
        return None
    acid = m.GetSubstructMatches(COOH)
    if acid and pos[0].GetSymbol() == 'N' and not pos[0].GetIsAromatic():
        o = m.GetAtomWithIdx(acid[0][2])
        o.SetFormalCharge(-1); o.SetNumExplicitHs(0); o.SetNoImplicit(True)
        try:
            Chem.SanitizeMol(m); return Chem.MolToSmiles(m)
        except Exception:
            return None
    a = pos[0]
    a.SetFormalCharge(a.GetFormalCharge() - 1)
    a.SetNumExplicitHs(a.GetTotalNumHs() - 1); a.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(m); return Chem.MolToSmiles(m)
    except Exception:
        return None


def make_pair(smiles, ion_type):
    """(acid_form, base_form, which calibration line) -- same rules as the DFT pipeline."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None, None
    if ion_type == 'anion':
        base = Chem.MolToSmiles(_reion.reionize(m))
        acid = add_one_proton(base)
        kind = 'base' if Chem.MolFromSmiles(base).HasSubstructMatch(BASIC_N) else 'acid'
    else:
        acid, base = smiles, remove_one_proton(smiles)
        pos = [a for a in m.GetAtoms() if a.GetFormalCharge() > 0 and a.GetTotalNumHs() > 0]
        cooh = bool(pos) and m.HasSubstructMatch(COOH) and pos[0].GetSymbol() == 'N' and not pos[0].GetIsAromatic()
        kind = 'acid' if cooh else 'base'
    return acid, base, kind


def best_conformer(smiles, n_conf=5, seed=42):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None, None
    mh = Chem.AddHs(m)
    if mh.GetNumAtoms() == 1:
        a = mh.GetAtomWithIdx(0)
        return np.array([a.GetAtomicNum()]), np.zeros((1, 3)), Chem.GetFormalCharge(m)
    p = AllChem.ETKDGv3(); p.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, params=p))
    if not cids:
        p.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, params=p))
    if not cids:
        return None, None, None
    if AllChem.MMFFHasAllMoleculeParams(mh):
        res = AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=1000)
    else:
        res = AllChem.UFFOptimizeMoleculeConfs(mh, maxIters=1000)
    best = cids[int(np.argmin([e for _, e in res]))]
    conf = mh.GetConformer(best)
    nums = np.array([a.GetAtomicNum() for a in mh.GetAtoms()])
    pos = conf.GetPositions() * ANG2BOHR          # tblite expects Bohr
    return nums, pos, Chem.GetFormalCharge(m)


def xtb_single(smiles, solvent='water'):
    """GFN2-xTB energy (Hartree) and dipole magnitude (Debye) in ALPB implicit water."""
    nums, pos, charge = best_conformer(smiles)
    if nums is None:
        return None
    calc = Calculator('GFN2-xTB', nums, pos, charge=float(charge))
    if solvent:
        calc.add('alpb-solvation', solvent)
    calc.set('verbosity', 0)
    res = calc.singlepoint()
    dip = np.asarray(res.get('dipole'), float)
    return dict(E=float(res.get('energy')), dipole=float(np.linalg.norm(dip) * AU2DEBYE),
                charge=int(charge), n_atoms=int(len(nums)))


def load():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def run(smiles_list, minutes=4.0, solvent='water'):
    done = load()
    todo = [s for s in dict.fromkeys(smiles_list) if s and s not in done]
    todo.sort(key=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms() if Chem.MolFromSmiles(s) else 999)
    t0 = time.time(); n = 0
    for s in todo:
        if (time.time() - t0) / 60 > minutes:
            break
        try:
            r = xtb_single(s, solvent)
            done[s] = r if r else {'error': 'no conformer'}
        except Exception as e:
            done[s] = {'error': repr(e)[:120]}
        n += 1
        if n % 25 == 0:
            json.dump(done, open(CACHE, 'w'))
    json.dump(done, open(CACHE, 'w'))
    print(f'computed {n} this run | cached {len(done)} | remaining {len(todo) - n}')
    return len(todo) - n
