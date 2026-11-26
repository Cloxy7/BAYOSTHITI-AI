"""Write a plain-English justification for every suggested counter-ion (rank 1..10) of every drug.
No literature needed: the reason is built from the model's own parts.
  - analogy  : known drugs similar to this drug that already use this ion (2D Tanimoto)
  - transfer : the pKa label (salt likely / permanent cation / ambiguous)
  - driver   : which score group (2D / 3D / pKa / ESP) pushed this ion up most
  - strength : the score and the rank out of all possible ions for this drug
  - safety   : GRAS-SCOGS / pharma class tag
Writes: justified_recommendations.csv (all drugs), and prints one drug's full top-10 plus some rank-6 examples.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import features; features.inject(globals())
from rdkit import Chem
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s

T = pd.read_csv('top10_recommendations_all_drugs.csv')
ncand = pd.read_csv('all_scores_leave_drug_out.csv', usecols=['drug']).groupby('drug').size().to_dict()  # real pool of reachable ions per drug
# safety tag
SF = pd.read_csv('../Ion_safety_GRAS_SCOGS_all_ions.csv'); AC = pd.read_csv('../ions_acceptability.csv'); AC['can'] = AC.smiles.map(can)
tier = dict(zip(SF.ion_name, SF.safety_list_tier)); acc = dict(zip(AC.can, AC.acceptability))
def safety_tag(name, smi):
    t = str(tier.get(name, '')); a = acc.get(can(smi))
    if t.startswith(('A:', 'B:')): return 'on the FDA GRAS/SCOGS safety list'
    if a == 'pharma-established salt former': return 'an established pharmaceutical salt former'
    if a == 'biocompatible / excipient / food-derived': return 'biocompatible (excipient / food-derived)'
    if a == 'active drug ion (dual-active IL)': return 'itself an active drug ion (dual-active)'
    if a == 'research-only (not pharma-acceptable)': return 'research-only (not pharma-acceptable)'
    return 'safety not classified'

DRV = {'c_2D similarity': '2D', 'c_3D similarity': '3D', 'c_pKa': 'pKa', 'c_ESP complementarity': 'ESP'}
def justify(r):
    i = int(r.drug); j = int(r.ion); dn = r.drug_name; io = r.ion_name if isinstance(r.ion_name, str) else r.ion_smiles[:24]
    users = sorted([k for k in use if j in pos[k] and PARENT[k] != PARENT[i]], key=lambda k: -S2[i, k])
    top = [(D.name[k], S2[i, k]) for k in users[:3]]
    # driver = biggest positive contribution
    contrib = {DRV[c]: r[c] for c in DRV}; drv = max(contrib, key=contrib.get)
    parts = []
    # analogy (only meaningful if a similar drug really uses it)
    if top and top[0][1] >= 0.2:
        names = ', '.join(f'{n} ({s:.2f})' for n, s in top if s >= 0.2)
        parts.append(f'Drugs similar to {dn} already use {io}: {names}.')
    elif len(users) >= 3:
        parts.append(f'{io} is a widely used counter-ion ({len(users)} other drugs use it), though none is very similar to {dn}.')
    else:
        parts.append(f'Few known drugs use {io}, and none is close to {dn}, so this suggestion leans on the chemistry below, not analogy.')
    # transfer
    lab = r.pKa_label
    if 'permanent' in lab: parts.append(f'{io} is a permanent cation, so it needs no proton transfer and pairs with any acid.')
    elif 'salt likely' in lab: parts.append(f'A proton should transfer (ΔpKa {r.dpKa_value:.1f}, salt likely).')
    elif 'co-crystal' in lab: parts.append(f'ΔpKa is low ({r.dpKa_value:.1f}); this is more likely a co-crystal than a salt.')
    elif 'ambiguous' in lab: parts.append(f'ΔpKa is in the grey zone ({r.dpKa_value:.1f}); salt or co-crystal is uncertain.')
    else: parts.append('Drug pKa is missing, so the proton transfer could not be scored.')
    # driver + strength
    drv_txt = {'2D': 'structural similarity to known pairs', '3D': 'a similar 3D shape to known pairs', 'pKa': 'the pKa match', 'ESP': 'the charge-surface match'}[drv]
    parts.append(f'The score is driven mainly by {drv_txt}.')
    parts.append(f'Score {r.score:.1f}, rank {int(r["rank"])} of {ncand[i]} possible ions for {dn} (compare scores only within this drug).')
    parts.append(f'Safety: {io} is {safety_tag(r.ion_name if isinstance(r.ion_name,str) else "", r.ion_smiles)}.')
    return ' '.join(parts)

T['justification'] = T.apply(justify, axis=1)
T[['drug_name', 'rank', 'ion_name', 'ion_smiles', 'score', 'is_reported_pair', 'pKa_label', 'justification']].to_csv('justified_recommendations.csv', index=False)
print('rows', len(T), 'drugs', T.drug.nunique(), '-> justified_recommendations.csv\n')

ex = 'Ciprofloxacin'
g = T[T.drug_name == ex].sort_values('rank')
print('=' * 30, ex, 'top 10', '=' * 30)
for _, r in g.iterrows():
    print(f"\n#{int(r['rank'])}  {r.ion_name if isinstance(r.ion_name,str) else r.ion_smiles[:24]}  (score {r.score:.1f}, {'published' if r.is_reported_pair else 'NEW'})")
    print('   ' + r.justification)
print('\n' + '=' * 30, 'a few rank-6 suggestions', '=' * 30)
for _, r in T[(T['rank'] == 6)].sort_values('score', ascending=False).head(4).iterrows():
    print(f"\n{r.drug_name} -> {r.ion_name if isinstance(r.ion_name,str) else r.ion_smiles[:24]} (rank 6, score {r.score:.1f})")
    print('   ' + r.justification)
