"""Match patent pairs to the 652 reported pairs used by the recommender (by name or SMILES, either ion order)."""
import pandas as pd, re
src = open('match_to_652.py').read(); PAT = eval(src[src.index('PAT = ') + 6:src.index("new = pd.read_csv")].strip())
PAT['diisopropanolamine'] += '|cc\\(o\\)c\\[nh2\\+\\]cc\\(c\\)o'; PAT['triisopropanolamine'] += '|cc\\(o\\)c\\[nh\\+\\]\\(cc\\(c\\)o\\)cc\\(c\\)o'
PAT['diethanolamine'] += '|occ\\[nh2\\+\\]cco'; PAT['triethanolamine'] += '|occ\\[nh\\+\\]\\(cco\\)cco'; PAT['choline'] += '|c\\[n\\+\\]\\(c\\)\\(c\\)cco'
PAT['loxoprofen'] += '|cc\\(c\\(=o\\)\\[o-\\]\\)c1ccc\\(cc2cccc2=o\\)cc1'
F = '../final_recommender/'
D = pd.read_csv(F + 'drugs.csv'); I = pd.read_csv(F + 'ions.csv'); Pos = pd.read_csv(F + 'positives.csv')
P2 = pd.read_csv('../Pairs_652_with_dpKa_reliability_and_safety.csv')
nm = P2.groupby('counter_ion_smiles').counter_ion.first().to_dict(); nm2 = P2.groupby('drug_smiles').drug.first().to_dict()
def it(j): return f"{I.name[j]} | {nm.get(I.smiles[j], '')} | {nm2.get(I.smiles[j], '')} | {I.smiles[j]}".lower()
def dt(i): return f"{D.name[i]} | {nm2.get(D.smiles[i], '')} | {nm.get(D.smiles[i], '')} | {D.smiles[i]}".lower()
lab = lambda i, j: f"{D.name[i]} + {I.name[j] if isinstance(I.name[j], str) else nm.get(I.smiles[j], I.smiles[j])}"
pairs = [(dt(i), it(j), lab(i, j)) for i, j in zip(Pos.drug_idx, Pos.ion_idx)]
mols = [dt(i) for i in range(len(D))] + [it(j) for j in range(len(I))]
n = pd.read_csv('patent_pairs_new_search.csv')
rank = {'prepared (example or table)': 0, 'cited prior art': 1, 'listed as an ionic liquid (not shown as prepared)': 2}
n = n.assign(r=n.evidence.map(rank)).sort_values('r').drop_duplicates(['drug', 'counter_ion']).drop(columns='r')
def find(a, b):
    return '; '.join(sorted({l for d, i, l in pairs if (re.search(PAT[a], d) and re.search(PAT[b], i)) or (re.search(PAT[b], d) and re.search(PAT[a], i))}))
n['matching_pair_in_652'] = [find(a, b) for a, b in zip(n.drug, n.counter_ion)]; n['in_our_652'] = n.matching_pair_in_652 != ''
kn = lambda x: any(re.search(PAT[x], m) for m in mols)
n['drug_in_dataset'] = n.drug.map(kn); n['ion_in_dataset'] = n.counter_ion.map(kn)
n = n.drop(columns=[c for c in n.columns if c.startswith('Unnamed')])
n.to_csv('patent_pairs_unique.csv', index=False)
S = n.groupby('evidence').agg(pairs=('drug', 'size'), already_in_652=('in_our_652', 'sum')); S['new'] = S.pairs - S.already_in_652; print(S)
new = n[~n.in_our_652]; print('total', len(n), 'new', len(new), 'new with both molecules known', int((new.drug_in_dataset & new.ion_in_dataset).sum()))
print('patents', n.patent.nunique())
