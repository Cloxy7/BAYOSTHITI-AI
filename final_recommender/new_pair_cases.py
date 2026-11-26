"""Literature check on pairs that are NOT in our 652 pairs. For each (drug, ion): rank among all candidate ions, score, dpKa, the contribution of each feature group,
and the most similar known drugs that already use this ion (the 'why' behind the score). Writes: new_pairs_evidence.csv"""
import numpy as np, pandas as pd
import features; features.inject(globals())
SCORES = pd.read_csv('all_scores_leave_drug_out.csv')
CASES = [('Ibuprofen', 'Lidocainium'), ('Cinnarizine', 'Tosylate (p-toluenesulfonate)'), ('Valproate', 'Cholinium'), ('Albendazole', 'Tartrate'), ('Aspirin', 'Lysine'), ('Propranolol', 'Oxalate')]
rows = []
for dn, inn in CASES:
    i = int(D.index[D.name == dn][0]); j = int(I.index[I.name == inn][0]); g = SCORES[SCORES.drug == i].sort_values('rank')
    r = g[g.ion == j].iloc[0]; reported = [I.name[k] for k in sorted(pos[i])]
    users = [k for k in use if j in pos[k] and PARENT[k] != PARENT[i]]; users = sorted(users, key=lambda k: -S2[i, k])[:3]
    rows.append(dict(drug=dn, ion=inn, in_our_652_pairs=bool(j in pos[i]), rank=int(r['rank']), n_candidates=len(g), score=r.score, dpKa=r.dpKa_value, pKa_label=r.pKa_label, sim2D=r.sim2D, sim3D=r.sim3D,
                     c_2D=r['c_2D similarity'], c_3D=r['c_3D similarity'], c_pKa=r['c_pKa'], c_ESP=r['c_ESP complementarity'],
                     similar_known_drugs_using_this_ion='; '.join(f'{D.name[k]} (sim {S2[i,k]:.2f})' for k in users), ions_reported_for_this_drug_in_our_data='; '.join(str(x) for x in reported)))
E = pd.DataFrame(rows).round(3); E.to_csv('new_pairs_evidence.csv', index=False); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120); print(E.T.to_string())
