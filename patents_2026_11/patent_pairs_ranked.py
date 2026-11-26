"""Where does the recommender (leave-drug-out ranking) place the new patent pairs whose drug and ion are both in our data?"""
import pandas as pd, re, numpy as np
exec(open('match_final.py').read().split("n = pd.read_csv('patent_pairs_new_search.csv')")[0])
A = pd.read_csv(F + 'all_scores_leave_drug_out.csv'); ncand = A.groupby('drug').size()
n = pd.read_csv('patent_pairs_unique.csv'); new = n[(~n.in_our_652) & n.drug_in_dataset & n.ion_in_dataset].copy()
def idx_d(x): return [i for i in range(len(D)) if re.search(PAT[x], dt(i))]
def idx_i(x): return [j for j in range(len(I)) if re.search(PAT[x], it(j))]
out = []
for _, r in new.iterrows():
    best = None
    for a, b in [(r.drug, r.counter_ion), (r.counter_ion, r.drug)]:
        for i in idx_d(a):
            for j in idx_i(b):
                g = A[(A.drug == i) & (A.ion == j)]
                if len(g):
                    rk = int(g['rank'].iloc[0]); c = int(ncand[i])
                    if best is None or rk < best[0]: best = (rk, c, D.name[i], float(g.score.iloc[0]))
    out.append(dict(drug=r.drug, counter_ion=r.counter_ion, evidence=r.evidence, patent=r.patent,
                    rank=best[0] if best else None, n_candidates=best[1] if best else None, scored_as_drug=best[2] if best else None, score=best[3] if best else None))
O = pd.DataFrame(out); O['in_top10'] = O['rank'] <= 10; O['random_chance_top10'] = 10 / O.n_candidates
O.to_csv('patent_pairs_new_ranked.csv', index=False)
s = O.dropna(subset=['rank'])
print('new pairs with both molecules known', len(O), '| could be ranked', len(s), '| not rankable (ion has no other user / same-charge issue)', len(O) - len(s))
for ev, g in s.groupby('evidence'):
    print(f'{ev}: n={len(g)}, top10={g.in_top10.mean():.2f}, top20={(g["rank"]<=20).mean():.2f}, median rank={g["rank"].median():.0f}, random top10={g.random_chance_top10.mean():.2f}')
print(f'ALL: n={len(s)}, top10={s.in_top10.mean():.2f}, top20={(s["rank"]<=20).mean():.2f}, median rank={s["rank"].median():.0f} of median {s.n_candidates.median():.0f}, random top10={s.random_chance_top10.mean():.2f}')
print(s.sort_values('rank')[['drug', 'counter_ion', 'rank', 'n_candidates', 'evidence']].to_string(index=False))
