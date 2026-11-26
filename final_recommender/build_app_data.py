"""Builds app_data.json for the web app: top-20 counter-ions for each of the 138 known drugs (each treated as unseen), plus model facts and the literature checks."""
import numpy as np, pandas as pd, json, reason
import features; features.inject(globals())
SAFE = dict(zip(range(ni), I.safety_tag)) if 'safety_tag' in I.columns else {}
SCORES = pd.read_csv('all_scores_leave_drug_out.csv'); TOP = 20
ncand = SCORES.groupby('drug').size().to_dict()          # full reachable-ion pool per drug (for the rank "x of N")
drugs = []
for i in use:
    g = SCORES[SCORES.drug == i].sort_values('rank').head(TOP); rows = []
    for r in g.to_dict('records'):
        j = int(r['ion']); allus = sorted([k for k in use if j in pos[k] and PARENT[k] != PARENT[i]], key=lambda k: -S2[i, k]); us = allus[:2]
        nm = r['ion_name'] if isinstance(r['ion_name'], str) else ''
        disp = nm or r['ion_smiles'][:28]
        c = [round(float(r['c_2D similarity']), 2), round(float(r['c_3D similarity']), 2), round(float(r['c_pKa']), 2), round(float(r['c_ESP complementarity']), 2)]
        rs = reason.make_reason(str(D.name[i]), disp, int(r['rank']), int(ncand[i]), float(r['score']), r['pKa_label'],
                                None if pd.isna(r['dpKa_value']) else float(r['dpKa_value']), c,
                                [(D.name[k], float(S2[i, k])) for k in allus], len(allus), SAFE.get(j, 'unclassified'), bool(r['is_reported_pair']))
        rows.append(dict(rk=int(r['rank']), ion=disp, smi=r['ion_smiles'], sc=round(float(r['score']), 2), rep=int(r['is_reported_pair']), lab=r['pKa_label'],
                         dp=None if pd.isna(r['dpKa_value']) else round(float(r['dpKa_value']), 2), s2=round(float(r['sim2D']), 2), s3=round(float(r['sim3D']), 2),
                         c=c, sim=[[D.name[k], round(float(S2[i, k]), 2)] for k in us],
                         just=rs['chem'], safe=rs['safe']))
    n_rep = len(pos[i]); drugs.append(dict(name=str(D.name[i]), smi=D.smiles[i], role='base (cation)' if D.sign[i] > 0 else 'acid (anion)', pka=None if pd.isna(D.pKa[i]) else float(D.pKa[i]),
                                          src='known drug, ranked as if unseen', ncand=int(len(SCORES[SCORES.drug == i])), nrep=n_rep, reported=sorted({str(I.name[j]) if isinstance(I.name[j], str) else I.smiles[j][:20] for j in pos[i]}), rows=rows))
EQ = pd.read_csv('results_score_equation.csv'); EQ0 = EQ[EQ.model.str.contains('fixed')].iloc[0]; M = pd.read_csv('results_setA_all_models.csv').set_index('model')
AB = pd.read_csv('ablation_equation.csv'); LO = pd.read_csv('leave_one_drug_out_summary.csv').set_index('metric').value
facts = dict(hit10=round(float(EQ0.Hit10), 3), hit10_lo=round(float(EQ0.Hit10_CI_low), 3), hit10_hi=round(float(EQ0.Hit10_CI_high), 3), mrr=round(float(EQ0.MRR), 3), rnd=round(float(M.loc['Random'].Hit10), 3),
             lodo_hit10=round(float(LO['Hit@10']), 3), lodo_mrr=round(float(LO['MRR']), 3), lodo_hit5=round(float(LO['Hit@5']), 3), lodo_hit1=round(float(LO['Hit@1']), 3),
             share={r.family_removed: float(r['share_of_total_drop_%']) for _, r in AB.iterrows()}, n_drugs=len(use))
E = pd.read_csv('new_pairs_evidence.csv').set_index(['drug', 'ion'])
def ev(d, ion): r = E.loc[(d, ion)]; return dict(rank=int(r['rank']), n=int(r.n_candidates), score=float(r.score), dp=None if pd.isna(r.dpKa) else float(r.dpKa), lab=r.pKa_label, sim=r.similar_known_drugs_using_this_ion, c=[float(r.c_2D), float(r.c_3D), float(r.c_pKa), float(r.c_ESP)])
cases = [
 dict(drug='Ibuprofen', ion='Lidocainium', top10=True, **ev('Ibuprofen', 'Lidocainium'),
      why='Ibuprofen is an acid and lidocaine is a base, so a proton can move from one to the other (our delta pKa is 4.8, "salt likely"). Most of the score comes from 2D similarity: loxoprofenate and ketoprofen, the two drugs most like ibuprofen in our data, already have lidocainium published as their counter-ion.',
      paper='Panić J., Tot A., Janković N., Drid P., Gadžurić S., Vraneš M. Physicochemical and structural properties of lidocaine-based ionic liquids with anti-inflammatory anions. RSC Advances, 2020. DOI 10.1039/c9ra08815f',
      url='https://pubs.rsc.org/en/content/articlehtml/2020/ra/c9ra08815f',
      found='The paper makes lidocaine ibuprofenate ([Lid][Ibp]) and lidocaine salicylate as ionic liquids with anti-inflammatory anions.',
      caveat='The authors note that some papers call lidocaine ibuprofenate a deep eutectic mixture, and their IR data suggest the conversion to an ionic liquid is incomplete for the ibuprofen pair.'),
 dict(drug='Cinnarizine', ion='Tosylate (p-toluenesulfonate)', top10=True, **ev('Cinnarizine', 'Tosylate (p-toluenesulfonate)'),
      why='Cinnarizine is a base and tosylic acid is a strong acid, so delta pKa is large (10.1, "salt likely"). The other three groups each add a little. Structural similarity is low for this pair, so the rank comes from many small pushes and not from one strong one.',
      paper='Kavitha C. N., Butcher R. J., Jasinski J. P., Yathirajan H. S., Dayananda A. S. Cinnarizinium bis(p-toluenesulfonate) dihydrate. Acta Crystallographica Section E, 2013, 69, o485–o486. DOI 10.1107/S1600536813003991',
      url='https://doi.org/10.1107/S1600536813003991',
      found='A crystal structure of a doubly protonated cinnarizinium cation with two p-toluenesulfonate anions and two water molecules.',
      caveat='This is a crystal-structure report, so it shows the salt exists but gives no solubility or stability data. The salt has two tosylates per drug, while the model scores one drug–ion pair.'),
 dict(drug='Valproate', ion='Cholinium', top10=True, **ev('Valproate', 'Cholinium'),
      why='Cholinium is a permanent cation, so no proton transfer is needed and the model gives it a full pKa score for any acid. Our data already has two other quaternary cations with valproate (cetylpyridinium and hexadecyltrimethylammonium).',
      paper='Ferraz R., Costa A., Marrucho I., Branco L., Prudêncio C., Noronha J. P., Petrovski Ž. Development of novel ionic liquids based on valproate anion. Book of Abstracts, 6th Spanish-Portuguese-Japanese Organic Chemistry Symposium, Lisbon, 2012, p. 122.',
      url='https://recipp.ipp.pt/entities/publication/821ce80f-6e03-4e59-bd12-4d871c89a9d2/full',
      found='The abstract describes valproate ionic liquids made with cetylpyridinium, choline and imidazolium cations.',
      caveat='This is a conference abstract, not a full paper. I could not see the compound list, so the choline–valproate pair is implied by the abstract and not named in a table.'),
 dict(drug='Albendazole', ion='Tartrate', top10=False, **ev('Albendazole', 'Tartrate'),
      why='The model puts this pair in the middle of the list. Our albendazole pKa (2.8) is almost equal to the tartrate pKa (2.74), so delta pKa is 0.06 ("ambiguous") and the pKa group lowers the score.',
      paper='Yan H., Zhong X., Liu Y. Improving the solubility, stability, and bioavailability of albendazole through synthetic salts. Molecules, 2024, 29(15), 3571. DOI 10.3390/molecules29153571',
      url='https://pmc.ncbi.nlm.nih.gov/articles/PMC11314343',
      found='Salts with fumaric acid, D-tartaric acid and hydrochloric acid. The D-tartrate salt was about 585 times more soluble than albendazole at pH 2 and gave about 5.2 times the AUC in rats.',
      caveat='The authors use a pKa of 9.51 for albendazole, which is not the value in our table. This is an example of why delta pKa is shown as a label and not used as a filter.'),
 dict(drug='Aspirin', ion='Lysine', top10=False, **ev('Aspirin', 'Lysine'),
      why='Lysine is a base and aspirin is an acid (delta pKa 7.15, "salt likely"), and aspirin is similar to salicylic acid and ketoprofen, which already use lysine. Many other amino-acid cations score even higher, which pushes lysine down to rank 31.',
      paper='Lysine acetylsalicylate (aspirin DL-lysine), a more soluble form of aspirin given by injection. Wikipedia entry (secondary source); DrugBank salt entry DBSALT003034.',
      url='https://en.wikipedia.org/wiki/Lysine_acetylsalicylate',
      found='Lysine acetylsalicylate is described as a more soluble form of aspirin developed for intravenous use.',
      caveat='I only found a secondary source for this one, so it is weaker than the others.'),
 dict(drug='Propranolol', ion='Oxalate', top10=False, **ev('Propranolol', 'Oxalate'),
      why='Delta pKa is large (7.94, "salt likely"), but propranolol is not similar to any drug in our data that uses oxalate (only pipemidic acid, similarity 0.11), so the structural part of the score is low.',
      paper='Stepanovs D., Jure M., Yanichev A., Belyakov S., Mishnev A. Molecular salts of propranolol with dicarboxylic acids: diversity of stoichiometry, supramolecular structures and physicochemical properties. CrystEngComm, 2015, 17, 9023–9028. DOI 10.1039/C5CE01408E',
      url='https://pubs.rsc.org/en/content/articlehtml/2015/ce/c5ce01408e',
      found='Salts of propranolol with oxalic, fumaric and maleic acids (1:1 and 2:1 for oxalate). Melting points about twice that of propranolol, and aqueous solubility clearly higher than the base.',
      caveat='Fumarate and maleate are not in our 291-ion list, so only oxalate could be scored.')]
json.dump(dict(drugs=drugs, facts=facts, cases=cases), open('app_data.json', 'w'), separators=(',', ':'))
import os; print('drugs', len(drugs), 'bytes', os.path.getsize('app_data.json'), facts)
