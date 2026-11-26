"""Export everything the static site needs into one JSON file.

Runs off the precomputed tables through the same Recommender the chat layer uses, so the website, the
chat agent and the report cannot disagree. No RDKit needed.

    python -m web.build_data          # -> web/site_data.json

With RDKit installed the explanation text is richer: `_similar_users` can rank the drugs that already use
an ion by 2D similarity, so a card reads "Loxoprofenate (0.49), Ketoprofen (0.41) already use it" instead
of "a widely used counter-ion (6 other drugs use it)". Re-run this with RDKit present to upgrade it.
"""
import json
import os
import sys

REC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REC not in sys.path:
    sys.path.insert(0, REC)

import pandas as pd  # noqa: E402

from agent.tools import Recommender  # noqa: E402
from rag.corpus import build_corpus  # noqa: E402

TOP = 10
HERE = os.path.dirname(os.path.abspath(__file__))


def main(out=None):
    out = out or os.path.join(HERE, 'site_data.json')
    R = Recommender()
    rdkit_ok = True
    try:
        import rdkit  # noqa: F401
    except ImportError:
        rdkit_ok = False

    drugs = []
    for i in sorted(R.D.index, key=lambda i: str(R.D.name[i])):
        rec = R._recommend_known(i, str(R.D.name[i]), TOP)
        if not rec['recommendations']:
            continue
        rows = []
        for r in rec['recommendations']:
            ex = R.explain(str(R.D.name[i]), r['ion'] if not r['name_missing'] else r['ion_smiles'])
            rows.append(dict(
                rk=r['rank'], ion=r['ion'], smi=r['ion_smiles'], sc=r['score'], rep=int(r['already_reported']),
                lab=r['pKa_label'], dp=r['dpKa'], s2=r['sim2D'], s3=r['sim3D'], safe=r['safety'],
                nm=int(r['name_missing']),
                c=[r['contributions']['structure_2D'], r['contributions']['shape_3D'],
                   r['contributions']['pKa'], r['contributions']['electrostatics']],
                why=ex.get('chemistry', ''), users=ex.get('other_drugs_using_this_ion', [])[:6]))
        drugs.append(dict(name=rec['drug'], role=rec['role'], pka=rec['drug_pKa'],
                          ncand=rec['n_candidates'], smi=str(R.D.smiles[i]),
                          rep=sorted({str(R.I.name[j]) if isinstance(R.I.name[j], str) else str(R.I.smiles[j])[:22]
                                      for j in R.POS[R.POS.drug_idx == i].ion_idx}),
                          rows=rows))

    corpus = [dict(t=d['text'], c=d['citation'], k=d['kind'], d=d['drug'], i=d['ion'])
              for d in build_corpus()]

    eq = pd.read_csv(os.path.join(REC, 'results_score_equation.csv'))
    fixed = eq[eq.model.str.contains('fixed')].iloc[0]
    models = pd.read_csv(os.path.join(REC, 'results_setA_all_models.csv'))
    ab = pd.read_csv(os.path.join(REC, 'ablation_equation.csv'))
    lodo = pd.read_csv(os.path.join(REC, 'leave_one_drug_out_summary.csv')).set_index('metric').value

    facts = dict(
        n_drugs=len(drugs), n_ions=len(R.I), n_pairs=len(R.POS), n_passages=len(corpus),
        hit1=round(float(fixed.Hit1), 3), hit5=round(float(fixed.Hit5), 3), hit10=round(float(fixed.Hit10), 3),
        ci=[round(float(fixed.Hit10_CI_low), 3), round(float(fixed.Hit10_CI_high), 3)],
        mrr=round(float(fixed.MRR), 3),
        random=round(float(models[models.model == 'Random'].iloc[0].Hit10), 3),
        lodo_hit10=round(float(lodo['Hit@10']), 3),
        models=[dict(name=r.model, hit10=round(float(r.Hit10), 3),
                     lo=round(float(r.Hit10_CI_low), 3), hi=round(float(r.Hit10_CI_high), 3))
                for r in models.sort_values('Hit10', ascending=False).itertuples()],
        equation=[dict(name=r.model, hit10=round(float(r.Hit10), 3), fixed=bool('fixed' in r.model))
                  for r in eq.sort_values('Hit10', ascending=False).itertuples()],
        ablation=[dict(name=r.family_removed, share=round(float(r._10), 1)) for r in ab.itertuples()],
        rdkit_used=rdkit_ok,
    )

    payload = dict(facts=facts, drugs=drugs, corpus=corpus)
    with open(out, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    mb = os.path.getsize(out) / 1e6
    print(f'{out}  {mb:.2f} MB  |  {len(drugs)} drugs x top {TOP}  |  {len(corpus)} passages'
          f'  |  explanations: {"full" if rdkit_ok else "no-RDKit fallback"}')
    return out


if __name__ == '__main__':
    main()
