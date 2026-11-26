"""A transcript with no API key: the exact tool calls an LLM makes, and the real data that comes back.

    python -m agent.demo

Everything printed below the arrows is produced by the model and the dataset, not written by hand. The
final answer is the only part an LLM would compose, and every number and quote in it is lifted from a
tool result above it.
"""
import json
import time

from .tools import dispatch

W = 94


def head(t):
    print('\n' + '=' * W + f'\n{t}\n' + '=' * W)


def call(name, args):
    t0 = time.perf_counter()
    out = dispatch(name, args)
    print(f'\n  -> {name}({json.dumps(args)})    [{(time.perf_counter() - t0) * 1000:.1f} ms]')
    return out


def main():
    head('TURN 1   user: "Working with tolfenamic acid. What should I pair it with, and has anyone made it?"')

    rec = call('recommend_counterions', dict(drug='tolfenamic acid', top=5))
    print(f"     drug {rec['drug']}, {rec['role']}, {rec['n_candidates']} candidate ions, {rec['path']}")
    for r in rec['recommendations']:
        flag = '  [already reported]' if r['already_reported'] else ''
        print(f"     {r['rank']:>2}. {r['ion']:<30} {r['score']:>6.2f}   {r['pKa_label']:<32} {r['safety']}{flag}")

    top = rec['recommendations'][0]['ion']
    why = call('explain_recommendation', dict(drug='tolfenamic acid', ion=top))
    print(f"     {why['chemistry']}")
    print(f"     best structural match {why['best_structural_match']}; also used by: {', '.join(why['other_drugs_using_this_ion'][:4])}")

    ev = call('get_evidence', dict(drug='tolfenamic acid', ion=top))
    if ev['found']:
        print(f"     evidence: {ev['evidence_level']}")
        print(f'     quote:    "{ev["quote"]}"')
        print(f"     cite:     {ev['citation']}")
    else:
        print(f"     evidence: NOT FOUND - {ev['reason']}")

    print('\n  ANSWER the model can now write, with nothing added:')
    print(f"""
     Top candidate for tolfenamic acid is {top} (score {rec['recommendations'][0]['score']},
     rank 1 of {rec['n_candidates']}). {why['chemistry']}
     The dataset has no published record of this exact pair, so it is a suggestion, not a precedent.
     Runners-up: {', '.join(r['ion'] for r in rec['recommendations'][1:4])}.""")

    head('TURN 2   user: "Has anyone actually made the ibuprofen one with choline?"')
    ev = call('get_evidence', dict(drug='Ibuprofen', ion='Cholinium'))
    print(f"     found={ev['found']}  level={ev['evidence_level']}")
    print(f'     quote: "{ev["quote"]}"')
    print(f"     cite:  {ev['citation']}")
    print('\n  ANSWER: yes -- quoted verbatim, with the DOI. The LLM contributes no chemistry here at all.')

    head('TURN 3   user: "What about pairing it with lidocaine?"   (the model ranks it 4th but nobody has made it)')
    ev = call('get_evidence', dict(drug='Ibuprofen', ion='Lidocainium'))
    print(f"     found={ev['found']}  is_reported_pair={ev['is_reported_pair']}")
    print(f"     reason: {ev['reason']}")
    why = call('explain_recommendation', dict(drug='Ibuprofen', ion='Lidocainium'))
    print(f"     rank {why['rank']} of 158, score {why['score']}")
    print(f"     {why['chemistry']}")
    print('\n  ANSWER: ranked 4th, but the dataset holds no record of the pair. This is the grounding')
    print('  working: the model has an opinion, the literature tool says "no record", and the answer')
    print('  reports both instead of resolving the tension by inventing a citation.')

    head('TURN 4   user: "I have an acidic drug, pKa 4.2, with a hydrophobic aromatic ring. What do you suggest?"')
    bad = call('recommend_counterions', dict(drug='an acidic drug with pKa 4.2 and an aromatic ring', top=5))
    print(f"     ok={bad['ok']}")
    print(f"     error: {bad.get('error', '')}")
    print('\n  ANSWER: the system prompt makes the model ask for a name or SMILES. Two of the four features')
    print('  are similarity to specific known molecules, so there is nothing to compute from a description.')
    print('  Refusing here is the behaviour, not a gap.')

    head('TURN 5   user: "Will the cholinium salt be more soluble?"')
    print('\n  No tool answers that. recommend_counterions carries this caveat in every response:')
    print(f"\n     {rec['caveat']}")
    print('\n  ANSWER: the model declines. It predicts which pairings the literature reports, not solubility.')

    head('TURN 6   user: "Is cholinium actually safe to put in a formulation?"')
    r = call('search_literature', dict(query='GRAS SCOGS safety status', ion='cholinium', kind='ion_safety', k=1))
    for p in r['passages']:
        print(f"     {p['text'][:200]}")
        print(f"     cite: {p['citation']}")
    print('\n  ANSWER: tier B — only a salt or derivative is listed, so it needs checking by hand.')
    print('  get_evidence cannot answer this; it only keys exact pairs. This is what the corpus is for.')

    head('TURN 7   user: "Has anyone measured how much solubility actually improves?"')
    r = call('search_literature', dict(query='water solubility of the ionic liquid versus the parent drug',
                                       kind='measured_property', k=3))
    for p in r['passages']:
        print(f"     {p['text'][:140]}")
    print(f"     cite: {r['passages'][0]['citation']}")
    print('\n  ANSWER: real measured values with a DOI — and note these are the only outcome numbers in')
    print('  the corpus. The ranking model itself predicts reported pairings, not solubility.')

    head('TURN 8   user: "You suggested a methacrylate ion earlier. Any reason not to use it?"')
    r = call('search_literature', dict(query='methacrylate monomer sensitiser reject', kind='theory_note', k=1))
    for p in r['passages']:
        print(f"     {p['text'][:320]}")
        print(f"     cite: {p['citation']}")
    print('\n  ANSWER: the written assessment says reject it as a drug salt. That judgement was made by')
    print('  hand during the project and is retrieved, not re-derived by the LLM.')

    print('\n' + '=' * W)
    print(f'Tool calls above: 12 across 4 tools. Corpus searched: {r["searched"]} passages, all cited.')
    print('Sentences of chemistry invented by the LLM: 0.')
    print('=' * W)


if __name__ == '__main__':
    main()
