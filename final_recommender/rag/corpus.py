"""Turn the project's CSVs into retrievable passages, each carrying its own citation.

Nothing is paraphrased on the way in. A passage is a short piece of prose built from fields that were
already written down, plus the provenance needed to cite it. If a field is empty the passage is skipped
rather than padded, so retrieval can never surface a sentence that says nothing.

    from rag.corpus import build_corpus
    docs = build_corpus()
    docs[0]['text'], docs[0]['citation'], docs[0]['source_file']
"""
import os

import pandas as pd

REC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(REC)


def _read(path):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    try:
        return pd.read_csv(full, engine='python', on_bad_lines='skip')
    except Exception:
        return None


def _s(v):
    return '' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def _doc(kind, text, source_file, citation='', drug='', ion='', extra=None):
    d = dict(kind=kind, text=' '.join(text.split()), source_file=source_file,
             citation=citation, drug=drug, ion=ion)
    if extra:
        d.update(extra)
    return d


# ---------------------------------------------------------------- the seven sources

def _reported_pairs():
    """The 652 curated pairs. Each carries a verbatim quote and a PMID/DOI -- the backbone of the corpus."""
    df = _read('Pairs_652_with_dpKa_reliability_and_safety.csv')
    out = []
    if df is None:
        return out
    for r in df.itertuples():
        drug, ion = _s(r.drug), _s(getattr(r, 'ion_name', '')) or _s(r.counter_ion)
        quote, ref = _s(r.proof_quote), _s(r.proof_ref)
        if not quote:
            continue
        stoich = _s(r.stoichiometry)
        lead = f'{drug} paired with {ion}.'
        if stoich:
            lead += f' Stoichiometry {stoich}.'
        out.append(_doc('reported_pair', f'{lead} Reported in the literature: "{quote}"',
                        'Pairs_652_with_dpKa_reliability_and_safety.csv', ref, drug, ion,
                        dict(evidence_level=_s(r.evidence_level), verdict=_s(getattr(r, 'verdict', '')))))
        extra_q, extra_r = _s(getattr(r, 'extra_fulltext_quote', '')), _s(getattr(r, 'extra_fulltext_ref', ''))
        if extra_q:
            out.append(_doc('reported_pair_fulltext', f'{drug} with {ion}, further full-text support: "{extra_q}"',
                            'Pairs_652_with_dpKa_reliability_and_safety.csv', extra_r or ref, drug, ion))
    return out


def _measured_outcomes():
    """Measured solubility and toxicity of API ionic liquids -- the only place the corpus holds an outcome."""
    df = _read('api_il_sol_tox/API_IL_solubility_toxicity_values.csv')
    out = []
    if df is None:
        return out
    for r in df.itertuples():
        drug, ion, prop, val = _s(r.drug), _s(r.counter_ion), _s(r.property), _s(r.value)
        if not (drug and prop and val):
            continue
        unit, cond = _s(r.unit), _s(r.conditions)
        txt = f'Measured {prop} of {drug} {_s(r.compound_type)}'
        if ion:
            txt += f' with {ion}'
        txt += f': {val} {unit}'.rstrip()
        if cond:
            txt += f' ({cond})'
        where = _s(r.where_in_paper)
        if where:
            txt += f'. Reported in {where}'
        out.append(_doc('measured_property', txt + '.', 'api_il_sol_tox/API_IL_solubility_toxicity_values.csv',
                        _s(r.doi) or _s(r.reference), drug, ion, dict(property=prop, value=val, unit=unit)))
    return out


def _patents():
    """Pairs taken from 7 patents, with where this model ranked each one."""
    df = _read('patents_2026_11/patent_pairs_new_ranked.csv')
    out = []
    if df is None:
        return out
    for r in df.itertuples():
        drug, ion, pat = _s(r.drug), _s(r.counter_ion), _s(r.patent)
        if not (drug and ion and pat):
            continue
        txt = f'Patent evidence: {drug} with {ion} was {_s(r.evidence)} in patent {pat}.'
        rank, n = _s(r.rank), _s(r.n_candidates)
        if rank and n:
            txt += f' The recommender ranks this pair {rank.split(".")[0]} of {n.split(".")[0]} for {_s(r.scored_as_drug) or drug}.'
        out.append(_doc('patent', txt, 'patents_2026_11/patent_pairs_new_ranked.csv', pat, drug, ion))
    return out


def _theory_notes():
    """Hand-written assessments of the model's unreported suggestions, including the reasons to reject one."""
    df = _read('final_recommender/new_pair_theory_notes.csv')
    out = []
    if df is None:
        return out
    for r in df.itertuples():
        drug, ion = _s(r.drug_name), _s(r.ion_label)
        bits = [(f'Ionic-liquid likeness', _s(r.il_likeness)), ('Safety', _s(r.safety_note)),
                ('Literature status', _s(r.literature_status)), ('Verdict', _s(getattr(r, 'verdict', '')))]
        body = ' '.join(f'{k}: {v}.' for k, v in bits if v)
        if not body:
            continue
        out.append(_doc('theory_note', f'Assessment of {drug} with {ion}. {body}',
                        'final_recommender/new_pair_theory_notes.csv', _s(r.source), drug, ion))
    return out


def _ion_safety():
    """FDA GRAS / SCOGS status per counter-ion."""
    df = _read('Ion_safety_GRAS_SCOGS_all_ions.csv')
    out = []
    if df is None:
        return out
    for r in df.itertuples():
        ion, tier = _s(r.ion_name), _s(r.safety_list_tier)
        if not (ion and tier):
            continue
        txt = f'Safety status of the counter-ion {ion}: {tier}.'
        for label, col in [('SCOGS substance', 'scogs_substance'), ('GRAS notice substance', 'grn_substance'),
                           ('GRAS notice numbers', 'grn_numbers'), ('listed examples', 'b_examples')]:
            v = _s(getattr(r, col, ''))
            if v:
                txt += f' {label}: {v}.'
        n = _s(r.n_pairs)
        if n:
            txt += f' It appears in {n.split(".")[0]} reported pairs in this dataset.'
        out.append(_doc('ion_safety', txt, 'Ion_safety_GRAS_SCOGS_all_ions.csv',
                        'FDA GRAS notices and SCOGS database', '', ion))
    return out


def _measured_single(path, what):
    """Measured melting point, logS and LD50 of single drugs or single counter-ions."""
    df = _read(path)
    out = []
    if df is None:
        return out
    for r in df.itertuples():
        name = _s(r.name)
        if not name:
            continue
        bits = []
        for label, col, unit in [('melting point', 'mp_C', ' °C'), ('log solubility (logS)', 'logS', ''),
                                 ('oral LD50', 'ld50_mgkg', ' mg/kg')]:
            v = _s(getattr(r, col, ''))
            if v:
                match = _s(getattr(r, col + '_match', ''))
                bits.append(f'{label} {v}{unit}' + (f' (measured on the {match})' if match else ''))
        if not bits:
            continue
        src = _s(getattr(r, 'mp_source', '')) or _s(getattr(r, 'ld50_source', ''))
        cid = _s(getattr(r, 'pubchem_cid', ''))
        out.append(_doc('measured_single', f'Measured properties of the {what} {name}: ' + '; '.join(bits) + '.',
                        path, (src + (f'; PubChem CID {cid.split(".")[0]}' if cid else '')).strip('; '),
                        name if what == 'drug' else '', name if what == 'counter-ion' else ''))
    return out


def _pka_sources():
    """Where each drug's pKa came from. The report never predicts a drug pKa, so provenance matters."""
    df = _read('final_recommender/drug_pKa_sources_138.csv')
    out = []
    if df is None:
        return out
    for r in df.itertuples():
        name, pka = _s(r.name), _s(r.pKa_used)
        if not (name and pka):
            continue
        txt = f'pKa of {name} used in this work: {pka} ({_s(r.pKa_type) or "unspecified type"}, {_s(r.charge)}).'
        note = _s(r.note)
        if note:
            txt += f' Note: {note}.'
        cat = _s(r.source_category)
        if cat:
            txt += f' Source category: {cat}.'
        out.append(_doc('pka_source', txt, 'final_recommender/drug_pKa_sources_138.csv',
                        _s(r.source_as_recorded), name, ''))
    return out


def _pka_methods():
    """How the counter-ion pKa method was chosen: four methods benchmarked on the 138 ions with a measurement."""
    df = _read('pKa_methods/pKa_models_metrics_comparison_with_DFT.csv')
    out = []
    if df is None:
        return out
    for r in df.itertuples():
        model = _s(r.model)
        if not model:
            continue
        bits = [f'{k} {_s(getattr(r, k, ""))}' for k in ('n', 'MAE', 'RMSE', 'R2') if _s(getattr(r, k, ''))]
        out.append(_doc('pka_method', f'pKa prediction benchmark. Method {model} on {_s(r.test_set)}: '
                                      + ', '.join(bits) + '.',
                        'pKa_methods/pKa_models_metrics_comparison_with_DFT.csv',
                        'benchmark run in this project', '', ''))
    return out


SOURCES = [_reported_pairs, _measured_outcomes, _patents, _theory_notes, _ion_safety, _pka_sources, _pka_methods]


def build_corpus():
    docs = []
    for fn in SOURCES:
        docs.extend(fn())
    docs.extend(_measured_single('measured_properties/drugs_measured_properties.csv', 'drug'))
    docs.extend(_measured_single('measured_properties/counter_ions_measured_properties.csv', 'counter-ion'))
    for n, d in enumerate(docs):
        d['id'] = n
    return docs


if __name__ == '__main__':
    from collections import Counter
    docs = build_corpus()
    print(f'{len(docs)} passages')
    for kind, n in Counter(d['kind'] for d in docs).most_common():
        print(f'  {kind:<24} {n:>5}')
    cited = sum(1 for d in docs if d['citation'])
    print(f'\npassages carrying a citation: {cited} ({100 * cited / len(docs):.0f}%)')
