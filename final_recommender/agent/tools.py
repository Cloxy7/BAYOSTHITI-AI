"""The three functions an LLM is allowed to call.

Design rules, which are the whole point of the layer:
  * the model does the chemistry, the LLM does the talking. No scoring logic lives here.
  * every sentence of chemistry comes from reason.make_reason (the same function the web app uses).
  * every literature claim is a verbatim quote plus its DOI/PMID, looked up by key. Nothing is generated.
  * when a lookup misses, the tool says so. There is no path that returns a plausible-looking guess.

Two speeds:
  fast  - one of the 138 drugs already in the dataset. Reads the precomputed ranking. No RDKit, sub-millisecond.
  slow  - any other SMILES. Calls recommend_new_drug, which needs RDKit and a few seconds for conformers.

    from agent.tools import Recommender
    R = Recommender()
    R.recommend('ibuprofen', top=5)
    R.evidence('Ibuprofen', 'Docusate')
    R.explain('Ibuprofen', 'Lidocainium')
"""
import difflib
import os
import re
import sys

import pandas as pd

REC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(REC_DIR)
if REC_DIR not in sys.path:
    sys.path.insert(0, REC_DIR)

import reason  # noqa: E402  (lives in final_recommender/)

_STEREO = (re.compile(r'\[C@@?H\]'), re.compile(r'\[C@@?\]'),
           re.compile(r'\[([A-Za-z][a-z]?)@@?H\]'), re.compile(r'\[([A-Za-z][a-z]?)@@?\]'))

# 87 of the 291 counter-ions have no name in ions.csv, and they fill 279 of the 1,380 top-ten slots.
# Rather than printing a bare SMILES at a chemist, give the class it belongs to.
_CLASSES = [
    (re.compile(r'\[n\+\]1cc'), 'pyridinium cation'),
    (re.compile(r'\[n\+\]1c[nc]'), 'imidazolium cation'),
    (re.compile(r'\[P\+\]'), 'phosphonium cation'),
    (re.compile(r'\[N\+\]\((?:C|\[)'), 'quaternary ammonium cation'),
    (re.compile(r'\[NH[0-9]?\+\]'), 'protonated amine cation'),
    (re.compile(r'S\(=O\)\(=O\)\[O-\]'), 'sulfonate anion'),
    (re.compile(r'OS\(=O\)\(=O\)\[O-\]'), 'sulfate anion'),
    (re.compile(r'P\(=O\)\(\[O-\]\)'), 'phosphate anion'),
    (re.compile(r'C\(=O\)\[O-\]'), 'carboxylate anion'),
    (re.compile(r'\[O-\]'), 'anion'),
]


def describe(smiles):
    """A short structural class for an unnamed ion, so a chat answer never shows a bare SMILES alone."""
    s = str(smiles)
    for pat, label in _CLASSES:
        if pat.search(s):
            n = s.count('C') + s.count('c')
            size = 'long-chain ' if n >= 12 else ''
            return f'unnamed {size}{label}'
    return 'unnamed ion'


def _flat(smiles):
    """Stereo-insensitive SMILES key. The model tables store flat SMILES; the evidence table keeps
    [C@H] and /C=C/, so a plain string join silently drops every chiral molecule. Used only when
    RDKit is unavailable -- with RDKit we key on the InChIKey skeleton instead, which is exact."""
    s = str(smiles)
    s = _STEREO[0].sub('C', s)
    s = _STEREO[1].sub('C', s)
    s = _STEREO[2].sub(r'[\1H]', s)
    s = _STEREO[3].sub(r'[\1]', s)
    return s.replace('/', '').replace('\\', '').strip()


class Recommender:
    def __init__(self, rec_dir=REC_DIR, pairs_csv=None):
        self.dir = rec_dir
        j = lambda f: os.path.join(rec_dir, f)
        self.D = pd.read_csv(j('drugs.csv'))
        self.I = pd.read_csv(j('ions.csv'))
        self.POS = pd.read_csv(j('positives.csv'))
        self.SCORES = pd.read_csv(j('all_scores_leave_drug_out.csv'))
        self.PAIRS = pd.read_csv(pairs_csv or os.path.join(ROOT, 'Pairs_652_with_dpKa_reliability_and_safety.csv'))
        self._names = {str(n).lower(): i for i, n in enumerate(self.D.name)}
        self._safety = dict(zip(range(len(self.I)), self.I.safety_tag)) if 'safety_tag' in self.I else {}
        self._users = {}
        for r in self.POS.itertuples():
            self._users.setdefault(r.ion_idx, []).append(r.drug_idx)
        self._key, self.match_method = self._build_evidence_index()
        self._F = None
        self._index = None

    # ---------------------------------------------------------------- evidence index

    def _build_evidence_index(self):
        """Key the 652 quotes by (drug, ion). Prefer the InChIKey skeleton; fall back to a stereo-stripped
        SMILES, which resolves about 84% of pairs. The method used is reported so the gap is never silent."""
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.inchi import MolToInchiKey
            RDLogger.DisableLog('rdApp.*')

            def k(s):
                m = Chem.MolFromSmiles(str(s))
                return MolToInchiKey(m).split('-')[0] if m is not None else None
            method = 'inchikey-skeleton'
        except ImportError:
            k = _flat
            method = 'stereo-stripped-smiles'
        idx = {}
        for r in self.PAIRS.itertuples():
            a, b = k(r.drug_smiles), k(r.counter_ion_smiles)
            if a and b:
                idx.setdefault((a, b), []).append(r)
        self._k = k
        return idx, method

    # ---------------------------------------------------------------- resolution

    @staticmethod
    def _name_variants(s):
        """drugs.csv names the ionic form ('Tolfenamate', 'Lidocainium'), but a chemist types the neutral
        name ('tolfenamic acid', 'lidocaine'). Generate the obvious morphological bridges."""
        s = str(s).lower().strip()
        out = {s}
        if s.endswith(' acid'):
            base = s[:-5].strip()
            out.add(base)
            if base.endswith('ic'):
                out.add(base[:-2] + 'ate')          # tolfenamic acid -> tolfenamate
        if s.endswith('e'):
            out.add(s[:-1] + 'ium')                 # lidocaine -> lidocainium
            out.add(s[:-1] + 'ate')
        out.add(s + 'ate')                          # ibuprofen -> ibuprofenate
        out.add(s + 'ium')                          # amantadine -> amantadinium
        return out

    def resolve_drug(self, drug):
        """Name or SMILES -> (index, name) for a known drug, or (None, None)."""
        s = str(drug).strip()
        variants = self._name_variants(s)
        for v in sorted(variants, key=len, reverse=True):
            if v in self._names:
                i = self._names[v]
                return i, str(self.D.name[i])
        for v in variants:
            hit = difflib.get_close_matches(v, self._names, n=1, cutoff=0.88)
            if hit:
                i = self._names[hit[0]]
                return i, str(self.D.name[i])
        flat = _flat(s)
        for i, sm in enumerate(self.D.smiles):
            if _flat(sm) == flat:
                return i, str(self.D.name[i])
        return None, None

    def _ion_index(self, ion):
        s = str(ion).strip().lower()
        for i, n in enumerate(self.I.name):
            if str(n).strip().lower() == s:
                return i
        hit = difflib.get_close_matches(s, [str(n).lower() for n in self.I.name], n=1, cutoff=0.86)
        if hit:
            return [str(n).lower() for n in self.I.name].index(hit[0])
        flat = _flat(s)
        for i, sm in enumerate(self.I.smiles):
            if _flat(sm) == flat:
                return i
        return None

    # ---------------------------------------------------------------- tool 1

    def recommend(self, drug, pKa=None, top=10):
        """Rank the counter-ions for a drug. Returns a dict the caller can hand straight to an LLM."""
        i, name = self.resolve_drug(drug)
        if i is not None:
            out = self._recommend_known(i, name, top)
            if str(drug).strip().lower() != name.lower():
                out['matched_from'] = (f'you said {drug!r}; the dataset stores the ionic form as {name!r} '
                                       '— say so in your answer')
            return out
        return self._recommend_new(str(drug), pKa, top)

    def _recommend_known(self, i, name, top):
        g = self.SCORES[self.SCORES.drug == i].sort_values('rank')
        rows = []
        for r in g.head(top).to_dict('records'):
            named = isinstance(r['ion_name'], str) and r['ion_name'].strip()
            rows.append(dict(
                rank=int(r['rank']), ion=r['ion_name'] if named else describe(r['ion_smiles']),
                name_missing=not bool(named),
                ion_smiles=r['ion_smiles'], score=round(float(r['score']), 2),
                already_reported=bool(r['is_reported_pair']), pKa_label=r['pKa_label'],
                dpKa=None if pd.isna(r['dpKa_value']) else round(float(r['dpKa_value']), 2),
                sim2D=round(float(r['sim2D']), 3), sim3D=round(float(r['sim3D']), 3),
                safety=str(self._safety.get(int(r['ion']), 'unclassified')),
                contributions=dict(structure_2D=round(float(r['c_2D similarity']), 3),
                                   shape_3D=round(float(r['c_3D similarity']), 3),
                                   pKa=round(float(r['c_pKa']), 3),
                                   electrostatics=round(float(r['c_ESP complementarity']), 3))))
        return dict(ok=True, drug=name, matched_from=None, source='known drug, ranked as if unseen (leave-drug-out)',
                    path='fast: precomputed ranking, no RDKit',
                    role='base (cation)' if self.D.sign[i] > 0 else 'acid (anion)',
                    drug_pKa=None if pd.isna(self.D.pKa[i]) else float(self.D.pKa[i]),
                    n_candidates=int(len(g)), recommendations=rows,
                    caveat='Scores compare only within one drug. The model predicts which pairs the literature '
                           'reports, not solubility, toxicity or bioavailability.')

    def _recommend_new(self, smiles, pKa, top):
        # A description is not a molecule. Two of the four features are similarity to specific known
        # molecules, so there is nothing to compute from "an acidic drug with an aromatic ring".
        # Catch that here rather than letting it reach the SMILES parser.
        if ' ' in smiles.strip() or not re.search(r'[A-Za-z]', smiles):
            return dict(ok=False, error=f'{smiles!r} is neither one of the 138 known drug names nor a SMILES string',
                        hint='ask the user for the drug name or its SMILES. The model scores one specific '
                             'molecule against specific known molecules; it cannot score a description of '
                             'properties such as "acidic, pKa 4.2, aromatic ring".')
        try:
            import recommend_new_drug as rnd
        except ImportError as e:
            return dict(ok=False, error=f'{smiles!r} is not among the 138 known drugs, and scoring a new '
                                        f'molecule needs RDKit, which is not installed ({e})',
                        hint='pip install -r requirements.txt')
        cwd = os.getcwd()
        try:
            os.chdir(self.dir)
            table, meta = rnd.recommend(smiles, name='new drug', pKa=pKa, top=top)
        except ValueError as e:
            return dict(ok=False, error=str(e),
                        hint='give the charged form that would sit in the salt: a deprotonated acid ([O-]) '
                             'or a protonated base ([NH+])')
        finally:
            os.chdir(cwd)
        rows = []
        for r in table.head(top).to_dict('records'):
            rows.append(dict(rank=int(r['rank']), ion=r['ion_name'] or r['ion_smiles'][:28], ion_smiles=r['ion_smiles'],
                             score=round(float(r['score']), 2), already_reported=False, pKa_label=r['pKa_label'],
                             dpKa=None if pd.isna(r['dpKa_value']) else round(float(r['dpKa_value']), 2),
                             sim2D=round(float(r['sim2D']), 3), sim3D=round(float(r['sim3D']), 3),
                             safety=str(r['safety']), reason=r['reason']))
        return dict(ok=True, drug=meta['name'], smiles=meta['smiles'], role=meta['role'],
                    source='new molecule, scored against all 138 known drugs',
                    path='slow: RDKit conformers + fit', drug_pKa=meta['pKa'],
                    n_candidates=meta['n_candidates'], recommendations=rows,
                    caveat=meta.get('esp_note', '') + ' Scores compare only within one drug.')

    # ---------------------------------------------------------------- tool 2

    def evidence(self, drug, ion):
        """Verbatim literature support for one pair. Never paraphrases, never invents."""
        i, name = self.resolve_drug(drug)
        j = self._ion_index(ion)
        if i is None:
            return dict(found=False, reason=f'{drug!r} is not one of the 138 drugs in the dataset')
        if j is None:
            return dict(found=False, reason=f'{ion!r} is not one of the 291 counter-ions in the dataset')
        hits = self._key.get((self._k(self.D.smiles[i]), self._k(self.I.smiles[j])), [])
        if not hits:
            reported = j in set(self.POS[self.POS.drug_idx == i].ion_idx)
            return dict(found=False, drug=name, ion=str(self.I.name[j]),
                        reason=('the pair is in the dataset but its quote could not be keyed '
                                f'(match method: {self.match_method})') if reported
                               else 'no reported pairing of this drug and this ion in the 652-pair dataset',
                        is_reported_pair=reported)
        r = hits[0]
        return dict(found=True, drug=name, ion=str(self.I.name[j]),
                    evidence_level=str(r.evidence_level), quote=str(r.proof_quote), citation=str(r.proof_ref),
                    stoichiometry=None if pd.isna(r.stoichiometry) else str(r.stoichiometry),
                    match_method=self.match_method,
                    note='quote reproduced verbatim from the source named in `citation`')

    # ---------------------------------------------------------------- tool 3

    def explain(self, drug, ion):
        """Plain-English reason for one suggestion, written by reason.make_reason -- the same function
        behind the web app, so the chat answer and the app cannot drift apart."""
        i, name = self.resolve_drug(drug)
        if i is None:
            return dict(ok=False, error=f'{drug!r} is not one of the 138 drugs in the dataset')
        j = self._ion_index(ion)
        if j is None:
            return dict(ok=False, error=f'{ion!r} is not one of the 291 counter-ions in the dataset')
        g = self.SCORES[(self.SCORES.drug == i) & (self.SCORES.ion == j)]
        if not len(g):
            return dict(ok=False, error=f'{self.I.name[j]} is not a reachable candidate for {name} '
                                        '(no other drug in the dataset uses it, so it cannot be ranked)')
        r = g.iloc[0]
        disp = str(r['ion_name']) if isinstance(r['ion_name'], str) else r['ion_smiles'][:28]
        users = [k for k in self._users.get(j, []) if k != i]
        sim_users, n_users = self._similar_users(i, users)
        card = reason.make_reason(
            name, disp, int(r['rank']), int((self.SCORES.drug == i).sum()), float(r['score']),
            r['pKa_label'], None if pd.isna(r['dpKa_value']) else float(r['dpKa_value']),
            [float(r['c_2D similarity']), float(r['c_3D similarity']), float(r['c_pKa']), float(r['c_ESP complementarity'])],
            sim_users, n_users, self._safety.get(j, 'unclassified'), bool(r['is_reported_pair']))
        return dict(ok=True, drug=name, ion=disp, rank=int(r['rank']), score=round(float(r['score']), 2),
                    chemistry=card['chem'], safety=card['safe'], full=card['full'],
                    best_structural_match=round(float(r['sim2D']), 3),
                    other_drugs_using_this_ion=[str(self.D.name[k]) for k in users])

    # ---------------------------------------------------------------- tool 4

    @property
    def index(self):
        """BM25 index over all 1,628 passages. Built once, ~90 ms, then ~2 ms a query."""
        if self._index is None:
            from rag.index import Index
            self._index = Index.build()
        return self._index

    def search_literature(self, query, k=5, kind=None, drug=None, ion=None):
        """Free-text search across every evidence source in the project.

        Hybrid on purpose. `evidence` answers "has this exact pair been made" by key, which is precise but
        only answers that one question. This answers everything else -- measured solubilities, patents,
        safety records, pKa provenance, the hand-written reasons a suggestion was rejected -- and returns
        passages with their citations rather than a synthesised answer.
        """
        if drug:
            i, nm = self.resolve_drug(drug)
            if i is not None:
                drug = nm
        hits = self.index.search(query, k=k, kind=kind, drug=drug, ion=ion)
        if not hits:
            return dict(found=False, query=query,
                        reason='nothing in the project corpus matched that query',
                        searched=len(self.index.docs), filters=dict(kind=kind, drug=drug, ion=ion))
        return dict(found=True, query=query, searched=len(self.index.docs), n=len(hits),
                    passages=[dict(text=h['text'], citation=h['citation'], kind=h['kind'],
                                   source_file=h['source_file'], relevance=h['score'],
                                   drug=h['drug'] or None, ion=h['ion'] or None) for h in hits],
                    note='passages are reproduced from the project data, not generated. Quote them or cite '
                         'them; do not extend them.')

    def _similar_users(self, i, users):
        """(name, sim2D) for the drugs that use this ion, best first. Needs the similarity matrix, so
        without RDKit the names are returned with no similarity and reason.make_reason falls back to its
        'widely used counter-ion' wording rather than quoting a number it does not have."""
        if not users:
            return [], 0
        try:
            if self._F is None:
                import features
                self._F = features.load(self.dir)
            S2 = self._F.S2
            ranked = sorted(users, key=lambda k: -S2[i, k])
            return [(str(self.D.name[k]), float(S2[i, k])) for k in ranked[:3]], len(users)
        except ImportError:
            return [], len(users)


TOOLS = [
    dict(name='recommend_counterions',
         description='Rank the counter-ions most likely to form a salt or ionic liquid with a drug. Accepts one '
                     'of the 138 known drug names, or a charged SMILES for any other molecule. Returns ranked '
                     'ions with scores and the four contributions behind each score.',
         input_schema=dict(type='object', properties=dict(
             drug=dict(type='string', description='drug name, or the charged SMILES of a new molecule'),
             pKa=dict(type='number', description='measured pKa of the drug; only used for a new SMILES'),
             top=dict(type='integer', description='how many to return (default 10)')), required=['drug'])),
    dict(name='get_evidence',
         description='Return the verbatim literature quote and citation for one drug/counter-ion pair from the '
                     '652-pair dataset. Returns found=false when there is no record. Use this before stating '
                     'that a pair has been made.',
         input_schema=dict(type='object', properties=dict(
             drug=dict(type='string'), ion=dict(type='string')), required=['drug', 'ion'])),
    dict(name='search_literature',
         description='Free-text search over every evidence source in the project: the 652 reported pairs with '
                     'their quotes, measured solubility and toxicity values, patent records, FDA GRAS/SCOGS '
                     'safety status, measured melting points and LD50s, drug pKa provenance, and written '
                     'assessments of unreported suggestions. Returns passages with citations. Use it for any '
                     'question get_evidence cannot key, such as "how safe is this ion" or "what solubility gain '
                     'has been measured". Optional filters narrow the pool before ranking.',
         input_schema=dict(type='object', properties=dict(
             query=dict(type='string', description='natural-language query'),
             k=dict(type='integer', description='how many passages (default 5)'),
             kind=dict(type='string', enum=['reported_pair', 'reported_pair_fulltext', 'measured_property',
                                            'measured_single', 'patent', 'ion_safety', 'theory_note',
                                            'pka_source', 'pka_method'],
                       description='restrict to one evidence type'),
             drug=dict(type='string', description='restrict to one drug'),
             ion=dict(type='string', description='restrict to one counter-ion')), required=['query'])),
    dict(name='explain_recommendation',
         description='Plain-English reason the model ranked one counter-ion where it did, including which '
                     'similar drugs already use it and what the pKa difference implies.',
         input_schema=dict(type='object', properties=dict(
             drug=dict(type='string'), ion=dict(type='string')), required=['drug', 'ion'])),
]

_R = None


def dispatch(name, args):
    """Route one tool call. The only entry point the chat loop needs."""
    global _R
    if _R is None:
        _R = Recommender()
    if name == 'recommend_counterions':
        return _R.recommend(args['drug'], args.get('pKa'), args.get('top', 10))
    if name == 'get_evidence':
        return _R.evidence(args['drug'], args['ion'])
    if name == 'search_literature':
        return _R.search_literature(args['query'], args.get('k', 5), args.get('kind'),
                                    args.get('drug'), args.get('ion'))
    if name == 'explain_recommendation':
        return _R.explain(args['drug'], args['ion'])
    return dict(error=f'no such tool: {name}')


SYSTEM = """You help formulation chemists choose a counter-ion for a drug.

You have no chemistry knowledge of your own in this conversation. Every claim you make must come from a tool result:
- ranks, scores and pKa labels come from recommend_counterions
- the reason a given ion ranks where it does comes from explain_recommendation
- whether a pair has actually been made comes from get_evidence, quoted verbatim with its citation
- anything else -- safety status, measured solubility, patents, why a suggestion was rejected -- comes from
  search_literature, and you quote or cite its passages rather than extending them

Rules:
- Never state that a pair exists in the literature unless get_evidence returned found=true. If it returned
  found=false, say that the dataset has no record of it. Do not reason from memory about whether it is likely.
- Never invent a score, a rank or a pKa.
- Scores are comparable only within one drug. Never compare scores across drugs.
- If the user describes a drug by its properties instead of naming it or giving a SMILES, ask for the name or
  SMILES. The model scores a specific molecule against specific known molecules; it cannot score a description.
- The model predicts which pairings the literature reports. It does not predict solubility, toxicity or
  bioavailability. Say so if the user asks for those.
- Be brief. Give the shortlist, then the reason for the top one or two, then the evidence status."""
