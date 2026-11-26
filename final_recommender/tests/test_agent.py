"""The grounding guarantees of the tool layer. No RDKit needed: every test uses the fast path.

These are the claims the agent makes about itself, turned into assertions. If one fails, the chat
layer can say something the model and the dataset do not support.
"""
import pytest

from agent.tools import Recommender, describe, dispatch


@pytest.fixture(scope='module')
def R(rec_dir):
    return Recommender(rec_dir)


# ---------------------------------------------------------------- resolution

@pytest.mark.parametrize('query,expected', [
    ('ibuprofen', 'Ibuprofen'),
    ('Ibuprofen', 'Ibuprofen'),
    ('tolfenamic acid', 'Tolfenamate'),      # dataset stores the anion
    ('mefenamic acid', 'Mefenamate'),
    ('lidocaine', 'Lidocainium'),            # dataset stores the cation
    ('cinnarizine', 'Cinnarizine'),
])
def test_chemists_spelling_resolves_to_the_dataset_name(R, query, expected):
    _, name = R.resolve_drug(query)
    assert name == expected


def test_a_description_is_not_treated_as_a_molecule(R):
    """Two of the four features are similarity to specific known molecules, so a property description
    cannot be scored. It must be refused, never passed to the SMILES parser."""
    out = R.recommend('an acidic drug with pKa 4.2 and a hydrophobic aromatic ring')
    assert out['ok'] is False
    assert 'SMILES' in out['error'] or 'smiles' in out['error']
    assert 'description' in out.get('hint', '')


def test_an_unknown_name_does_not_silently_match_something_else(R):
    i, _ = R.resolve_drug('zzzznotadrug')
    assert i is None


# ---------------------------------------------------------------- recommend

def test_known_drug_uses_the_precomputed_path(R):
    out = R.recommend('ibuprofen', top=5)
    assert out['ok'] and out['drug'] == 'Ibuprofen'
    assert 'no RDKit' in out['path']
    assert len(out['recommendations']) == 5
    assert [r['rank'] for r in out['recommendations']] == [1, 2, 3, 4, 5]


def test_reported_pairs_are_flagged_as_such(R):
    out = R.recommend('ibuprofen', top=10)
    assert any(r['already_reported'] for r in out['recommendations']), \
        'the flag that separates a precedent from a suggestion is missing'


def test_every_recommendation_carries_its_four_contributions(R):
    out = R.recommend('cinnarizine', top=5)
    for r in out['recommendations']:
        assert set(r['contributions']) == {'structure_2D', 'shape_3D', 'pKa', 'electrostatics'}


def test_the_scope_caveat_rides_on_every_answer(R):
    """The model predicts reported pairings, not solubility. The caveat must not be optional."""
    out = R.recommend('aspirin', top=3)
    assert 'solubility' in out['caveat']
    assert 'only within one drug' in out['caveat']


def test_unnamed_ions_get_a_class_not_a_bare_smiles(R):
    """87 of the 291 ions have no name; a chat answer must not print a raw SMILES on its own."""
    out = R.recommend('tolfenamic acid', top=10)
    for r in out['recommendations']:
        if r['name_missing']:
            assert r['ion'].startswith('unnamed')
            assert r['ion'] != r['ion_smiles']


@pytest.mark.parametrize('smiles,expected', [
    ('C[N+](C)(CCO)CCO', 'quaternary ammonium'),
    ('CCCCCCCCCCCCCCCC[n+]1ccccc1', 'pyridinium'),
    ('CC(=O)[O-]', 'carboxylate'),
    ('CS(=O)(=O)[O-]', 'sulfonate'),
])
def test_ion_class_labels(smiles, expected):
    assert expected in describe(smiles)


# ---------------------------------------------------------------- evidence

def test_a_reported_pair_returns_a_verbatim_quote_and_a_citation(R):
    e = R.evidence('Ibuprofen', 'Cholinium')
    assert e['found'] is True
    assert e['quote'].strip()
    assert any(tag in e['citation'] for tag in ('DOI', 'PMID', 'PMC'))


def test_an_unreported_pair_returns_found_false_and_nothing_else(R):
    """The anti-hallucination guarantee: no quote, no citation, no hedged prose -- just 'no record'."""
    e = R.evidence('Ibuprofen', 'Lidocainium')
    assert e['found'] is False
    assert 'quote' not in e and 'citation' not in e
    assert e['is_reported_pair'] is False


def test_evidence_refuses_unknown_names_instead_of_guessing(R):
    assert R.evidence('zzzznotadrug', 'Cholinium')['found'] is False
    assert R.evidence('Ibuprofen', 'zzzznotanion')['found'] is False


def test_evidence_resolves_most_of_the_dataset(R):
    """The model tables store flat SMILES and the evidence table keeps stereochemistry, so the join needs
    a stereo-insensitive key. Without one it silently drops every chiral molecule."""
    found = sum(R.evidence(str(R.D.name[r.drug_idx]), str(R.I.name[r.ion_idx]))['found']
                for r in R.POS.head(120).itertuples() if isinstance(R.I.name[r.ion_idx], str))
    assert found >= 60, f'only {found} of the first ~120 reported pairs resolved to a quote'


# ---------------------------------------------------------------- explain

def test_explanation_comes_from_reason_py(R):
    """The chat answer and the web app must not drift: both call reason.make_reason."""
    import reason
    out = R.explain('Ibuprofen', 'Lidocainium')
    assert out['ok'] and out['rank'] == 4
    assert out['chemistry'] in out['full']
    assert out['safety'] in reason.SAFE_PHRASE.values()


def test_explain_refuses_an_unreachable_candidate(R):
    """An ion no other drug uses cannot be ranked for this drug, and the tool must say so."""
    bad = R.explain('Ibuprofen', 'zzzznotanion')
    assert bad['ok'] is False


# ---------------------------------------------------------------- dispatch

def test_dispatch_routes_every_tool():
    assert dispatch('recommend_counterions', dict(drug='ibuprofen', top=3))['ok']
    assert dispatch('get_evidence', dict(drug='Ibuprofen', ion='Cholinium'))['found']
    assert dispatch('explain_recommendation', dict(drug='Ibuprofen', ion='Cholinium'))['ok']
    assert dispatch('search_literature', dict(query='measured solubility'))['found']
    assert 'error' in dispatch('no_such_tool', {})


def test_tool_schemas_match_the_dispatcher():
    from agent.tools import TOOLS
    assert {t['name'] for t in TOOLS} == {'recommend_counterions', 'get_evidence',
                                          'search_literature', 'explain_recommendation'}
    for t in TOOLS:
        assert t['description'] and t['input_schema']['required']
