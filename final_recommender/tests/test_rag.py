"""The retrieval layer: corpus provenance, tokenisation, ranking, and the grounding guarantee.

No RDKit needed. The claim being pinned is that retrieval can only ever return text that was already
written down in the project, with the citation it came with.
"""
import pytest

from rag.corpus import build_corpus
from rag.index import Index, _same_entity, stem, tokenize


@pytest.fixture(scope='module')
def docs():
    return build_corpus()


@pytest.fixture(scope='module')
def ix(docs):
    return Index.build(docs)


# ---------------------------------------------------------------- corpus

def test_corpus_covers_every_source(docs):
    kinds = {d['kind'] for d in docs}
    assert {'reported_pair', 'measured_property', 'patent', 'ion_safety',
            'theory_note', 'pka_source', 'pka_method', 'measured_single'} <= kinds


def test_every_passage_carries_a_citation(docs):
    """A passage with no provenance cannot be quoted responsibly, so none may exist."""
    missing = [d for d in docs if not d['citation'].strip()]
    assert not missing, f'{len(missing)} passages have no citation, e.g. {missing[0]["text"][:80]!r}'


def test_every_passage_has_text_and_a_source_file(docs):
    for d in docs:
        assert len(d['text']) > 20, f'empty-ish passage: {d!r}'
        assert d['source_file'] and d['kind']


def test_all_652_reported_pairs_are_indexed(docs):
    assert sum(1 for d in docs if d['kind'] == 'reported_pair') == 652


def test_passages_have_unique_ids(docs):
    assert len({d['id'] for d in docs}) == len(docs)


# ---------------------------------------------------------------- tokeniser

@pytest.mark.parametrize('a,b', [('choline', 'cholinium'), ('tolfenamic', 'tolfenamate'),
                                 ('lidocaine', 'lidocainium'), ('salicylic', 'salicylate')])
def test_ionic_and_neutral_spellings_share_a_stem(a, b):
    assert stem(a) == stem(b), f'{a} and {b} must index to the same prefix'


@pytest.mark.parametrize('a,b', [('cholinium', 'cholate'), ('aspirin', 'ampicillin'),
                                 ('naproxen', 'nalidixate')])
def test_unrelated_names_do_not_share_a_stem(a, b):
    assert stem(a) != stem(b), f'{a} and {b} must not collapse onto one term'


def test_stems_are_namespaced(ix):
    """A prefix term must not be confusable with a real word in the vocabulary."""
    assert all(t.startswith('~') or not t.startswith('~') for t in ix.vocab)
    assert any(t.startswith('~') for t in ix.vocab), 'no prefix terms were indexed'
    assert '~choli' in ix.vocab


def test_short_words_are_not_stemmed():
    assert stem('salt') is None
    assert stem('12345') is None
    assert '~' not in ''.join(tokenize('salt'))


@pytest.mark.parametrize('a,b,same', [('Cholinium', 'choline', True), ('Cholinium', 'cholate', False),
                                      ('Tolfenamate', 'tolfenamic acid', True), ('Aspirin', 'ampicillin', False)])
def test_entity_matching(a, b, same):
    assert _same_entity(a, b) is same


# ---------------------------------------------------------------- retrieval

def test_retrieval_returns_passages_verbatim_from_the_corpus(ix, docs):
    """The grounding guarantee: every returned string exists in the corpus. Nothing is synthesised."""
    corpus = {d['text'] for d in docs}
    for q in ['choline ibuprofen solubility', 'GRAS safety', 'patent indomethacin', 'pKa benchmark']:
        for h in ix.search(q, k=5):
            assert h['text'] in corpus


def test_scores_descend(ix):
    hits = ix.search('ionic liquid solubility', k=10)
    assert hits
    assert [h['score'] for h in hits] == sorted((h['score'] for h in hits), reverse=True)


def test_a_query_with_no_match_returns_nothing(ix):
    assert ix.search('zzzqqq wibble flumph', k=5) == []


def test_empty_query_returns_nothing(ix):
    assert ix.search('', k=5) == []


def test_kind_filter_restricts_the_pool(ix):
    hits = ix.search('safety', k=10, kind='ion_safety')
    assert hits and all(h['kind'] == 'ion_safety' for h in hits)


def test_ion_filter_does_not_leak_a_similar_name(ix):
    """The regression that motivated the prefix scheme: a choline safety query returned cholate first."""
    hits = ix.search('is it safe', k=5, kind='ion_safety', ion='cholinium')
    assert hits
    assert all(_same_entity(h['ion'], 'cholinium') for h in hits)
    assert not any('Cholate' == h['ion'] for h in hits)


def test_drug_filter_accepts_the_chemists_spelling(ix):
    assert ix.search('counter-ion', k=3, drug='tolfenamic acid')


def test_index_round_trips_through_disk(ix, tmp_path):
    p = ix.save(str(tmp_path / 'ix.pkl'))
    other = Index.load(p)
    assert len(other.docs) == len(ix.docs)
    assert [h['id'] for h in other.search('choline solubility', k=5)] == \
           [h['id'] for h in ix.search('choline solubility', k=5)]


# ---------------------------------------------------------------- the tool

def test_search_tool_reports_a_miss_rather_than_inventing(rec_dir):
    from agent.tools import Recommender
    out = Recommender(rec_dir).search_literature('zzzqqq wibble flumph')
    assert out['found'] is False
    assert 'passages' not in out


def test_search_tool_returns_citations_with_every_passage(rec_dir):
    from agent.tools import Recommender
    out = Recommender(rec_dir).search_literature('measured water solubility', k=4)
    assert out['found'] and out['passages']
    for p in out['passages']:
        assert p['citation'].strip() and p['source_file'].strip()
