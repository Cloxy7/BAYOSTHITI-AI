"""BM25 retrieval over the passage corpus.

Lexical, not dense, on purpose. The queries here are chemical names and the corpus is 1,628 short
passages full of them; an exact match on "tolfenamate" matters more than semantic neighbourhood, and
a sparse index needs no model, no GPU and no network. The one piece of domain knowledge in it is the
tokeniser, which indexes a chemical stem beside each word so that choline/cholinium and
tolfenamic/tolfenamate land on the same term.

    from rag.index import Index
    ix = Index.build()
    ix.search('choline ibuprofen solubility', k=5)
"""
import math
import pickle
import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from .corpus import build_corpus

K1, B = 1.5, 0.75
_WORD = re.compile(r'[a-z0-9]+')
STEM_MIN, STEM_LEN = 7, 5   # words of 7+ letters also index their 5-letter prefix


def stem(word):
    """A 5-letter prefix, which is what bridges the ionic and neutral spellings of a chemical name:
    choline/cholinium -> choli, tolfenamic/tolfenamate -> tolfe, lidocaine/lidocainium -> lidoc.

    Suffix stripping was tried first and collapsed cholinium and cholate onto the same term, which put
    cholate at the top of a safety query about choline. A fixed prefix keeps those apart."""
    return word[:STEM_LEN] if len(word) >= STEM_MIN and not word.isdigit() else None


def tokenize(text):
    out = []
    for w in _WORD.findall(str(text).lower()):
        if len(w) < 2:
            continue
        out.append(w)
        s = stem(w)
        if s:
            out.append('~' + s)     # namespaced so a prefix can never collide with a real word
    return out


def _same_entity(a, b):
    """Is this the same drug or ion, written two ways? Exact token overlap, or a shared 6-letter prefix
    on a word of 7+ letters. Deliberately stricter than the retrieval tokeniser."""
    aw = [w for w in _WORD.findall(str(a).lower()) if len(w) > 2]
    bw = [w for w in _WORD.findall(str(b).lower()) if len(w) > 2]
    if set(aw) & set(bw):
        return True
    return any(x[:6] == y[:6] for x in aw for y in bw if len(x) >= 7 and len(y) >= 7)


class Index:
    def __init__(self, docs, vocab, matrix, idf, lengths):
        self.docs, self.vocab, self.X, self.idf, self.len = docs, vocab, matrix, idf, lengths
        self.avg = float(lengths.mean()) if len(lengths) else 0.0

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, docs=None):
        docs = docs if docs is not None else build_corpus()
        toks = [tokenize(d['text']) for d in docs]
        vocab = {}
        for t in toks:
            for w in t:
                vocab.setdefault(w, len(vocab))
        rows, cols, vals = [], [], []
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                rows.append(i); cols.append(vocab[w]); vals.append(c)
        X = csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)), dtype=np.float32)
        n = len(docs)
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        lengths = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        return cls(docs, vocab, X.tocsc(), idf, lengths)

    # ---------------------------------------------------------------- search

    def search(self, query, k=5, kind=None, drug=None, ion=None):
        """Top-k passages. kind/drug/ion narrow the pool before scoring, which is what makes a question
        like 'is this ion safe' retrieve safety records rather than whichever passage mentions it most."""
        qt = [t for t in tokenize(query) if t in self.vocab]
        if not qt:
            return []
        scores = np.zeros(len(self.docs), dtype=np.float32)
        norm = K1 * (1 - B + B * self.len / max(self.avg, 1e-9))
        for w, qc in Counter(qt).items():
            col = self.X[:, self.vocab[w]]
            idx, tf = col.indices, col.data
            scores[idx] += self.idf[self.vocab[w]] * (tf * (K1 + 1) / (tf + norm[idx])) * (1 + math.log(qc))
        mask = self._mask(kind, drug, ion)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        hits = np.argsort(-scores)[:k]
        return [dict(score=round(float(scores[i]), 3), **{j: v for j, v in self.docs[i].items()})
                for i in hits if scores[i] > 0]

    def _mask(self, kind, drug, ion):
        if kind is None and drug is None and ion is None:
            return None
        m = np.ones(len(self.docs), dtype=bool)
        if kind:
            kinds = {kind} if isinstance(kind, str) else set(kind)
            m &= np.array([d['kind'] in kinds for d in self.docs])
        for field, want in (('drug', drug), ('ion', ion)):
            if want:
                m &= np.array([bool(d[field]) and _same_entity(d[field], want) for d in self.docs])
        return m

    # ---------------------------------------------------------------- persistence

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(docs=self.docs, vocab=self.vocab, X=self.X, idf=self.idf, len=self.len), f)
        return path

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return cls(d['docs'], d['vocab'], d['X'], d['idf'], d['len'])


if __name__ == '__main__':
    import sys, time
    t0 = time.perf_counter()
    ix = Index.build()
    print(f'{len(ix.docs)} passages, {len(ix.vocab)} terms, built in {(time.perf_counter() - t0) * 1000:.0f} ms\n')
    q = ' '.join(sys.argv[1:]) or 'choline ibuprofen solubility'
    t0 = time.perf_counter()
    hits = ix.search(q, k=5)
    print(f'query: {q!r}   [{(time.perf_counter() - t0) * 1000:.1f} ms]\n')
    for h in hits:
        print(f"  [{h['score']:>6}] ({h['kind']}) {h['text'][:150]}")
        print(f"           cite: {h['citation'][:100]}\n")
